from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys
from typing import Callable


@dataclass(frozen=True)
class NCMLossSelection:
    loss: Callable
    implementation: str
    reason: str
    pooling: str

    def __call__(self, solutions, prefixes):
        return self.loss(solutions, prefixes)


def _original_type(value, module_name, class_name, filename):
    module = sys.modules.get(module_name)
    source = getattr(module, "__file__", None)
    expected = Path(__file__).resolve().parent / "vendor/ncm/models" / filename
    return (module is not None and source is not None
            and Path(source).resolve() == expected.resolve()
            and type(value) is getattr(module, class_name, None))


def _unsupported_reason(model, tokenizer, solution_length, prefix_length):
    if solution_length != 89 or prefix_length != 91:
        return "Frozen target reuse requires 89 solution tokens and 91 prefix tokens."
    if not _original_type(tokenizer, "models.vqvae", "VQVAE", "vqvae.py"):
        return "Frozen target reuse requires the vendored original VQVAE implementation."
    if not _original_type(model, "models.next_concept", "NextConceptTransformer", "next_concept.py"):
        return "Frozen target reuse requires the vendored original NextConceptTransformer implementation."
    vq = tokenizer.vq
    if not _original_type(vq, "models.quant", "MultiscaleResidualVectorQuantizer", "quant.py"):
        return "Frozen target reuse requires the original residual quantizer implementation."
    from ncm_fast_targets import LEVELS

    if (tuple(vq.levels) != LEVELS or vq.embed_dim != 256
            or tuple(vq.vqvae_vocab_sizes) != (16,) * len(LEVELS)
            or tokenizer.embed_dim != 256 or tokenizer.context_length != 89):
        return "The tokenizer does not use the verified original 322-code hierarchy."
    if (any(module.training for module in tokenizer.modules())
            or any(parameter.requires_grad for parameter in tokenizer.parameters())
            or getattr(vq, "_stoch_encode", False)):
        return "Frozen target reuse requires an entirely frozen, eval-mode, deterministic tokenizer."
    if (tuple(model.levels) != LEVELS or model.prefix_len != 91
            or not model.query_level0 or model.l0_ar or model.predict_level0
            or model.use_ctx_encoder or model.use_post_upsample_conv
            or model.shared_output_head or model.vqvae_local[0] is not tokenizer
            or tuple(model.vocab_sizes) != (16,) * len(LEVELS)):
        return "The generator does not use the verified prefix and level-query contract."
    config = model.config
    required = {"n_layer": 8, "n_head": 8, "n_embd": 512, "dropout": 0.1,
                "bias": True, "attn_pattern": "block_diagonal", "use_rope": True,
                "use_level_encoding": True, "use_qk_norm": True,
                "raw_token_prefix": True, "tie_heads_to_codebook": False}
    if any(getattr(config, key, None) != value for key, value in required.items()):
        return "The generator configuration differs from the verified original configuration."
    import torch

    parameters = list(tokenizer.parameters()) + list(model.parameters())
    if not parameters or any(parameter.device.type != "mps" or parameter.dtype != torch.float32
                             for parameter in parameters):
        return "Frozen target reuse is verified for colocated FP32 MPS model and tokenizer parameters."
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        return "The tokenizer and generator must be on one device."
    if any(buffer.device != device or (buffer.is_floating_point() and buffer.dtype != torch.float32)
           for buffer in tokenizer.buffers()):
        return "The tokenizer buffers must use the same device and FP32 floating-point format."
    return None


class _ForwardOnlyModel:
    def __init__(self, model, forward_context):
        self.model = model
        self.forward_context = forward_context

    def build_l0_query_input(self, batch_size):
        return self.model.build_l0_query_input(batch_size)

    def __call__(self, inputs):
        with self.forward_context():
            logits = self.model(inputs)
        return logits.float()


def select_ncm_loss(model, tokenizer, reference_loss: Callable, *, solution_length: int,
                    prefix_length: int, pooling: str = "reference",
                    forward_context: Callable | None = None,
                    objective: str = "task") -> NCMLossSelection:
    """Bind a task's reference loss or the narrow original-NCM frozen target path.

    The reference callable receives model, tokenizer, solutions, and prefixes.
    Selection performs no forward pass, encode, sampling, or RNG operation.
    The optional context wraps only a fast-path generator forward. The reference
    callable owns its forward precision context. A fast objective must be explicitly
    declared as mean-ce-all-codes, without masking, weights, or auxiliary terms.
    """
    if not callable(reference_loss):
        raise TypeError("Provide a task-owned callable reference loss.")
    if pooling not in ("reference", "metal"):
        raise ValueError("Frozen pooling must be reference or metal.")
    if objective not in ("task", "mean-ce-all-codes"):
        raise ValueError("Objective must be task or mean-ce-all-codes.")
    if type(solution_length) is not int or solution_length < 1 or type(prefix_length) is not int or prefix_length < 0:
        raise ValueError("Solution length must be positive and prefix length must be nonnegative integers.")
    reason = _unsupported_reason(model, tokenizer, solution_length, prefix_length)
    if reason is None and objective != "mean-ce-all-codes":
        reason = "Task-owned loss retained. Frozen target reuse requires explicit objective='mean-ce-all-codes'."
    if reason is not None:
        return NCMLossSelection(partial(reference_loss, model, tokenizer), "reference", reason, "reference")
    from ncm_fast_targets import fast_generator_loss

    selected_model = model if forward_context is None else _ForwardOnlyModel(model, forward_context)
    return NCMLossSelection(partial(fast_generator_loss, selected_model, tokenizer, pooling_kernel=pooling),
                            "frozen-targets", "The original MPS NCM hierarchy satisfies the verified frozen-target contract. "
                            "Metal pooling still requires its guarded tensor shapes and strides.", pooling)

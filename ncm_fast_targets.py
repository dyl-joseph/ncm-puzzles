"""Build frozen NCM training codes and contexts in one residual pass."""

import torch
import torch.nn.functional as F

from ncm_torch_ops import NCMMPSSafeOps, _arguments
from ncm_area_pool import POOLING_KERNELS, area89, supported_area89


LEVELS = (1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89)


class NCMFastMPSSafeOps(NCMMPSSafeOps):
    def __init__(self, devices=("mps",), *, pooling_kernel="reference"):
        if pooling_kernel not in POOLING_KERNELS:
            raise ValueError("Frozen pooling kernel must be reference or metal.")
        super().__init__(devices)
        self.pooling_kernel = pooling_kernel

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.interpolate:
            values = _arguments(
                args, kwargs,
                ("input", "size", "scale_factor", "mode", "align_corners",
                 "recompute_scale_factor", "antialias"),
                (None, None, None, "nearest", None, None, False),
            )
            if values is not None:
                x, size = values["input"], values["size"]
                if isinstance(size, (tuple, list)) and len(size) == 1:
                    size = size[0]
                if (isinstance(x, torch.Tensor) and x.ndim == 3
                        and x.device.type in self.devices
                        and all(dimension > 0 for dimension in x.shape)
                        and values["mode"] == "area"
                        and type(size) is int and size > 0
                        and values["scale_factor"] is None
                        and values["align_corners"] is None
                        and values["recompute_scale_factor"] in (None, False)
                        and values["antialias"] is False):
                    if size == x.shape[-1]:
                        return x
                    if self.pooling_kernel == "metal" and supported_area89(x, size):
                        return area89(x, size)
        return super().__torch_function__(func, types, args, kwargs)


def _validate_frozen_tokenizer(tokenizer, solutions):
    if (solutions.ndim != 2 or solutions.shape[0] < 1 or solutions.shape[1] != 89
            or solutions.dtype != torch.long):
        raise ValueError("Frozen NCM targets require nonempty int64 [batch, 89] solutions.")
    if any(module.training for module in tokenizer.modules()):
        raise ValueError("Fast NCM targets require an entirely eval-mode tokenizer.")
    if any(parameter.requires_grad for parameter in tokenizer.parameters()):
        raise ValueError("Fast NCM targets require a frozen tokenizer.")
    vq = tokenizer.vq
    if (tuple(vq.levels) != LEVELS or vq.embed_dim != 256
            or tuple(vq.vqvae_vocab_sizes) != (16,) * len(LEVELS)):
        raise ValueError("Fast NCM targets support the original 322-code Sudoku hierarchy.")
    if getattr(vq, "_stoch_encode", False):
        raise ValueError("Fast NCM targets require deterministic encoding.")
    if any(parameter.dtype != torch.float32 or parameter.device != solutions.device
           for parameter in tokenizer.parameters()):
        raise ValueError("Fast NCM targets require float32 tokenizer parameters on the input device.")


@torch.no_grad()
def frozen_targets_and_context(tokenizer, solutions):
    """Return exact [B, 322] codes and [B, 321, 256] teacher contexts."""
    _validate_frozen_tokenizer(tokenizer, solutions)
    vq = tokenizer.vq
    z_rest = tokenizer._encode_pre_quant(solutions).float().detach().clone()
    batch, length, channels = z_rest.shape
    if (length, channels) != (89, 256):
        raise ValueError("The frozen encoder must produce [batch, 89, 256] vectors.")
    # Match idx_to_ncm_input's contiguous layout to preserve reduction order.
    z_hat = torch.zeros(batch, length, channels, device=z_rest.device)
    codes, contexts = [], []
    for level_index, level in enumerate(LEVELS):
        if level_index == len(LEVELS) - 1:
            flat_input = z_rest.reshape(-1, channels)
        else:
            flat_input = vq._down_for_quant(z_rest, level_index, level).reshape(-1, channels)
        distances = vq._distances(flat_input, vq.embeddings[level_index].weight)
        indices = torch.argmin(distances, dim=1).view(batch, level)
        codes.append(indices)
        if level_index == len(LEVELS) - 1:
            break
        quantized = vq._up_to_full(vq.embeddings[level_index](indices), level_index, length)
        quantized = vq._apply_quant_resi(quantized, level_index)
        z_hat.add_(quantized)
        z_rest.sub_(quantized)
        contexts.append(F.interpolate(
            z_hat.transpose(-1, -2), size=(LEVELS[level_index + 1],), mode="area"
        ).transpose(-1, -2))
    return torch.cat(codes, dim=-1), torch.cat(contexts, dim=1)


def fast_generator_loss(model, tokenizer, solutions, prefixes, *, pooling_kernel="reference"):
    with torch.no_grad(), NCMFastMPSSafeOps(pooling_kernel=pooling_kernel):
        targets, contexts = frozen_targets_and_context(tokenizer, solutions)
        prefix_embeddings = tokenizer.encoder.wte(prefixes)
    query = model.build_l0_query_input(solutions.shape[0])
    inputs = torch.cat((prefix_embeddings, query, contexts), dim=1)
    if inputs.shape[1:] != (413, 256) or targets.shape != (solutions.shape[0], 322):
        raise ValueError("The original conditional hierarchy must contain all 322 target codes.")
    logits = model(inputs)
    if logits.shape != (solutions.shape[0], 322, 16):
        raise ValueError("The generator must produce 322 sixteen-way predictions.")
    loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
    return loss, {"code_ce": loss.detach()}

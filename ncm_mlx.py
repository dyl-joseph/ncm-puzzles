import os
from pathlib import Path

os.environ["MLX_ENABLE_TF32"] = "0"

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


LEVELS = (1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89)
BLOCK_SUFFIXES = {
    "ln_1.weight": (512,), "ln_1.bias": (512,),
    "attn.c_attn.weight": (1536, 512), "attn.c_attn.bias": (1536,),
    "attn.c_proj.weight": (512, 512), "attn.c_proj.bias": (512,),
    "attn.q_norm.weight": (64,), "attn.k_norm.weight": (64,),
    "ln_2.weight": (512,), "ln_2.bias": (512,),
    "mlp.c_fc.weight": (2048, 512), "mlp.c_fc.bias": (2048,),
    "mlp.c_proj.weight": (512, 2048), "mlp.c_proj.bias": (512,),
}
MODEL_CONFIG = {
    "n_layer": 8, "n_head": 8, "bias": True, "dropout": 0.1,
    "n_embd": 512, "context_length": 180, "vocab_size": [16] * 12,
    "attn_pattern": "block_diagonal", "use_positional_encoding": True,
    "use_level_encoding": True, "prefix_len": 91, "use_rope": True,
    "rope_base": 10000, "use_qk_norm": True, "raw_token_prefix": True,
    "query_level0": True, "tie_heads_to_codebook": False,
}
VQVAE_CONFIG = {
    "n_layers": 4, "context_length": 89, "embed_dim": 256,
    "in_vocab_size": 12, "vqvae_vocab_size": [16] * 12,
    "compression_factor": 1, "dropout": 0.3, "dec_layers": 1,
    "pre_quant_dropout": 0.0, "pre_quant_groupnorm": 8,
    "pre_quant_norm_type": "group", "use_qk_norm": True,
    "vector_quantizer_config": {
        "clss": "multiscale_residual_vector_quantizer", "decay": 0.99,
        "epsilon": 1e-5, "commitment_cost": 0.25,
        "learned_l1_sampling": True, "learned_all_sampling": False,
        "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0,
                       "learnable_ratio": False},
        "levels": {"use_manual_levels": True, "manual_levels": list(LEVELS)},
        "learned_l1_cascade": True, "learned_l1_cascade_base": 4,
        "learned_l1_cascade_norm": False,
        "learned_l1_cascade_norm_final": False,
        "aux": {"fine_drop": {"prob": 0.5, "min_keep": 1}},
    },
    "checkpoint_path": "tokenizer.pt",
}


def as_mlx(tensor):
    return mx.array(tensor.detach().cpu().contiguous().numpy(), dtype=mx.float32)


def interpolation_operator(input_length, output_length, mode):
    basis = torch.eye(input_length, dtype=torch.float32).unsqueeze(0)
    operator = F.interpolate(basis, size=output_length, mode=mode)[0].T.contiguous()
    return as_mlx(operator)


class MLXPredictor:
    def __init__(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        config = OmegaConf.to_container(OmegaConf.load(checkpoint_dir / "config.yaml"))
        if config["model"] != MODEL_CONFIG or config["tokenizer"]["vqvae"] != VQVAE_CONFIG:
            raise ValueError("MLX inference supports only the exact published hard Sudoku config")
        if not mx.metal.is_available():
            raise RuntimeError("An MLX Metal GPU is required")
        mx.set_default_device(mx.gpu)
        self.device = str(mx.default_device())
        self.generator = self._load_generator(checkpoint_dir / "generator.pt")
        self.tokenizer = self._load_tokenizer(checkpoint_dir / "tokenizer.pt")
        self.up = {i: interpolation_operator(n, 89, "linear")
                   for i, n in enumerate(LEVELS) if 0 < i < 11}
        self.down = {i: interpolation_operator(89, n, "area")
                     for i, n in enumerate(LEVELS) if i > 0}
        self.rope = []
        self.masks = []
        inverse_frequency = 1.0 / (10000 ** (torch.arange(32, dtype=torch.float32) / 32))
        for length in LEVELS:
            positions = torch.cat([torch.arange(91), torch.arange(length)]).float()
            frequencies = positions[:, None] * inverse_frequency[None, :]
            self.rope.append((as_mlx(torch.cat([frequencies.cos()] * 2, dim=-1)),
                              as_mlx(torch.cat([frequencies.sin()] * 2, dim=-1))))
            mask = np.zeros((1, 1, 91 + length, 91 + length), dtype=np.float32)
            mask[:, :, :91, 91:] = -np.inf
            self.masks.append(mx.array(mask))
        self.up0_matrix = mx.transpose(self.tokenizer["vq.learned_up_l1.0.weight"], (0, 2, 1)).reshape(256, 89 * 256)
        self.residual_weights = [mx.transpose(self.tokenizer[f"vq.quant_resi.{i}.weight"], (0, 2, 1)) for i in range(12)]
        mx.eval(*self.generator.values(), *self.tokenizer.values(), self.up0_matrix,
                *self.residual_weights, *self.up.values(), *self.down.values(),
                *[array for pair in self.rope for array in pair], *self.masks)

    @staticmethod
    def _read(path, required, unused):
        state = torch.load(path, map_location="cpu", weights_only=True)["model"]
        if set(state) != set(required) | set(unused):
            raise ValueError(f"Unsupported checkpoint keys in {path.name}: missing={(set(required) | set(unused)) - set(state)}, extra={set(state) - set(required) - set(unused)}")
        result = {}
        for key, shape in required.items():
            tensor = state[key]
            if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
                raise ValueError(f"Unsupported parameter {key}: {tensor.shape}, {tensor.dtype}")
            result[key] = as_mlx(tensor)
        return result

    def _load_generator(self, path):
        required = {
            "l0_queries": (1, 1, 256), "transformer.wte.weight": (512, 256),
            "transformer.wte.bias": (512,), "transformer.wle.weight": (12, 512),
            "transformer.ln_f.weight": (512,), "transformer.ln_f.bias": (512,),
        }
        for i in range(8):
            required.update({f"transformer.blocks.{i}.{key}": shape for key, shape in BLOCK_SUFFIXES.items()})
        for i in range(12):
            required[f"lm_heads.{i}.weight"] = (16, 512)
            required[f"lm_heads.{i}.bias"] = (16,)
        return self._read(path, required, {"lvl_1L", "attn_mask_for_masking"})

    def _load_tokenizer(self, path):
        required = {
            "encoder.wte.weight": (12, 256),
            "vq.learned_up_l1.0.weight": (256, 256, 89),
            "vq.learned_up_l1.0.bias": (256,),
            "decoder.ln_f.weight": (256,),
            "decoder.output_layer.weight": (12, 256),
        }
        for key, shape in BLOCK_SUFFIXES.items():
            shape = tuple(n // 2 if n != 64 else 64 for n in shape)
            required[f"decoder.middle_block.{key}"] = shape
        unused = {"encoder.wpe.weight", "vq.learned_down_l1.0.weight", "vq.learned_down_l1.0.bias"}
        for i in range(12):
            required[f"vq.embeddings.{i}.weight"] = (16, 256)
            required[f"vq.quant_resi.{i}.weight"] = (256, 256, 3)
            required[f"vq.quant_resi.{i}.bias"] = (256,)
            unused.update({f"vq.ema_cluster_size_{i}", f"vq.ema_w_{i}", f"vq.codebook_hit_{i}"})
        return self._read(path, required, unused)

    @staticmethod
    def linear(x, weights, name):
        result = x @ weights[name + ".weight"].T
        bias = weights.get(name + ".bias")
        return result if bias is None else result + bias

    @staticmethod
    def norm(x, weights, name):
        return mx.fast.layer_norm(x, weights[name + ".weight"], weights.get(name + ".bias"), 1e-6)

    def block(self, x, weights, name, heads, mask=None, rope=None):
        batch, length, channels = x.shape
        h = self.norm(x, weights, name + ".ln_1")
        q, k, v = mx.split(self.linear(h, weights, name + ".attn.c_attn"), 3, axis=-1)
        q, k, v = [t.reshape(batch, length, heads, 64).transpose(0, 2, 1, 3) for t in (q, k, v)]
        q = mx.fast.rms_norm(q, weights[name + ".attn.q_norm.weight"], 1e-6)
        k = mx.fast.rms_norm(k, weights[name + ".attn.k_norm.weight"], 1e-6)
        if rope is not None:
            cos, sin = rope
            q = q * cos + mx.concatenate([-q[..., 32:], q[..., :32]], axis=-1) * sin
            k = k * cos + mx.concatenate([-k[..., 32:], k[..., :32]], axis=-1) * sin
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
        h = h.transpose(0, 2, 1, 3).reshape(batch, length, channels)
        x = x + self.linear(h, weights, name + ".attn.c_proj")
        h = self.linear(self.norm(x, weights, name + ".ln_2"), weights, name + ".mlp.c_fc")
        h = 0.5 * h * (1 + mx.erf(h * (2 ** -0.5)))
        return x + self.linear(h, weights, name + ".mlp.c_proj")

    def generator_logits(self, prefix_embedding, target, level):
        h = self.linear(target, self.generator, "transformer.wte")
        h = h + self.generator["transformer.wle.weight"][level]
        h = mx.concatenate([prefix_embedding, h], axis=1)
        for i in range(8):
            h = self.block(h, self.generator, f"transformer.blocks.{i}", 8,
                           self.masks[level], self.rope[level])
        h = self.norm(h, self.generator, "transformer.ln_f")
        return self.linear(h[:, 91:], self.generator, f"lm_heads.{level}")

    def residual(self, embedded, level):
        if level == 0:
            h = (embedded[:, 0] @ self.up0_matrix).reshape(-1, 89, 256)
            h = h + self.tokenizer["vq.learned_up_l1.0.bias"]
        elif level < 11:
            h = self.up[level] @ embedded
        else:
            h = embedded
        transformed = mx.conv1d(h, self.residual_weights[level], padding=1)
        transformed = transformed + self.tokenizer[f"vq.quant_resi.{level}.bias"]
        return 0.5 * h + 0.5 * transformed

    def decoder_logits(self, reconstruction):
        h = self.block(reconstruction, self.tokenizer, "decoder.middle_block", 4)
        h = self.norm(h, self.tokenizer, "decoder.ln_f")
        return self.linear(h, self.tokenizer, "decoder.output_layer")

    def infer(self, prefix_rows, trace=False):
        if not prefix_rows or any(len(row) != 91 or any(type(t) is not int or not 0 <= t < 12 for t in row) for row in prefix_rows):
            raise ValueError("prefix_rows must contain rectangular [batch,91] integer IDs in [0,11]")
        prefix = mx.array(prefix_rows, dtype=mx.int32)
        prefix_embedding = self.linear(self.tokenizer["encoder.wte.weight"][prefix], self.generator, "transformer.wte")
        target = mx.broadcast_to(self.generator["l0_queries"], (len(prefix_rows), 1, 256))
        reconstruction = mx.zeros((len(prefix_rows), 89, 256), dtype=mx.float32)
        all_codes, all_logits = [], []
        mx.eval(prefix_embedding)
        for level in range(12):
            logits = self.generator_logits(prefix_embedding, target, level)
            codes = mx.argmax(logits, axis=-1)
            embedded = self.tokenizer[f"vq.embeddings.{level}.weight"][codes]
            reconstruction = reconstruction + self.residual(embedded, level)
            all_codes.append(codes)
            if trace:
                all_logits.append(logits)
            mx.eval(codes, reconstruction, *(all_logits[-1:] if trace else []))
            if level < 11:
                target = self.down[level + 1] @ reconstruction
        logits = self.decoder_logits(reconstruction)
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(predictions, logits)
        if trace:
            return {"predictions": predictions, "codes": mx.concatenate(all_codes, axis=1),
                    "level_logits": all_logits, "decoder_logits": logits,
                    "reconstruction": reconstruction}
        return predictions

    def predict(self, prefix_rows):
        return self.infer(prefix_rows).tolist()

"""Small MLX GPU port of the released NCM algorithm for action packets."""

import mlx.core as mx
import mlx.nn as nn


LEVELS = (1, 3, 6, 12)
WIDTH = 64
CODEBOOK_SIZE = 64
CONFIG = {
    "implementation": "MLX GPU, trained from scratch",
    "reference": "vendor/ncm/models/{vqvae,vae,quant,next_concept,common}.py",
    "target_length": 12,
    "levels": list(LEVELS),
    "latent_width": WIDTH,
    "codebook_sizes": [CODEBOOK_SIZE] * len(LEVELS),
    "encoder_layers": 0,
    "encoder_positional": True,
    "compression_factor": 1,
    "pre_quant_norm": None,
    "decoder_layers": 1,
    "decoder_heads": 4,
    "decoder_positional": False,
    "ema_decay": 0.99,
    "ema_epsilon": 1e-5,
    "commitment_cost": 0.25,
    "learned_l1_sampling": False,
    "learned_all_sampling": False,
    "quant_resi": False,
    "fine_level_dropout": 0.0,
    "dropout": 0.0,
    "generator_width": WIDTH,
    "generator_layers": 2,
    "generator_heads": 4,
    "context_embedding": "independent and trainable",
    "position_encoding": "learned absolute, reset within every target level",
    "use_rope": False,
    "use_qk_norm": True,
    "activation": "exact GELU",
    "attention": "prefix bidirectional, target block diagonal, padded keys masked",
    "query_level0": True,
    "tie_output_heads": False,
    "pretrained_weights": False,
    "published_sudoku_deviations": [
        "12 action tokens replace the 89-token Sudoku target",
        "4 levels replace the published 12-level hierarchy",
        "64-dimensional latents and 64-entry codebooks replace 256 and 16",
        "64-wide two-block generator replaces 512-wide eight-block generator",
        "area and linear resampling replace learned level-zero sampling",
        "quantization residual convolutions, group normalization, and dropout disabled",
        "learned absolute positions replace RoPE",
        "independent trainable prefix embedding replaces tokenizer prefix embedding",
        "no published checkpoint used and no numerical parity claim",
    ],
}


def validate_device():
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("This experiment requires MLX Metal GPU as the default device")
    return str(mx.default_device())


def area_down(x, output_length):
    input_length = x.shape[1]
    if input_length == output_length:
        return x
    position = mx.arange(input_length)[None, :]
    bins = mx.arange(output_length)[:, None]
    start = bins * input_length // output_length
    end = ((bins + 1) * input_length + output_length - 1) // output_length
    weights = ((position >= start) & (position < end)).astype(x.dtype)
    weights = weights / (end - start).astype(x.dtype)
    return weights @ x


def linear_up(x, output_length):
    input_length = x.shape[1]
    if input_length == output_length:
        return x
    source = (mx.arange(output_length, dtype=mx.float32) + 0.5)
    source = mx.clip(source * (input_length / output_length) - 0.5, 0, input_length - 1)
    left = mx.floor(source).astype(mx.int32)
    right = mx.minimum(left + 1, input_length - 1)
    fraction = (source - left.astype(mx.float32))[None, :, None]
    return x[:, left, :] * (1 - fraction) + x[:, right, :] * fraction


class Block(nn.Module):
    def __init__(self, width=WIDTH, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = nn.RMSNorm(width // heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(width // heads, eps=1e-6)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.expand = nn.Linear(width, 4 * width)
        self.contract = nn.Linear(4 * width, width)

    def __call__(self, x, mask=None):
        batch, length, width = x.shape
        q, k, v = mx.split(self.qkv(self.norm1(x)), 3, axis=-1)
        q, k, v = [a.reshape(batch, length, self.heads, width // self.heads)
                   .transpose(0, 2, 1, 3) for a in (q, k, v)]
        q, k = self.q_norm(q), self.k_norm(k)
        attention = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(width // self.heads) ** -0.5, mask=mask)
        attention = attention.transpose(0, 2, 1, 3).reshape(batch, length, width)
        x = x + self.projection(attention)
        h = self.expand(self.norm2(x))
        h = 0.5 * h * (1 + mx.erf(h * (2 ** -0.5)))
        return x + self.contract(h)


class ResidualQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebooks = [mx.random.normal((CODEBOOK_SIZE, WIDTH)) for _ in LEVELS]
        self.ema_counts = [mx.zeros((CODEBOOK_SIZE,)) for _ in LEVELS]
        self.ema_sums = [mx.array(book) for book in self.codebooks]
        self.freeze(keys=["codebooks", "ema_counts", "ema_sums"], recurse=False)

    def quantize(self, z, update=False):
        residual = mx.stop_gradient(z)
        reconstruction = mx.zeros_like(z)
        codes = []
        loss = mx.array(0.0)
        for level_index, length in enumerate(LEVELS):
            flat = area_down(residual, length).reshape(-1, WIDTH)
            book = mx.stop_gradient(self.codebooks[level_index])
            distances = (mx.sum(flat ** 2, axis=-1, keepdims=True)
                         + mx.sum(book ** 2, axis=-1)[None, :] - 2 * flat @ book.T)
            ids = mx.argmin(distances, axis=-1).astype(mx.int32)
            if update:
                assignments = (ids[:, None] == mx.arange(CODEBOOK_SIZE)[None, :]).astype(z.dtype)
                counts = 0.99 * self.ema_counts[level_index] + 0.01 * mx.sum(assignments, axis=0)
                sums = 0.99 * self.ema_sums[level_index] + 0.01 * assignments.T @ flat
                total = mx.sum(counts)
                smooth = (counts + 1e-5) / (total + CODEBOOK_SIZE * 1e-5) * total
                book = mx.stop_gradient(sums / smooth[:, None])
                self.ema_counts[level_index] = mx.stop_gradient(counts)
                self.ema_sums[level_index] = mx.stop_gradient(sums)
                self.codebooks[level_index] = book
            ids = ids.reshape(z.shape[0], length)
            codes.append(ids)
            contribution = linear_up(book[ids], LEVELS[-1])
            reconstruction = reconstruction + contribution
            residual = residual - contribution
            loss = loss + 0.25 * mx.mean((mx.stop_gradient(reconstruction) - z) ** 2)
            loss = loss + mx.mean((reconstruction - mx.stop_gradient(z)) ** 2)
        straight_through = z + mx.stop_gradient(reconstruction - z)
        return straight_through, loss / len(LEVELS), mx.concatenate(codes, axis=1)

    def reconstruction(self, codes):
        z = mx.zeros((codes.shape[0], LEVELS[-1], WIDTH))
        offset = 0
        for index, length in enumerate(LEVELS):
            book = mx.stop_gradient(self.codebooks[index])
            z = z + linear_up(book[codes[:, offset:offset + length]], LEVELS[-1])
            offset += length
        return z

    def contexts(self, codes):
        z = mx.zeros((codes.shape[0], LEVELS[-1], WIDTH))
        offset = 0
        result = []
        for index, length in enumerate(LEVELS[:-1]):
            book = mx.stop_gradient(self.codebooks[index])
            z = z + linear_up(book[codes[:, offset:offset + length]], LEVELS[-1])
            result.append(area_down(z, LEVELS[index + 1]))
            offset += length
        return mx.stop_gradient(mx.concatenate(result, axis=1))


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        validate_device()
        self.vocab_size = vocab_size
        self.encoder_tokens = nn.Embedding(vocab_size, WIDTH)
        self.encoder_positions = nn.Embedding(LEVELS[-1], WIDTH)
        self.encoder_tokens.weight = mx.random.normal((vocab_size, WIDTH))
        self.encoder_positions.weight = mx.random.normal((LEVELS[-1], WIDTH))
        self.quantizer = ResidualQuantizer()
        self.decoder_block = Block()
        self.decoder_norm = nn.LayerNorm(WIDTH, eps=1e-6, bias=False)
        self.decoder_head = nn.Linear(WIDTH, vocab_size, bias=False)

    def latent(self, targets):
        return self.encoder_tokens(targets) + self.encoder_positions(mx.arange(LEVELS[-1]))[None]

    def decoder_logits(self, latent):
        return self.decoder_head(self.decoder_norm(self.decoder_block(latent)))

    def loss(self, targets):
        latent, quantization_loss, _ = self.quantizer.quantize(self.latent(targets), update=self.training)
        reconstruction_loss = nn.losses.cross_entropy(self.decoder_logits(latent), targets, reduction="mean")
        return reconstruction_loss + quantization_loss

    def encode(self, targets):
        return mx.stop_gradient(self.quantizer.quantize(self.latent(targets), update=False)[2])

    def decode(self, codes):
        return self.decoder_logits(self.quantizer.reconstruction(mx.stop_gradient(codes)))

    def contexts(self, codes):
        return self.quantizer.contexts(mx.stop_gradient(codes))


class Generator(nn.Module):
    def __init__(self, vocab_size: int, prefix_len: int, tokenizer: Tokenizer):
        super().__init__()
        validate_device()
        object.__setattr__(self, "_tokenizer", tokenizer)
        self.prefix_len = prefix_len
        self.context_embedding = nn.Embedding(vocab_size, WIDTH)
        self.position_embedding = nn.Embedding(max(prefix_len, max(LEVELS)), WIDTH)
        self.level_embedding = nn.Embedding(len(LEVELS), WIDTH)
        self.level0_query = mx.random.normal((1, 1, WIDTH), scale=WIDTH ** -0.5)
        self.target_projection = nn.Linear(WIDTH, WIDTH)
        self.blocks = [Block() for _ in range(CONFIG["generator_layers"])]
        self.norm = nn.LayerNorm(WIDTH, eps=1e-6)
        self.heads = [nn.Linear(WIDTH, CODEBOOK_SIZE) for _ in LEVELS]

    def _mask(self, prefixes, lengths, level_ids):
        prefix_levels = mx.full((self.prefix_len,), -1, dtype=mx.int32)
        target_levels = mx.concatenate([mx.full((length,), level, dtype=mx.int32)
                                        for length, level in zip(lengths, level_ids)])
        levels = mx.concatenate([prefix_levels, target_levels])
        key_is_prefix = levels[None, :] == -1
        same_target_level = (levels[:, None] == levels[None, :]) & (levels[:, None] >= 0)
        allowed = key_is_prefix | same_target_level
        valid_keys = mx.concatenate([prefixes != 0, mx.ones((prefixes.shape[0], sum(lengths)), dtype=mx.bool_)], axis=1)
        return mx.where(allowed[None, None, :, :] & valid_keys[:, None, None, :], 0.0, -mx.inf)

    def _logits(self, prefixes, chunks, level_ids):
        prefix = self.context_embedding(prefixes)
        prefix = prefix + self.position_embedding(mx.arange(self.prefix_len))[None]
        target_chunks = []
        for chunk, level in zip(chunks, level_ids):
            target = self.target_projection(chunk)
            target = target + self.position_embedding(mx.arange(chunk.shape[1]))[None]
            target = target + self.level_embedding(mx.array(level))
            target_chunks.append(target)
        x = mx.concatenate([prefix, *target_chunks], axis=1)
        mask = self._mask(prefixes, [chunk.shape[1] for chunk in chunks], level_ids)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x[:, self.prefix_len:])
        logits = []
        offset = 0
        for chunk, level in zip(chunks, level_ids):
            logits.append(self.heads[level](x[:, offset:offset + chunk.shape[1]]))
            offset += chunk.shape[1]
        return logits

    def teacher_logits(self, prefixes, codes):
        contexts = mx.stop_gradient(self._tokenizer.contexts(mx.stop_gradient(codes)))
        chunks = [mx.broadcast_to(self.level0_query, (prefixes.shape[0], 1, WIDTH))]
        offset = 0
        for length in LEVELS[1:]:
            chunks.append(contexts[:, offset:offset + length])
            offset += length
        return self._logits(prefixes, chunks, list(range(len(LEVELS))))

    def loss(self, prefixes, codes):
        logits = mx.concatenate(self.teacher_logits(prefixes, codes), axis=1)
        return nn.losses.cross_entropy(logits, mx.stop_gradient(codes), reduction="mean")

    def generate(self, prefixes):
        batch = prefixes.shape[0]
        z = mx.zeros((batch, LEVELS[-1], WIDTH))
        chunk = mx.broadcast_to(self.level0_query, (batch, 1, WIDTH))
        codes = []
        for index, length in enumerate(LEVELS):
            logits = self._logits(prefixes, [chunk], [index])[0]
            ids = mx.argmax(logits, axis=-1).astype(mx.int32)
            codes.append(ids)
            book = mx.stop_gradient(self._tokenizer.quantizer.codebooks[index])
            z = z + linear_up(book[ids], LEVELS[-1])
            if index + 1 < len(LEVELS):
                chunk = area_down(z, LEVELS[index + 1])
        return mx.concatenate(codes, axis=1)

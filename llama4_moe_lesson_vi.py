"""Mini LLaMA 4-style Mixture-of-Experts Transformer lesson.

Chạy nhanh:
    python llama4_moe_lesson_vi.py --steps 20 --generate 80

Mục đích của file này là giúp bạn nhớ bài học:
- Transformer decoder = embedding -> attention có causal mask -> FFN/MoE -> logits.
- LLaMA-style model thường dùng RMSNorm và RoPE.
- MoE thay FFN thường bằng nhiều expert; router chọn top-k expert cho từng token.
- Shared expert luôn chạy để giữ một đường xử lý ổn định.

Đây là model đồ chơi để học kiến trúc, không phải bản triển khai LLaMA 4 thật.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


TRAIN_TEXT = (
    "Facebook was founded in a dorm room at Harvard by Mark Zuckerberg and his roommates. "
    "What began as a small social network became a global platform, connecting billions of "
    "people across the world. Over the years, it expanded into messaging, photos, groups, "
    "marketplaces, and many other products."
)


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    block_size: int = 32
    num_experts: int = 4
    top_k: int = 2
    expert_multiplier: int = 2
    dropout: float = 0.0


class CharTokenizer:
    """Tokenizer mức ký tự: dễ học vì mỗi ký tự là một token."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class RMSNorm(nn.Module):
    """RMSNorm giống tinh thần LLaMA: normalize theo root-mean-square, không trừ mean."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x.square(), dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Positional Embedding cho Q/K.

    Input shape: (batch, heads, time, head_dim). RoPE quay từng cặp chiều để mã hóa vị trí.
    """
    batch, heads, time, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim phải chẵn để dùng RoPE")

    device = x.device
    half_dim = head_dim // 2
    positions = torch.arange(time, device=device).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    angles = torch.outer(positions, inv_freq)  # (time, half_dim)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention có causal mask và RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim phải chia hết cho num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :time, :time] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.out(attended)


class ExpertMLP(nn.Module):
    """Một expert là gated MLP: SiLU(gate(x)) * up(x) -> down."""

    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.up = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFeedForward(nn.Module):
    """MoE layer: router chọn top-k expert cho từng token, rồi cộng thêm shared expert."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        expert_dim = config.hidden_dim * config.expert_multiplier
        self.top_k = config.top_k
        self.router = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            ExpertMLP(config.hidden_dim, expert_dim) for _ in range(config.num_experts)
        )
        self.shared_expert = ExpertMLP(config.hidden_dim, expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        flat_x = x.view(batch * time, channels)

        router_logits = self.router(flat_x)
        top_values, top_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)

        routed_output = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_rows, selected_slots = torch.where(top_indices == expert_id)
            if token_rows.numel() == 0:
                continue
            expert_input = flat_x[token_rows]
            expert_weight = top_weights[token_rows, selected_slots].unsqueeze(-1)
            routed_output[token_rows] += expert(expert_input) * expert_weight

        shared_output = self.shared_expert(flat_x)
        return (routed_output + shared_output).view(batch, time, channels)


class TransformerBlock(nn.Module):
    """Một decoder block: RMSNorm -> Attention -> residual -> RMSNorm -> MoE -> residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.moe_norm = RMSNorm(config.hidden_dim)
        self.moe = MoEFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.moe_norm(x))
        return x


class MiniLlama4MoE(nn.Module):
    """Mini language model để ghi nhớ bài học LLaMA 4/MoE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if idx.size(1) > self.config.block_size:
            raise ValueError("Sequence dài hơn block_size")

        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def make_dataset(text: str, tokenizer: CharTokenizer, block_size: int, device: str):
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    xs, ys = [], []
    for start in range(len(ids) - block_size):
        xs.append(ids[start : start + block_size])
        ys.append(ids[start + 1 : start + block_size + 1])
    return torch.stack(xs), torch.stack(ys)


def sample_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    rows = torch.randint(0, x.size(0), (batch_size,), device=x.device)
    return x[rows], y[rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mini LLaMA 4-style MoE lesson model")
    parser.add_argument("--steps", type=int, default=200, help="Số bước train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--generate", type=int, default=120, help="Số ký tự sinh thêm")
    parser.add_argument("--prompt", type=str, default="Facebook was founded")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(1337)
    tokenizer = CharTokenizer(TRAIN_TEXT)
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    train_x, train_y = make_dataset(TRAIN_TEXT, tokenizer, config.block_size, args.device)

    model = MiniLlama4MoE(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    print(f"vocab_size={tokenizer.vocab_size}, train_examples={train_x.size(0)}, device={args.device}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for step in range(1, args.steps + 1):
        xb, yb = sample_batch(train_x, train_y, args.batch_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % max(1, args.steps // 5) == 0:
            print(f"step={step:04d} loss={loss.item():.4f}")

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    generated = model.generate(prompt_ids, max_new_tokens=args.generate, temperature=0.9)
    print("\n--- generated text ---")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()

"""
Generate sample text from a nanochat checkpoint to evaluate quality.
Usage: python generate_samples.py [--step 87000] [--depth 12] [--temperature 0.8] [--max-tokens 80]
"""
import argparse
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_checkpoint

parser = argparse.ArgumentParser(description="Generate samples from nanochat checkpoint")
parser.add_argument("--step", type=int, default=87000, help="Checkpoint step to load")
parser.add_argument("--depth", type=int, default=12, help="Model depth")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
parser.add_argument("--max-tokens", type=int, default=80, help="Max tokens to generate")
parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = get_tokenizer()

n_embd = args.depth * 64
n_head = n_embd // 128
cfg = GPTConfig(
    sequence_len=2048, vocab_size=32768,
    n_layer=args.depth, n_head=n_head, n_kv_head=n_head,
    n_embd=n_embd, window_pattern="L",
)
model = GPT(cfg).to(device)

checkpoint_dir = f"/home/lzw/.cache/nanochat/base_checkpoints/d{args.depth}"
model_data, _, _ = load_checkpoint(checkpoint_dir, args.step, device, load_optimizer=False, rank=0)
model.load_state_dict(model_data, strict=True)
model.eval()

engine = Engine(model, tokenizer)

prompts = [
    "The capital of France is",
    "Once upon a time",
    "The meaning of life is",
    "In 2025, artificial intelligence",
    "The recipe for chocolate cake requires",
    "def fibonacci(n):",
    "The theory of relativity states that",
    "import torch",
    "The quick brown fox",
    "Machine learning is",
]

print(f"Model: d{args.depth}, step={args.step}, device={device}")
print(f"Temperature={args.temperature}, top_k={args.top_k}, max_tokens={args.max_tokens}")
print("=" * 60)

for prompt in prompts:
    tokens = tokenizer(prompt, prepend="<|bos|>")
    with torch.no_grad():
        sample, _ = engine.generate_batch(
            tokens, num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    text = tokenizer.decode(sample[0])
    print(f"\n--- {prompt} ---")
    print(text)

print("\n" + "=" * 60)
print("Done.")

"""Pre-compute text embeddings for subset prompts using Qwen3-Embedding-4B on CPU."""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# VoxTell vendor path
VOXTELL_VENDOR = Path(__file__).resolve().parent.parent / "生医工大赛-demo" / "third_party" / "voxtell_pkg" / "voxtell_wheel"
if str(VOXTELL_VENDOR) not in sys.path:
    sys.path.insert(0, str(VOXTELL_VENDOR))

from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction  # noqa: E402


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-train", type=Path, required=True)
    parser.add_argument("--manifest-val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_jsonl(args.manifest_train) + load_jsonl(args.manifest_val)
    print(f"Total prompts: {len(all_rows)}")

    existing = set(p.stem for p in args.output_dir.glob("*.pt"))
    to_process = [r for r in all_rows if r["id"] not in existing]
    if existing:
        print(f"Skipping {len(existing)} existing, {len(to_process)} remaining")

    if not to_process:
        print("All done.")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading {args.model_name} on {device} ({args.dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=dtype).to(device).eval()

    prompts = [r["prompt"] for r in to_process]
    ids = [r["id"] for r in to_process]

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Encoding"):
        batch = prompts[i:i + args.batch_size]
        batch_ids = ids[i:i + args.batch_size]

        wrapped = wrap_with_instruction(batch)
        tokens = tokenizer(wrapped, padding=True, truncation=True, max_length=8192, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])

        for j, fid in enumerate(batch_ids):
            torch.save(embeddings[j].detach().float().cpu().clone(), args.output_dir / f"{fid}.pt")

    print(f"Done. Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

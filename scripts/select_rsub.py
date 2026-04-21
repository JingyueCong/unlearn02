"""Select R_sub: retain samples most semantically similar to the forget set.

Output: a JSON file with a list of retain indices (into the retain split used
for training). These indices identify samples that A2 learns to remember and
that A1 also treats as 'forget' so their contributions cancel at inference.

Usage:
    python scripts/select_rsub.py \
        --forget_split forget10_perturbed \
        --retain_split retain90 \
        --retain_num 400 \
        --k 80 \
        --out data/rsub/forget10_k80.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


@torch.no_grad()
def encode(model, tokenizer, texts, device, batch_size=32, max_len=256):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = F.normalize(pooled, dim=-1)
        embs.append(pooled.cpu())
    return torch.cat(embs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forget_split", required=True, help="e.g. forget10_perturbed")
    ap.add_argument("--retain_split", default=None, help="auto from forget_split if omitted")
    ap.add_argument("--retain_num", type=int, default=400,
                    help="how many trailing retain samples are used in training "
                         "(matches ToFU_DataModule default)")
    ap.add_argument("--k", type=int, required=True, help="R_sub size")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L12-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--use_question_and_answer", action="store_true",
                    help="embed q+a instead of q only")
    args = ap.parse_args()

    if args.retain_split is None:
        raw = args.forget_split.split("_")[0].replace("forget", "")
        args.retain_split = "retain" + str(100 - int(raw)).zfill(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModel.from_pretrained(args.encoder).to(device).eval()

    forget = load_dataset("locuslab/TOFU", args.forget_split)["train"]
    retain = load_dataset("locuslab/TOFU", args.retain_split)["train"]

    # Mirror ToFU_DataModule: only the trailing retain_num rows are used.
    retain_num = min(args.retain_num, len(forget))
    retain_train_start = len(retain) - retain_num
    retain_used = retain.select(range(retain_train_start, len(retain)))

    def to_text(ds):
        if args.use_question_and_answer:
            return [f"Q: {x['question']}\nA: {x['answer']}" for x in ds]
        return [x["question"] for x in ds]

    forget_emb = encode(model, tokenizer, to_text(forget), device)
    retain_emb = encode(model, tokenizer, to_text(retain_used), device)

    # For each retain item: max cosine similarity to any forget item
    sim = retain_emb @ forget_emb.T  # (R, F)
    max_sim, _ = sim.max(dim=1)

    k = min(args.k, len(retain_used))
    topk = torch.topk(max_sim, k=k).indices.tolist()
    topk = sorted(topk)

    # Indices are into the SELECTED retain slice (length == retain_num),
    # which is the slice ToFU_DataModule actually loads.
    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "forget_split": args.forget_split,
                "retain_split": args.retain_split,
                "retain_num": retain_num,
                "retain_slice_start": retain_train_start,
                "k": k,
                "indices": topk,
                "max_sim": [float(max_sim[i]) for i in topk],
            },
            f,
            indent=2,
        )
    print(f"Wrote {k} R_sub indices to {args.out}")


if __name__ == "__main__":
    main()

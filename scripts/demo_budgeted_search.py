#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from contextfocus.inference.budgeted_search import SearchConfig, budgeted_latent_activation_search
from contextfocus.utils import load_hf_causal_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--vectors_dir", type=str, required=True)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--context", type=str, required=True)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--probe_tokens", type=int, default=24)
    ap.add_argument("--final_tokens", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    args = ap.parse_args()

    bundle = load_hf_causal_lm(args.model_id, dtype=args.dtype)

    cfg = SearchConfig(budget=args.budget, probe_tokens=args.probe_tokens, final_tokens=args.final_tokens)
    out = budgeted_latent_activation_search(
        bundle,
        vectors_dir=args.vectors_dir,
        question=args.question,
        context=args.context,
        cfg=cfg,
    )

    print(json.dumps({k: v for k, v in out.items() if k != "candidates"}, indent=2))
    if "candidates" in out:
        print("Candidates:")
        for c in out["candidates"]:
            print(c)
    print("\n---\nTEXT\n---\n")
    print(out["text"])


if __name__ == "__main__":
    main()

"""
contextfocus_backend.py
=======================
Backend wrapper for the Axiom Streamlit app.

Provides two clean API functions:
  - load_environment()  → ModelBundle
  - run_inference()     → dict

This module imports from the existing `src/contextfocus/` package.
Run Streamlit with: PYTHONPATH=src streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Try processor for multimodal models like Gemma-3
try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None

# ── package imports ──────────────────────────────────────────────────────────
# The src/contextfocus package must be on the Python path.
# When launching via `PYTHONPATH=src streamlit run app.py` this is automatic.
try:
    from contextfocus.utils import (  # type: ignore[import]
        ModelBundle,
        decode,
        get_eos_id,
        get_model_hidden_size,
        get_transformer_blocks,
        tokenize_chat,
        tokenize_text,
        set_seed,
    )
    from contextfocus.prompting.templates import (  # type: ignore[import]
        PromptParts,
        build_openended_messages,
        build_openended_prompt,
        can_use_chat_template,
    )
    from contextfocus.inference.budgeted_search import (  # type: ignore[import]
        SearchConfig,
        budgeted_latent_activation_search,
    )
    from contextfocus.steering.steerer import load_vector  # type: ignore[import]
    _PKG_AVAILABLE = True
except ImportError as _e:
    _PKG_AVAILABLE = False
    _PKG_ERROR = str(_e)


# ─────────────────────────────────────────────────────────────────────────────
# Model cache directory
# ─────────────────────────────────────────────────────────────────────────────

# Models downloaded from HuggingFace are saved here permanently.
# On first run: downloads & caches (~6-7 GB for a 3B model).
# On subsequent runs: loads directly from disk — no internet needed.
MODEL_CACHE_DIR = str(Path(__file__).parent / "model_downloads")
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_device() -> tuple[torch.device, torch.dtype]:
    """
    Returns (device, torch_dtype) as torch.device objects:
      - CUDA  → cuda,  float16
      - MPS   → mps,   float16  (Apple Silicon)
      - CPU   → cpu,   float32
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# load_environment
# ─────────────────────────────────────────────────────────────────────────────

def load_environment(
    model_name: str,
    vectors_dir: str,
) -> "ModelBundle":
    """
    Load the HuggingFace causal LM and all layer steering vectors.

    Parameters
    ----------
    model_name : str
        HF model ID, e.g. "meta-llama/Llama-3.2-3B-Instruct" or
        "google/gemma-3-4b-it".
    vectors_dir : str
        Directory containing layer_000.pt … layer_NNN.pt steering vectors.

    Returns
    -------
    ModelBundle
        Dataclass with .model, .tokenizer, .device attributes.

    Notes
    -----
    - Uses BitsAndBytes 4-bit quantization for T4 GPU.
    - Apple Silicon (MPS) → float16; pure CPU → float32.
    - HF_TOKEN / HF_TOK env vars used for gated model access.
    """
    if not _PKG_AVAILABLE:
        raise ImportError(
            f"contextfocus package not found. "
            f"Run with: PYTHONPATH=src streamlit run app.py\n"
            f"Original error: {_PKG_ERROR}"
        )

    set_seed(7)
    device, torch_dtype = _detect_device()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOK")

    print(f"[Axiom] Loading '{model_name}' → device={device}, dtype={torch_dtype}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    # Try AutoProcessor first (Gemma-3 multimodal), fall back to AutoTokenizer
    tokenizer = None
    if AutoProcessor is not None and "gemma-3" in model_name.lower():
        try:
            tokenizer = AutoProcessor.from_pretrained(
                model_name, token=token, trust_remote_code=True
            )
        except Exception:
            tokenizer = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=MODEL_CACHE_DIR,
        )
    # Always ensure pad_token is set
    tk = getattr(tokenizer, "tokenizer", tokenizer)
    if getattr(tk, "pad_token", None) is None:
        tk.pad_token = tk.eos_token

    # ── Model ─────────────────────────────────────────────────────────────
    # device_map="auto" maps all layers automatically
    # cache_dir ensures the model is saved to ./model_downloads
    print(f"[Axiom] Cache dir: {MODEL_CACHE_DIR}")
    
    # Configure 4-bit quantization
    quant_config = None
    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else {"" : device},
            quantization_config=quant_config,
            torch_dtype=None if quant_config else torch_dtype,
            token=token,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
        )
    except Exception as e:
        if "gemma-3" in model_name.lower():
            try:
                from transformers import Gemma3ForConditionalGeneration
                model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto" if torch.cuda.is_available() else {"" : device},
                    quantization_config=quant_config,
                    torch_dtype=None if quant_config else torch_dtype,
                    token=token,
                    cache_dir=MODEL_CACHE_DIR,
                )
            except Exception:
                raise e
        else:
            raise e

    model.eval()
    actual_device = next(model.parameters()).device
    print(f"[Axiom] Model ready on {actual_device}")

    # ── Pre-load steering vectors ─────────────────────────────────────────
    vdir = Path(vectors_dir)
    vectors: Dict[int, torch.Tensor] = {}
    if vdir.exists():
        for pt_file in sorted(vdir.glob("layer_*.pt")):
            try:
                idx = int(pt_file.stem.split("_")[1])
                vectors[idx] = torch.load(pt_file, map_location="cpu",
                                          weights_only=True)
            except Exception:
                pass
        print(f"[Axiom] Loaded {len(vectors)} steering vectors from {vdir}")
    else:
        warnings.warn(f"[Axiom] Vectors dir not found: {vdir}. Steering disabled.")

    bundle = ModelBundle(model=model, tokenizer=tokenizer, device=actual_device)
    bundle._vectors_cache = vectors  # type: ignore[attr-defined]
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Baseline generation helper (no steering)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_baseline(bundle: "ModelBundle", question: str, max_new_tokens: int = 64) -> str:
    """
    Vanilla generation with NO context (tests parametric memory only).
    """
    model = bundle.model
    tok = bundle.tokenizer
    parts = PromptParts(system="", context="", question=question)

    if can_use_chat_template(tok):
        msgs = build_openended_messages(parts)
        inputs = tokenize_chat(tok, msgs, max_length=1024).to(model.device)
    else:
        prompt = build_openended_prompt(parts)
        inputs = tokenize_text(tok, prompt, max_length=1024).to(model.device)

    input_len = int(inputs["input_ids"].shape[-1])
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=get_eos_id(tok),
    )
    return decode(tok, out[0, input_len:])


# ─────────────────────────────────────────────────────────────────────────────
# run_inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    bundle: "ModelBundle",
    vectors_dir: str,
    question: str,
    context: str,
    use_bplis: bool = False,
    budget: int = 6,
    probe_tokens: int = 20,
    final_tokens: int = 64,
) -> dict:
    """
    Main inference entry point for the Axiom UI.

    Runs two generation passes:
    1. Baseline (no context, no steering) → parametric memory answer
    2. Steered (BCILS + budgeted search with injected context) → faithful answer

    Parameters
    ----------
    bundle : ModelBundle
        Loaded model + tokenizer.
    vectors_dir : str
        Path to directory of layer_xxx.pt vectors.
    question : str
        User's question.
    context : str
        Injected context (may conflict with parametric knowledge).
    use_bplis : bool
        Whether to enable B-PLIS CMA-ES latent vector search.
    budget : int
        Max candidate (layer, multiplier) pairs to probe.
    probe_tokens : int
        Tokens per probe in budgeted search.
    final_tokens : int
        Tokens for the final steered generation.

    Returns
    -------
    dict with keys:
        base_text         str   — baseline (no-context) generation
        steered_text      str   — steered (context-faithful) generation
        conflict_detected bool  — whether JSD gating triggered
        intervention_layer int|None — BCILS-chosen layer
        divergence        float — JS divergence score
        used_search       bool  — same as conflict_detected
    """
    if not _PKG_AVAILABLE:
        raise ImportError(
            f"contextfocus package not found. Run: PYTHONPATH=src streamlit run app.py"
        )

    # 1. Baseline (no context)
    base_text = _generate_baseline(bundle, question, max_new_tokens=final_tokens)

    # 2. Steered via budgeted search
    cfg = SearchConfig(
        budget=budget,
        probe_tokens=probe_tokens,
        final_tokens=final_tokens,
        multipliers=(1.0, 2.0, 3.0),
        top_k_layers=6,
        js_threshold=0.08,
        max_length=1024,
        use_bplis=use_bplis,
        bplis_generations=5 if use_bplis else 0,   # keep budget short for demo
        bplis_popsize=6 if use_bplis else 0,
        bplis_intrinsic_dim=64,
        householder_theta=0.6,
    )

    result = budgeted_latent_activation_search(
        bundle,
        vectors_dir=vectors_dir,
        question=question,
        context=context,
        cfg=cfg,
    )

    steered_text = result.get("text", "")
    used_search = result.get("used_search", False)
    divergence = result.get("divergence", 0.0)
    layer = result.get("layer", None)

    # Normalise layer value (can be list when B-PLIS wins)
    if isinstance(layer, list):
        intervention_layer = layer[0] if layer else None
    else:
        intervention_layer = layer

    return {
        "base_text": base_text,
        "steered_text": steered_text,
        "conflict_detected": bool(used_search),
        "intervention_layer": intervention_layer,
        "divergence": float(divergence),
        "used_search": bool(used_search),
        "raw_result": result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RAG helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_chunks(pdf_bytes: bytes, chunk_size: int = 500) -> List[str]:
    """
    Extract text from a PDF (bytes) and split into chunks.

    Tries pdfplumber first, falls back to PyPDF2.
    Returns a list of text chunks.
    """
    text = ""
    try:
        import pdfplumber  # type: ignore[import]
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        pass

    if not text.strip():
        try:
            import PyPDF2  # type: ignore[import]
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = "\n\n".join(
                (page.extract_text() or "") for page in reader.pages
            )
        except ImportError:
            raise ImportError(
                "Install either pdfplumber or PyPDF2: pip install pdfplumber"
            )

    # Split by double newlines first, then by fixed chunk_size
    raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    # Merge very short fragments and split very long ones
    chunks: List[str] = []
    buf = ""
    for chunk in raw_chunks:
        if len(buf) + len(chunk) < chunk_size:
            buf = (buf + " " + chunk).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = chunk
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) > 20]


def retrieve_best_chunk(query: str, chunks: List[str]) -> str:
    """
    TF-IDF retrieval: returns the single chunk most relevant to the query.
    Falls back to Jaccard similarity if scikit-learn is not installed.
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]
        import numpy as np  # type: ignore[import]

        corpus = chunks + [query]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(corpus)
        q_vec = tfidf[-1]
        chunk_vecs = tfidf[:-1]
        scores = cosine_similarity(q_vec, chunk_vecs)[0]
        best_idx = int(np.argmax(scores))
        return chunks[best_idx]
    except ImportError:
        pass

    # Fallback: Jaccard
    q_tokens = set(query.lower().split())
    best_score = -1.0
    best_chunk = chunks[0]
    for chunk in chunks:
        c_tokens = set(chunk.lower().split())
        if not c_tokens:
            continue
        score = len(q_tokens & c_tokens) / len(q_tokens | c_tokens)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk

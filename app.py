"""
app.py — Axiom
==============
Streamlit frontend for the ContextFocus Dynamic Steering demo.

Launch:
    cd "ContextFocus_replica-main (Copy)"
    PYTHONPATH=src streamlit run app.py

Requirements (beyond base project):
    pip install streamlit pdfplumber scikit-learn
"""
from __future__ import annotations

import sys, os

# Set HF_TOKEN environment variable for Kaggle connection
os.environ["HF_TOKEN"] = "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"

# Ensure src/ package is importable when launched directly
_ROOT = os.path.dirname(__file__)
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import time
from pathlib import Path
from typing import Optional, List

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Axiom",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "**Axiom** — ContextFocus Dynamic Steering Demo. Runs inference-time BCILS layer selection on GPU.",
    },
)

from contextfocus_backend import (
    load_environment,
    run_inference,
    extract_pdf_chunks,
    retrieve_best_chunk,
    MODEL_CACHE_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f1a 100%);
}

/* Hide Streamlit default footer but keep header for sidebar toggle */
#MainMenu, footer { visibility: hidden; }

/* Axiom title gradient */
.axiom-title {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c3aed, #2563eb, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}

.axiom-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin-top: 4px;
    font-weight: 400;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 3px solid #7c3aed;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* Result card base */
.result-card {
    border-radius: 12px;
    padding: 18px 20px;
    line-height: 1.7;
    font-size: 0.95rem;
    min-height: 100px;
}

.result-card-base {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.25);
    color: #fca5a5;
}

.result-card-steered {
    background: rgba(16, 185, 129, 0.06);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #6ee7b7;
}

.result-card-rag {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.30);
    color: #c4b5fd;
}

/* Column headers */
.col-header-base {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #ef4444;
    margin-bottom: 10px;
}

.col-header-steered {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #10b981;
    margin-bottom: 10px;
}

/* Metrics bar */
.metrics-bar {
    display: flex;
    gap: 16px;
    margin-top: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 18px;
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #94a3b8;
}

.metric-val {
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}

.conflict-yes { color: #fbbf24; }
.conflict-no  { color: #34d399; }

/* Sidebar styling */
.sidebar-model-badge {
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #c4b5fd;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
}

/* History card */
.history-meta {
    font-size: 0.78rem;
    color: #475569;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* Divider */
.axiom-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 32px 0;
}

/* Stagger fade-in animation */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.4s ease forwards; }

/* Status dot */
.status-dot-green { display: inline-block; width:8px; height:8px; border-radius:50%; background:#10b981; margin-right:6px; box-shadow: 0 0 6px #10b981; }
.status-dot-amber { display: inline-block; width:8px; height:8px; border-radius:50%; background:#f59e0b; margin-right:6px; box-shadow: 0 0 6px #f59e0b; }
.status-dot-red   { display: inline-block; width:8px; height:8px; border-radius:50%; background:#ef4444; margin-right:6px; box-shadow: 0 0 6px #ef4444; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "sandbox_history": [],
        "rag_context": "",
        "system_warmed_up": False,
        "pdf_chunks": [],        # chunks extracted from uploaded PDF
        "rag_history": [],
        "last_sandbox_result": None,
        "last_rag_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Axiom")
    st.markdown("*ContextFocus Dynamic Steering*")
    st.divider()

    st.markdown("### ⚙️ Model Configuration")

    model_name = st.text_input(
        "Model ID",
        value="meta-llama/Llama-3.2-3B-Instruct",
        help="Any HuggingFace causal LM. Gated models need HF_TOKEN env var.",
    )
    vectors_dir = st.text_input(
        "Vectors Directory",
        value="./vectors",
        help="Path to layer_000.pt … layer_NNN.pt steering vectors.",
    )

    hf_token_input = st.text_input(
        "HF Token (optional)",
        value="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",
        type="password",
        help="For gated models. Or set HF_TOKEN env var.",
    )
    if hf_token_input:
        os.environ["HF_TOKEN"] = hf_token_input

    # Cache dir info
    _cache_size = ""
    try:
        from pathlib import Path as _P
        _cache_path = _P(MODEL_CACHE_DIR)
        if _cache_path.exists():
            _bytes = sum(f.stat().st_size for f in _cache_path.rglob("*") if f.is_file())
            _gb = _bytes / (1024 ** 3)
            _cache_size = f"{_gb:.1f} GB" if _gb >= 0.1 else f"{_bytes // (1024**2)} MB"
    except Exception:
        pass

    st.markdown(
        f'<div style="font-size:0.78rem;color:#475569;margin-top:6px">'
        f'💾 Model cache: <code style="color:#7c3aed">model_downloads/</code>'
        f'{(" · " + _cache_size) if _cache_size else ""}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 📄 RAG Document")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload any PDF. The app will chunk it and use TF-IDF to retrieve the most relevant passage for your question.",
    )

    if uploaded_pdf is not None:
        with st.spinner("Parsing PDF…"):
            try:
                chunks = extract_pdf_chunks(uploaded_pdf.read())
                st.session_state.pdf_chunks = chunks
                st.success(f"✓ {len(chunks)} chunks extracted")
            except Exception as e:
                st.error(f"PDF parse error: {e}")

    st.divider()

    # Advanced settings
    with st.expander("🛠️ Advanced Settings", expanded=False):
        budget = st.slider("Search Budget", 2, 12, 6, help="Max candidate (layer, multiplier) pairs to probe.")
        probe_tokens = st.slider("Probe Tokens", 10, 40, 20, help="Tokens per probe generation (speed vs quality).")
        final_tokens = st.slider("Final Tokens", 32, 128, 64, help="Max tokens in the final output.")
        use_bplis = st.checkbox(
            "Enable B-PLIS (CMA-ES)",
            value=False,
            help="Synthesises a query-specific Δh via CMA-ES. Much slower (~80 forward passes).",
        )

    st.divider()
    st.caption("**Axiom** · ContextFocus Research Demo  \n© 2026 · CPU/MPS compatible")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading 3B Model & Vectors...")
def initialize_system(model_name: str, vectors_dir: str):
    """Cached model + vectors loader. Streamlit calls this once per session."""
    bundle = load_environment(model_name, vectors_dir)
    return bundle

try:
    bundle = initialize_system(model_name, vectors_dir)
    st.session_state.system_warmed_up = True
    _model_ok = True
except Exception as _load_err:
    st.session_state.system_warmed_up = False
    _model_ok = False
    _load_err_msg = str(_load_err)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="axiom-title">Axiom: Absolute Context Adherence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="axiom-subtitle" style="margin-top: 10px; margin-bottom: 20px; line-height: 1.5;">'
    'Developed as part of a college research initiative improving the ContextFocus architecture, '
    'Axiom demonstrates how to mathematically force an LLM to accept provided evidence as absolute truth. '
    'By leveraging Bayesian Causal Intervention for Layer Selection (BCILS) and Householder rotations, '
    'Axiom intercepts the model\'s internal thought process, overriding stubborn hallucinations and '
    'locking its attention onto your context.<br><br>'
    f'<code style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;color:#7c3aed">{model_name.split("/")[-1]}</code>'
    '</div>',
    unsafe_allow_html=True,
)

if not _model_ok:
    st.error(f"❌ Model failed to load: `{_load_err_msg}`\n\nCheck the Model ID and vectors directory in the sidebar.")
    st.stop()

# System status bar
_vdir = Path(vectors_dir)
_n_vectors = len(list(_vdir.glob("layer_*.pt"))) if _vdir.exists() else 0
_device = str(bundle.device)

status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    st.markdown(f'<span class="status-dot-green"></span><span style="font-size:0.8rem;color:#64748b">Model Ready</span>', unsafe_allow_html=True)
with status_col2:
    st.markdown(f'<span style="font-size:0.8rem;color:#64748b">Device: <code style="color:#7c3aed">{_device}</code></span>', unsafe_allow_html=True)
with status_col3:
    st.markdown(f'<span style="font-size:0.8rem;color:#64748b">Vectors: <code style="color:#7c3aed">{_n_vectors}</code></span>', unsafe_allow_html=True)
with status_col4:
    st.markdown(f'<span style="font-size:0.8rem;color:#64748b">B-PLIS: <code style="color:#7c3aed">{"ON" if use_bplis else "OFF"}</code></span>', unsafe_allow_html=True)

st.markdown('<hr class="axiom-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — SANDBOX
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧪 Sandbox — A/B Context Intervention</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#475569;font-size:0.88rem;margin-bottom:20px">'
    'Inject a custom context that contradicts the model\'s parametric knowledge. '
    'The <b>BCILS selector</b> dynamically picks the optimal transformer layer to steer — '
    'compare what the model \"knows\" vs what the context says.</p>',
    unsafe_allow_html=True,
)

with st.form("sandbox_form", clear_on_submit=False):
    sb_ctx = st.text_area(
        "💉 Inject Custom Context",
        value="The sky is made of green cheese.",
        height=90,
        help="This context will be injected into the model's residual stream via activation steering.",
    )
    sb_q = st.text_input(
        "❓ Ask a Question",
        value="What is the sky made of?",
        help="Question the model will answer — with and without context steering.",
    )
    run_btn = st.form_submit_button("⚡ Run Context Intervention", use_container_width=True)

if run_btn:
    if not sb_q.strip():
        st.warning("Please enter a question.")
    elif not sb_ctx.strip():
        st.warning("Please enter a context to inject.")
    else:
        with st.spinner("Calculating causal gradients and generating..."):
            try:
                _t0 = time.time()
                result = run_inference(
                    bundle,
                    vectors_dir=vectors_dir,
                    question=sb_q.strip(),
                    context=sb_ctx.strip(),
                    use_bplis=use_bplis,
                    budget=budget,
                    probe_tokens=probe_tokens,
                    final_tokens=final_tokens,
                )
                _elapsed = time.time() - _t0
                result["_elapsed"] = _elapsed
                result["_question"] = sb_q.strip()
                result["_context"] = sb_ctx.strip()
                st.session_state.last_sandbox_result = result
                # Prepend to history
                st.session_state.sandbox_history.insert(0, {
                    "query": sb_q.strip(),
                    "context": sb_ctx.strip(),
                    "base_ans": result["base_text"],
                    "steered_ans": result["steered_text"],
                    "metrics": {
                        "conflict": result["conflict_detected"],
                        "layer": result["intervention_layer"],
                        "divergence": result["divergence"],
                        "elapsed": _elapsed,
                    },
                })
            except Exception as _e:
                st.error(f"Inference error: `{_e}`")
                st.stop()

# Display latest result
if st.session_state.last_sandbox_result is not None:
    r = st.session_state.last_sandbox_result

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="col-header-base">🧠 Base Model (Parametric Memory)</div>', unsafe_allow_html=True)
        # st.info gives the spec-correct blue/gray box
        st.info(r["base_text"] or "*(no output)*")
    with col2:
        st.markdown('<div class="col-header-steered">✨ Steered Model (BCILS Active)</div>', unsafe_allow_html=True)
        # st.success gives the spec-correct green box
        st.success(r["steered_text"] or "*(no output)*")

    # Metrics strip
    _conflict = r["conflict_detected"]
    _layer = r["intervention_layer"]
    _jsd = r.get("divergence", 0.0)
    _elapsed = r.get("_elapsed", 0.0)

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            if _conflict:
                st.markdown(
                    '<div class="metric-pill conflict-yes">⚠️ Conflict Detected<br>'
                    '<span style="font-size:0.75rem;color:#64748b">JSD Gating: Active</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="metric-pill conflict-no">✅ No Conflict<br>'
                    '<span style="font-size:0.75rem;color:#64748b">JSD Gating: Bypassed</span></div>',
                    unsafe_allow_html=True,
                )
        with m2:
            layer_str = f"Layer {_layer}" if _layer is not None else "N/A"
            st.markdown(
                f'<div class="metric-pill">🎯 Intervention Layer<br>'
                f'<span class="metric-val">{layer_str}</span></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-pill">📏 JS Divergence<br>'
                f'<span class="metric-val">{_jsd:.4f}</span></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-pill">⏱️ Inference Time<br>'
                f'<span class="metric-val">{_elapsed:.1f}s</span></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — RAG PLAYGROUND
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="axiom-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📚 RAG Playground</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#475569;font-size:0.88rem;margin-bottom:20px">'
    'Upload a PDF in the sidebar, ask a question about it, and watch BCILS steer the model '
    'to faithfully follow the retrieved passage using TF-IDF chunk retrieval.</p>',
    unsafe_allow_html=True,
)

if not st.session_state.pdf_chunks:
    st.info("📄 Upload a PDF document in the sidebar to enable the RAG Playground.")
else:
    st.markdown(
        f'<span style="font-size:0.83rem;color:#64748b">📑 {len(st.session_state.pdf_chunks)} chunks loaded · '
        'TF-IDF retrieval enabled</span>',
        unsafe_allow_html=True,
    )

    with st.form("rag_form", clear_on_submit=False):
        rag_q = st.text_input(
            "❓ Ask a question about the document",
            placeholder="e.g. What are the main findings of the study?",
        )
        rag_btn = st.form_submit_button("🔍 Search & Answer", use_container_width=True)

    if rag_btn:
        if not rag_q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Calculating causal gradients and generating..."):
                try:
                    best_chunk = retrieve_best_chunk(rag_q.strip(), st.session_state.pdf_chunks)
                    st.session_state.rag_context = best_chunk

                    _t0 = time.time()
                    rag_result = run_inference(
                        bundle,
                        vectors_dir=vectors_dir,
                        question=rag_q.strip(),
                        context=best_chunk,
                        use_bplis=True,  # always use B-PLIS for RAG for best grounding
                        budget=budget,
                        probe_tokens=probe_tokens,
                        final_tokens=final_tokens,
                    )
                    rag_result["_elapsed"] = time.time() - _t0
                    rag_result["_question"] = rag_q.strip()
                    rag_result["_chunk"] = best_chunk
                    st.session_state.last_rag_result = rag_result
                    st.session_state.rag_history.insert(0, {
                        "query": rag_q.strip(),
                        "chunk": best_chunk,
                        "answer": rag_result["steered_text"],
                        "metrics": {
                            "conflict": rag_result["conflict_detected"],
                            "layer": rag_result["intervention_layer"],
                            "divergence": rag_result["divergence"],
                            "elapsed": rag_result["_elapsed"],
                        }
                    })
                except Exception as _e:
                    st.error(f"RAG inference error: `{_e}`")

    if st.session_state.last_rag_result is not None:
        rr = st.session_state.last_rag_result

        with st.expander("📄 View Retrieved Context", expanded=False):
            st.markdown(
                f'<div style="font-size:0.88rem;color:#94a3b8;line-height:1.7;'
                f'background:rgba(255,255,255,0.025);padding:14px;border-radius:8px;">'
                f'{rr["_chunk"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="col-header-steered" style="margin-top:16px">✨ Steered Answer (BCILS + RAG)</div>', unsafe_allow_html=True)
        # st.success per spec: "output the final generated text in a large success box"
        st.success(rr["steered_text"] or "*(no output)*")

        # Metrics
        with st.container(border=True):
            rm1, rm2, rm3, rm4 = st.columns(4)
            with rm1:
                if rr["conflict_detected"]:
                    st.markdown('<div class="metric-pill conflict-yes">⚠️ Conflict Detected<br><span style="font-size:0.75rem;color:#64748b">JSD Gating: Active</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-pill conflict-no">✅ No Conflict<br><span style="font-size:0.75rem;color:#64748b">JSD Gating: Bypassed</span></div>', unsafe_allow_html=True)
            with rm2:
                _rl = rr["intervention_layer"]
                st.markdown(f'<div class="metric-pill">🎯 Layer<br><span class="metric-val">{"Layer " + str(_rl) if _rl is not None else "N/A"}</span></div>', unsafe_allow_html=True)
            with rm3:
                st.markdown(f'<div class="metric-pill">📏 JSD<br><span class="metric-val">{rr["divergence"]:.4f}</span></div>', unsafe_allow_html=True)
            with rm4:
                st.markdown(f'<div class="metric-pill">⏱️ Time<br><span class="metric-val">{rr.get("_elapsed", 0):.1f}s</span></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — EXPERIMENT HISTORY
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.sandbox_history:
    # The most recent is already shown above; show the rest in expanders
    older = st.session_state.sandbox_history[1:]
    if older:
        st.markdown('<hr class="axiom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🕑 Previous Experiments</div>', unsafe_allow_html=True)
        for i, h in enumerate(older):
            _m = h.get("metrics", {})
            label = f"#{len(older) - i}  ·  \"{h['query'][:60]}{'…' if len(h['query']) > 60 else ''}\""
            with st.expander(label, expanded=False):
                st.markdown(
                    f'<div class="history-meta">Context: "{h["context"][:80]}{"…" if len(h["context"]) > 80 else ""}"  '
                    f'·  Layer {_m.get("layer","?")}  ·  JSD {_m.get("divergence", 0):.4f}  '
                    f'·  {_m.get("elapsed", 0):.1f}s</div>',
                    unsafe_allow_html=True,
                )
                hc1, hc2 = st.columns(2)
                with hc1:
                    st.markdown('<div class="col-header-base" style="font-size:0.72rem">Base Model</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-card result-card-base" style="min-height:60px;font-size:0.88rem">{h["base_ans"]}</div>', unsafe_allow_html=True)
                with hc2:
                    st.markdown('<div class="col-header-steered" style="font-size:0.72rem">Steered Model</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-card result-card-steered" style="min-height:60px;font-size:0.88rem">{h["steered_ans"]}</div>', unsafe_allow_html=True)

# RAG history
if st.session_state.rag_history and len(st.session_state.rag_history) > 1:
    older_rag = st.session_state.rag_history[1:]
    st.markdown('<div class="section-header" style="margin-top:24px">🕑 Previous RAG Queries</div>', unsafe_allow_html=True)
    for i, h in enumerate(older_rag):
        _m = h.get("metrics", {})
        with st.expander(f"RAG #{len(older_rag) - i}  ·  \"{h['query'][:60]}\"", expanded=False):
            st.markdown(f'<div class="history-meta">Layer {_m.get("layer","?")} · JSD {_m.get("divergence",0):.4f} · {_m.get("elapsed",0):.1f}s</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-card result-card-rag" style="font-size:0.88rem">{h["answer"]}</div>', unsafe_allow_html=True)

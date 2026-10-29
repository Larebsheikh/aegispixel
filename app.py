import io

import streamlit as st
from PIL import Image

from audit import ask_gemma, build_audit_prompt, build_benchmark_prompt
from metrics import calculate_defense_metrics, compute_report, extract_metadata
from shield import noise_shield, pgd_available, pgd_shield

st.set_page_config(page_title="AegisPixel | AI Image Immunizer", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.shield-header{text-align:center;color:#1E3A8A;font-weight:800;font-size:2.2rem}
.shield-sub{text-align:center;color:#64748B;margin-bottom:1.5rem}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="shield-header">🛡️ AegisPixel</div>', unsafe_allow_html=True)
st.markdown('<div class="shield-sub">Adversarial Image Immunization &amp; Deepfake Defense — '
            'Powered by Google Gemma</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Google AI Studio API Key", type="password",
                            help="Free at aistudio.google.com")
    model_name = st.text_input("Gemma model", value="gemma-2-9b-it",
                               help="Change if your key exposes a different Gemma model.")
    epsilon = st.slider("Perturbation Epsilon (ε, /255)", 1, 25, 8)
    modes = ["Fast noise (NumPy)"]
    if pgd_available():
        modes.append("PGD encoder attack (PhotoGuard-style)")
    mode = st.radio("Shield mode", modes)
    steps = st.slider("PGD steps", 10, 100, 40, 10) if mode.startswith("PGD") else 0
    if not pgd_available():
        st.caption("Install `requirements-pgd.txt` to unlock the PGD mode.")
    st.markdown("---")
    st.caption("Hacktoberfest Hack Day 2026 | Open-Source AI Defense")

tab_shield, tab_bench = st.tabs(["🛡️ Shield & Audit", "📊 Benchmark Lab"])

with tab_shield:
    up = st.file_uploader("Upload a portrait/photo", type=["png", "jpg", "jpeg"])
    if up:
        metadata = extract_metadata(up)
        original = Image.open(up).convert("RGB")
        key = (up.name, up.size, epsilon, mode, steps)

        if st.session_state.get("key") != key:
            if mode.startswith("PGD"):
                bar = st.progress(0.0, text="Running PGD against the SD VAE encoder…")
                shielded = pgd_shield(original, epsilon, steps,
                                      progress_cb=lambda f, l: bar.progress(f, text=f"PGD loss {l:.4f}"))
                bar.empty()
                original = original.resize(shielded.size, Image.LANCZOS)
            else:
                shielded = noise_shield(original, epsilon)
            st.session_state.update(key=key, shielded=shielded, original=original,
                                    report=compute_report(original, shielded), audit=None)

        original, shielded = st.session_state["original"], st.session_state["shielded"]
        report = st.session_state["report"]

        c1, c2 = st.columns(2)
        c1.subheader("📷 Original")
        c1.image(original, use_container_width=True)
        c2.subheader("🛡️ Immunized")
        c2.image(shielded, use_container_width=True)
        buf = io.BytesIO()
        shielded.save(buf, format="PNG")  # PNG: lossless, no EXIF carried over
        c2.download_button("⬇️ Download Immunized PNG", buf.getvalue(),
                           "immunized_photo.png", "image/png")

        st.markdown("### 📐 Measured Metrics")
        cols = st.columns(4)
        for i, (k, v) in enumerate(report.items()):
            cols[i % 4].metric(k, v)
        with st.expander("🗂️ Original file metadata (privacy check)"):
            st.json(metadata)
        st.caption("⚠️ Share the PNG, not a re-compressed JPEG: JPEG/blur strips part of the "
                   "perturbation (see survival metrics). Protection is not guaranteed.")

        if st.button("Run Gemma AI Vulnerability & Privacy Audit 🔍"):
            if not api_key:
                st.error("Enter your Google AI Studio API key in the sidebar.")
            else:
                with st.spinner("Gemma is generating the security report…"):
                    try:
                        prompt = build_audit_prompt(mode, epsilon, report, metadata)
                        st.session_state["audit"] = ask_gemma(api_key, model_name, prompt)
                    except Exception as e:
                        st.error(f"Error querying Gemma: {e}")
        if st.session_state.get("audit"):
            st.success("✅ Security Audit Complete")
            st.markdown(st.session_state["audit"])

with tab_bench:
    st.markdown("Enter results from your own test run (e.g. diffusion edits on shielded vs. "
                "unshielded images). Metrics are computed in code; Gemma only interprets them.")
    a, b = st.columns(2)
    method = a.text_input("Perturbation method", "PGD / sign noise")
    eps_txt = b.text_input("Epsilon", f"{epsilon}/255")
    c = st.columns(4)
    tp = c[0].number_input("TP", 0, value=46)
    fp = c[1].number_input("FP", 0, value=4)
    tn = c[2].number_input("TN", 0, value=44)
    fn = c[3].number_input("FN", 0, value=6)
    m = calculate_defense_metrics(int(tp), int(fp), int(tn), int(fn))
    mc = st.columns(len(m))
    for col, (k, v) in zip(mc, m.items()):
        col.metric(k, v)
    if st.button("Run Gemma Benchmark Audit 📋"):
        if not api_key:
            st.error("Enter your Google AI Studio API key in the sidebar.")
        else:
            with st.spinner("Gemma is analyzing the benchmark…"):
                try:
                    n = int(tp + fp + tn + fn)
                    st.markdown(ask_gemma(api_key, model_name, build_benchmark_prompt(
                        eps_txt, method, n, int(tp), int(fp), int(tn), int(fn), m)))
                except Exception as e:
                    st.error(f"Error querying Gemma: {e}")

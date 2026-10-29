# 🛡️ AegisPixel: Adversarial AI Privacy Shield & Threat Explainer

Built for **Hacktoberfest Hack Day 2026 (MLH × Zeppelin Labs × DEV)**.
Inspired by MIT CSAIL's [PhotoGuard](https://arxiv.org/abs/2302.06588), AegisPixel adds
imperceptible, ε-bounded perturbations to a portrait to make unauthorized generative edits
(deepfakes, inpainting, face-swaps) harder, then uses **Google Gemma** to explain the math and
audit the result.

## Features
- **Shield:** two modes
  - *Fast noise* (NumPy/Pillow): high-frequency ±ε sign noise, runs anywhere.
  - *PGD encoder attack* (optional, PyTorch + diffusers): PhotoGuard-style Projected Gradient
    Descent that pushes the Stable Diffusion VAE latent of your image toward a gray target:
    `min ‖E(x+δ) − z_target‖²  s.t. ‖δ‖∞ ≤ ε`.
- **Measured metrics:** PSNR, L∞, mean change, and how much of the perturbation survives
  JPEG q90/q75 and Gaussian blur, combined into a heuristic *Shield Integrity Score*.
- **Gemma audit:** metadata (EXIF/GPS) privacy check + a red-team report grounded in the
  measured numbers (`gemma-2-9b-it` via Google AI Studio, configurable in the sidebar).
- **Benchmark Lab:** enter TP/FP/TN/FN from your own tests; Precision, Recall, Specificity,
  Accuracy and F1 are computed in code and interpreted by Gemma.
- Side-by-side comparison, ε slider, one-click PNG download.

## Quick start
```bash
git clone https://github.com/<your-username>/aegispixel.git
cd aegispixel
pip install -r requirements.txt            # fast mode
# pip install -r requirements-pgd.txt      # optional: PGD mode (downloads sd-vae-ft-mse)
streamlit run app.py
```
Get a free API key at [aistudio.google.com](https://aistudio.google.com/), paste it in the sidebar.

## Honest limitations
- Fast-noise mode is **not** optimized against any model; treat its protection as weak.
- PGD mode targets one encoder (SD VAE); transfer to other models is not guaranteed.
- JPEG re-compression, resizing and blur can strip perturbations (see survival metrics).
  Share the downloaded PNG where possible.
- No adversarial shield is a guarantee; combine with other privacy practices.

## Structure
`app.py` UI · `shield.py` immunization · `metrics.py` metrics & EXIF · `audit.py` Gemma prompts

## License
MIT

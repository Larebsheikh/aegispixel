"""Gemma security audit prompts + API call."""
from __future__ import annotations


def build_audit_prompt(mode: str, epsilon: int, report: dict, metadata: dict) -> str:
    rep = "\n".join(f"- {k}: {v}" for k, v in report.items())
    meta = "\n".join(f"- {k}: {v}" for k, v in metadata.items()) or "- none found"
    return f"""You are an expert AI Red Teaming and Computer Vision Security Analyst.
A user applied an adversarial shield inspired by MIT CSAIL PhotoGuard to a personal photo.

SHIELD MODE: {mode}
EPSILON: {epsilon}/255
MEASURED METRICS (computed locally, treat as ground truth):
{rep}

ORIGINAL FILE METADATA:
{meta}

Rules: base every claim on the numbers above. Do NOT guarantee protection. If the mode is
'Fast noise', state clearly that the noise is not optimized against any encoder and gives
weak/unverified protection. If the mode is 'PGD', explain that it targets the Stable Diffusion
VAE encoder but may not transfer to other models or survive JPEG/resizing.

Write a concise Markdown report with these sections:
### 🛡️ Defense Mechanism Analysis
Explain the math: an encoder E maps x to a latent; PGD finds delta with ||delta||_inf <= eps
minimizing ||E(x+delta) - z_target||. Explain imperceptibility vs. effectiveness (use PSNR).
### ⚡ Threat Defense Score
Interpret the Shield Integrity Score and the survival-under-JPEG/blur numbers. Give a verdict:
[Weak / Moderate / Strong] with a one-line justification.
### 🔍 Metadata Privacy Findings
Comment on any metadata leaks (GPS, device, timestamps) found above.
### 🔒 Recommendations
3 short, practical actions (e.g. strip EXIF, upload PNG, higher epsilon trade-offs, watermarking).
"""


def build_benchmark_prompt(epsilon: str, method: str, n: int, tp: int, fp: int, tn: int,
                           fn: int, metrics: dict) -> str:
    m = "\n".join(f"- {k}: {v}" for k, v in metrics.items())
    return f"""You are an AI Security Researcher auditing "AegisPixel", an adversarial image immunization pipeline.
Setup: epsilon={epsilon}, method={method}, N={n}.
Confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}.
Verified metrics (already computed in code, do not recompute differently):
{m}

Write Markdown with: ### 📊 Metrics Interpretation (what each number means for this defense),
### ⚠️ Failure-Mode Analysis (false negatives: JPEG re-encoding, smoothing, diffusion purification;
false positives: visible artifacts at high epsilon), ### 📋 Deployment Rating
[Production Ready / Conditionally Viable / Vulnerable] based on the F1 score, plus 2 hardening
strategies (e.g. Expectation over Transformation, ensembles of encoders).
"""


def ask_gemma(api_key: str, model_name: str, prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name).generate_content(prompt).text

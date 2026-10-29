"""Measurable quality/robustness metrics and metadata extraction."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter
from PIL.ExifTags import GPSTAGS, TAGS


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def psnr(a: Image.Image, b: Image.Image) -> float:
    mse = float(np.mean((_arr(a) - _arr(b)) ** 2))
    return float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=1.0))


def _survival(orig: Image.Image, shielded: Image.Image, transform) -> float:
    """Fraction of the perturbation still present after a preprocessing step
    (projection of the residual onto the original perturbation, clipped to 0..1)."""
    d = _arr(shielded) - _arr(orig)
    denom = float(np.sum(d * d))
    if denom == 0:
        return 0.0
    r = _arr(transform(shielded)) - _arr(transform(orig))
    return float(np.clip(np.sum(r * d) / denom, 0.0, 1.0))


def compute_report(orig: Image.Image, shielded: Image.Image) -> dict:
    """Measured values only; nothing here is guessed."""
    if orig.size != shielded.size:
        orig = orig.resize(shielded.size, Image.LANCZOS)
    d = _arr(shielded) - _arr(orig)
    p = psnr(orig, shielded)
    jpeg75 = _survival(orig, shielded, lambda i: _jpeg(i, 75))
    jpeg90 = _survival(orig, shielded, lambda i: _jpeg(i, 90))
    blur = _survival(orig, shielded, _blur)
    imperceptibility = float(np.clip((p - 30) / 15, 0, 1))
    robustness = (jpeg75 + jpeg90 + blur) / 3
    return {
        "PSNR (dB)": round(p, 2) if np.isfinite(p) else 99.0,
        "L∞ (max pixel change)": int(np.abs(d).max()),
        "Mean abs change": round(float(np.abs(d).mean()), 3),
        "Perturbation surviving JPEG q90": round(jpeg90, 3),
        "Perturbation surviving JPEG q75": round(jpeg75, 3),
        "Perturbation surviving Gaussian blur": round(blur, 3),
        "Shield Integrity Score (0-100)": round(100 * (0.3 * imperceptibility + 0.7 * robustness), 1),
    }


def extract_metadata(img_file) -> dict:
    """EXIF metadata of the ORIGINAL upload (privacy leaks like GPS/device)."""
    out: dict = {}
    try:
        img_file.seek(0)
        img = Image.open(img_file)
        out["Format"], out["Size"] = img.format, f"{img.size[0]}x{img.size[1]}"
        exif = img.getexif()
        for tag_id, val in exif.items():
            name = TAGS.get(tag_id, str(tag_id))
            if name in ("Make", "Model", "Software", "DateTime", "Artist", "Copyright"):
                out[name] = str(val)[:80]
        gps = exif.get_ifd(0x8825)
        if gps:
            out["GPS data present"] = "YES: " + ", ".join(GPSTAGS.get(k, str(k)) for k in gps)
    except Exception as e:  # metadata is best-effort
        out["Metadata error"] = str(e)
    finally:
        img_file.seek(0)
    return out


def calculate_defense_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Confusion-matrix metrics for the Benchmark Lab."""
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {"Precision": round(precision, 4), "Recall": round(recall, 4),
            "Specificity": round(specificity, 4), "Accuracy": round(accuracy, 4),
            "F1 Score": round(f1, 4)}

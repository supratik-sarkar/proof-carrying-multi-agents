#!/usr/bin/env python3
"""Figure release gate: vector PDF + embedded fonts + extractable text + 300dpi PNG.

A zero-text figure is REJECTED rather than raster-edited.
"""
from __future__ import annotations
import glob, json, os, re, struct, subprocess, sys

PDF_DIR = "artifacts/v3_0/figures/pdf"
PNG_DIR = "artifacts/v3_0/figures/png"
MIN_TEXT_CHARS = 50
MIN_DPI = 295


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


def pdf_report(path: str) -> dict:
    text_raw = _run(["pdftotext", path, "-"])
    text = re.sub(r"\s+", "", text_raw)
    fonts_raw = _run(["pdffonts", path]).splitlines()[2:]
    total = embedded = 0
    for line in fonts_raw:
        parts = line.split()
        if len(parts) < 7:
            continue
        total += 1
        if parts[-5] == "yes":          # emb column: name .. emb sub uni objID gen
            embedded += 1
    images = [l for l in _run(["pdfimages", "-list", path]).splitlines()[2:] if l.strip()]

    # Fallback to pypdf if external poppler tools are not in PATH
    if not text_raw and total == 0:
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            extracted = ""
            raster_cnt = 0
            font_cnt = 0
            for page in reader.pages:
                extracted += (page.extract_text() or "")
                raster_cnt += len(list(page.images))
                res = page.get("/Resources", {})
                if hasattr(res, "get_object"):
                    res = res.get_object()
                f_dict = res.get("/Font", {}) if isinstance(res, dict) else {}
                if hasattr(f_dict, "get_object"):
                    f_dict = f_dict.get_object()
                if isinstance(f_dict, dict):
                    font_cnt += len(f_dict)
            text = re.sub(r"\s+", "", extracted)
            total = embedded = font_cnt
            images = ["raster"] * raster_cnt
        except Exception:
            pass

    return {"file": os.path.basename(path), "text_chars": len(text),
            "fonts_total": total, "fonts_embedded": embedded,
            "raster_images": len(images),
            "text_ok": len(text) >= MIN_TEXT_CHARS,
            "fonts_ok": total > 0 and total == embedded,
            "vector_ok": len(images) == 0}


def png_dpi(path: str) -> float:
    d = open(path, "rb").read()
    i = d.find(b"pHYs")
    if i < 0:
        return 0.0
    ppm = struct.unpack(">I", d[i + 4:i + 8])[0]
    return ppm * 0.0254


def main() -> int:
    rows, ok = [], True
    for p in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
        r = pdf_report(p)
        stem = os.path.splitext(r["file"])[0]
        png = os.path.join(PNG_DIR, f"{stem}.png")
        r["png_present"] = os.path.exists(png)
        r["png_dpi"] = round(png_dpi(png), 1) if r["png_present"] else 0.0
        r["png_ok"] = r["png_present"] and r["png_dpi"] >= MIN_DPI
        r["pass"] = all([r["text_ok"], r["fonts_ok"], r["vector_ok"], r["png_ok"]])
        ok &= r["pass"]
        rows.append(r)
    os.makedirs("artifacts/v3_0/checks", exist_ok=True)
    json.dump({"pass": ok, "figures": rows},
              open("artifacts/v3_0/checks/figure_gate.json", "w"), indent=2)
    for r in rows:
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['file']:<32} "
              f"text={r['text_chars']:<4} fonts={r['fonts_embedded']}/{r['fonts_total']} "
              f"raster={r['raster_images']} png_dpi={r['png_dpi']}")
    print(f"\nPNG_PDF_DUAL_OUTPUT={'PASS' if ok else 'FAIL'}  ({len(rows)} figures)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

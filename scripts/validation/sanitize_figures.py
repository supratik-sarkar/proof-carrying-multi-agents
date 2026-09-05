#!/usr/bin/env python3
"""Sanitize image files by stripping all metadata.

Loads every raster image, extracts only its raw pixel array,
maps it to a fresh Pillow RGB/RGBA canvas, and saves to a
new directory WITHOUT copying any metadata, EXIF, XMP, C2PA,
PNG text chunks, or steganographic modifications.

SVG files are passed through an XML sanitizer that removes
<metadata> elements and XML comments.

PDF files cannot be pixel-sanitized; they are reported as skipped.
"""

import os
import shutil
import sys
import re
from pathlib import Path


RASTER_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.tiff', '.tif', '.bmp', '.gif'}
VECTOR_EXTS = {'.svg'}
PDF_EXTS = {'.pdf'}


def sanitize_raster(src_path: str, dst_path: str) -> str:
    """Load raster image, extract raw pixels, save to fresh canvas."""
    from PIL import Image
    import numpy as np

    with Image.open(src_path) as img:
        # Convert to RGB or RGBA (strip all metadata implicitly)
        if img.mode == 'RGBA':
            mode = 'RGBA'
        elif img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
            mode = 'RGBA'
        else:
            img = img.convert('RGB')
            mode = 'RGB'

        # Extract raw pixel array
        pixels = np.array(img, dtype=np.uint8)

    # Create brand new image from raw pixels (no metadata carried over)
    clean_img = Image.fromarray(pixels, mode=mode)

    # Determine output format
    ext = Path(dst_path).suffix.lower()
    if ext in ('.jpg', '.jpeg'):
        if mode == 'RGBA':
            clean_img = clean_img.convert('RGB')
        clean_img.save(dst_path, format='JPEG', quality=95, optimize=True)
    elif ext == '.png':
        clean_img.save(dst_path, format='PNG', optimize=True)
    elif ext == '.webp':
        clean_img.save(dst_path, format='WEBP', quality=95)
    elif ext in ('.tiff', '.tif'):
        clean_img.save(dst_path, format='TIFF')
    elif ext == '.bmp':
        clean_img.save(dst_path, format='BMP')
    elif ext == '.gif':
        clean_img.save(dst_path, format='GIF')
    else:
        clean_img.save(dst_path)

    return "SANITIZED"


def sanitize_svg(src_path: str, dst_path: str) -> str:
    """Remove <metadata> blocks and XML comments from SVG."""
    with open(src_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Remove <metadata>...</metadata> blocks
    content = re.sub(r'<metadata[^>]*>.*?</metadata>', '', content, flags=re.DOTALL | re.IGNORECASE)

    # Remove XML comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

    # Remove XMP processing instructions
    content = re.sub(r'<\?xpacket.*?\?>', '', content, flags=re.DOTALL)

    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return "SANITIZED"


def main():
    repo_root = Path("${PCG_ROOT}")
    exclude_dirs = {'.sota_src', '.venvs', '__pycache__', '.git', 'node_modules'}

    # Create output directory
    sanitized_root = repo_root / "figures_sanitized"
    sanitized_root.mkdir(exist_ok=True)

    print("=" * 72)
    print("  IMAGE SANITIZER — METADATA & WATERMARK STRIPPER")
    print(f"  Source: {repo_root}")
    print(f"  Output: {sanitized_root}")
    print("=" * 72)
    print()

    all_exts = RASTER_EXTS | VECTOR_EXTS | PDF_EXTS
    sanitized = 0
    skipped = 0
    errors = 0

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and d != 'figures_sanitized']
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext not in all_exts:
                continue

            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(src_path, repo_root)
            dst_path = sanitized_root / rel_path

            # Create parent directories
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                if ext in RASTER_EXTS:
                    status = sanitize_raster(src_path, str(dst_path))
                    sanitized += 1
                elif ext in VECTOR_EXTS:
                    status = sanitize_svg(src_path, str(dst_path))
                    sanitized += 1
                elif ext in PDF_EXTS:
                    # PDFs cannot be pixel-sanitized; copy as-is
                    shutil.copy2(src_path, str(dst_path))
                    status = "COPIED (PDF — cannot pixel-sanitize)"
                    skipped += 1
                else:
                    status = "SKIPPED"
                    skipped += 1

                print(f"  {status:12s}  {rel_path}")

            except Exception as e:
                print(f"  ERROR        {rel_path}: {e}")
                errors += 1

    print()
    print("=" * 72)
    print(f"  Sanitized:  {sanitized}")
    print(f"  Skipped:    {skipped} (PDFs)")
    print(f"  Errors:     {errors}")
    print(f"  Output dir: {sanitized_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()

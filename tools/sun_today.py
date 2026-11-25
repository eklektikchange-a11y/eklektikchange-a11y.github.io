#!/usr/bin/env python3
"""Build a/sun_today.webp + a/sun_today.json from NASA SDO latest images.

Sources (NASA, public domain):
  - AIA 304 Å  — chromospheric / transition-region structure (fibrils, filaments, plage)
  - HMI continuum (HMIIC) — real sunspot umbra / penumbra

The disc is cropped tightly, limb-flattened, high-pass sharpened, recolored to a
deep-orange → bright yellow-orange H-alpha palette, and darkened where HMI shows spots.
"""
from __future__ import annotations
import io, json, os, re, sys, urllib.request
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageFilter

AIA_URL = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_0304.jpg"
HMI_URL = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_HMIIC.jpg"
AIA_TIMES = "https://sdo.gsfc.nasa.gov/assets/img/latest/times0304.txt"
HMI_TIMES = "https://sdo.gsfc.nasa.gov/assets/img/latest/timesHMIIC.txt"
UA = "EklektikWorld/1.0 (sun-today; educational)"

# H-alpha style luminance → RGB (values 0..1)
PAL = np.array([
    [0.00, 0.10, 0.015, 0.008],
    [0.12, 0.42, 0.07,  0.015],
    [0.30, 0.82, 0.22,  0.03],
    [0.48, 1.00, 0.48,  0.07],
    [0.64, 1.00, 0.68,  0.16],
    [0.78, 1.00, 0.84,  0.32],
    [0.90, 1.00, 0.94,  0.58],
    [1.00, 1.00, 0.99,  0.90],
], dtype=np.float32)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def parse_time(txt: str, key: str) -> str | None:
    # e.g. "0304: 20261125_122830"  or "HMIIC: 20261125_120000"
    m = re.search(rf"{re.escape(key)}\s*:\s*(\d{{8}})_(\d{{6}})", txt)
    if not m:
        return None
    return f"{m.group(1)}T{m.group(2)}"


def iso(tag: str | None) -> str | None:
    if not tag or len(tag) < 15:
        return None
    try:
        dt = datetime.strptime(tag, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def date_only(iso_s: str | None, fallback: str = "") -> str:
    if iso_s and len(iso_s) >= 10:
        return iso_s[:10]
    return fallback


def to_gray(im: Image.Image) -> np.ndarray:
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    # AIA 304 is almost pure red; HMI is orange — use luma but keep red weight high
    return a[..., 0] * 0.65 + a[..., 1] * 0.30 + a[..., 2] * 0.05


def fit_disc(g: np.ndarray, caption_h: int = 48) -> tuple[float, float, float]:
    """Center and radius of the solar disc from a continuum-like image."""
    h, w = g.shape
    work = g.copy()
    work[-caption_h:] = 0
    # threshold relative to the bright core
    mid = work[h // 2, w // 4: 3 * w // 4]
    hi = np.percentile(mid[mid > 10], 75) if np.any(mid > 10) else 128
    mask = work > max(12.0, hi * 0.18)
    # erode a little so prominences and glare don't pull the centroid
    try:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask.astype(np.uint8), k, iterations=2).astype(bool)
    except Exception:
        pass
    ys, xs = np.where(mask)
    if len(xs) < 1000:
        return (w - 1) / 2, (h - 1) / 2, min(w, h) * 0.45
    cx, cy = float(xs.mean()), float(ys.mean())
    # radius: median distance of limb pixels (mask edge)
    try:
        import cv2
        edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ey, ex = np.where(edge > 0)
        # keep only the outer ring (ignore sunspot holes)
        d = np.hypot(ex - cx, ey - cy)
        r = float(np.median(d[d > np.percentile(d, 70)])) if d.size else float(d.mean())
    except Exception:
        # fallback: max extent of the mask along mid-rows
        rr = np.hypot(xs - cx, ys - cy)
        r = float(np.percentile(rr, 99.2))
    return cx, cy, r


def radial_profile(g: np.ndarray, cx: float, cy: float, r: float, n: int = 256) -> np.ndarray:
    """Annular p60 of g, as a function of radius in [0, r] (n bins), smoothed."""
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    rho = np.hypot(xx - cx, yy - cy) / max(r, 1e-6)
    prof = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a, b = i / n, (i + 1) / n
        sel = (rho >= a) & (rho < b)
        if np.count_nonzero(sel) > 30:
            prof[i] = np.percentile(g[sel], 60)
        else:
            prof[i] = prof[i - 1] if i else 1.0
    # fill zeros, then smooth
    for i in range(1, n):
        if prof[i] <= 1e-3:
            prof[i] = prof[i - 1]
    # gaussian-ish smooth
    k = np.array([1, 4, 8, 4, 1], dtype=np.float32); k /= k.sum()
    pad = np.pad(prof, 2, mode="edge")
    prof = np.convolve(pad, k, mode="valid")
    return prof


def apply_profile(g: np.ndarray, cx: float, cy: float, r: float, prof: np.ndarray) -> np.ndarray:
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    rho = np.hypot(xx - cx, yy - cy) / max(r, 1e-6)
    t = np.clip(rho, 0, 0.999) * (len(prof) - 1)
    i0 = np.floor(t).astype(np.int32)
    f = t - i0
    p = prof[i0] * (1 - f) + prof[np.minimum(i0 + 1, len(prof) - 1)] * f
    p = np.maximum(p, np.percentile(prof[prof > 0], 10) * 0.3)
    return g / p


def colorize(L: np.ndarray) -> np.ndarray:
    """Map 0..1 luminance to the H-alpha palette."""
    x = np.clip(L, 0, 1)
    stops, cols = PAL[:, 0], PAL[:, 1:4]
    out = np.empty(L.shape + (3,), dtype=np.float32)
    # vectorized piecewise lerp
    idx = np.searchsorted(stops, x, side="right") - 1
    idx = np.clip(idx, 0, len(stops) - 2)
    x0, x1 = stops[idx], stops[idx + 1]
    t = np.clip((x - x0) / np.maximum(x1 - x0, 1e-6), 0, 1)[..., None]
    out[:] = cols[idx] * (1 - t) + cols[idx + 1] * t
    return out


def highpass(L: np.ndarray, sigma: float, amt: float) -> np.ndarray:
    im = Image.fromarray(np.clip(L * 255, 0, 255).astype(np.uint8), "L")
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32) / 255.0
    return L + (L - blur) * amt


def build(aia_im: Image.Image, hmi_im: Image.Image, out_w: int = 2048) -> np.ndarray:
    # use HMI (clean limb, no prominences) for geometry
    hg = to_gray(hmi_im)
    ag = to_gray(aia_im)
    # if sizes differ, scale HMI to AIA
    if hg.shape != ag.shape:
        hmi_im = hmi_im.resize(aia_im.size, Image.Resampling.LANCZOS)
        hg = to_gray(hmi_im)
    cx, cy, r = fit_disc(hg)
    # slightly shrink so we stay inside the limb (avoid caption / off-limb AIA glow)
    r_use = r * 0.975
    print(f"disc cx={cx:.1f} cy={cy:.1f} r={r:.1f}", flush=True)

    a_flat = apply_profile(ag, cx, cy, r_use, radial_profile(ag, cx, cy, r_use))
    h_flat = apply_profile(hg, cx, cy, r_use, radial_profile(hg, cx, cy, r_use))

    # contrast of AIA structure
    yy, xx = np.ogrid[:ag.shape[0], :ag.shape[1]]
    rho = np.hypot(xx - cx, yy - cy) / r_use
    disc = rho <= 1.0
    vals = a_flat[disc]
    lo, hi = np.percentile(vals, [3, 99.2])
    L = np.clip((a_flat - lo) / max(hi - lo, 1e-6), 0, 1)
    # two-scale high-pass: network (~8px) + fibrils (~2px)
    # high-pass only well inside the disc (the limb is noisy in AIA 304)
    fade = np.clip((0.93 - rho) / 0.10, 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    Lp = highpass(L, sigma=6.0, amt=0.70)
    Lp = highpass(Lp, sigma=1.8, amt=0.90)
    L = np.clip(L * (1 - fade) + Lp * fade, 0, 1)
    L = np.power(np.clip(L, 0, 1), 0.88)

    rgb = colorize(L)

    # HMI spots: pixels darker than the flattened photosphere
    hv = h_flat[disc]
    # flattened photosphere hovers near 1; spots << 1
    p10, p50 = np.percentile(hv, [8, 50])
    # penumbra / umbra thresholds
    pen = np.clip((0.92 - h_flat) / max(0.92 - 0.55, 1e-6), 0, 1)
    umb = np.clip((0.62 - h_flat) / max(0.62 - 0.28, 1e-6), 0, 1)
    pen = np.where(disc, pen, 0)
    umb = np.where(disc, umb, 0)
    # soften
    def blur1(a, s):
        im = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8), "L")
        return np.asarray(im.filter(ImageFilter.GaussianBlur(radius=s)), dtype=np.float32) / 255.0
    pen, umb = blur1(pen, 1.2), blur1(umb, 0.8)
    # darken towards deep umber; keep a hint of the orange so it doesn't go grey
    umber = np.array([0.10, 0.04, 0.015], dtype=np.float32)
    pcol = np.array([0.35, 0.14, 0.04], dtype=np.float32)
    rgb = rgb * (1 - 0.72 * pen[..., None]) + pcol * (0.35 * pen[..., None])
    rgb = rgb * (1 - 0.88 * umb[..., None]) + umber * (0.25 * umb[..., None])

    # crop to a tight square around the disc, scale so the disc radius is 0.98 * half-size
    # (shader uses 0.5 + 0.5*d / 0.98)
    margin = r_use * (1 / 0.98 - 1)
    x0, y0 = cx - r_use - margin, cy - r_use - margin
    x1, y1 = cx + r_use + margin, cy + r_use + margin
    # paste into a black canvas of the crop size, then resize
    H, W = rgb.shape[:2]
    # bilinear sample via PIL
    rgb8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    # mask softly at the limb so corners are black and the edge isn't a hard cut
    # sharp 2-px antialiased disc; no dark 'ring' — the shader adds limb glow
    px = r_use * 2.0 / max(rgb.shape[0], 1)
    edge = np.clip((1.0 - rho) / max(2.5 * px / r_use, 1e-4) + 0.5, 0, 1)
    rgb8 = (rgb8.astype(np.float32) * edge[..., None]).astype(np.uint8)
    im = Image.fromarray(rgb8, "RGB")
    # crop (clamped) then resize
    pad = 8
    box = (int(np.floor(x0) - pad), int(np.floor(y0) - pad), int(np.ceil(x1) + pad), int(np.ceil(y1) + pad))
    # if the box exceeds the image, pad
    cx0, cy0, cx1, cy1 = box
    canv = Image.new("RGB", (cx1 - cx0, cy1 - cy0), (0, 0, 0))
    src_box = (max(cx0, 0), max(cy0, 0), min(cx1, W), min(cy1, H))
    dest = (src_box[0] - cx0, src_box[1] - cy0)
    canv.paste(im.crop(src_box), dest)
    out = canv.resize((out_w, out_w), Image.Resampling.LANCZOS)
    return np.asarray(out), (cx, cy, r_use, 0.98)


def save_webp(arr: np.ndarray, path: str, q: int = 80) -> int:
    im = Image.fromarray(arr, "RGB")
    im.save(path, "WEBP", quality=q, method=6, exact=True)
    return os.path.getsize(path)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--aia", default="")
    ap.add_argument("--hmi", default="")
    ap.add_argument("--aia-time", default="")
    ap.add_argument("--hmi-time", default="")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--force-commit-stamp", action="store_true", help="always refresh fetched_at (for workflow_dispatch)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def parse_arg_time(s):
        if not s: return None
        if s.endswith("Z") and len(s) >= 20: return s
        t = s.replace("-", "").replace(":", "").replace("Z", "")
        if "T" not in t and len(t) >= 14:
            t = t[:8] + "T" + t[8:14]
        return iso(t) or s
    if args.aia:
        aia = Image.open(args.aia).convert("RGB")
        hmi = Image.open(args.hmi).convert("RGB")
        aia_t = parse_arg_time(args.aia_time)
        hmi_t = parse_arg_time(args.hmi_time)
    else:
        print("fetching SDO latest…", flush=True)
        aia = Image.open(io.BytesIO(fetch(AIA_URL))).convert("RGB")
        hmi = Image.open(io.BytesIO(fetch(HMI_URL))).convert("RGB")
        try:
            aia_t = iso(parse_time(fetch(AIA_TIMES).decode("utf-8", "replace"), "0304"))
        except Exception as e:
            print("times0304:", e); aia_t = None
        try:
            hmi_t = iso(parse_time(fetch(HMI_TIMES).decode("utf-8", "replace"), "HMIIC"))
        except Exception as e:
            print("timesHMIIC:", e); hmi_t = None

    # fallback observation time: HMI/AIA last-modified isn't passed; use today UTC if missing
    obs = aia_t or hmi_t or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = date_only(obs, datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    arr, meta = build(aia, hmi, 2048)
    webp = os.path.join(args.outdir, "sun_today.webp")
    n = save_webp(arr, webp, 80)
    info = {
        "date": day,
        "utc": obs,
        "aia": aia_t,
        "hmi": hmi_t,
        "src": "NASA SDO / AIA 304 + HMI continuum",
        "disc": 0.98,
        "bytes": n,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    js = os.path.join(args.outdir, "sun_today.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("wrote", webp, n, "json", info, flush=True)
    if args.preview:
        Image.fromarray(arr).resize((512, 512), Image.Resampling.LANCZOS).save(
            os.path.join(args.outdir, "preview_today.png"))
        # close-up of a mid-disc patch
        Image.fromarray(arr).crop((700, 700, 1300, 1300)).resize((512, 512)).save(
            os.path.join(args.outdir, "preview_today_close.png"))


if __name__ == "__main__":
    main()

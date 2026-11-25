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

# H-alpha style luminance → RGB (values 0..1). Highlights stay warm yellow, never white.
PAL = np.array([
    [0.00, 0.22, 0.030, 0.010],
    [0.14, 0.58, 0.100, 0.020],
    [0.30, 0.86, 0.230, 0.035],
    [0.46, 0.98, 0.380, 0.060],
    [0.60, 1.00, 0.540, 0.100],
    [0.74, 1.00, 0.720, 0.180],
    [0.86, 1.00, 0.850, 0.280],
    [1.00, 1.00, 0.920, 0.400],
], dtype=np.float32)

# source disc used in the texture, as a fraction of the fitted photospheric radius
R_IN = 0.97
# in the output texture the disc fills this fraction of the half-size (1px of aa beyond)
R_OUT = 0.998




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


def _rough_center(work: np.ndarray, caption_h: int) -> tuple[float, float]:
    h, w = work.shape
    mid = work[h // 2 - 30: h // 2 + 30, w // 2 - 30: w // 2 + 30]
    hi = float(np.percentile(mid, 80)) if mid.size else 128.0
    th = max(hi * 0.35, 8.0)
    xs = np.where(work[h // 2] > th)[0]
    ys = np.where(work[:, w // 2] > th)[0]
    if len(xs) > 50 and len(ys) > 50:
        return (float(xs[0] + xs[-1]) / 2.0, float(ys[0] + ys[-1]) / 2.0)
    return (w - 1) / 2.0, (h - 1) / 2.0


def _median_radial_profile(work: np.ndarray, cx: float, cy: float, caption_h: int) -> tuple[np.ndarray, float]:
    """Azimuthal median vs integer radius. Prominences occupy few azimuths, so they drop out."""
    h, w = work.shape
    yy, xx = np.ogrid[:h, :w]
    rho = np.hypot(xx - cx, yy - cy)
    rmax = int(max(8, min(cx, cy, w - 1 - cx, h - 1 - caption_h - cy)))
    prof = np.zeros(rmax, dtype=np.float32)
    last = 0.0
    for i in range(rmax):
        sel = (rho >= i) & (rho < i + 1) & (yy < h - caption_h)
        if np.count_nonzero(sel) > 24:
            last = float(np.median(work[sel]))
        prof[i] = last
    if rmax >= 15:
        k = np.hanning(11).astype(np.float32); k /= k.sum()
        prof = np.convolve(prof, k, mode="same")
    return prof.astype(np.float32), float(rmax)


def _radius_from_profile(prof: np.ndarray) -> float:
    n = len(prof)
    if n < 20:
        return float(n)
    inner = prof[int(0.12 * n): max(int(0.50 * n), int(0.12 * n) + 4)]
    ref = float(np.median(inner)) if inner.size else float(np.max(prof) or 1.0)
    if ref < 1e-3:
        return float(n * 0.4)
    high = np.where(prof > 0.70 * ref)[0]
    if len(high) == 0:
        return float(n * 0.4)
    start = int(np.percentile(high, 55))
    level = 0.42 * ref
    below = np.where(prof[start:] < level)[0]
    if len(below) == 0:
        return float(high[-1])
    i = start + int(below[0])
    if i <= 0:
        return float(i)
    z0, z1 = float(prof[i - 1]), float(prof[i])
    frac = (z0 - level) / max(z0 - z1, 1e-6)
    return float(i - 1 + np.clip(frac, 0, 1))


def fit_disc(g: np.ndarray, caption_h: int = 70) -> tuple[float, float, float]:
    """Fit the solar disc of one SDO JPEG.

    HMI continuum and AIA 304 'latest_2048' JPEGs do NOT share a crop or plate scale:
    HMI nearly fills the frame; AIA is a wider FOV with off-limb prominences. Always fit
    each image separately. Radius comes from the drop in the azimuthal-median profile
    (prominences are rejected as outliers), then a Kasa refinement on inlier limb rays.
    """
    h, w = g.shape
    work = g.copy()
    work[max(0, h - caption_h):] = 0
    work[:6] = 0
    cx, cy = _rough_center(work, caption_h)
    # two-pass centre: after a first radius, re-estimate from inlier limb points
    prof, _ = _median_radial_profile(work, cx, cy, caption_h)
    r0 = _radius_from_profile(prof)
    level = float(np.median(prof[int(0.15 * r0): max(int(0.55 * r0), 4)])) * 0.45 if r0 > 20 else 20.0

    angs = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    pts = []
    tmax = r0 * 1.35 if r0 > 20 else min(w, h) * 0.55
    ts = np.arange(0.0, tmax, 0.5)
    for a in angs:
        ca, sa = np.cos(a), np.sin(a)
        xsamp, ysamp = cx + ts * ca, cy + ts * sa
        valid = (xsamp >= 1) & (xsamp < w - 2) & (ysamp >= 1) & (ysamp < h - 2 - caption_h)
        if np.count_nonzero(valid) < 20:
            continue
        t = ts[valid]
        x, y = xsamp[valid], ysamp[valid]
        x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        fx, fy = x - x0, y - y0
        z = (work[y0, x0] * (1 - fx) * (1 - fy) + work[y0, x0 + 1] * fx * (1 - fy)
             + work[y0 + 1, x0] * (1 - fx) * fy + work[y0 + 1, x0 + 1] * fx * fy)
        # limb = first crossing near the expected radius (skip inner dark filaments)
        near = (t > 0.75 * r0) if r0 > 20 else (t > 8)
        below = np.where(near & (z < level))[0]
        if len(below) == 0:
            continue
        i = int(below[0])
        if i < 1:
            continue
        z0, z1 = float(z[i - 1]), float(z[i])
        frac = (level - z0) / (z1 - z0) if z1 != z0 else 0.0
        rr = float(t[i - 1] + np.clip(frac, 0, 1) * (t[i] - t[i - 1]))
        # prominences land well outside; reject them
        if r0 > 20 and abs(rr - r0) > 0.06 * r0:
            continue
        pts.append((cx + rr * ca, cy + rr * sa, rr))
    if len(pts) < 40:
        return cx, cy, float(r0)
    P = np.array(pts, dtype=np.float64)
    rmed = float(np.median(P[:, 2]))
    keep = np.abs(P[:, 2] - rmed) < 0.025 * rmed
    P = P[keep]
    if len(P) < 30:
        return cx, cy, rmed
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack((2 * x, 2 * y, np.ones(len(P))))
    bvec = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    cx2, cy2, c0 = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = float(np.sqrt(max(c0 + cx2 * cx2 + cy2 * cy2, 1.0)))
    if abs(cx2 - cx) > 24 or abs(cy2 - cy) > 24 or abs(r2 - rmed) > 0.05 * rmed:
        return cx, cy, rmed
    return cx2, cy2, r2


def radial_profile(g: np.ndarray, cx: float, cy: float, r: float, n: int = 320, cap: float = 0.995) -> np.ndarray:
    """Azimuthal median of g vs radius in [0, r]; bins beyond `cap` are extrapolated."""
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    rho = np.hypot(xx - cx, yy - cy) / max(r, 1e-6)
    prof = np.zeros(n, dtype=np.float32)
    last = 1.0
    last_ok = 0
    for i in range(n):
        a, b = i / n, (i + 1) / n
        if a >= cap:
            break
        sel = (rho >= a) & (rho < b)
        if np.count_nonzero(sel) > 40:
            prof[i] = float(np.median(g[sel]))
            last, last_ok = prof[i], i
        else:
            prof[i] = last
    # extrapolate the outermost bins from the last stable interior (avoid limb / off-limb)
    for i in range(last_ok + 1, n):
        prof[i] = last
    # smooth
    k = np.array([1, 4, 10, 4, 1], dtype=np.float32); k /= k.sum()
    pad = np.pad(prof, 2, mode="edge")
    prof = np.convolve(pad, k, mode="valid")
    return prof.astype(np.float32)


def apply_profile(g: np.ndarray, cx: float, cy: float, r: float, prof: np.ndarray) -> np.ndarray:
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    rho = np.hypot(xx - cx, yy - cy) / max(r, 1e-6)
    t = np.clip(rho, 0, 0.999) * (len(prof) - 1)
    i0 = np.floor(t).astype(np.int32)
    f = (t - i0).astype(np.float32)
    p = prof[i0] * (1 - f) + prof[np.minimum(i0 + 1, len(prof) - 1)] * f
    p = np.maximum(p, float(np.percentile(prof[prof > 0], 5)) * 0.4)
    return g / p



def azimuth_fill(g: np.ndarray, cx: float, cy: float, r: float, rim: float = 0.94) -> np.ndarray:
    """Copy the intensity at `rim` outward by azimuth, so a later blur does not see a black off-disc."""
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    rho = np.hypot(xx - cx, yy - cy) / max(r, 1e-6)
    ang = np.arctan2(yy - cy, xx - cx)
    nang = 360
    bins = ((ang + np.pi) / (2 * np.pi) * nang).astype(np.int32) % nang
    ring = (rho >= rim - 0.04) & (rho <= min(rim + 0.02, 0.995))
    az = np.full(nang, np.median(g[rho <= 1.0]), dtype=np.float64)
    for i in range(nang):
        sel = ring & (bins == i)
        if np.count_nonzero(sel) >= 6:
            az[i] = float(np.median(g[sel]))
    # wrap-smooth ~12 deg
    k = 13
    pad = np.r_[az[-k:], az, az[:k]]
    ker = np.hanning(k); ker /= ker.sum()
    az = np.convolve(pad, ker, mode="valid")[k // 2: k // 2 + nang]
    fill = az[bins].astype(np.float32)
    return np.where(rho <= rim, g, fill)


def large_scale_flatten(g: np.ndarray, cx: float, cy: float, r: float, sig_frac: float = 0.16) -> np.ndarray:
    """Divide out the smooth 2D background (EW imbalance, remaining limb glow, large plage pedestals)."""
    work = azimuth_fill(g, cx, cy, r, rim=0.95)
    lo, hi = float(np.percentile(work, 0.5)), float(np.percentile(work, 99.5))
    t = np.clip((work - lo) / max(hi - lo, 1e-6), 0, 1)
    im = Image.fromarray((t * 255).astype(np.uint8), "L")
    bg = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=max(r * sig_frac, 8.0))), dtype=np.float32) / 255.0
    bg = bg * (hi - lo) + lo
    bg = np.maximum(bg, float(np.percentile(work, 10)) * 0.5)
    return g / bg


def colorize(L: np.ndarray) -> np.ndarray:
    """Map 0..1 luminance to the H-alpha palette."""
    x = np.clip(L, 0, 1)
    stops, cols = PAL[:, 0], PAL[:, 1:4]
    idx = np.searchsorted(stops, x, side="right") - 1
    idx = np.clip(idx, 0, len(stops) - 2)
    x0, x1 = stops[idx], stops[idx + 1]
    t = np.clip((x - x0) / np.maximum(x1 - x0, 1e-6), 0, 1)[..., None]
    return cols[idx] * (1 - t) + cols[idx + 1] * t


def highpass(L: np.ndarray, sigma: float, amt: float) -> np.ndarray:
    im = Image.fromarray(np.clip(L * 255, 0, 255).astype(np.uint8), "L")
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32) / 255.0
    return L + (L - blur) * amt


def s_curve(x: np.ndarray, mid: float = 0.48, k: float = 1.35) -> np.ndarray:
    """Gentle contrast around mid-grey; keeps ends pinned."""
    # sigmoid-ish around mid, then rescale to 0..1 of the same domain
    y = 1.0 / (1.0 + np.exp(-k * 8.0 * (x - mid)))
    y0 = 1.0 / (1.0 + np.exp(-k * 8.0 * (0.0 - mid)))
    y1 = 1.0 / (1.0 + np.exp(-k * 8.0 * (1.0 - mid)))
    return (y - y0) / max(y1 - y0, 1e-6)


def soft_highlight(x: np.ndarray, start: float = 0.82, range_: float = 0.22) -> np.ndarray:
    """Roll off bright plage so it stays yellow, never blows to white."""
    t = np.clip((x - start) / max(range_, 1e-6), 0, 1)
    # smooth shoulder: start + range * (1-e^{-2t})/ (1-e^{-2})  but never above start+0.16
    shoulder = start + 0.16 * (1.0 - np.exp(-2.2 * t)) / (1.0 - np.exp(-2.2))
    return np.where(x > start, np.minimum(x, shoulder), x)


def sample_disc(src: np.ndarray, cx: float, cy: float, r_src: float, out_w: int, r_out: float) -> np.ndarray:
    """Resample a circular region of `src` so it fills a disc of radius r_out * (out_w/2) in the output."""
    # inverse map: output pixel -> source
    half = (out_w - 1) / 2.0
    # map output rho=1 (at r_out * out_w/2) to source radius r_src
    scale = r_src / (r_out * (out_w / 2.0))
    yy, xx = np.ogrid[:out_w, :out_w]
    sx = cx + (xx - half) * scale
    sy = cy + (yy - half) * scale
    # bilinear, black if out of bounds
    h, w = src.shape[:2]
    ch = 1 if src.ndim == 2 else src.shape[2]
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    fx = (sx - x0).astype(np.float32)
    fy = (sy - y0).astype(np.float32)
    ok = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    def grab(arr):
        if arr.ndim == 2:
            v = (arr[y0c, x0c] * (1 - fx) * (1 - fy) + arr[y0c, x0c + 1] * fx * (1 - fy)
                 + arr[y0c + 1, x0c] * (1 - fx) * fy + arr[y0c + 1, x0c + 1] * fx * fy)
            return np.where(ok, v, 0.0).astype(np.float32)
        out = np.zeros((out_w, out_w, arr.shape[2]), dtype=np.float32)
        for k in range(arr.shape[2]):
            out[..., k] = grab(arr[..., k])
        return out
    return grab(src)


def build(aia_im: Image.Image, hmi_im: Image.Image, out_w: int = 2048):
    # Do not resize one image to the other's pixel size: the two JPEGs have different FOVs.
    hg = to_gray(hmi_im)
    ag = to_gray(aia_im)
    cx_a, cy_a, r_a = fit_disc(ag)
    cx_h, cy_h, r_h = fit_disc(hg)
    r_src_a = r_a * R_IN
    r_src_h = r_h * R_IN
    print(
        f"AIA disc cx={cx_a:.2f} cy={cy_a:.2f} r={r_a:.2f}  using {R_IN}*R={r_src_a:.2f}",
        flush=True,
    )
    print(
        f"HMI disc cx={cx_h:.2f} cy={cy_h:.2f} r={r_h:.2f}  using {R_IN}*R={r_src_h:.2f}",
        flush=True,
    )

    # flatten each image in its own disc coordinates
    a_rad = apply_profile(ag, cx_a, cy_a, r_src_a, radial_profile(ag, cx_a, cy_a, r_src_a, cap=0.99))
    a_flat = large_scale_flatten(a_rad, cx_a, cy_a, r_src_a, sig_frac=0.10)
    h_flat = apply_profile(hg, cx_h, cy_h, r_src_h, radial_profile(hg, cx_h, cy_h, r_src_h, cap=0.99))

    def blur_f(arr, sig):
        lo, hi = float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5))
        t = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        im = Image.fromarray((t * 255).astype(np.uint8), "L")
        b = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=sig)), dtype=np.float32) / 255.0
        return b * (hi - lo) + lo
    a_den = 0.68 * a_flat + 0.32 * blur_f(a_flat, 1.1)

    yy, xx = np.ogrid[:ag.shape[0], :ag.shape[1]]
    rho_s = np.hypot(xx - cx_a, yy - cy_a) / r_src_a
    # flattened field is ~1; compress leftover bright outliers more at the used edge
    w_edge = np.clip((rho_s - 0.78) / 0.20, 0, 1)
    w_edge = w_edge * w_edge * (3 - 2 * w_edge)
    over = np.maximum(a_den - 1.12, 0.0)
    a_den = a_den - over * (0.45 + 0.50 * w_edge)

    inside = a_den[rho_s <= 0.92]
    lo, mid, hi = np.percentile(inside, [8, 50, 99.0])
    Llin = np.clip((a_den - lo) / max(hi - lo, 1e-6), 0, 1)
    mid_lin = float(np.clip((mid - lo) / max(hi - lo, 1e-6), 0.05, 0.95))
    # power that lands the quiet-sun median on deep orange (~0.46)
    pwr = np.log(0.46) / np.log(mid_lin)
    pwr = float(np.clip(pwr, 0.45, 1.8))
    L = np.power(np.clip(Llin, 0, 1), pwr)
    # fibril-scale unsharp (no fine grain); keep a floor so close-ups aren't two-tone holes
    L = highpass(L, sigma=8.0, amt=0.48)
    L = np.clip(L, 0.10, 1)
    L = s_curve(L, mid=0.48, k=0.70)
    L = soft_highlight(L, start=0.84, range_=0.20)
    L = np.clip(L, 0, 1)

    # HMI spots on the source grid
    pen = np.clip((0.90 - h_flat) / max(0.90 - 0.55, 1e-6), 0, 1)
    umb = np.clip((0.60 - h_flat) / max(0.60 - 0.28, 1e-6), 0, 1)
    def blur1(a, sig):
        im = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8), "L")
        return np.asarray(im.filter(ImageFilter.GaussianBlur(radius=sig)), dtype=np.float32) / 255.0
    pen, umb = blur1(pen, 1.1), blur1(umb, 0.7)

    # gentle H-alpha limb brightening on luminance (shader adds the glowing ring)
    limb = 1.0 + 0.06 * np.clip(rho_s, 0, 1) ** 2 + 0.05 * np.clip(rho_s, 0, 1) ** 8
    L = soft_highlight(np.clip(L * limb, 0, 1), start=0.80, range_=0.24)
    L = np.clip(L, 0, 0.94)

    rgb = colorize(L)
    # never let highlights go white (keep a yellow-orange ceiling) before spots
    rgb[..., 1] = np.minimum(rgb[..., 1], rgb[..., 0] * 0.92)
    rgb[..., 2] = np.minimum(rgb[..., 2], rgb[..., 0] * 0.50)
    rgb = np.clip(rgb, 0, 1)

    # resample AIA color and HMI spots, each from its own disc, onto the same output circle
    out = sample_disc(rgb, cx_a, cy_a, r_src_a, out_w, R_OUT)
    pen_o = sample_disc(pen, cx_h, cy_h, r_src_h, out_w, R_OUT)
    umb_o = sample_disc(umb, cx_h, cy_h, r_src_h, out_w, R_OUT)
    umber = np.array([0.12, 0.04, 0.015], dtype=np.float32)
    pcol = np.array([0.40, 0.16, 0.04], dtype=np.float32)
    out = out * (1 - 0.70 * pen_o[..., None]) + pcol * (0.30 * pen_o[..., None])
    out = out * (1 - 0.86 * umb_o[..., None]) + umber * (0.22 * umb_o[..., None])
    out = np.clip(out, 0, 1)
    yy, xx = np.ogrid[:out_w, :out_w]
    rho = np.hypot(xx - (out_w - 1) / 2.0, yy - (out_w - 1) / 2.0) / (out_w / 2.0)
    # mild blur only at the very edge (kills AIA/JPEG speckle, keeps interior fibrils)
    def blur_rgb(arr, sig):
        im = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), "RGB")
        b = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=sig)), dtype=np.float32) / 255.0
        return b
    w_edge = np.clip((rho - 0.93) / 0.06, 0, 1)
    w_edge = w_edge * w_edge * (3 - 2 * w_edge)
    out = out * (1 - w_edge[..., None]) + blur_rgb(out, 1.6) * w_edge[..., None]
    # Fill *outside* the disc with the mean limb color (not black). Bilinear
    # samples at the cap edge and any slight UV warp then stay warm orange.
    ring = (rho >= 0.955) & (rho <= 0.996)
    if np.any(ring):
        fill = out[ring].mean(axis=0).astype(np.float32)
    else:
        fill = np.array([0.82, 0.46, 0.12], dtype=np.float32)
    aa = np.clip((R_OUT - rho) / (1.25 / (out_w / 2.0)) + 0.5, 0, 1)
    out = out * aa[..., None] + fill * (1.0 - aa)[..., None]
    out = np.clip(out, 0, 1)
    return (out * 255.0 + 0.5).astype(np.uint8), (cx_a, cy_a, r_a, r_src_a, R_OUT)


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
        "disc": R_OUT,
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
        # diagnostic: draw fitted used-disc on each source
        from PIL import ImageDraw
        for tag, im, cx, cy, r in (
            ("aia", aia, *fit_disc(to_gray(aia))),
            ("hmi", hmi, *fit_disc(to_gray(hmi))),
        ):
            ov = im.convert("RGB").copy()
            d = ImageDraw.Draw(ov)
            d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0, 255, 0), width=3)
            d.ellipse([cx-r*R_IN, cy-r*R_IN, cx+r*R_IN, cy+r*R_IN], outline=(255, 255, 0), width=3)
            ov.resize((700, 700), Image.Resampling.LANCZOS).save(
                os.path.join(args.outdir, f"fit_{tag}.png"))


if __name__ == "__main__":
    main()

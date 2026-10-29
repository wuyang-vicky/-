# -*- coding: utf-8 -*-
"""纹样解析：平面(二值矢量)/实物照片(多色描摹) 自动分流，并输出花纹特征。"""
import math
import numpy as np
import cv2

CREAM = "#F7F1E3"
GOLD = "#C9A227"
NEAR_WHITE = "#FFFDF5"
INK = "#3A2318"


def _hex(bgr):
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return "#%02X%02X%02X" % (r, g, b)


def _dominant(img):
    sm = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    K = 6
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(lab, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=K)
    i = int(np.argmax(counts))
    return centers[i].astype(np.float32), float(counts[i]) / counts.sum()


def _is_flat(img):
    _, bgfrac = _dominant(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > 60) & (hsv[..., 2] > 50)
    if m.sum() < 400:
        return True
    hist = np.bincount((hsv[..., 0][m] // 10).astype(np.int64), minlength=18).astype(np.float64)
    return bool(bgfrac > 0.45 or hist.max() / hist.sum() > 0.62)


def _border_clean(mask, min_comp):
    h, w = mask.shape
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.INTER_NEAREST)
    keep = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < min_comp:
            continue
        if x <= 1 or y <= 1 or x + ww >= w - 1 or y + hh >= h - 1:
            continue
        keep.append(i)
    if not keep:
        return None
    return (np.isin(lab, keep).astype(np.uint8) * 255)


def _clean(mask, min_comp):
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.INTER_NEAREST)
    keep = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_comp]
    if not keep:
        return None
    return (np.isin(lab, keep).astype(np.uint8) * 255)


def _contours_pts(mask, eps_ratio=0.0012, min_area=6):
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = max(0.8, min(3.0, eps_ratio * cv2.arcLength(c, True)))
        p = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(p) >= 3:
            out.append(p)
    return out


def _poly_to_d(p):
    return "M " + " L ".join("%d %d" % (int(x), int(y)) for x, y in p) + " Z"


def _svg(w, h, body):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" '
            'preserveAspectRatio="xMidYMid meet"><rect width="%d" height="%d" fill="%s"/>%s</svg>'
            % (w, h, w, h, CREAM, "".join(body)))
def _lineart_svg(w, h, polys):
    body = ['<path fill="none" stroke="%s" stroke-width="1" stroke-linejoin="round" d="%s"/>'
            % (INK, _poly_to_d(p)) for p in polys]
    return _svg(w, h, body)


def _color_only_svg(img, K=6, block=6):
    long = 200
    sc = long / max(img.shape[:2])
    sm = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    H, W = sm.shape[:2]
    pix = sm.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(pix, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    q = centers[labels].reshape(H, W, 3).astype(np.uint8)
    rects = []
    for yy in range(0, H, block):
        for xx in range(0, W, block):
            b, g, r = [int(v) for v in q[yy, xx]]
            rects.append('<rect x="%d" y="%d" width="%d" height="%d" fill="#%02X%02X%02X"/>'
                         % (xx, yy, block, block, r, g, b))
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" '
            'preserveAspectRatio="xMidYMid meet"><rect width="%d" height="%d" fill="#EFE7D2"/>%s</svg>'
            % (W, H, W, H, "".join(rects)))


def _symmetry(mask):
    h, w = mask.shape
    hh = h - (h % 2)
    ww = w - (w % 2)
    m = mask[:hh, :ww]
    left, right = m[:, :ww // 2], m[:, ww // 2:]
    ml = (left == cv2.flip(right, 1)).mean()
    top, bot = m[:hh // 2, :], m[hh // 2:, :]
    mv = (top == cv2.flip(bot, 0)).mean()
    if ml > 0.85 and mv > 0.85:
        return "中心对称(四向对称)"
    if ml > 0.85:
        return "左右镜像对称"
    if mv > 0.85:
        return "上下镜像对称"
    return "自由构图/不对称"


def _repetition(mask):
    m = cv2.resize((mask > 0).astype(np.float32), (128, 128))
    f = np.fft.fft2(m - m.mean())
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac = np.fft.fftshift(ac)
    if ac.max() > 0:
        ac = ac / ac.max()
    H, W = ac.shape
    y0, x0 = H // 2, W // 2
    band = ac.copy()
    band[max(0, y0 - 6):y0 + 7, max(0, x0 - 6):x0 + 7] = 0
    return int(min(100, round(float(band.max()) * 100)))


def _features(polys, mask, img):
    n = len(polys)
    H, W = mask.shape
    total_len = 0
    cats = {}
    for p in polys:
        pi = p.astype(np.int32)
        area = cv2.contourArea(pi)
        peri = cv2.arcLength(pi, True)
        total_len += peri
        if area < (H * W) * 0.0009:
            continue
        x, y, bw, bh = cv2.boundingRect(pi)
        solidity = area / max(1, bw * bh)
        circ = 4 * math.pi * area / max(1, peri * peri)
        ar = max(bw, bh) / max(1, min(bw, bh))
        if circ > 0.7 and 0.7 <= ar <= 1.5:
            c = "团花/宝相花"
        elif ar > 1.8:
            c = "缠枝/卷草"
        elif solidity < 0.62:
            c = "几何/回纹"
        else:
            c = "散点/辅纹"
        cats[c] = cats.get(c, 0) + 1
    complexity = int(min(100, total_len / max(1, (H + W) * 2) * 25))
    rep = _repetition(mask)
    fg = mask > 0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    fgL = float(lab[..., 0][fg].mean()) if fg.sum() else 0.0
    bgL = float(lab[..., 0][~fg].mean()) if (~fg).sum() else 0.0
    contrast = round(fgL / bgL, 2) if bgL > 0 else 0
    return {
        "contour_count": int(n),
        "total_edge_len": int(total_len),
        "complexity": int(complexity),
        "repetition": int(rep),
        "contrast": contrast,
        "fg_ratio": round(float(fg.mean()), 3),
        "motifs": cats,
    }


def _finalize(img, mask, polys, base_svg, palette, mode, h, w, h0, w0, layers):
    lineart = _lineart_svg(w, h, polys)
    color_only = _color_only_svg(img)
    feat = _features(polys, mask, img)
    feat["symmetry"] = _symmetry(mask)
    return {
        "svg": base_svg, "palette": palette, "symmetry": feat["symmetry"],
        "contours": len(polys), "size": [w0, h0], "mode": mode, "layers": int(layers),
        "lineart_svg": lineart, "color_only_svg": color_only, "features": feat,
    }
def _parse_flat(img, h0, w0):
    h, w = img.shape[:2]
    bg, _ = _dominant(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.linalg.norm(lab - bg, axis=2)
    fg = (d > 22).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = _border_clean(fg, max(40, int(w * h * 0.0005)))
    if fg is None:
        return None
    polys = _contours_pts(fg)
    if not polys:
        return None
    col = _hex(np.median(img[fg > 0], axis=0))
    body = ['<path fill-rule="evenodd" fill="%s" stroke="%s" stroke-width="0.6" d="%s"/>'
            % (col, col, _poly_to_d(p)) for p in polys]
    base = _svg(w, h, body)
    return _finalize(img, fg, polys, base, [col], "flat", h, w, h0, w0, 1)


def _build(labels, centers, h, w, bg, area_thr, min_comp):
    layers = []
    polys = []
    mask = np.zeros((h, w), np.uint8)
    for ci in range(len(centers)):
        if ci in bg:
            continue
        b, g, r = [int(v) for v in centers[ci]]
        sat = max(b, g, r) - min(b, g, r)
        val = max(b, g, r)
        if sat < 18 and val > 226:
            fill, stroke, sw = NEAR_WHITE, GOLD, 1.1
        else:
            fill, stroke, sw = _hex(centers[ci]), _hex(centers[ci]), 0.7
        m = ((labels == ci).reshape(h, w).astype(np.uint8)) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = _clean(m, min_comp)
        if m is None:
            continue
        if int((m > 0).sum()) < area_thr:
            continue
        cps = _contours_pts(m, 0.005, 14)
        if not cps:
            continue
        mask = np.maximum(mask, m)
        polys.extend(cps)
        ps = [_poly_to_d(p) for p in cps]
        layers.append((int((m > 0).sum()), fill, stroke, sw, ps))
    layers.sort(key=lambda t: -t[0])
    return layers[:9], polys, mask


def _parse_multi(img, h0, w0, k, max_side):
    h, w = img.shape[:2]
    sm = cv2.bilateralFilter(img, 9, 65, 65)
    sm = cv2.medianBlur(sm, 3)
    pix = sm.reshape(-1, 3).astype(np.float32)
    sample = pix[::4] if pix.shape[0] > 60000 else pix
    K = int(min(k, max(4, sample.shape[0] // 40)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(sample, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    c32 = centers.astype(np.int32)
    labels = (((pix.astype(np.int32)[:, None, :] - c32[None, :, :]) ** 2).sum(2)).argmin(1)
    bpx = np.concatenate([sm[0], sm[-1], sm[:, 0], sm[:, -1]]).astype(np.int32)
    med = np.median(bpx, axis=0)
    bg = set()
    hsv = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    for ci in range(K):
        if float(np.linalg.norm(c32[ci] - med)) < 70:
            bg.add(ci)
        if hsv[ci][1] < 38 and 50 < hsv[ci][2] < 210:
            bg.add(ci)
    area_thr = (w * h) * 0.0006
    min_comp = max(24, int(w * h * 0.0003))
    layers, polys, mask = _build(labels, centers, h, w, bg, area_thr, min_comp)
    guard = 0
    while len(layers) < 3 and guard < 2:
        area_thr *= 0.45
        min_comp = int(min_comp * 0.6)
        layers, polys, mask = _build(labels, centers, h, w, bg, area_thr, min_comp)
        guard += 1
    if not polys:
        return None
    body = ['<path fill-rule="evenodd" fill="%s" stroke="%s" stroke-width="%.1f" d="%s"/>'
            % (fill, stroke, sw, " ".join(ps)) for _, fill, stroke, sw, ps in layers]
    counts = np.bincount(labels, minlength=K)
    palette = [_hex(centers[i]) for i in np.argsort(-counts)][:5]
    base = _svg(w, h, body)
    return _finalize(img, mask, polys, base, palette, "multi", h, w, h0, w0, len(layers))


def parse_image_bytes_v2(data, k=12, max_side=700):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h0, w0 = img.shape[:2]
    sc = float(max_side) / max(h0, w0)
    if sc < 1.0:
        img = cv2.resize(img, (int(w0 * sc), int(h0 * sc)), interpolation=cv2.INTER_AREA)
    if _is_flat(img):
        res = _parse_flat(img, h0, w0)
        if res is not None:
            return res
    return _parse_multi(img, h0, w0, k, max_side)

# -*- coding: utf-8 -*-
"""绣纹智绘 - 故宫纹样解析与创新生成演示 (Flask 后端, 零标注零训练)"""
import os, sys, io, json, random
BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, send_file
from imgparse import parse_image_bytes_v2

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = r"G:\剪纸素材\剪纸素材"
if not os.path.isdir(SAMPLE_DIR):
    SAMPLE_DIR = os.path.join(BASE, "samples")
app = Flask(__name__, template_folder=os.path.join(BASE, "templates"))

RED = "#9D2933"
GOLD = "#E8C36B"
CREAM = "#F7F1E3"
DARK = "#5C1A1B"

def parse_image_bytes(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    fg = cv2.bitwise_not(gray) if border.mean() > 127 else gray
    g = cv2.GaussianBlur(fg, (5, 5), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ker = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker, iterations=2)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (w * h) * 0.0008
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = 0.012 * cv2.arcLength(c, True)
        p = cv2.approxPolyDP(c, eps, True)
        pts = [(int(x), int(y)) for x, y in p.reshape(-1, 2)]
        if len(pts) >= 3:
            polys.append(pts)
    paths = ["M " + " L ".join("%d %d" % (x, y) for x, y in pts) + " Z" for pts in polys]
    return {"svg": build_svg(w, h, paths), "palette": dominant_colors(img, 5),
            "symmetry": detect_symmetry(bw), "contours": len(polys), "size": [w, h]}

def build_svg(w, h, paths):
    body = "".join('<path d="%s" fill="%s" fill-opacity="0.92"/>' % (d, RED) for d in paths)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" '
            'preserveAspectRatio="xMidYMid meet"><rect width="%d" height="%d" fill="%s"/>%s</svg>'
            ) % (w, h, w, h, CREAM, body)

def dominant_colors(img, k):
    small = cv2.resize(img, (80, 80))
    pix = small.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pix, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    out = ["#%02X%02X%02X" % (int(r), int(g), int(b))
           for i in np.argsort(-counts) for b, g, r in [centers[i]]]
    return out

def detect_symmetry(bw):
    h, w = bw.shape
    left, right = bw[:, :w // 2], bw[:, w // 2:]
    ml = (left == cv2.flip(right, 1)).mean() if left.shape == right.shape else 0
    top, bot = bw[:h // 2, :], bw[h // 2:, :]
    mv = (top == cv2.flip(bot, 0)).mean() if top.shape == bot.shape else 0
    if ml > 0.85 and mv > 0.85:
        return "中心对称(四向对称)"
    if ml > 0.85:
        return "左右镜像对称"
    if mv > 0.85:
        return "上下镜像对称"
    return "自由构图/不对称"
FIELD_KINDS = ("binglie", "shuiwen")  # 满版生成类（细分 / 参数化机制），不遵循母题复制式对称

def _sanitize_points(pts, cap=2000):
    """校验并限幅客户端传入的母题点；非法则返回 None（回退默认母题）。"""
    if not isinstance(pts, list) or len(pts) > cap:
        return None
    out = []
    for p in pts:
        if not (isinstance(p, list) and len(p) == 2):
            return None
        try:
            xf, yf = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            return None
        xf = max(-2.0, min(2.0, xf))
        yf = max(-2.0, min(2.0, yf))
        out.append([xf, yf])
    return out

def motif(kind, cx, cy, s, rng, color, points=None):
    el = []
    if kind == "lotus":
        petals = rng.randint(6, 10)
        for i in range(petals):
            a = 2 * np.pi * i / petals
            px = cx + s * 0.55 * np.cos(a)
            py = cy + s * 0.55 * np.sin(a)
            deg = a * 180 / np.pi
            el.append('<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" fill="%s" transform="rotate(%.1f %.1f %.1f)"/>'
                      % (px, py, s * 0.30, s * 0.14, color, deg, px, py))
        el.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (cx, cy, s * 0.20, color))
    elif kind == "cloud":
        bumps = rng.randint(3, 5)
        for i in range(bumps):
            x = cx - s * 0.6 + i * (s * 1.2 / (bumps - 1))
            r = s * (0.22 if i in (0, bumps - 1) else 0.30)
            el.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (x, cy, r, color))
        el.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (cx, cy - s * 0.18, s * 0.16, color))
    elif kind == "meander":
        n = rng.randint(4, 6)
        pts = ["%.1f,%.1f" % (cx + s * 0.7 * np.cos(2 * np.pi * i / n),
                              cy + s * 0.7 * np.sin(2 * np.pi * i / n)) for i in range(n + 1)]
        el.append('<polygon points="%s" fill="none" stroke="%s" stroke-width="%.1f"/>' % (" ".join(pts), color, s * 0.08))
    elif kind == "baoxiang":
        # 宝相花：内外两圈花瓣 + 中心花蕊（辐射式团花复合母题）
        for ring, (pc, rr, pr) in enumerate([(rng.randint(8, 12), s * 0.5, s * 0.16),
                                             (rng.randint(6, 8), s * 0.30, s * 0.13)]):
            for i in range(pc):
                a = 2 * np.pi * i / pc + ring * 0.22
                px = cx + rr * np.cos(a)
                py = cy + rr * np.sin(a)
                deg = a * 180 / np.pi
                el.append('<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" fill="%s" transform="rotate(%.1f %.1f %.1f)"/>'
                          % (px, py, pr, pr * 0.45, color, deg, px, py))
        el.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (cx, cy, s * 0.15, color))
    elif kind == "lianzhu":
        # 联珠纹：环列小圆珠 + 外环
        n = rng.randint(10, 16)
        for i in range(n):
            a = 2 * np.pi * i / n
            x = cx + (s * 0.5) * np.cos(a)
            y = cy + (s * 0.5) * np.sin(a)
            el.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (x, y, s * 0.10, color))
        el.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="none" stroke="%s" stroke-width="%.1f"/>'
                  % (cx, cy, s * 0.55, color, s * 0.05))
    elif kind == "guibei":
        # 龟背锦：六边形单元（可平铺为六边锦地）
        R = s * 0.5
        pts = ["%.1f,%.1f" % (cx + R * np.cos(2 * np.pi * i / 6 + np.pi / 6),
                              cy + R * np.sin(2 * np.pi * i / 6 + np.pi / 6)) for i in range(6)]
        el.append('<polygon points="%s" fill="none" stroke="%s" stroke-width="%.1f"/>' % (" ".join(pts), color, s * 0.06))
        el.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (cx, cy, s * 0.12, color))
    elif kind == "custom" and points:
        # 自定义母题：用解析得到的主轮廓（归一化点）填充为实体基元
        d = "M " + " L ".join("%.1f %.1f" % (cx + x * s, cy + y * s) for x, y in points) + " Z"
        el.append('<path d="%s" fill="%s" fill-rule="evenodd"/>' % (d, color))
    else:
        for rot in (0, 45):
            el.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" transform="rotate(%d %.1f %.1f)"/>'
                      % (cx - s * 0.45, cy - s * 0.45, s * 0.9, s * 0.9, color, rot, cx, cy))
        el.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (cx, cy, s * 0.14, color))
    return el


def _gen_header(size, with_ring=True, bg=RED):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" '
             'preserveAspectRatio="xMidYMid meet"><rect width="%d" height="%d" fill="%s"/>'
             % (size, size, size, size, bg)]
    if with_ring:
        c = size // 2
        parts.append('<circle cx="%d" cy="%d" r="%d" fill="none" stroke="%s" stroke-width="6"/>' % (c, c, int(size * 0.46), GOLD))
        parts.append('<circle cx="%d" cy="%d" r="%d" fill="none" stroke="%s" stroke-width="2"/>' % (c, c, int(size * 0.42), GOLD))
    return parts


def generate_papercut(symmetry, kind, layers, seed, size=640, motif_points=None):
    if kind in FIELD_KINDS:
        return generate_field(kind, seed, size, layers)
    if not isinstance(motif_points, list):
        motif_points = None
    rng = random.Random(seed)
    cx = cy = size / 2
    def draw(k, x, y, sc):
        return motif(k, x, y, sc, rng, GOLD, points=(motif_points if k == "custom" else None))
    parts = _gen_header(size, with_ring=(symmetry in ("radial", "mirror", "four")))
    if symmetry == "radial":
        folds = rng.randint(4, 8)
        for layer in range(1, layers + 1):
            R = size * 0.12 * layer
            for f in range(folds):
                ang = 2 * np.pi * f / folds + layer * 0.25
                parts += draw(kind, cx + R * np.cos(ang), cy + R * np.sin(ang), R * 0.5)
    elif symmetry == "mirror":
        for layer in range(1, layers + 1):
            R = size * 0.14 * layer
            for sx in (-1, 1):
                parts += draw(kind, cx + sx * R * 0.62, cy - R * 0.15 * layer, R * 0.5)
    elif symmetry == "four":
        for layer in range(1, layers + 1):
            R = size * 0.14 * layer
            for dx in (-1, 1):
                for dy in (-1, 1):
                    parts += draw(kind, cx + dx * R * 0.5, cy + dy * R * 0.5, R * 0.45)
    elif symmetry == "grid":
        # 四方连续：母题在二维网格上平铺（含交错偏移）
        step = size / (layers + 2)
        for gy in range(layers + 2):
            off = step * 0.5 if gy % 2 else 0
            for gx in range(layers + 3):
                parts += draw(kind, step * (gx + 0.5) + off, step * (gy + 0.5), step * 0.42)
    elif symmetry == "strip":
        # 二方连续：母题沿水平方向重复成带
        step = size / (layers + 2)
        for gy in (0.28, 0.5, 0.72):
            for gx in range(layers + 3):
                parts += draw(kind, step * (gx + 0.5), size * gy, step * 0.42)
    else:
        for layer in range(1, layers + 1):
            R = size * 0.14 * layer
            for dx in (-1, 1):
                for dy in (-1, 1):
                    parts += draw(kind, cx + dx * R * 0.5, cy + dy * R * 0.5, R * 0.45)
    parts += draw(kind, cx, cy, size * 0.16)
    parts.append('</svg>')
    return "".join(parts)


def generate_field(kind, seed, size=640, layers=3):
    rng = random.Random(seed)
    parts = _gen_header(size, with_ring=False)
    if kind == "binglie":
        # 冰裂纹：随机撒点 + 最近邻锯齿连线（空间细分机制，非母题复制）
        n = 12 + layers * 3
        pts = [(rng.uniform(0.08, 0.92) * size, rng.uniform(0.08, 0.92) * size) for _ in range(n)]
        used = set()
        for i, p in enumerate(pts):
            d = sorted(range(n), key=lambda j: (pts[j][0] - p[0]) ** 2 + (pts[j][1] - p[1]) ** 2)
            for j in d[1:3]:
                key = (min(i, j), max(i, j))
                if key in used:
                    continue
                used.add(key)
                a, b = p, pts[j]
                mx = (a[0] + b[0]) / 2 + rng.uniform(-size * 0.04, size * 0.04)
                my = (a[1] + b[1]) / 2 + rng.uniform(-size * 0.04, size * 0.04)
                parts.append('<polyline points="%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="none" stroke="%s" stroke-width="2"/>'
                             % (a[0], a[1], mx, my, b[0], b[1], GOLD))
    else:
        # 海水江崖：正弦波带 + 中央立崖（参数化曲线机制）
        rows = 5 + layers
        for r in range(rows):
            y0 = size * (0.20 + 0.60 * r / rows)
            amp = size * 0.03
            period = size / (4 + rng.randint(0, 2))
            phase = rng.uniform(0, 6.28)
            pts = ["%.1f,%.1f" % (x, y0 + amp * np.sin(2 * np.pi * x / period + phase))
                   for x in range(0, size + 1, 8)]
            parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (" ".join(pts), GOLD))
        parts.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="none" stroke="%s" stroke-width="2"/>'
                     % (size * 0.38, size * 0.85, size * 0.5, size * 0.60, size * 0.62, size * 0.85, GOLD))
    parts.append('</svg>')
    return "".join(parts)
@app.route("/")
def index():
    return send_file(os.path.join(BASE, "templates", "index.html"), mimetype="text/html")

@app.route("/api/parse", methods=["POST"])
def api_parse():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no image"}), 400
    res = parse_image_bytes_v2(f.read())
    if not res:
        return jsonify({"error": "decode failed"}), 400
    return jsonify(res)

@app.route("/api/generate", methods=["POST"])
def api_gen():
    p = request.get_json(force=True)
    mp = _sanitize_points(p.get("motif_points"))
    svg = generate_papercut(p.get("symmetry", "radial"), p.get("motif", "lotus"),
                            max(1, int(p.get("layers", 3))), int(p.get("seed", 1)),
                            motif_points=mp)
    return jsonify({"svg": svg})

@app.route("/api/gallery")
def api_gallery():
    if not os.path.isdir(SAMPLE_DIR):
        return jsonify([])
    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(exts)][:30]
    return jsonify(files)

@app.route("/sample/<path:name>")
def sample(name):
    return send_file(os.path.join(SAMPLE_DIR, name))

def _free_port(start=5000, limit=10):
    import socket
    for p in range(start, start + limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start

if __name__ == "__main__":
    port = _free_port(5000)
    print("\n绣纹智绘已启动 -> 请打开浏览器访问  http://127.0.0.1:%d  （Ctrl+C 停止服务）\n" % port)
    app.run(host="0.0.0.0", port=port, debug=False)

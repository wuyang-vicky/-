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
def motif(kind, cx, cy, s, rng, color):
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
    else:
        for rot in (0, 45):
            el.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" transform="rotate(%d %.1f %.1f)"/>'
                      % (cx - s * 0.45, cy - s * 0.45, s * 0.9, s * 0.9, color, rot, cx, cy))
        el.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (cx, cy, s * 0.14, color))
    return el

def generate_papercut(symmetry, kind, layers, seed, size=640):
    rng = random.Random(seed)
    cx = cy = size / 2
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" '
             'preserveAspectRatio="xMidYMid meet"><rect width="%d" height="%d" fill="%s"/>'
             % (size, size, size, size, RED)]
    parts.append('<circle cx="%d" cy="%d" r="%d" fill="none" stroke="%s" stroke-width="6"/>' % (cx, cy, size * 0.46, GOLD))
    parts.append('<circle cx="%d" cy="%d" r="%d" fill="none" stroke="%s" stroke-width="2"/>' % (cx, cy, size * 0.42, GOLD))
    if symmetry == "radial":
        folds = rng.randint(4, 8)
        for layer in range(1, layers + 1):
            R = size * 0.12 * layer
            for f in range(folds):
                ang = 2 * np.pi * f / folds + layer * 0.25
                parts += motif(kind, cx + R * np.cos(ang), cy + R * np.sin(ang), R * 0.5, rng, GOLD)
    elif symmetry == "mirror":
        for layer in range(1, layers + 1):
            R = size * 0.14 * layer
            for sx in (-1, 1):
                parts += motif(kind, cx + sx * R * 0.62, cy - R * 0.15 * layer, R * 0.5, rng, GOLD)
    else:
        for layer in range(1, layers + 1):
            R = size * 0.14 * layer
            for dx in (-1, 1):
                for dy in (-1, 1):
                    parts += motif(kind, cx + dx * R * 0.5, cy + dy * R * 0.5, R * 0.45, rng, GOLD)
    parts += motif(kind, cx, cy, size * 0.16, rng, GOLD)
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
    svg = generate_papercut(p.get("symmetry", "radial"), p.get("motif", "lotus"),
                            max(1, int(p.get("layers", 3))), int(p.get("seed", 1)))
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

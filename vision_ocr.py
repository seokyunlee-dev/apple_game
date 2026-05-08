"""
캡처된 이미지에서 격자(Grid)를 분리하고 숫자(1~9)를 인식해 2차원 리스트로 매핑합니다.

입력: capture.png (또는 임의 경로)
출력: grid.json (grid[y][x] = number, 빈 칸은 0)

사용 예:
  python vision_ocr.py "captures/capture_20260430_165643_908994.png" --out grid.json --debug-dir debug
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception as e:
    raise SystemExit(f"OpenCV(cv2)가 필요합니다. 원인: {e}")

try:
    import numpy as np  # type: ignore
except Exception as e:
    raise SystemExit(f"numpy가 필요합니다. 원인: {e}")


@dataclass(frozen=True)
class GridDetection:
    x_lines: List[int]
    y_lines: List[int]

    @property
    def cols(self) -> int:
        return max(0, len(self.x_lines) - 1)

    @property
    def rows(self) -> int:
        return max(0, len(self.y_lines) - 1)


def _write_image(debug_dir: Optional[Path], name: str, img: np.ndarray) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / name
    try:
        cv2.imwrite(str(out_path), img)
    except Exception:
        pass


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    p = str(path)
    try:
        data = np.fromfile(p, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(p, flags)


def preprocess_for_grid(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    return gray, bin_inv


def _extract_lines(bin_inv: np.ndarray, *, v_len: int, h_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    vertical = cv2.erode(bin_inv, vertical_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vertical_kernel, iterations=2)
    horizontal = cv2.erode(bin_inv, horizontal_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=2)
    grid = cv2.bitwise_or(vertical, horizontal)
    return vertical, horizontal, grid


def _peaks_from_projection(proj: np.ndarray, *, min_gap: int, rel_threshold: float = 0.55) -> List[int]:
    proj = proj.astype(np.float32)
    if proj.size == 0 or proj.max() <= 0:
        return []
    mask = proj >= (proj.max() * rel_threshold)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    lines: List[int] = []
    start = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        if i == prev + 1:
            prev = int(i)
        else:
            lines.append((start + prev) // 2)
            start = prev = int(i)
    lines.append((start + prev) // 2)
    merged: List[int] = []
    for x in sorted(lines):
        if not merged or x - merged[-1] >= min_gap:
            merged.append(x)
        else:
            merged[-1] = (merged[-1] + x) // 2
    return merged


def detect_grid_lines(bgr: np.ndarray, *, debug_dir: Optional[Path] = None) -> GridDetection:
    gray, bin_inv = preprocess_for_grid(bgr)
    _write_image(debug_dir, "01_gray.png", gray)
    _write_image(debug_dir, "02_bin_inv.png", bin_inv)
    h, w = bin_inv.shape[:2]
    scales = [0.25, 0.20, 0.16, 0.12, 0.10]
    best_det = None
    for scale in scales:
        vertical, horizontal, grid_mask = _extract_lines(bin_inv, v_len=max(18, int(h * scale)), h_len=max(18, int(w * scale)))
        x_lines = _peaks_from_projection(vertical.sum(axis=0), min_gap=max(10, w // 45))
        y_lines = _peaks_from_projection(horizontal.sum(axis=1), min_gap=max(10, h // 45))
        def _with_bounds(ls, b0, b1):
            out = sorted(set(ls + [b0, b1]))
            return [x for x in out if b0 <= x <= b1]
        x2, y2 = _with_bounds(x_lines, 0, w - 1), _with_bounds(y_lines, 0, h - 1)
        det = GridDetection(x_lines=x2, y_lines=y2)
        if 2 <= det.rows <= 40 and 2 <= det.cols <= 40:
            best_det = det
            break
        best_det = det
    return best_det if best_det else GridDetection([0, w-1], [0, h-1])


def _cluster_1d(values: List[float], *, max_gap: float) -> List[float]:
    if not values: return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= max_gap: clusters[-1].append(v)
        else: clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def detect_grid_by_apples(bgr: np.ndarray, *, debug_dir: Optional[Path] = None) -> GridDetection:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, (0, 70, 50), (12, 255, 255)), cv2.inRange(hsv, (165, 70, 50), (180, 255, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    _write_image(debug_dir, "06_red_mask.png", mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        m = cv2.moments(c)
        if m["m00"] > 0: centers.append((m["m10"]/m["m00"], m["m01"]/m["m00"]))
    if len(centers) < 6: raise RuntimeError("Apples not found")
    xs, ys = [c[0] for c in centers], [c[1] for c in centers]
    def get_gap(vs):
        vs = sorted(vs)
        ds = [vs[i+1]-vs[i] for i in range(len(vs)-1) if vs[i+1]-vs[i] > 1.0]
        return float(np.median(ds)) if ds else 20.0
    x_gap, y_gap = get_gap(xs), get_gap(ys)
    xc, yc = sorted(_cluster_1d(xs, max_gap=x_gap*0.45)), sorted(_cluster_1d(ys, max_gap=y_gap*0.45))
    def to_edges(cs, b0, b1):
        if len(cs) < 2: return [b0, b1]
        mids = [(cs[i]+cs[i+1])/2.0 for i in range(len(cs)-1)]
        edges = [cs[0]-(cs[1]-cs[0])/2.0] + mids + [cs[-1]+(cs[-1]-cs[-2])/2.0]
        return [int(max(b0, min(b1, round(e)))) for e in edges]
    return GridDetection(to_edges(xc, 0, bgr.shape[1]-1), to_edges(yc, 0, bgr.shape[0]-1))


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def preprocess_cell_for_ocr_variants(cell_bgr: np.ndarray) -> List[np.ndarray]:
    h, w = cell_bgr.shape[:2]
    if h < 5 or w < 5: return [np.zeros((32, 32), dtype=np.uint8)]
    lab = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l)
    cell_enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(cell_enhanced, cv2.COLOR_BGR2HSV)
    white_mask = cv2.bitwise_and(cv2.threshold(hsv[:,:,2], 180, 255, cv2.THRESH_BINARY)[1], cv2.threshold(hsv[:,:,1], 90, 255, cv2.THRESH_BINARY_INV)[1])
    gray = cv2.cvtColor(cell_enhanced, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.bitwise_and(cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4), cv2.threshold(hsv[:,:,1], 90, 255, cv2.THRESH_BINARY_INV)[1])
    def post(img):
        img = cv2.dilate(cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
        up = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        return _ensure_uint8(cv2.threshold(up, 127, 255, cv2.THRESH_BINARY)[1])
    return [post(white_mask), post(thresh_inv), _ensure_uint8(cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))]


def crop_to_content(img: np.ndarray, min_area: int = 40) -> np.ndarray:
    if img.ndim != 2: return img
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return img
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area: return img
    x, y, w, h = cv2.boundingRect(c)
    return img[max(0, y-4):min(img.shape[0], y+h+4), max(0, x-4):min(img.shape[1], x+w+4)]


def normalize_bin(img, size=32):
    img = (img > 0).astype(np.uint8) * 255
    if img.size == 0: return np.zeros((size, size), dtype=np.uint8)
    img = crop_to_content(img, min_area=20)
    h, w = img.shape[:2]
    canvas = np.zeros((max(h, w), max(h, w)), dtype=np.uint8)
    canvas[(canvas.shape[0]-h)//2 : (canvas.shape[0]-h)//2+h, (canvas.shape[1]-w)//2 : (canvas.shape[1]-w)//2+w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)


def template_score(a, b):
    a, b = a.astype(np.float32).reshape(-1), b.astype(np.float32).reshape(-1)
    if a.size != b.size or a.size == 0: return -1.0
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-6 else -1.0


def vec32(img):
    v = normalize_bin(img).astype(np.float32).reshape(-1) / 255.0
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def cosine(a, b):
    return float(np.dot(a, b)) if a.size == b.size and a.size > 0 else -1.0


def count_holes(img: np.ndarray) -> int:
    if img.ndim != 2: return 0
    cnts, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None: return 0
    return sum(1 for h in hier[0] if h[3] != -1)


def _fix_digits(img, d):
    """
    Hole count (topology)를 이용해 5/6, 8, 9 등을 교정합니다.
    """
    if img.ndim != 2: return d
    holes = count_holes(img)
    
    # 1. 확실한 케이스: 구멍 2개는 무조건 8
    if holes == 2: return 8
    
    # 2. 5와 6 구분: 5는 구멍이 없고, 6은 1개임
    if d == 5 and holes == 1: return 6
    if d == 6 and holes == 0: return 5
    
    # 3. 8 보정: 8로 인식됐는데 구멍이 하나도 없으면 3일 확률이 높음
    if d == 8 and holes == 0: return 3
    # 8로 인식됐고 구멍이 1개면, 8의 한쪽이 막혔거나 진짜 6/9일 수 있음. 
    # 여기서는 8로 유지하는 것이 안전할 수 있음 (OCR 엔진을 믿음)
    
    # 4. 1과 7 구분 (상단 가로획 비율)
    if d == 7:
        h, w = img.shape[:2]
        fg = (img > 0).astype(np.uint8)
        tot = int(fg.sum())
        if tot > 0:
            ratio = float(fg[:max(1, int(h*0.22)), :].sum()) / tot
            if ratio < 0.16: return 1
            if h/max(1, w) > 2.8 and ratio < 0.22: return 1
            
    return d


class DigitOCR:
    def __init__(self):
        self._tesseract = None
        self._easy = None
        self.available_engines = []
        try:
            import pytesseract
            self._tesseract = pytesseract
            self._tesseract.get_tesseract_version()
            self.available_engines.append("pytesseract")
        except: self._tesseract = None
        if not self._tesseract: self._ensure_easy()

    def _ensure_easy(self):
        if self._easy: return
        try:
            import easyocr
            base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "."
            mdir = Path(base) / "apple_game_easyocr"
            mdir.mkdir(parents=True, exist_ok=True)
            self._easy = easyocr.Reader(["en"], gpu=False, model_storage_directory=str(mdir), download_enabled=True, verbose=False)
            if "easyocr" not in self.available_engines: self.available_engines.append("easyocr")
        except: pass

    def _read_with_tesseract(self, img):
        if not self._tesseract: return None
        try:
            txt = self._tesseract.image_to_string(img, config="--oem 3 --psm 10 -c tessedit_char_whitelist=123456789").strip()
            txt = "".join(c for c in txt if c.isdigit())
            if txt: return int(txt[0]), 0.8
        except: pass
        return None

    def _read_with_easyocr(self, img):
        self._ensure_easy()
        if not self._easy: return None
        try:
            res = self._easy.readtext(img, detail=1, allowlist="123456789", decoder="greedy", batch_size=1, mag_ratio=1.5)
            if res:
                txt = "".join(c for c in str(res[0][1]) if c.isdigit())
                if txt: return int(txt[0]), float(res[0][2])
        except: pass
        return None

    def read_digit_with_conf(self, cell):
        vars = preprocess_cell_for_ocr_variants(cell)
        cands = []
        for img in vars:
            is_bin = (img.ndim == 2 and np.unique(img).size <= 2)
            proc = crop_to_content(img) if is_bin else img
            for f in [self._read_with_tesseract, self._read_with_easyocr]:
                r = f(proc)
                if r:
                    d, c = r
                    # 이진화된 이미지(proc) 또는 첫 번째 변체(vars[0])를 보정용으로 사용
                    d = _fix_digits(proc if is_bin else vars[0], d)
                    cands.append((d, c, proc if is_bin else vars[0]))
        if not cands: return 0, 0.0, crop_to_content(vars[0])
        return max(cands, key=lambda x: x[1])


def iter_cells(bgr, det, margin_ratio=0.10):
    h, w = bgr.shape[:2]
    for y in range(det.rows):
        y0, y1 = det.y_lines[y], det.y_lines[y+1]
        for x in range(det.cols):
            x0, x1 = det.x_lines[x], det.x_lines[x+1]
            mx, my = int((x1-x0)*margin_ratio), int((y1-y0)*margin_ratio)
            yield x, y, (x0+mx, y0+my, x1-mx, y1-my), bgr[y0+my:y1-my, x0+mx:x1-mx]


def build_grid(image_path, debug_dir=None):
    bgr = imread_unicode(image_path)
    if bgr is None: raise FileNotFoundError(image_path)
    det = detect_grid_lines(bgr, debug_dir=debug_dir)
    if det.rows <= 1 or det.cols <= 1: det = detect_grid_by_apples(bgr, debug_dir=debug_dir)
    ocr = DigitOCR()
    grid = [[0 for _ in range(det.cols)] for _ in range(det.rows)]
    ocr_pass = {}
    for x, y, bbox, cell in iter_cells(bgr, det):
        d, c, b = ocr.read_digit_with_conf(cell)
        ocr_pass[(x, y)] = (d, c, b)
        if debug_dir: _write_image(debug_dir, f"cell_{y:02d}_{x:02d}_bin.png", b)
    protos = {d: [] for d in range(1, 10)}
    for (x, y), (d, c, b) in ocr_pass.items():
        if 1 <= d <= 9 and c >= 0.8:
            protos[d].append(normalize_bin(b))
            if len(protos[d]) > 6: protos[d] = protos[d][:6]
    for y in range(det.rows):
        for x in range(det.cols):
            d, c, b = ocr_pass.get((x, y), (0, 0, None))
            if d == 0 or c < 0.6:
                best_d, best_s = 0, -1.0
                cand = normalize_bin(b) if b is not None else None
                if cand is not None:
                    for pd in range(1, 10):
                        for p in protos[pd]:
                            s = template_score(cand, p)
                            if s > best_s: best_s, best_d = s, pd
                if best_d != 0 and best_s > 0.3: d = best_d
            grid[y][x] = d
    cells = []
    for x, y, bbox, _ in iter_cells(bgr, det):
        d, c, _ = ocr_pass.get((x, y), (grid[y][x], 0, None))
        cells.append({"x": x, "y": y, "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}, "value": grid[y][x], "conf": c})
    return {"image_path": str(image_path), "grid": grid, "rows": det.rows, "cols": det.cols, "ocr_engines": ocr.available_engines, "x_lines": det.x_lines, "y_lines": det.y_lines, "cells": cells}


def calibrate_and_override_grid(*, correct_grid, ocr_pass, rows, cols):
    train = []
    for y in range(rows):
        for x in range(cols):
            d = correct_grid[y][x]
            if 1 <= d <= 9: train.append((d, vec32(ocr_pass[(x, y)][2])))
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            v = vec32(ocr_pass[(x, y)][2])
            sims = sorted([(cosine(v, tv), td) for (td, tv) in train], reverse=True, key=lambda t: t[0])
            votes = {}
            for s, d in sims[:3]: votes[d] = votes.get(d, 0.0) + s
            out[y][x] = max(votes.items(), key=lambda kv: kv[1])[0] if votes else ocr_pass[(x, y)][0]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default="grid.json")
    parser.add_argument("--debug-dir")
    parser.add_argument("--calibrate-from")
    args = parser.parse_args()
    ipath, opath = Path(args.image).resolve(), Path(args.out).resolve()
    ddir = Path(args.debug_dir).resolve() if args.debug_dir else None
    res = build_grid(ipath, debug_dir=ddir)
    if args.calibrate_from:
        corr = json.loads(Path(args.calibrate_from).read_text(encoding="utf-8"))
        bgr = imread_unicode(ipath)
        det = GridDetection(res["x_lines"], res["y_lines"])
        ocr = DigitOCR()
        ocr_pass = { (x,y): ocr.read_digit_with_conf(c) for x, y, b, c in iter_cells(bgr, det) }
        res["grid"] = calibrate_and_override_grid(correct_grid=corr["grid"], ocr_pass=ocr_pass, rows=res["rows"], cols=res["cols"])
    opath.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {opath}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

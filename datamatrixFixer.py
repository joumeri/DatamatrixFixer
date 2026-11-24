#!/usr/bin/env python3
"""
DataMatrix Fixer - vioittuneiden DataMatrix (ECC200) -koodien luku.

Kaksi menetelmää yhdessä:
  1) OMA DEKOODERI (ei tarvitse ehdokaslistaa): kuva -> perspektiivikorjaus -> moduulimatriisi
     -> ECC200-sijoittelu -> Reed-Solomon (virheet + poistot epävarmoille koodisanoille) -> teksti.
  2) GENEROI JA VERTAA (jos annat sarjanumeroehdokkaat): ehdokkaat enkoodataan ja verrataan
     kuvista yhdistettyyn evidenssiin. Toimii vielä silloinkin, kun virheitä on liikaa korjattavaksi.

Useita kuvia yhdistetään moduulikohtaisesti. Symbolikoko: --size 20 | 8x18 | 18,20,22 | auto

Käyttö:
    python datamatrixFixer.py --size auto --images kuvat/
    python datamatrixFixer.py --size 20 --images kuvat/ --each
    python datamatrixFixer.py serials.txt --size 18,20 --images kuvat/
    python datamatrixFixer.py serials.txt --size auto --camera 0     (r = nollaa, q = lopeta)

Asennus:
    py -m pip install opencv-python numpy setuptools pylibdmtx zxing-cpp
    (pylibdmtx ja zxing-cpp ovat valinnaisia: oma dekooderi toimii ilman niitä.
     pylibdmtx tarvitaan vain ehdokasvertailuun.)
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

try:
    from pylibdmtx.pylibdmtx import encode as dmtx_encode, decode as dmtx_decode
    HAVE_DMTX = True
except Exception as _e:                      # ImportError, DLL-virhe, distutils...
    dmtx_encode = dmtx_decode = None
    HAVE_DMTX = False
    _DMTX_ERR = str(_e)

try:
    import zxingcpp
except Exception:
    zxingcpp = None

K = 16          # pikseliä per moduuli perspektiivikorjatussa kuvassa
SCHEMES = ["Ascii", "C40", "Text", "X12", "Edifact", "Base256"]
OFFS = (-0.5, -0.25, 0.0, 0.25, 0.5)
SCALES = (0.95, 1.0, 1.05)
OFFS_COARSE = (-0.3, 0.0, 0.3)


# ============================================================================
# Oma ECC200-dekooderi: sijoittelu, Reed-Solomon (virheet + poistot), datan purku
# ============================================================================

# (rivit, sarakkeet): (pystyalueita, vaakaalueita, datakoodisanoja, ecc-koodisanoja, lohkoja)
SYMBOLS = {
    (10, 10): (1, 1, 3, 5, 1),       (12, 12): (1, 1, 5, 7, 1),
    (14, 14): (1, 1, 8, 10, 1),      (16, 16): (1, 1, 12, 12, 1),
    (18, 18): (1, 1, 18, 14, 1),     (20, 20): (1, 1, 22, 18, 1),
    (22, 22): (1, 1, 30, 20, 1),     (24, 24): (1, 1, 36, 24, 1),
    (26, 26): (1, 1, 44, 28, 1),     (32, 32): (2, 2, 62, 36, 1),
    (36, 36): (2, 2, 86, 42, 1),     (40, 40): (2, 2, 114, 48, 1),
    (44, 44): (2, 2, 144, 56, 1),    (48, 48): (2, 2, 174, 68, 1),
    (52, 52): (2, 2, 204, 84, 2),    (64, 64): (4, 4, 280, 112, 2),
    (72, 72): (4, 4, 368, 144, 4),   (80, 80): (4, 4, 456, 192, 4),
    (88, 88): (4, 4, 576, 224, 4),   (96, 96): (4, 4, 696, 272, 4),
    (104, 104): (4, 4, 816, 336, 6), (120, 120): (6, 6, 1050, 408, 6),
    (132, 132): (6, 6, 1304, 496, 8), (144, 144): (6, 6, 1558, 620, 10),
    (8, 18): (1, 1, 5, 7, 1),        (8, 32): (1, 2, 10, 11, 1),
    (12, 26): (1, 1, 16, 14, 1),     (12, 36): (1, 2, 22, 18, 1),
    (16, 36): (1, 2, 32, 24, 1),     (16, 48): (1, 2, 49, 28, 1),
}

# ---------- GF(256), polynomi 0x12D ----------
EXP = [0] * 512
LOG = [0] * 256
_x = 1
for _i in range(255):
    EXP[_i] = _x
    LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x12D
for _i in range(255, 512):
    EXP[_i] = EXP[_i - 255]


def gmul(a, b):
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]


def ginv(a):
    return EXP[255 - LOG[a]]


def pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] ^= gmul(x, y)
    return out


def peval(p, x):
    """p matalimmasta korkeimpaan, x != 0"""
    r = 0
    lx = LOG[x]
    for i, c in enumerate(p):
        if c:
            r ^= EXP[(LOG[c] + lx * i) % 255]
    return r


def syndromes(cw, nec):
    S = []
    for j in range(1, nec + 1):
        s = 0
        for c in cw:
            s = gmul(s, EXP[j]) ^ c
        S.append(s)
    return S


def _bm(s):
    C, B, L, m, b = [1], [1], 0, 1, 1
    for n in range(len(s)):
        d = s[n]
        for i in range(1, L + 1):
            if i < len(C):
                d ^= gmul(C[i], s[n - i])
        if d == 0:
            m += 1
            continue
        T = C[:]
        coef = gmul(d, ginv(b))
        need = len(B) + m
        if len(C) < need:
            C = C + [0] * (need - len(C))
        for i, v in enumerate(B):
            C[i + m] ^= gmul(coef, v)
        if 2 * L <= n:
            L, B, b, m = n + 1 - L, T, d, 1
        else:
            m += 1
    C = (C + [0] * (L + 1))[:L + 1]
    return C, L


def rs_correct(cw, nec, erasures=(), spare=0):
    """Reed-Solomon (juuret alpha^1..alpha^nec). Palauttaa (korjattu, virheiden määrä) tai None."""
    n = len(cw)
    erasures = sorted(set(erasures))
    e = len(erasures)
    if e > nec:
        return None
    S = syndromes(cw, nec)
    if not any(S):
        return list(cw), 0
    G = [1]
    for p in erasures:
        G = pmul(G, [1, EXP[(n - 1 - p) % 255]])
    T = pmul(S, G)[:nec]
    sigma, L = _bm(T[e:])
    if 2 * L + e > nec - spare:
        return None
    Lam = pmul(G, sigma)
    while len(Lam) > 1 and Lam[-1] == 0:
        Lam.pop()
    deg = len(Lam) - 1
    roots = []
    for p in range(n):
        xinv = EXP[(255 - (n - 1 - p)) % 255]
        if peval(Lam, xinv) == 0:
            roots.append(p)
    if len(roots) != deg:
        return None
    Om = pmul(S, Lam)[:nec]
    out = list(cw)
    for p in roots:
        xinv = EXP[(255 - (n - 1 - p)) % 255]
        num = peval(Om, xinv)
        dl = [Lam[i] if i % 2 == 1 else 0 for i in range(1, len(Lam))]  # Lam'(x) (char 2)
        den = peval(dl, xinv)
        if den == 0:
            return None
        out[p] ^= gmul(num, ginv(den))
    if any(syndromes(out, nec)):
        return None
    return out, len(roots)


# ---------- ECC200-sijoittelu ----------
_PLACE_CACHE = {}


def placement(nrow, ncol):
    key = (nrow, ncol)
    if key in _PLACE_CACHE:
        return _PLACE_CACHE[key]
    arr = [[0] * ncol for _ in range(nrow)]
    pos = {}

    def module(row, col, chr_, bit):
        if row < 0:
            row += nrow
            col += 4 - ((nrow + 4) % 8)
        if col < 0:
            col += ncol
            row += 4 - ((ncol + 4) % 8)
        arr[row][col] = 1
        pos[(chr_, bit)] = (row, col)

    def utah(row, col, chr_):
        for b, (dr, dc) in enumerate([(-2, -2), (-2, -1), (-1, -2), (-1, -1),
                                      (-1, 0), (0, -2), (0, -1), (0, 0)], 1):
            module(row + dr, col + dc, chr_, b)

    def corner(chr_, cells):
        for b, (r, c) in enumerate(cells, 1):
            module(r, c, chr_, b)

    chr_, row, col = 1, 4, 0
    while True:
        if row == nrow and col == 0:
            corner(chr_, [(nrow-1, 0), (nrow-1, 1), (nrow-1, 2), (0, ncol-2),
                          (0, ncol-1), (1, ncol-1), (2, ncol-1), (3, ncol-1)]); chr_ += 1
        if row == nrow - 2 and col == 0 and ncol % 4:
            corner(chr_, [(nrow-3, 0), (nrow-2, 0), (nrow-1, 0), (0, ncol-4),
                          (0, ncol-3), (0, ncol-2), (0, ncol-1), (1, ncol-1)]); chr_ += 1
        if row == nrow - 2 and col == 0 and ncol % 8 == 4:
            corner(chr_, [(nrow-3, 0), (nrow-2, 0), (nrow-1, 0), (0, ncol-2),
                          (0, ncol-1), (1, ncol-1), (2, ncol-1), (3, ncol-1)]); chr_ += 1
        if row == nrow + 4 and col == 2 and ncol % 8 == 0:
            corner(chr_, [(nrow-1, 0), (nrow-1, ncol-1), (0, ncol-3), (0, ncol-2),
                          (0, ncol-1), (1, ncol-3), (1, ncol-2), (1, ncol-1)]); chr_ += 1
        while True:
            if row < nrow and col >= 0 and not arr[row][col]:
                utah(row, col, chr_); chr_ += 1
            row -= 2; col += 2
            if not (row >= 0 and col < ncol):
                break
        row += 1; col += 3
        while True:
            if row >= 0 and col < ncol and not arr[row][col]:
                utah(row, col, chr_); chr_ += 1
            row += 2; col -= 2
            if not (row < nrow and col >= 0):
                break
        row += 3; col += 1
        if not (row < nrow or col < ncol):
            break
    _PLACE_CACHE[key] = (pos, chr_ - 1)
    return _PLACE_CACHE[key]


def data_region(E, rows, cols):
    """Poimii datamoduulit (ilman reunakuvioita) yhdeksi matriisiksi."""
    vr, hc = SYMBOLS[(rows, cols)][:2]
    rh, rw = rows // vr - 2, cols // hc - 2
    out = np.zeros((vr * rh, hc * rw), E.dtype)
    for i in range(vr):
        for j in range(hc):
            out[i*rh:(i+1)*rh, j*rw:(j+1)*rw] = \
                E[i*(rh+2)+1:i*(rh+2)+1+rh, j*(rw+2)+1:j*(rw+2)+1+rw]
    return out


def codewords_from_evidence(E, rows, cols):
    """E: + = tumma. Palauttaa (koodisanat, luottamus per koodisana)."""
    core = data_region(E, rows, cols)
    pos, ncw = placement(*core.shape)
    cws, conf = [], []
    for c in range(1, ncw + 1):
        v, m = 0, 9.0
        for b in range(1, 9):
            r, cc = pos[(c, b)]
            val = float(core[r, cc])
            v = (v << 1) | (1 if val > 0 else 0)
            m = min(m, abs(val))
        cws.append(v)
        conf.append(m)
    return cws, conf


def rs_symbol(cws, conf, rows, cols, max_erase_frac=0.75):
    """Dekoodaa koko symbolin (lohkot). Kokeilee poistoja matalan luottamuksen koodisanoille."""
    vr, hc, nd, ne, nb = SYMBOLS[(rows, cols)]
    if len(cws) != nd + ne:
        return None
    nec = ne // nb
    blocks = []
    for b in range(nb):
        di = list(range(b, nd, nb))
        ei = list(range(nd + b, nd + ne, nb))
        blocks.append((di + ei, len(di)))
    order = sorted(range(len(cws)), key=lambda i: conf[i])
    max_er = int(nec * max_erase_frac)
    tried = set()
    for k in [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 60, 90, 130, 200, 400]:
        low = [i for i in order[:k] if conf[i] < 0.9]
        key = tuple(sorted(low))
        if key in tried:
            continue
        tried.add(key)
        out = list(cws)
        total_err = 0
        ok = True
        for idx, nd_b in blocks:
            blk = [cws[i] for i in idx]
            er = [j for j, i in enumerate(idx) if i in set(low)]
            if len(er) > max_er:
                # liikaa poistoja tässä lohkossa: pidä vain matalimman luottamuksen
                er = sorted(er, key=lambda j: conf[idx[j]])[:max_er]
            r = rs_correct(blk, nec, er, spare=min(4, nec // 3))
            if r is None:
                ok = False
                break
            fixed, ne_found = r
            total_err += ne_found
            for j, i in enumerate(idx):
                out[i] = fixed[j]
        if ok:
            return out[:nd], total_err, len(low)
    return None


# ---------- datan purku ----------
_C40_S2 = "!\"#$%&'()*+,-./:;<=>?@[\\]^_"


def decode_data(cw):
    out, info, i = [], [], 0

    def unpack(a, b):
        v = a * 256 + b - 1
        return [v // 1600, (v % 1600) // 40, v % 40]

    def unm(v, p):
        return (v - (149 * p) % 255 - 1) % 256

    while i < len(cw):
        c = cw[i]; i += 1
        if c == 0:
            continue
        if c <= 128:
            out.append(chr(c - 1))
        elif c == 129:
            info.append("PAD"); break
        elif c <= 229:
            out.append("%02d" % (c - 130))
        elif c in (230, 239):
            text = c == 239
            info.append("TEXT" if text else "C40")
            shift, up = 0, False
            while i + 1 < len(cw) and cw[i] != 254:
                for v in unpack(cw[i], cw[i + 1]):
                    if shift == 0:
                        if v < 3:
                            shift = v + 1; continue
                        ch = " " if v == 3 else (str(v - 4) if v <= 13 else
                                                  chr((97 if text else 65) + v - 14))
                    elif shift == 1:
                        ch = chr(v)
                    elif shift == 2:
                        if v <= 26: ch = _C40_S2[v]
                        elif v == 27: ch = "<FNC1>"
                        elif v == 30: up = True; shift = 0; continue
                        else: ch = "?"
                    else:
                        tbl = ["`"] + [chr((65 if text else 97) + k) for k in range(26)] + \
                              ["{", "|", "}", "~", "\x7f"]
                        ch = tbl[v] if v < len(tbl) else "?"
                    if up and len(ch) == 1:
                        ch = chr(ord(ch) + 128); up = False
                    out.append(ch); shift = 0
                i += 2
            if i < len(cw) and cw[i] == 254:
                i += 1
        elif c == 238:
            info.append("X12")
            while i + 1 < len(cw) and cw[i] != 254:
                for v in unpack(cw[i], cw[i + 1]):
                    out.append(["\r", "*", ">", " "][v] if v < 4 else
                               (str(v - 4) if v <= 13 else chr(65 + v - 14)))
                i += 2
            if i < len(cw) and cw[i] == 254:
                i += 1
        elif c == 240:
            info.append("EDIFACT")
            bits, done = "", False
            while i < len(cw) and not done:
                bits += format(cw[i], "08b"); i += 1
                while len(bits) >= 6:
                    v = int(bits[:6], 2); bits = bits[6:]
                    if v == 31:
                        done = True; break
                    out.append(chr(v if v >= 32 else v + 64))
        elif c == 231:
            info.append("BASE256")
            d1 = unm(cw[i], i + 1); i += 1
            if d1 == 0: ln = len(cw) - i
            elif d1 < 250: ln = d1
            else:
                ln = 250 * (d1 - 249) + unm(cw[i], i + 1); i += 1
            for _ in range(ln):
                if i >= len(cw): break
                out.append(chr(unm(cw[i], i + 1))); i += 1
        elif c == 232:
            out.append("<FNC1>")
        elif c == 235:
            if i < len(cw):
                out.append(chr(cw[i] - 1 + 128)); i += 1
        else:
            info.append("CW%d" % c)
    return "".join(out), info


def decode_evidence(E, rows, cols):
    """Oma dekoodaus evidenssimatriisista (+ = tumma, itseisarvo = luottamus).
    Palauttaa dict(text, info, errors, erasures) tai None."""
    if (rows, cols) not in SYMBOLS:
        return None
    cws, conf = codewords_from_evidence(E, rows, cols)
    r = rs_symbol(cws, conf, rows, cols)
    if r is None:
        return None
    data, nerr, nera = r
    try:
        text, info = decode_data(data)
    except Exception:
        return None
    if not text or any((ord(ch) < 32 and ch not in "\r\n\t\x1d\x1e\x04") for ch in text if len(ch) == 1):
        return None          # tyhjä / roskaa -> hylkää (suoja vääriä positiivisia vastaan)
    return dict(text=text, info=info, errors=nerr, erasures=nera, data=data)


# ============================================================================
# Kuvankäsittely
# ============================================================================
def parse_sizes(text):
    """'auto' -> None; muuten lista (rivit, sarakkeet)."""
    if text.strip().lower() == "auto":
        return None
    sizes = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" in part:
            r, c = part.split("x")
            sizes.append((int(r), int(c)))
        else:
            sizes.append((int(part), int(part)))
    for s in sizes:
        if s not in SYMBOLS:
            sys.exit(f"Tuntematon ECC200-koko {s[0]}x{s[1]}. Sallitut: "
                     + ", ".join(f"{r}x{c}" for r, c in sorted(SYMBOLS)))
    return sizes


def order_corners(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def refine_quad(contour, init, tols=(0.08, 0.05, 0.03)):
    """Tarkentaa kulmat sovittamalla suorat sivuille (konveksin kuoren tiheytetyt pisteet)
    ja leikkaamalla ne. Kestää puuttuvat kulmamoduulit ja kellokuvion lovet."""
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    dense = []
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        n = max(2, int(np.linalg.norm(b - a)))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        dense.append(a + (b - a) * t)
    dense = np.vstack(dense)
    pts = order_corners(init).astype(np.float64)
    for tol in tols:
        lines = []
        for k in range(4):
            a, b = pts[k], pts[(k + 1) % 4]
            v = b - a
            L = float(np.linalg.norm(v))
            if L < 1:
                return init
            u = v / L
            nrm = np.array([-u[1], u[0]])
            rel = dense - a
            t = rel @ u / L
            dist = np.abs(rel @ nrm)
            sel = (t > 0.1) & (t < 0.9) & (dist < tol * L)
            if sel.sum() < 10:
                return init
            vx, vy, x0, y0 = cv2.fitLine(dense[sel], cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
        corners = []
        for k in range(4):
            p1, d1 = lines[(k - 1) % 4]
            p2, d2 = lines[k]
            A = np.array([d1, -d2]).T
            if abs(np.linalg.det(A)) < 1e-6:
                return init
            sv = np.linalg.solve(A, p2 - p1)
            corners.append(p1 + d1 * sv[0])
        pts = np.array(corners)
    return pts.astype(np.float32)


def locate_quads(gray, aspect_target=1.0, max_quads=3):
    """Etsii koodin nelikulmioehdokkaat (alkuperäisen kuvan koordinaateissa)."""
    scale = 1.0
    work = gray
    if max(gray.shape) > 2000:
        scale = 2000.0 / max(gray.shape)
        work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(work, (3, 3), 0)
    h, w = work.shape
    side = min(h, w)
    bs = max(15, (side // 8) | 1)
    binaries = []
    for polarity in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        binaries.append(cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              polarity, bs, 6))
        _, o = cv2.threshold(blur, 0, 255, polarity + cv2.THRESH_OTSU)
        binaries.append(o)
    cands = []
    for th in binaries:
        for ks in (3, 7, 13):
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((ks, ks), np.uint8))
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 300 or area > 0.8 * h * w:
                    continue
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if min(rw, rh) < 1:
                    continue
                ratio = max(rw, rh) / min(rw, rh)
                if not (aspect_target / 1.5 < ratio < aspect_target * 1.5):
                    continue
                if area / (rw * rh) < 0.5:
                    continue
                box = cv2.boxPoints(rect).astype(np.float32)
                cands.append((area, box, c))
    cands.sort(key=lambda t: -t[0])
    out = []
    for _, box, c in cands:
        q = order_corners(refine_quad(c, box))
        tol = 0.03 * np.sqrt(abs(cv2.contourArea(q)) + 1)
        if all(np.linalg.norm(q - o, axis=1).mean() > tol for o in out):
            out.append(q)
        if len(out) >= max_quads:
            break
    return [q / scale for q in out]


class Geometry:
    """Symbolin koko + reunakuvion malli (L-etsijä + kellokuvio, kanoninen asento:
    yhtenäinen L vasemmalla ja alhaalla, kello ylhäällä ja oikealla)."""

    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.square = rows == cols
        dark = np.zeros((rows, cols), bool)
        light = np.zeros((rows, cols), bool)
        dark[:, 0] = True
        dark[rows - 1, :] = True
        for c in range(1, cols):
            (dark if c % 2 == 0 else light)[0, c] = True
        for r in range(0, rows - 1):
            (dark if (rows - 1 - r) % 2 == 0 else light)[r, cols - 1] = True
        self.dm, self.lm = dark, light
        self.W, self.H = cols * K, rows * K
        self.name = f"{rows}x{cols}"
        self.aspect = max(rows, cols) / min(rows, cols)

    def module_matrix(self, text, scheme):
        if not HAVE_DMTX:
            return None
        try:
            e = dmtx_encode(text.encode("latin-1"), scheme=scheme, size=self.name)
        except Exception:
            return None
        img = np.frombuffer(e.pixels, np.uint8).reshape(e.height, e.width, 3)[:, :, 0]
        ys, xs = np.where(img < 128)
        if len(ys) == 0:
            return None
        crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = crop.shape
        rr = ((np.arange(self.rows) + 0.5) * h / self.rows).astype(int)
        cc = ((np.arange(self.cols) + 0.5) * w / self.cols).astype(int)
        return crop[np.ix_(rr, cc)] < 128

    def warps(self, gray, quad):
        pts = order_corners(quad)
        variants = [pts] if self.square else [pts, np.roll(pts, -1, axis=0)]
        dst = np.array([[0, 0], [self.W, 0], [self.W, self.H], [0, self.H]], np.float32)
        for src in variants:
            M = cv2.getPerspectiveTransform(src, dst)
            yield cv2.warpPerspective(gray, M, (self.W, self.H), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)

    def sample(self, blur, ox, oy, sx, sy):
        xs = self.W / 2 + ((np.arange(self.cols) + 0.5) - self.cols / 2) * K * sx + ox * K
        ys = self.H / 2 + ((np.arange(self.rows) + 0.5) - self.rows / 2) * K * sy + oy * K
        xi = np.clip(np.rint(xs).astype(int), 0, self.W - 1)
        yi = np.clip(np.rint(ys).astype(int), 0, self.H - 1)
        return blur[np.ix_(yi, xi)].astype(np.float32)

    def dprime(self, v):
        d, l = v[self.dm], v[self.lm]
        return (l.mean() - d.mean()) / (d.std() + l.std() + 1e-6), d.mean(), l.mean()

    def _search(self, blurs, offs, scales):
        best = None
        rots = range(4) if self.square else (0, 2)
        for blur in blurs:
            for ox in offs:
                for oy in offs:
                    for sx in scales:
                        for sy in scales:
                            vals = self.sample(blur, ox, oy, sx, sy)
                            for inv in (False, True):
                                v = 255.0 - vals if inv else vals
                                for k in rots:
                                    r = np.rot90(v, k)
                                    dp, dmn, lmn = self.dprime(r)
                                    if best is None or dp > best["dp"]:
                                        best = dict(dp=dp, v=r, dm=dmn, lm=lmn, k=k, inv=inv)
        return best

    def _blurs(self, gray, quads):
        for quad in quads:
            for warped in self.warps(gray, quad):
                yield cv2.boxFilter(warped, -1, (K // 2 + 1, K // 2 + 1))

    def quick_score(self, gray, quads):
        """Karkea sopivuus (auto-koon valintaan)."""
        best = self._search(list(self._blurs(gray, quads)), OFFS_COARSE, (1.0,))
        return -9.0 if best is None else best["dp"]

    def analyze(self, gray, quads=None):
        """dict(dp, e) tai None. e = evidenssimatriisi kanonisessa asennossa (+ = tumma)."""
        quads = quads if quads is not None else locate_quads(gray, self.aspect)
        best = self._search(list(self._blurs(gray, quads)), OFFS, SCALES)
        if best is None or best["lm"] - best["dm"] < 1e-3:
            return None
        thr = (best["dm"] + best["lm"]) / 2
        half = (best["lm"] - best["dm"]) / 2
        return dict(dp=best["dp"], e=np.clip((thr - best["v"]) / half, -1, 1),
                    k=best["k"], inv=best["inv"])


_GEO = {}


def geo_for(rows, cols):
    if (rows, cols) not in _GEO:
        _GEO[(rows, cols)] = Geometry(rows, cols)
    return _GEO[(rows, cols)]


def auto_geometries(gray, top=3):
    """Arvaa symbolikoon: karkea reunakuvion sovitus kaikille tunnetuille koille."""
    scores = []
    quad_cache = {}
    for (rows, cols) in SYMBOLS:
        geo = geo_for(rows, cols)
        key = round(geo.aspect, 2)
        if key not in quad_cache:
            quad_cache[key] = locate_quads(gray, geo.aspect)
        quads = quad_cache[key]
        if not quads:
            continue
        scores.append((geo.quick_score(gray, quads), geo))
    scores.sort(key=lambda t: -t[0])
    return [g for s, g in scores[:top] if s > 0.5]


# ============================================================================
# Oma ECC200-enkooderi (ASCII-tila) ja mallisovitus (template fit)
# ============================================================================
def _gen_poly(nec):
    g = [1]
    for i in range(1, nec + 1):
        g = pmul(g, [EXP[i], 1])
    return g


def rs_encode(data, nec):
    g = _gen_poly(nec)
    rem = [0] * nec
    for c in data:
        f = c ^ rem[0]
        rem = rem[1:] + [0]
        if f:
            for j in range(nec):
                rem[j] ^= gmul(f, g[nec - 1 - j])
    return rem


def ascii_encode(text):
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c.isdigit() and i + 1 < len(text) and text[i + 1].isdigit():
            out.append(130 + int(text[i:i + 2]))
            i += 2
        else:
            o = ord(c)
            if o > 127:
                out += [235, o - 127]
            else:
                out.append(o + 1)
            i += 1
    return out


def _pad_to(cw, n):
    if len(cw) > n:
        return None
    cw = list(cw)
    if len(cw) < n:
        cw.append(129)
        while len(cw) < n:
            r = ((149 * (len(cw) + 1)) % 253) + 1
            t = 129 + r
            cw.append(t - 254 if t > 254 else t)
    return cw


def encode_symbol(text, rows, cols):
    """ASCII-enkoodaus -> moduulimatriisi kanonisessa asennossa (1 = tumma), tai None."""
    if (rows, cols) not in SYMBOLS:
        return None
    try:
        raw = ascii_encode(text)
    except Exception:
        return None
    vr, hc, nd, ne, nb = SYMBOLS[(rows, cols)]
    cw = _pad_to(raw, nd)
    if cw is None:
        return None
    nec = ne // nb
    full = [0] * (nd + ne)
    for b in range(nb):
        di = list(range(b, nd, nb))
        ei = list(range(nd + b, nd + ne, nb))
        blk = [cw[i] for i in di]
        ecc = rs_encode(blk, nec)
        for j, i in enumerate(di):
            full[i] = blk[j]
        for j, i in enumerate(ei):
            full[i] = ecc[j]
    rh, rw = rows // vr - 2, cols // hc - 2
    core = np.zeros((vr * rh, hc * rw), np.uint8)
    pos, ncw = placement(*core.shape)
    placed = np.zeros(core.shape, bool)
    for c in range(1, ncw + 1):
        for b in range(1, 9):
            r, cc = pos[(c, b)]
            core[r, cc] = (full[c - 1] >> (8 - b)) & 1
            placed[r, cc] = True
    if not placed[-1, -1]:
        core[-1, -1] = 1
        core[-2, -2] = 1
    S = np.zeros((rows, cols), np.uint8)
    for i in range(vr):
        for j in range(hc):
            r0, c0 = i * (rh + 2), j * (rw + 2)
            S[r0 + 1:r0 + 1 + rh, c0 + 1:c0 + 1 + rw] = \
                core[i * rh:(i + 1) * rh, j * rw:(j + 1) * rw]
            S[r0:r0 + rh + 2, c0] = 1
            S[r0 + rh + 1, c0:c0 + rw + 2] = 1
            for c in range(rw + 2):
                S[r0, c0 + c] = 1 if c % 2 == 0 else 0
            for r in range(rh + 2):
                S[r0 + r, c0 + rw + 1] = 1 if (rh + 1 - r) % 2 == 0 else 0
    return S


FT = 8          # mallin pikseliä per moduuli


def _kron(M):
    return np.kron(M.astype(np.float32), np.ones((FT, FT), np.float32))


def _render(S_img, blur):
    return cv2.GaussianBlur(_kron(S_img), (0, 0), blur)


def _ncc(a, b, w):
    x = a[w]
    y = b[w]
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


class FitMatcher:
    """Sovittaa ehdokkaan moduulimallin suoraan (matalaresoluutioiseen, sumeaan) kuvaan.
    Optimoi perspektiivimuunnoksen (4 kulmaa) kuvan ja mallin välisen korrelaation mukaan.
    Kestää epätarkan ruudukon, jossa suora näytteistys epäonnistuu. Puolet mallista voi
    olla vioittunutta: vertailu tehdään myös maskeilla, jotka jättävät pois reunakaistan."""

    def __init__(self, geo, serials, schemes):
        from scipy.optimize import minimize   # noqa: F401  (tarkistus)
        self.geo = geo
        self.items = []               # (teksti, skeema, matriisi kanonisessa asennossa)
        seen = set()
        for s in serials:
            mats = []
            S = encode_symbol(s, geo.rows, geo.cols)
            if S is not None:
                mats.append(("Ascii", S))
            if HAVE_DMTX:
                for sch in schemes:
                    if sch == "Ascii":
                        continue
                    m = geo.module_matrix(s, sch)
                    if m is not None:
                        mats.append((sch, m.astype(np.uint8)))
            for sch, M in mats:
                key = (s, M.tobytes())
                if key not in seen:
                    seen.add(key)
                    self.items.append((s, sch, M))
        self.stack = np.stack([m for _, _, m in self.items]) if self.items else None

    # --- apufunktiot ---
    @staticmethod
    def _orient(M, k):
        return np.rot90(M, -k)         # kanoninen -> kuvan asento (k kiertoa myötäpäivään)

    def _channels(self, bgr):
        ch = [("R", bgr[:, :, 2]), ("G", bgr[:, :, 1]), ("B", bgr[:, :, 0]),
              ("gray", cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))]
        flat = bgr.reshape(-1, 3).astype(np.float32)
        flat -= flat.mean(axis=0)
        try:
            _, vecs = cv2.PCACompute(flat, mean=None, maxComponents=1)
            p = (flat @ vecs[0]).reshape(bgr.shape[:2])
            ch.append(("PCA", (p - p.min()).astype(np.float32)))
        except Exception:
            pass
        return [(n, c.astype(np.float32)) for n, c in ch]

    def _warp(self, chan, quad0, p, th, tw):
        q = quad0 + np.float32(p).reshape(4, 2)
        dst = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
        return cv2.warpPerspective(chan, cv2.getPerspectiveTransform(q, dst), (tw, th),
                                   flags=cv2.INTER_LINEAR)

    def _fit(self, chan, quad0, tmpl, wmask, x0s):
        from scipy.optimize import minimize
        th, tw = tmpl.shape
        best = None
        for x0 in x0s:
            r = minimize(lambda p: -_ncc(tmpl, self._warp(chan, quad0, p, th, tw), wmask),
                         x0, method="Powell", options=dict(xtol=0.05, ftol=1e-4, maxiter=3000))
            if best is None or r.fun < best.fun:
                best = r
        return best

    def _strip_masks(self, k, strip=3):
        """Moduulimaskit kuvan asennossa: kaikki + yksi reunakaista pois kerrallaan."""
        rows, cols = (self.geo.rows, self.geo.cols) if k % 2 == 0 else (self.geo.cols, self.geo.rows)
        full = np.ones((rows, cols), bool)
        out = {"kaikki": full}
        for name, sl in (("ei oikeaa", (slice(None), slice(-strip, None))),
                         ("ei vasenta", (slice(None), slice(0, strip))),
                         ("ei ylintä", (slice(0, strip), slice(None))),
                         ("ei alinta", (slice(-strip, None), slice(None)))):
            m = full.copy()
            m[sl] = False
            out[name] = m
        return out

    def _edge_orient(self, chan, quad):
        """Asento neliösymbolille: kaksi vierekkäistä tasaista sivua (L) ja kaksi vuorottelevaa
        (kello). Palauttaa (k, laatu), k: kanoninen -> kuvan asento np.rot90(M, -k)."""
        n_ = self.geo.rows
        Wt = n_ * FT
        dst = np.float32([[0, 0], [Wt, 0], [Wt, Wt], [0, Wt]])
        w_ = cv2.warpPerspective(chan, cv2.getPerspectiveTransform(order_corners(quad), dst),
                                 (Wt, Wt), flags=cv2.INTER_LINEAR)
        a_, b_ = int(FT * 0.25), int(FT * 0.75)
        lo, hi = FT, Wt - FT
        prof = {"top": w_[a_:b_, lo:hi].mean(axis=0),
                "bottom": w_[Wt - b_:Wt - a_, lo:hi].mean(axis=0),
                "left": w_[lo:hi, a_:b_].mean(axis=1),
                "right": w_[lo:hi, Wt - b_:Wt - a_].mean(axis=1)}
        sd = {k_: float(np.std(v_)) + 1e-6 for k_, v_ in prof.items()}
        pairs = {("bottom", "left"): 0, ("left", "top"): 1, ("right", "top"): 2,
                 ("bottom", "right"): 3}
        best = None
        for (p1, p2), k_ in pairs.items():
            others = [x for x in sd if x not in (p1, p2)]
            ratio = (sd[others[0]] + sd[others[1]]) / (sd[p1] + sd[p2])
            if best is None or ratio > best[1]:
                best = (k_, ratio)
        return best

    def match(self, bgr, blur=None, refine=6):
        """Palauttaa dict(ranked=[(ncc, teksti, skeema, vaihtoehto)], info, agree) tai None."""
        if self.stack is None:
            return None
        geo = self.geo
        shared_c = (self.stack == self.stack[0]).all(axis=0)
        border_c = geo.dm | geo.lm
        base_c = shared_c | border_c
        chans = self._channels(bgr)
        # kanavat kontrastin mukaan (p90-p10); käytetään kahta kontrastisinta
        chans.sort(key=lambda t: -(np.percentile(t[1], 90) - np.percentile(t[1], 10)))
        chans = chans[:2]
        # paikannusehdokkaat usealta kanavalta
        quads = []
        for _, ch in chans:
            g8 = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            for q in locate_quads(g8, geo.aspect, max_quads=3):
                q = order_corners(q)
                if all(np.linalg.norm(q - o, axis=1).mean() > 2.0 for o in quads):
                    quads.append(q)
        if not quads:
            return None
        # suunnikasvariantit: yksi kulma kerrallaan korvataan kolmesta muusta lasketulla
        # (kestää esim. siirtyneen reunakaistan aiheuttaman ulkonevan kulman)
        extra = []
        for q in quads:
            for c in range(4):
                v = q.copy()
                v[c] = q[(c + 1) % 4] + q[(c + 3) % 4] - q[(c + 2) % 4]
                if all(np.linalg.norm(v - o, axis=1).mean() > 2.0 for o in quads + extra):
                    extra.append(v)
        quads = quads + extra
        h, w = bgr.shape[:2]
        blur = blur if blur is not None else max(1.5, FT * 0.30)
        # karkea haku: kanava x nelikulmio; asento reunojen tasaisuudesta (L = tasainen,
        # kello = vuorotteleva), neliösymbolille. Suorakulmaiselle kokeillaan kaikki asennot.
        proto = self.stack[0].astype(np.float32).copy()
        proto[~base_c] = 0.5
        pb = np.full((geo.rows, geo.cols), 0.5, np.float32)
        pb[geo.dm] = 1.0
        pb[geo.lm] = 0.0
        coarse = []
        for ci, (cn, ch) in enumerate(chans):
            for qi, q in enumerate(quads):
                if geo.square:
                    k, qual = self._edge_orient(ch, q)
                    ks = (k,)
                else:
                    ks, qual = (0, 1, 2, 3), 1.0
                best_c = (0.0, ks[0])
                for k in ks:
                    tb = _render(self._orient(pb, k), blur)
                    wb = _kron(self._orient(border_c, k)) > 0.5
                    th_, tw_ = tb.shape
                    c = _ncc(tb, self._warp(ch, q, np.zeros(8), th_, tw_), wb)
                    if abs(c) > abs(best_c[0]):
                        best_c = (c, k)
                coarse.append((qual * abs(best_c[0]), best_c[0], ci, qi, best_c[1]))
        coarse.sort(reverse=True)
        # hienosäätö parhaille (kanava, nelikulmio, asento) -kolmikoille: karkea -> tarkka malli
        fits = []
        used_q = set()
        for _, c0, ci, qi, k in coarse:
            if (ci, qi) in used_q:
                continue
            used_q.add((ci, qi))
            cn, ch = chans[ci]
            chan = ch if c0 >= 0 else 255.0 - ch
            So = self._orient(proto, k)
            wm = _kron(self._orient(base_c, k)) > 0.5
            rng = np.random.RandomState(ci * 7 + qi * 3 + k)
            r1 = self._fit(chan, quads[qi], _render(So, blur * 1.9), wm,
                           [np.zeros(8), rng.uniform(-2.5, 2.5, 8)])
            r2 = self._fit(chan, quads[qi], _render(So, blur), wm, [r1.x])
            fits.append((-r2.fun, r2.x, cn, chan, qi, k))
            if len(fits) >= 12:
                break
        fits.sort(key=lambda t: -t[0])

        def solve(fit):
            a_ncc, px, cn, chan, qi, k = fit
            quad0 = quads[qi]
            masks = self._strip_masks(k)
            sample_img = self._orient(self.stack[0], k)
            th, tw = sample_img.shape[0] * FT, sample_img.shape[1] * FT
            dst = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
            q_fit = quad0 + np.float32(px).reshape(4, 2)
            W0 = np.linalg.inv(cv2.getPerspectiveTransform(q_fit, dst)).astype(np.float32)
            tmpls = [_render(self._orient(M, k), blur) for _, _, M in self.items]
            tmpls_w = [_render(self._orient(M, k), blur * 1.9) for _, _, M in self.items]
            kmask = {mn: _kron(mm) > 0.5 for mn, mm in masks.items()}
            zero = np.zeros(8)

            def full_fit(idx, mn, starts):
                r1 = self._fit(chan, quad0, tmpls_w[idx], kmask[mn], starts)
                return self._fit(chan, quad0, tmpls[idx], kmask[mn], [r1.x])

            # 1a) halpa esikarsinta: kevyt sovitus (vain tarkka malli) kaikille ehdokkaille
            cheap = []
            for idx in range(len(self.items)):
                r = self._fit(chan, quad0, tmpls[idx], kmask["kaikki"], [px])
                cheap.append((-r.fun, idx))
            cheap.sort(reverse=True)
            # 1b) parhaille täysi kaksivaiheinen sovitus useammasta alkuarvauksesta
            res_all = []
            for cc0, idx in cheap[:max(25, refine)]:
                r = full_fit(idx, "kaikki", [px, zero])
                res_all.append((max(-r.fun, cc0), idx, r.x))
            res_all += [(cc0, idx, px) for cc0, idx in cheap[max(25, refine):]]
            res_all.sort(key=lambda t: -t[0])
            # 2) parhaille: sovitus myös niin, että yksi reunakaista jätetään pois
            final = {}
            for cc, idx, xk in res_all[:refine]:
                best = (cc, "kaikki", xk)
                for name in ("ei oikeaa", "ei vasenta", "ei ylintä", "ei alinta"):
                    r = full_fit(idx, name, [xk])
                    if -r.fun > best[0]:
                        best = (-r.fun, name, r.x)
                final[idx] = best
            ranked = [(v[0], self.items[i][0], self.items[i][1], v[1]) for i, v in final.items()]
            rest = [(cc, self.items[i][0], self.items[i][1], "kaikki")
                    for cc, i, _ in res_all[refine:]]
            ranked.sort(reverse=True)
            seen, uniq = set(), []
            for item in ranked + rest:
                if item[1] not in seen:
                    seen.add(item[1])
                    uniq.append(item)
            top_idx = max(final, key=lambda i: final[i][0])
            warped2 = self._warp(chan, quad0, final[top_idx][2], th, tw)
            Si = self._orient(self.items[top_idx][2], k)
            mod = warped2.reshape(Si.shape[0], FT, Si.shape[1], FT).mean(axis=(1, 3))
            thr = (mod[Si == 1].mean() + mod[Si == 0].mean()) / 2
            agree = ((mod - thr) * (Si.astype(np.float32) - 0.5)) > 0
            return dict(ranked=uniq, channel=cn, orient=k, ncc_align=a_ncc, agree=agree,
                        best_mask=final[top_idx][1], top=final[top_idx][0])

        sols, seen_fit = [], set()
        for fit in fits:
            key = (fit[2], fit[4], fit[5])
            if key in seen_fit:
                continue
            seen_fit.add(key)
            sols.append(solve(fit))
            if len(sols) >= 2:
                break
        return max(sols, key=lambda d_: d_["top"])


# ============================================================================
# Lukija: yksi symbolikoko, kertyvä evidenssi, valinnainen ehdokaskirjasto
# ============================================================================
class Reader:
    def __init__(self, geo, serials, schemes):
        self.geo = geo
        self.entries = []
        mats = []
        self.skipped = 0
        if serials and HAVE_DMTX:
            warned = False
            for s in serials:
                ok = False
                for sch in schemes:
                    m = geo.module_matrix(s, sch)
                    if m is None:
                        continue
                    if not (m[geo.dm].all() and not m[geo.lm].any()) and not warned:
                        print(f"VAROITUS ({geo.name}): enkoodatun symbolin reunakuvio ei vastaa "
                              "odotettua (tarkista pylibdmtx-versio).", file=sys.stderr)
                        warned = True
                    self.entries.append((s, sch))
                    mats.append(np.where(m[1:-1, 1:-1], 1.0, -1.0))
                    ok = True
                if not ok:
                    self.skipped += 1
        self.mats = np.stack(mats) if mats else None
        self.reset()

    def reset(self):
        self.s = np.zeros((self.geo.rows, self.geo.cols))
        self.w = 0.0
        self.frames = 0

    def add(self, res):
        w = min(res["dp"], 5.0) ** 2
        self.s += w * res["e"]
        self.w += w
        self.frames += 1

    @property
    def E(self):
        return self.s / self.w if self.w > 0 else np.zeros_like(self.s)

    def rank(self, E=None):
        if self.mats is None:
            return []
        E = self.E if E is None else E
        core = np.clip(E[1:-1, 1:-1], -1, 1)
        err = ((1 - core[None] * self.mats) / 2).mean(axis=(1, 2))
        best = {}
        for (s, sch), e in zip(self.entries, err):
            if s not in best or e < best[s][0]:
                best[s] = (float(e), s, sch)
        return sorted(best.values())

    def own_decode(self, E=None):
        E = self.E if E is None else E
        return decode_evidence(E, self.geo.rows, self.geo.cols)


class Pool:
    def __init__(self, serials, schemes):
        self.serials, self.schemes, self.readers = serials, schemes, {}

    def get(self, geo):
        if geo.name not in self.readers:
            rd = Reader(geo, self.serials, self.schemes)
            if self.serials:
                msg = f"Koko {geo.name}: {len(rd.entries)} kirjastomerkintää"
                if rd.skipped:
                    msg += f", {rd.skipped} ehdokasta ei mahdu"
                print(msg)
            self.readers[geo.name] = rd
        return self.readers[geo.name]

    def all(self):
        return list(self.readers.values())


def decide(ranked, max_err, min_margin):
    if not ranked:
        return None, "ei ehdokkaita"
    best = ranked[0]
    margin = (ranked[1][0] - best[0]) if len(ranked) > 1 else 1.0
    info = f"virhe {best[0]:.3f}, marginaali {margin:.3f}, skeema {best[2]}"
    if best[0] > max_err:
        return None, "liian huono täsmäys: " + info
    if margin < min_margin:
        return None, "epävarma (tarkista käsin): " + info
    return best[1], "OK: " + info


def direct_decode(gray):
    """Valinnainen nopea polku: libdmtx / zxing-cpp."""
    for img in (gray, 255 - gray):
        if HAVE_DMTX:
            try:
                res = dmtx_decode(img, timeout=300, max_count=1)
            except Exception:
                res = []
            if res:
                return res[0].data.decode("latin-1")
        if zxingcpp is not None:
            r = zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.DataMatrix,
                                       try_rotate=True, try_downscale=True)
            if r:
                return r[0].text
    return None


def load_serials(path):
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\r\n") for ln in f]       # ÄLÄ trimmaa välilyöntejä
    return [ln for ln in lines if ln != ""]


def describe_own(res, serial_set):
    note = ""
    if serial_set:
        note = "  [kuuluu ehdokkaisiin]" if res["text"] in serial_set else \
               "  [EI ole ehdokaslistassa!]"
    return (f"{res['text']!r}  (RS: {res['errors']} virhettä korjattu, "
            f"{res['erasures']} koodisanaa poistettu; {','.join(res['info'])}){note}")


def final_report(readers, args, serial_set):
    print("\n=== Yhdistetty tulos ===")
    rows = []
    for rd in readers:
        if rd.frames == 0:
            continue
        own = rd.own_decode()
        ranked = rd.rank()
        val, info = decide(ranked, args.max_err, args.min_margin) if ranked else (None, "")
        rows.append((rd, own, ranked, val, info))
    if not rows:
        print("Ei käyttökelpoisia kuvia (laatu alle --dp-min).")
        return False
    # parhaat ensin: oma dekoodaus onnistui > ehdokas OK > pienin virhe
    rows.sort(key=lambda t: (t[1] is None, (2 * t[1]['errors'] + t[1]['erasures']) if t[1] else 0,
                              t[3] is None, t[2][0][0] if t[2] else 9))
    for rd, own, ranked, val, info in rows:
        print(f"\nKoko {rd.geo.name} ({rd.frames} kuvaa):")
        print("  Oma dekooderi:", describe_own(own, serial_set) if own else
              "ei onnistunut (liikaa vioittuneita koodisanoja)")
        if ranked:
            for e, s, sch in ranked[:5]:
                print(f"  ehdokas {e:.3f}  {s!r}  ({sch})")
            print(f"  ehdokasvertailu -> {val!r}  {info}")
    rd, own, ranked, val, info = rows[0]
    if own:
        print(f"\nTULOS: {own['text']!r}  [koko {rd.geo.name}, oma dekooderi]")
        return True
    if val is not None:
        print(f"\nTULOS: {val!r}  [koko {rd.geo.name}, ehdokasvertailu]  {info}")
        return True
    print("\nTULOS: ei varmaa tulosta" + (f"  ({info})" if info else ""))
    return False


def run_fit(files, geos, serials, schemes, args):
    """Mallisovitus: sovittaa jokaisen ehdokkaan suoraan kuviin (pienet/sumeat/vioittuneet)."""
    if not serials:
        print("\nMallisovitus vaatii ehdokaslistan (serials.txt).")
        return False
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("\nMallisovitus vaatii scipyn: py -m pip install scipy")
        return False
    found = False
    for geo in geos:
        fm = FitMatcher(geo, serials, schemes)
        if fm.stack is None:
            print(f"\nMallisovitus [{geo.name}]: yksikään ehdokas ei mahdu symbolikokoon.")
            continue
        print(f"\n=== Mallisovitus, koko {geo.name} ({len(fm.items)} ehdokasta, "
              f"{len(files)} kuvaa) ===")
        scores, first = {}, None
        for fp in files:
            bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            res = fm.match(bgr)
            if res is None:
                print(f"{os.path.basename(fp)}: koodia ei löytynyt")
                continue
            top = res["ranked"][0]
            print(f"{os.path.basename(fp)}: kanava {res['channel']}, paras {top[1]!r} "
                  f"(korrelaatio {top[0]:.3f}, {top[3]})")
            if first is None:
                first = res
            for sc, txt, sch, mn in res["ranked"]:
                scores.setdefault(txt, []).append(sc)
        if not scores:
            continue
        nmax = max(len(v) for v in scores.values())
        agg = sorted(((float(np.mean(v)), t) for t, v in scores.items() if len(v) == nmax),
                     reverse=True)
        for sc, t in agg[:5]:
            print(f"  {sc:.3f}  {t!r}")
        best = agg[0]
        margin = best[0] - agg[1][0] if len(agg) > 1 else None
        ok = best[0] >= args.fit_min and (margin is None or margin >= args.fit_margin)
        mtxt = f", marginaali {margin:.3f}" if margin is not None else " (vain yksi ehdokas)"
        if ok:
            print(f"\nTULOS (mallisovitus): {best[1]!r}  [koko {geo.name}]  "
                  f"korrelaatio {best[0]:.3f}{mtxt}")
            found = True
        else:
            print(f"\nMallisovitus: ei varmaa tulosta. Paras {best[1]!r}, korrelaatio "
                  f"{best[0]:.3f}{mtxt} (rajat: {args.fit_min} / {args.fit_margin})")
        if first is not None:
            print("Yhteensopivuuskartta parhaalle (kuvan asennossa, X = poikkeaa; "
                  f"vioittunut kaista pois: {first['best_mask']}):")
            for row in first["agree"]:
                print("  " + "".join("." if v else "X" for v in row))
    return found


def geometries_for(gray, fixed):
    if fixed is not None:
        return [geo_for(r, c) for r, c in fixed]
    return auto_geometries(gray)


def process_images(files, fixed, pool, serial_set, args, serials=(), schemes=SCHEMES):
    direct_hits = []
    geo_dp = {}
    for fp in files:
        name = os.path.basename(fp)
        gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"{name}: ei voitu lukea")
            continue
        if not args.no_direct and (HAVE_DMTX or zxingcpp is not None):
            txt = direct_decode(gray)
            if txt is not None:
                print(f"{name}: suora dekoodaus -> {txt!r}")
                direct_hits.append(txt)
        geos = geometries_for(gray, fixed)
        if not geos:
            print(f"{name}: koodia/kokoa ei tunnistettu")
            continue
        results = []
        for geo in geos:
            res = geo.analyze(gray)
            if res is not None:
                results.append((geo, res))
        if not results:
            print(f"{name}: koodia ei löytynyt")
            continue
        for g_, r_ in results:
            old = geo_dp.get(g_.name, (g_, 0.0))
            geo_dp[g_.name] = (g_, old[1] + r_["dp"])
        usable_res = [(g_, r_) for g_, r_ in results if r_["dp"] >= args.dp_min]
        shown = usable_res if usable_res else [max(results, key=lambda t: t[1]["dp"])]
        for geo, res in shown:
            usable = res["dp"] >= args.dp_min
            rd = pool.get(geo)
            own = decode_evidence(res["e"] * 1.0, geo.rows, geo.cols)
            line = f"{name} [{geo.name}]: laatu {res['dp']:.2f}" + ("" if usable else " (heikko)")
            line += "  oma: " + (repr(own["text"]) if own else "ei")
            if rd.mats is not None:
                top = rd.rank(res["e"])
                if top:
                    line += f"  ehdokas: {top[0][1]!r} (virhe {top[0][0]:.3f})"
            print(line)
            if usable and not args.each:
                rd.add(res)
    if direct_hits:
        print(f"\nSuoria dekoodauksia {len(direct_hits)} kpl, yleisin: "
              f"{max(set(direct_hits), key=direct_hits.count)!r}")
    own_ok = False
    if not args.each:
        own_ok = final_report(pool.all(), args, serial_set)
    if serials and args.fit != "off" and (args.fit == "on" or not own_ok):
        if fixed is not None:
            geos = [geo_for(r, c) for r, c in fixed]
        else:
            ranked_geos = sorted(geo_dp.values(), key=lambda t: -t[1])
            geos = [g_ for g_, _ in ranked_geos[:1]]
        if geos:
            run_fit(files, geos, serials, schemes, args)
        else:
            print("\nMallisovitus: symbolikokoa ei tunnistettu, anna --size.")


def run_camera(fixed, pool, serial_set, args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Kameraa ei voitu avata")
    status = "..."
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for geo in geometries_for(gray, fixed):
            rd = pool.get(geo)
            res = geo.analyze(gray)
            if res and res["dp"] >= args.dp_min:
                rd.add(res)
                own = rd.own_decode()
                if own:
                    status = f"{geo.name} {rd.frames} kuvaa | {own['text']!r} | RS OK"
                elif rd.mats is not None:
                    val, info = decide(rd.rank(), args.max_err, args.min_margin)
                    status = f"{geo.name} {rd.frames} kuvaa | {val!r} | {info}"
                else:
                    status = f"{geo.name} {rd.frames} kuvaa | ei vielä"
        cv2.putText(frame, status[:90], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2)
        cv2.imshow("datamatrixFixer (r=nollaa, q=lopeta)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            for rd in pool.all():
                rd.reset()
            status = "nollattu"
    cap.release()
    cv2.destroyAllWindows()
    print(status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("serials", nargs="?", help="(valinnainen) tekstitiedosto, yksi ehdokas per rivi")
    ap.add_argument("--size", default="auto",
                    help="symbolikoko: 20, 16x16, 8x18, lista 18,20,22 tai auto (oletus)")
    ap.add_argument("--images", help="kansio tai glob-kuvio (esim. kuvat/*.png)")
    ap.add_argument("--camera", type=int, help="kameran indeksi")
    ap.add_argument("--each", action="store_true", help="käsittele kuvat erikseen")
    ap.add_argument("--no-direct", action="store_true", help="ohita libdmtx/zxing-nopea polku")
    ap.add_argument("--scheme", action="append", help="rajaa ehdokasenkoodauksen skeemaa")
    ap.add_argument("--dp-min", type=float, default=1.5, help="kuvan minimilaatu")
    ap.add_argument("--max-err", type=float, default=0.20)
    ap.add_argument("--min-margin", type=float, default=0.03)
    ap.add_argument("--fit", choices=["auto", "on", "off"], default="auto",
                    help="mallisovitus (pienet/sumeat kuvat): auto = kun oma dekooderi epäonnistuu")
    ap.add_argument("--fit-min", type=float, default=0.55, help="mallisovituksen min. korrelaatio")
    ap.add_argument("--fit-margin", type=float, default=0.08, help="min. marginaali toiseen")
    args = ap.parse_args()

    serials = load_serials(args.serials)
    if serials and not HAVE_DMTX:
        print("HUOM: pylibdmtx ei latautunut (" + _DMTX_ERR + ") -> ehdokasvertailu pois käytöstä, "
              "käytetään vain omaa dekooderia.", file=sys.stderr)
    fixed = parse_sizes(args.size)
    pool = Pool(serials, args.scheme or SCHEMES)

    if args.camera is not None:
        run_camera(fixed, pool, set(serials), args)
    elif args.images:
        if os.path.isdir(args.images):
            files = sorted(glob.glob(os.path.join(args.images, "*")))
        else:
            files = sorted(glob.glob(args.images))
        process_images(files, fixed, pool, set(serials), args, serials, args.scheme or SCHEMES)
    else:
        ap.error("anna --images tai --camera")


if __name__ == "__main__":
    main()

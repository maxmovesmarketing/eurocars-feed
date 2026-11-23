"""Foto 1 van Eurocars heeft een reclame-overlay die Hexon er al in de bron opzet
(geen schone versie beschikbaar). We maken hem zelf schoon:
precies masker per sjabloon (banner, oranje rand, adresbalk, Google-score, award-logo,
"wordt verwacht"-sticker) en invullen met big-LaMa op volle resolutie. Er wordt niets
van de auto weggesneden; alleen de witte opvulbalken van Hexon gaan eraf.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import cv2
import httpx
import numpy as np

from .config import ASSETS, IMG_DIR, MODEL_PATH, PUBLIC_BASE_URL
from .scraper import Vehicle, gallery

log = logging.getLogger(__name__)
_lama = None
_dev = "cpu"


def _model():
    global _lama, _dev
    if _lama is None:
        import torch

        _dev = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.set_num_threads(max(1, torch.get_num_threads()))
        _lama = torch.jit.load(str(MODEL_PATH), map_location="cpu").to(_dev).eval()
    return _lama


def _run_lama(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    lama = _model()
    h, w = img.shape[:2]
    H, W = (h + 7) // 8 * 8, (w + 7) // 8 * 8
    im = np.zeros((H, W, 3), np.float32)
    im[:h, :w] = img[..., ::-1] / 255.0
    m = np.zeros((H, W), np.float32)
    m[:h, :w] = mask > 0
    t = torch.from_numpy(im).permute(2, 0, 1)[None].to(_dev)
    mt = torch.from_numpy(m)[None, None].to(_dev)
    with torch.no_grad():
        out = lama(t, mt)[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(out * 255, 0, 255).astype(np.uint8)[:h, :w, ::-1]


def _blend(img, out, mask):
    soft = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 1.5)[..., None]
    soft = np.maximum(soft, (mask > 0)[..., None])
    return (out * soft + img * (1 - soft)).astype(np.uint8)


def _poly(m, pts, s):
    cv2.fillPoly(m, [np.array([[int(x * s), int(y * s)] for x, y in pts], np.int32)], 255)


def template(im: np.ndarray) -> str:
    h = im.shape[0]
    if im.mean() < 60:
        return "placeholder"  # zwarte "WORDT VERWACHT"-plaat, geen echte foto
    return "letterbox" if im[: int(h * 0.04)].mean() > 245 else "bar"


_rembg = None


def car_mask(im: np.ndarray) -> np.ndarray:
    """Voorgrond (de auto) volgens rembg/u2net; 255 = auto."""
    global _rembg
    from PIL import Image
    from rembg import new_session, remove

    if _rembg is None:
        _rembg = new_session("u2net")
    m = np.array(remove(Image.fromarray(im[..., ::-1]), session=_rembg, only_mask=True))
    return (m > 128).astype(np.uint8) * 255


def _off(t: str) -> int:
    return 40 if t == "letterbox" else 0


def _fixed_mask(im, t):
    """Vaste delen van het sjabloon: banner, oranje rand, adresbalk, award-logo."""
    h, w = im.shape[:2]
    s = w / 1024
    m = np.zeros((h, w), np.uint8)
    if t == "bar":
        _poly(m, [(726, 0), (1024, 0), (1024, 72), (696, 72)], s)
        _poly(m, [(990, 0), (1024, 0), (1024, 768), (990, 768)], s)
        _poly(m, [(0, 696), (1024, 696), (1024, 768), (0, 768)], s)
        _poly(m, [(735, 676), (1024, 676), (1024, 700), (725, 700)], s)
    else:
        _poly(m, [(736, 40), (1024, 40), (1024, 114), (698, 114)], s)
        _poly(m, [(986, 40), (1024, 40), (1024, 728), (986, 728)], s)
        _poly(m, [(730, 716), (1024, 716), (1024, 728), (730, 728)], s)
    logo = cv2.imread(str(ASSETS / f"logo_{w}_{t}.png"), 0)
    if logo is None or logo.shape != (h, w):
        ref = cv2.imread(str(ASSETS / f"logo_1024_{t}.png"), 0)
        logo = cv2.resize(ref, (w, h), interpolation=cv2.INTER_NEAREST) if ref is not None else None
    if logo is not None:
        # alleen als het award-logo er echt op staat (nieuwere sjablonen hebben het niet)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        gold = (hsv[..., 0] >= 5) & (hsv[..., 0] <= 32) & (hsv[..., 1] > 100) & (hsv[..., 2] > 100)
        if gold[logo > 0].mean() > 0.35:
            m = np.maximum(m, cv2.dilate(logo, np.ones((int(13 * s) | 1,) * 2, np.uint8)))
    return m


def _fill_components(m: np.ndarray, min_area: float) -> np.ndarray:
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, [c for c in cnts if cv2.contourArea(c) >= min_area], -1, 255, -1)
    return out


def _zone_mask(im, car, box, s):
    """Alles in een zone dat geen egale muur/vloer is en geen auto: score, stickers,
    linten, merklogo's, actie-emblemen. Werkt ook voor nieuwe campagnes."""
    x0, y0, x1, y1 = (int(v * s) for v in box)
    reg = im[y0:y1, x0:x1]
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.medianBlur(g.astype(np.uint8), 31).astype(np.float32)
    odd = (hsv[..., 1] > 70) | (np.abs(g - bg) > 35)
    m = np.zeros(im.shape[:2], np.uint8)
    m[y0:y1, x0:x1] = odd.astype(np.uint8) * 255
    m[car > 0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(15 * s) | 1,) * 2, np.uint8))
    m = _fill_components(m, 60 * s * s)
    return cv2.dilate(m, np.ones((int(9 * s) | 1,) * 2, np.uint8))


def _badge_mask(im, t):
    """Rond actie-embleem rechtsonder (bv. WINNAAR 26/27), ook als het over de auto staat."""
    h, w = im.shape[:2]
    s = w / 1024
    y0, x0 = int((440 + _off(t)) * s), int(700 * s)
    reg = im[y0:, x0:]
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    m = np.zeros((h, w), np.uint8)
    white = ((hsv[..., 1] < 45) & (hsv[..., 2] > 225)).mean()
    org = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 150) & (hsv[..., 2] > 170)).mean()
    if white < 0.08 or org < 0.08:
        return m
    gray = cv2.medianBlur(cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY), 5)
    cs = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, param1=120, param2=40,
                          minRadius=int(100 * s), maxRadius=int(155 * s))
    if cs is None:
        return m
    cx, cy, r = cs[0][0]
    # Hough vindt vaak de binnenring; de buitenrand (goud/oranje) ligt ~12% verder
    cv2.circle(m, (int(cx + x0), int(cy + y0)), int(r * 1.15 + 8 * s), 255, -1)
    return m


def _darkbox_mask(im, t):
    """Zwart tekstblok (bv. "Tweede Kerstdag geopend") boven de adresbalk."""
    h, w = im.shape[:2]
    s = w / 1024
    x0, y0, x1, y1 = int(560 * s), int((380 + _off(t)) * s), int(1000 * s), int((700 + _off(t)) * s)
    reg = im[y0:y1, x0:x1]
    dark = ((reg.max(-1) < 45) | (reg.min(-1) > 200)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((int(9 * s) | 1,) * 2, np.uint8))
    m = np.zeros((h, w), np.uint8)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > 9000 * s * s and cv2.contourArea(c) / (bw * bh) > 0.85 and bw > 1.4 * bh:
            pad = int(6 * s)
            cv2.rectangle(m, (x0 + x - pad, y0 + y - pad), (x0 + x + bw + pad, y0 + y + bh + pad), 255, -1)
    return m


def _promo_mask(im, t, car):
    """Bestickering op de vloer/muur rechtsonder in actiekleuren (oranje/goud/wit), geen auto."""
    h, w = im.shape[:2]
    s = w / 1024
    x0, y0 = int(690 * s), int((430 + _off(t)) * s)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    col = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 32) & (hsv[..., 1] > 110) & (hsv[..., 2] > 120)).astype(np.uint8) * 255
    m = np.zeros((h, w), np.uint8)
    m[y0:, x0:] = col[y0:, x0:]
    m[car > 0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(17 * s) | 1,) * 2, np.uint8))
    m = _fill_components(m, 400 * s * s)
    return cv2.dilate(m, np.ones((int(9 * s) | 1,) * 2, np.uint8))


def _sticker_mask(im):
    h, w = im.shape[:2]
    s = w / 1024
    tpl = cv2.resize(cv2.imread(str(ASSETS / "sticker_tpl.png")), None, fx=s, fy=s)
    reg = im[: int(h * 0.45), : int(w * 0.45)]
    r = cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    m = np.zeros((h, w), np.uint8)
    if mx < 0.6:
        return m
    ox, oy = loc[0] - int(70 * s), loc[1] - int(100 * s)
    x0, y0 = max(0, ox + int(10 * s)), max(0, oy + int(25 * s))
    x1, y1 = ox + int(365 * s), oy + int(260 * s)
    box = im[y0:y1, x0:x1]
    hb = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    sel = ((box.min(-1) > 236) | (box.max(-1) < 60) | ((hb[..., 0] < 25) & (hb[..., 1] > 120))).astype(np.uint8) * 255
    sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, np.ones((int(25 * s) | 1,) * 2, np.uint8))
    cnts, _ = cv2.findContours(sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sel, [c for c in cnts if cv2.contourArea(c) > 2000 * s * s], -1, 255, -1)
    core = np.zeros((h, w), np.uint8)
    core[y0:y1, x0:x1] = sel
    near = cv2.dilate(core, np.ones((int(18 * s) | 1,) * 2, np.uint8))
    far = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(90 * s) | 1,) * 2))
    hs = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    wall = ((hs[..., 1] < 40) & (hs[..., 2] > 110)).astype(np.uint8) * 255
    return cv2.morphologyEx(np.maximum(near, far & wall), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def is_cover(im: np.ndarray, car: np.ndarray) -> bool:
    """Auto onder een (donkere) hoes: grote egale donkere voorgrond zonder detail."""
    fg = car > 0
    if fg.mean() < 0.08:
        return False
    px = im[fg]
    dark = (px.max(-1) < 90).mean()
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = np.abs(cv2.Laplacian(g, cv2.CV_32F))[fg].mean()
    return dark > 0.75 and tex < 6


def _inpaint(im, m):
    """Grote gaten eerst op lage resolutie (structuur), daarna de rand scherp op volle resolutie."""
    h, w = im.shape[:2]
    s = w / 1024
    if not m.any():
        return im
    f = 512 / w
    small = cv2.resize(im, (512, int(h * f)), interpolation=cv2.INTER_AREA)
    mm = cv2.resize(m, (512, int(h * f)), interpolation=cv2.INTER_NEAREST)
    lo = cv2.resize(_run_lama(small, mm), (w, h), interpolation=cv2.INTER_CUBIC)
    guide = np.where(m[..., None] > 0, lo, im)
    inner = cv2.erode(m, np.ones((int(25 * s) | 1,) * 2, np.uint8))
    band = cv2.subtract(m, inner)
    out = _blend(guide, _run_lama(guide, band), band) if band.any() else guide
    return _blend(im, out, m)


def build_mask(im: np.ndarray, t: str, car: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    s = w / 1024
    o = _off(t)
    m = _fixed_mask(im, t)
    gpoly = [(6, 6), (122, 6), (122, 62), (6, 62)] if t == "bar" else [(12, 52), (116, 52), (116, 104), (12, 104)]
    _poly(m, gpoly, s)                                            # Google-score (altijd)
    m |= _zone_mask(im, car, (0, o, 360, 215 + o), s)            # linksboven: score, stickers, linten, logo's
    m |= _zone_mask(im, car, (0, o, 140, 330 + o), s)            # onderkant van schuine hoeklinten
    m |= _zone_mask(im, car, (620, 72 + o, 990, 150 + o), s)     # onder de banner
    m |= _sticker_mask(im)                                        # "WORDT VERWACHT"
    m |= _badge_mask(im, t)                                       # rond actie-embleem
    m |= _darkbox_mask(im, t)                                     # zwart tekstblok
    m |= _promo_mask(im, t, car)                                  # overige actiekleuren rechtsonder
    return cv2.dilate(m, np.ones((int(5 * s) | 1,) * 2, np.uint8))


def trim_borders(im: np.ndarray) -> np.ndarray:
    """Effen witte/zwarte opvulranden (Hexon) weghalen; nooit meer dan 20% per kant."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape

    def flat(line):
        return line.std() < 4 and (line.mean() > 243 or line.mean() < 14)

    def hard(a, b):  # echte opvulbalk eindigt in een harde overgang, overbelichte muur niet
        return float(np.abs(a - b).mean()) > 12

    top = 0
    while top < h * 0.2 and flat(g[top]):
        top += 1
    if top and (top >= h * 0.2 or not hard(g[top - 1], g[min(top + 1, h - 1)])):
        top = 0
    bot = h
    while bot > h * 0.8 and flat(g[bot - 1]):
        bot -= 1
    if bot < h and (bot <= h * 0.8 or not hard(g[bot], g[max(bot - 2, 0)])):
        bot = h
    left = 0
    while left < w * 0.2 and flat(g[:, left]):
        left += 1
    if left and (left >= w * 0.2 or not hard(g[:, left - 1], g[:, min(left + 1, w - 1)])):
        left = 0
    right = w
    while right > w * 0.8 and flat(g[:, right - 1]):
        right -= 1
    if right < w and (right <= w * 0.8 or not hard(g[:, right], g[:, max(right - 2, 0)])):
        right = w
    return im[top:bot, left:right]


def clean(im: np.ndarray) -> np.ndarray | None:
    """Schone versie van foto 1, of None als het geen bruikbare foto van de auto is
    (zwarte "wordt verwacht"-plaat of auto onder een hoes)."""
    t = template(im)
    if t == "placeholder":
        return None
    car = car_mask(im)
    if is_cover(im, car):
        return None
    h, w = im.shape[:2]
    s = w / 1024
    out = _inpaint(im, build_mask(im, t, car))
    # de oranje rand rechts bedekte daar alles; wegsnijden i.p.v. verzinnen
    out = out[:, : int(984 * s)]
    if t == "letterbox":
        out = out[int(43 * s): int(725 * s)]  # alleen de witte Hexon-opvulbalken
    return trim_borders(out)


ALGO = "v3"  # ophogen als de schoonmaak verandert: dan worden alle foto's opnieuw gemaakt


def _version(r: httpx.Response) -> str:
    return f"{ALGO}|{r.headers.get('last-modified', '')}|{r.headers.get('content-length', '')}"


async def prepare(client: httpx.AsyncClient, vehicles: list[Vehicle]) -> tuple[list[Vehicle], dict]:
    """Zet v.images (schone foto 1 + galerij). Auto's zonder echte foto vallen af."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = IMG_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    keep: list[Vehicle] = []
    stats = {"schoongemaakt": 0, "uit_cache": 0, "geen_foto": [], "mislukt": []}
    for v in vehicles:
        hid = v.stock_number
        if not v.photos:
            stats["geen_foto"].append(hid)
            continue
        dst = IMG_DIR / f"{hid}.jpg"
        src = v.photos[0]
        try:
            head = await client.head(src)
            ver = _version(head)
            if meta.get(hid) == "skip|" + ver:
                rest = gallery(v)
                if rest:
                    v.images = rest[:10]
                    keep.append(v)
                else:
                    stats["geen_foto"].append(hid)
                continue
            if not (dst.exists() and meta.get(hid) == ver):
                r = await client.get(src)
                r.raise_for_status()
                im = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                out = clean(im) if im is not None else None
                if out is None:
                    # zwarte plaat of auto onder een hoes: eerste echte galerijfoto wordt de hoofdfoto
                    rest = gallery(v)
                    if rest:
                        v.images = rest[:10]
                        stats.setdefault("hoofdfoto_uit_galerij", []).append(hid)
                        keep.append(v)
                    else:
                        stats["geen_foto"].append(hid)
                    meta[hid] = "skip|" + ver
                    continue
                cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
                meta[hid] = ver
                stats["schoongemaakt"] += 1
            else:
                stats["uit_cache"] += 1
        except Exception as e:  # noqa: BLE001
            log.warning("foto %s mislukt: %s", hid, e)
            stats["mislukt"].append(hid)
            if not dst.exists():
                rest = gallery(v)  # foto 1 bestaat niet (404): volgende foto wordt hoofdfoto
                if rest:
                    v.images = rest[:10]
                    keep.append(v)
                continue
        v.images = [f"{PUBLIC_BASE_URL}/img/{hid}.jpg?v={hashlib.md5(meta.get(hid, '').encode()).hexdigest()[:8]}"] + gallery(v)[:9]
        keep.append(v)
    # schone foto's van verkochte auto's opruimen
    live = {v.stock_number for v in vehicles}
    for p in IMG_DIR.glob("*.jpg"):
        if p.stem not in live:
            p.unlink(missing_ok=True)
            meta.pop(p.stem, None)
    meta_path.write_text(json.dumps(meta))
    return keep, stats

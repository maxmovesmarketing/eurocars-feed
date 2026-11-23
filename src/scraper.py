"""Eurocars (DealerLeasing-thema, foto's van Hexon) uitlezen: alle personenauto's."""

from __future__ import annotations

import asyncio
import html as htmllib
import logging
import re
from dataclasses import dataclass, field

import httpx

from .config import (BASE_URL, GENERIC_TAIL, HTTP_CONCURRENCY, HTTP_DELAY, HTTP_TIMEOUT,
                     LISTING_URL, USER_AGENT)

log = logging.getLogger(__name__)

DETAIL_RE = re.compile(r'href="(https://www\.eurocars\.nl/aanbod/([a-z0-9-]+)-(\d+)/)\?([^"]*)"')
PAGED_RE = re.compile(r"_paged=(\d+)")
ROW_RE = re.compile(r"<div class=row><div class=label>\s*(.*?)</div><div class=value>\s*(.*?)</div></div>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
H1_RE = re.compile(r'supply-title-and-meta__title--title[^>]*>(.*?)</h1>', re.S)
TYPE_RE = re.compile(r'class="?supply-title-and-meta__[a-z-]*type[^>]*>(.*?)<', re.S)
TYPE_FALLBACK_RE = re.compile(r'class="[^"]*type[^"]*"[^>]*>\s*([^<]{2,120}?)\s*<')
PRICE_RE = re.compile(r'class=price-inc>\s*&euro;|class=price-inc>\s*€\s*([\d.]+)')
PRICE_ANY_RE = re.compile(r'class=price-inc>\s*(?:€|&euro;)\s*([\d.]+)')
OPTIONS_RE = re.compile(r"id=opties-accessoires-modal>(.*?)</div></div></div>", re.S)
OPTION_ITEM_RE = re.compile(r"<div class=value>(.*?)</div>", re.S)
TEXT_RE = re.compile(r"<div class=supply-text>(.*?)<div class=grid-", re.S)
STORE_RE = re.compile(r"Vestiging ([A-Za-z' -]+?) - ")


@dataclass
class Vehicle:
    stock_number: str
    url: str
    store: str
    make: str | None = None
    model: str | None = None
    trim: str | None = None
    year: int | None = None
    mileage_km: int | None = None
    price_eur: int | None = None
    fuel: str | None = None
    transmission: str | None = None
    body: str | None = None
    color: str | None = None
    kenteken: str | None = None
    first_registration: str | None = None  # dd-mm-yyyy
    description: str | None = None
    options: list[str] = field(default_factory=list)
    photo_count: int = 0
    photos: list[str] = field(default_factory=list)  # foto 1..n zoals de site ze toont
    images: list[str] = field(default_factory=list)  # gevuld door photos.py


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(TAG_RE.sub(" ", s))).strip()


def _int(s: str | None) -> int | None:
    if not s:
        return None
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None


async def _get(client: httpx.AsyncClient, url: str) -> str:
    for attempt in range(3):
        try:
            r = await client.get(url)
            r.raise_for_status()
            return r.text
        except httpx.HTTPError as e:
            if attempt == 2:
                raise
            log.warning("retry %s: %s", url, e)
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


async def discover(client: httpx.AsyncClient) -> dict[str, tuple[str, str]]:
    """{hexon_id: (detail_url, vestiging)} over alle lijstpagina's."""
    first = await _get(client, LISTING_URL)
    last = max([int(n) for n in PAGED_RE.findall(first)] or [1])
    found: dict[str, tuple[str, str]] = {}

    def collect(page: str) -> None:
        for url, _slug, hid, query in DETAIL_RE.findall(page):
            q = htmllib.unescape(query)
            m = re.search(r"vestiging=([^&]+)", q)
            found.setdefault(hid, (url, m.group(1) if m else ""))

    collect(first)
    for n in range(2, last + 1):
        await asyncio.sleep(HTTP_DELAY)
        collect(await _get(client, f"{LISTING_URL}?_paged={n}"))
    log.info("lijst: %d pagina's, %d auto's", last, len(found))
    return found


def parse_detail(hid: str, url: str, store: str, page: str, brands: list[str]) -> Vehicle:
    page = re.sub(r"\s+", " ", page)  # de site zet regeleindes binnen tags
    v = Vehicle(stock_number=hid, url=url, store=store)
    specs = {}
    for label, value in ROW_RE.findall(page):
        specs.setdefault(_text(label).lower(), _text(value))

    title = _text(H1_RE.search(page).group(1)) if H1_RE.search(page) else specs.get("merk en model", "")
    make = next((b for b in brands if title.lower().startswith(b.lower() + " ") or title.lower() == b.lower()), None)
    if make is None and title:
        make = title.split(" ")[0]
    v.make = make
    v.model = title[len(make):].strip() if make and title.lower().startswith(make.lower()) else None
    m = TYPE_RE.search(page)
    trim = _text(m.group(1)) if m else None
    if not trim:
        for cand in TYPE_FALLBACK_RE.findall(page):
            cand = _text(cand)
            if cand and not cand.lower().startswith(("dit veld", "sorteren")):
                trim = cand
                break
    v.trim = trim or None

    m = PRICE_ANY_RE.search(page)
    v.price_eur = _int(m.group(1)) if m else None
    v.mileage_km = _int(specs.get("kilometerstand"))
    v.year = _int(specs.get("bouwjaar"))
    v.first_registration = specs.get("datumdeel 1")
    v.fuel = specs.get("brandstof")
    v.transmission = specs.get("transmissie")
    v.body = specs.get("carrosserievorm")
    color = specs.get("kleur")
    v.color = color.capitalize() if color else None
    v.kenteken = specs.get("kenteken")

    m = OPTIONS_RE.search(page)
    if m:
        v.options = [o for o in (_text(x) for x in OPTION_ITEM_RE.findall(m.group(1))) if o and o.lower() != "opties & toebehoren"]
    i = page.find("class=supply-text>")
    if i >= 0:
        seg = page[i:i + 12000]
        end = min([j for j in (seg.find("supply-row"), seg.find("<footer")) if j > 0] or [len(seg)])
        txt = _text("<" + seg[:end])
        txt = re.sub(r"^.*?Aanvullende informatie\s*", "", txt)
        cut = txt.find("Team Eurocars")
        v.description = (txt[: cut + len("Team Eurocars")] if cut > 0 else txt).strip() or None

    # Bestandsnamen: <id>-<n>.jpg of <id>-<n>-<versie>.jpg (nieuwere upload); geen -WxH-varianten.
    # De <head> noemt soms een verouderde versie (404); de galerij gebruikt de juiste,
    # dus per foto de URL die het vaakst op de pagina staat.
    counts: dict[int, dict[str, int]] = {}
    for url, n in re.findall(rf"(https://www\.eurocars\.nl/app/uploads/sites/6/hexon/{hid}/{hid}-(\d+)(?:-\d+)?\.jpg)", page):
        c = counts.setdefault(int(n), {})
        c[url] = c.get(url, 0) + 1
    v.photos = [max(c, key=c.get) for _, c in sorted(counts.items())]
    v.photo_count = len(v.photos)
    return v


def gallery(v: Vehicle) -> list[str]:
    """Foto 2..n zonder de vaste pandfoto's (altijd de laatste vier) aan het eind."""
    rest = v.photos[1:]
    return rest[:-GENERIC_TAIL] if len(v.photos) >= GENERIC_TAIL + 2 else rest


async def brand_list(client: httpx.AsyncClient) -> list[str]:
    try:
        data = (await client.get(f"{BASE_URL}/wp-json/store-locator/v1/stores")).json()
        names = {b["name"] for s in data.get("stores", []) for b in s.get("brands", [])}
    except Exception as e:  # noqa: BLE001
        log.warning("merkenlijst niet op te halen: %s", e)
        names = set()
    names |= {"Mercedes-Benz", "Land Rover", "Alfa Romeo", "Lynk & Co", "Aston Martin", "DS"}
    return sorted(names, key=len, reverse=True)


async def scrape_all() -> list[Vehicle]:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "nl-NL,nl;q=0.9"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers, follow_redirects=True) as client:
        brands = await brand_list(client)
        found = await discover(client)
        sem = asyncio.Semaphore(HTTP_CONCURRENCY)
        out: list[Vehicle] = []
        failed = 0

        async def one(hid: str, url: str, store: str) -> None:
            nonlocal failed
            async with sem:
                await asyncio.sleep(HTTP_DELAY)
                try:
                    page = await _get(client, url)
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    log.warning("detail mislukt %s: %s", url, e)
                    return
            v = parse_detail(hid, url, store, page, brands)
            if not v.store:
                m = STORE_RE.search(page)
                v.store = m.group(1) if m else ""
            out.append(v)

        await asyncio.gather(*(one(h, u, s) for h, (u, s) in found.items()))
        log.info("detail: %d gelezen, %d mislukt", len(out), failed)
        out.sort(key=lambda v: v.stock_number)
        return out

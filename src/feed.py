"""Google Vehicle Ads-feed (NL) voor Eurocars, meerdere vestigingen.

Per auto één store_code: die van de vestiging waar hij staat (Google: store_code moet
een locatie zijn uit het gekoppelde Bedrijfsprofiel; link_template bevat {store_code}).
"""

from __future__ import annotations

from datetime import datetime, timezone
from xml.sax.saxutils import escape

from .config import BASE_URL, DEALER_NAME, STORES
from .scraper import Vehicle

NS_G = "http://base.google.com/ns/1.0"

ENGINE_MAP = {
    "plug-in": "plug-in_hybrid",
    "hybride": "hybrid",
    "hybrid": "hybrid",
    "benzine/elektr": "hybrid",
    "elektrisch/benzine": "hybrid",
    "diesel/elektr": "hybrid",
    "elektr": "electric",
    "benzine": "gasoline",
    "diesel": "diesel",
    "lpg": "lpg",
    "cng": "natural_gas",
    "aardgas": "natural_gas",
    "waterstof": "other",
}
BODY_MAP = {
    "sedan": "sedan",
    "limousine": "sedan",
    "hatchback": "hatchback",
    "suv": "suv",
    "terreinwagen": "suv",
    "cross": "crossover",
    "stationwagon": "station_wagon",
    "stationwagen": "station_wagon",
    "station": "station_wagon",
    "combi": "station_wagon",
    "coupe": "coupe",
    "coupé": "coupe",
    "cabrio": "convertible",
    "roadster": "convertible",
    "mpv": "minivan",
    "bus": "full_size_van",
    "pick-up": "truck",
}


def _g(tag, value):
    if value is None or value == "":
        return ""
    return f"<g:{tag}>{escape(str(value))}</g:{tag}>"


def _cdata(tag, value):
    if not value:
        return ""
    return f"<g:{tag}><![CDATA[{value.replace(']]>', ']]]]><![CDATA[>')}]]></g:{tag}>"


def _map(table, value):
    if not value:
        return None
    low = value.lower()
    return next((mapped for key, mapped in table.items() if key in low), None)


def _dfr(raw):
    if raw and len(raw) >= 10 and raw[2] == "-" and raw[5] == "-":
        return f"{raw[6:10]}-{raw[3:5]}"
    return None


PHEV_RE = ("phev", "plug-in", "plug in", "e-hybrid", " gte", "e-tron hybrid", "recharge", " 4xe")


def engine(v: Vehicle) -> str | None:
    """De site noemt een plug-in vaak gewoon "Hybride"; de titel verraadt hem."""
    t = f" {title(v).lower()} "
    if any(k in t for k in PHEV_RE):
        return "plug-in_hybrid"
    return _map(ENGINE_MAP, v.fuel)


def title(v: Vehicle) -> str:
    return " ".join(b for b in (v.make, v.model, v.trim) if b).strip()[:150]


def store_code(v: Vehicle) -> str | None:
    s = STORES.get(v.store)
    return s["store_code"] if s else None


def _item(v: Vehicle) -> str:
    code = store_code(v)
    p = ["<item>", _g("google_product_category", "916"), _g("id", v.stock_number), _cdata("title", title(v)),
         _g("link", v.url), _g("link_template", f"{v.url}?store={{store_code}}"),
         _cdata("image_link", v.images[0])]
    p += [_cdata("additional_image_link", x) for x in v.images[1:10]]
    p += [_g("price", f"{v.price_eur} EUR"), _g("condition", "used"), _g("brand", v.make), _g("model", v.model),
          _cdata("trim", v.trim), _g("year", v.year), _g("color", v.color),
          _g("mileage", f"{v.mileage_km} km" if v.mileage_km is not None else None),
          _g("engine", engine(v)), _g("body_style", _map(BODY_MAP, v.body)),
          _g("date_first_registered", _dfr(v.first_registration)), _g("identifier_exists", "no")]
    if v.options:
        p.append(_cdata("vehicle_option", ", ".join(v.options[:200])))
    p.append(f"<g:vehicle_fulfillment><g:option>in_store</g:option><g:store_code>{escape(code)}</g:store_code></g:vehicle_fulfillment>")
    p += [_g("availability", "in_stock"), _g("custom_label_0", v.store),
          _cdata("description", (v.description or title(v))[:5000]), "</item>"]
    return "\n  ".join(x for x in p if x)


def eligible(v: Vehicle) -> list[str]:
    """Ontbrekende verplichte velden (leeg = mag de feed in)."""
    miss = []
    for name, val in (("prijs", v.price_eur), ("merk", v.make), ("model", v.model), ("bouwjaar", v.year),
                      ("kilometerstand", v.mileage_km), ("kleur", v.color), ("foto", v.images),
                      ("eerste toelating", _dfr(v.first_registration)), ("vestiging", store_code(v))):
        if not val:
            miss.append(name)
    return miss


def build_feed(vehicles: list[Vehicle]) -> bytes:
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y om %H:%M uur")
    items = "\n".join(_item(v) for v in vehicles)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:g="{NS_G}">
<channel>
<title>{escape(DEALER_NAME)}</title>
<link>{BASE_URL}</link>
<description>{escape(DEALER_NAME)} personenauto's. Laatst bijgewerkt op {now}.</description>
{items}
</channel>
</rss>
""".encode("utf-8")

"""Eurocars uitlezen, foto 1 schoonmaken en feed.xml + status.json schrijven."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone

import httpx

from .config import DATA_DIR, FEED_PATH, HTTP_TIMEOUT, STATUS_PATH, USER_AGENT
from .feed import build_feed, eligible
from .photos import prepare
from .scraper import scrape_all

log = logging.getLogger("eurocars-feed")


async def run() -> dict:
    vehicles = await scrape_all()
    if not vehicles:
        raise RuntimeError("geen auto's gevonden")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as c:
        with_photos, pstats = await prepare(c, vehicles)
    feed, skipped = [], {}
    for v in with_photos:
        miss = eligible(v)
        if miss:
            skipped[v.stock_number] = miss
        else:
            feed.append(v)
    # vangnet: niet overschrijven bij een halve run
    if FEED_PATH.exists():
        old = FEED_PATH.read_text().count("<item>")
        if old and len(feed) < old * 0.5:
            raise RuntimeError(f"maar {len(feed)} auto's tegen {old} live: feed niet overschreven")
    xml = build_feed(feed)
    tmp = FEED_PATH.with_suffix(".xml.tmp")
    tmp.write_bytes(xml)
    tmp.replace(FEED_PATH)
    return {
        "ok": True,
        "gevonden": len(vehicles),
        "in_feed": len(feed),
        "per_vestiging": dict(Counter(v.store for v in feed)),
        "fotos": {k: (len(x) if isinstance(x, list) else x) for k, x in pstats.items()},
        "hoofdfoto_uit_galerij": pstats.get("hoofdfoto_uit_galerij", []),
        "zonder_foto": pstats["geen_foto"],
        "foto_mislukt": pstats["mislukt"],
        "niet_in_feed": skipped,
        "feed_bytes": len(xml),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    try:
        status = asyncio.run(run())
    except Exception as e:  # noqa: BLE001
        log.exception("run mislukt")
        status = {"ok": False, "error": str(e)}
    status |= {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat()}
    STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    log.info("klaar: %s", {k: status.get(k) for k in ("ok", "gevonden", "in_feed", "per_vestiging", "error")})
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

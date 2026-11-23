import os
from pathlib import Path

BASE_URL = "https://www.eurocars.nl"
LISTING_URL = f"{BASE_URL}/personenautos/"
UPLOADS = f"{BASE_URL}/app/uploads/sites/6/hexon"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://maxmovesmarketing.github.io/eurocars-feed")

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
FEED_PATH = DATA_DIR / "feed.xml"
STATUS_PATH = DATA_DIR / "status.json"
IMG_DIR = DATA_DIR / "img"
MODEL_PATH = Path(os.getenv("LAMA_MODEL", "/app/model/big-lama.pt"))
ASSETS = Path(__file__).parent / "assets"

DEALER_NAME = "Eurocars"

# Vestiging (zoals de site hem in ?vestiging= noemt) -> store code.
# Afspraak: 4 cijfers van de postcode. Moet gelijk zijn aan de winkelcode van die
# locatie in het Google Bedrijfsprofiel dat aan Merchant Center gekoppeld is.
STORES = {
    "Helmond": {"store_code": "5705", "address": "Varenschut 7, 5705 DK Helmond", "phone": "0887001832"},
    "Asten": {"store_code": "5721", "address": "Kanaalweg 9, 5721 MZ Asten", "phone": "0887001827"},
    "Oss": {"store_code": "5347", "address": "Havenstraat 28, 5347 KK Oss", "phone": "0887001830"},
    "Geldrop": {"store_code": "5663", "address": "Emopad 29, 5663 PA Geldrop", "phone": "0887001828"},
}

HTTP_TIMEOUT = 30.0
HTTP_CONCURRENCY = int(os.getenv("HTTP_CONCURRENCY", "4"))
HTTP_DELAY = float(os.getenv("HTTP_DELAY", "0.4"))
USER_AGENT = "Mozilla/5.0 (compatible; EurocarsVehicleFeedBot/1.0)"
# De laatste foto's van elke auto zijn vaste pandfoto's van Eurocars, geen foto van de auto.
GENERIC_TAIL = int(os.getenv("GENERIC_TAIL", "4"))

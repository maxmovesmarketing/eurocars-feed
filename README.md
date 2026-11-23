# Eurocars Google Vehicle Ads-feed

Personenauto's van eurocars.nl als Google Vehicle Ads-feed, foto 1 zonder de reclame-overlay.

- Feed: https://maxmovesmarketing.github.io/eurocars-feed/feed.xml
- Status van de laatste run: https://maxmovesmarketing.github.io/eurocars-feed/status.json
- Elke nacht via GitHub Actions (`.github/workflows/feed.yml`), handmatig starten kan via "Run workflow".

Per auto de store code van zijn vestiging (4 cijfers postcode): Helmond 5705, Asten 5721, Oss 5347, Geldrop 5663.
Die codes moeten als winkelcode in het gekoppelde Google Bedrijfsprofiel staan.

Foto 1 wordt schoongemaakt (`src/photos.py`): vaste delen van het sjabloon, alles linksboven dat geen muur of auto is
(score, stickers, linten, merklogo's), het ronde actie-embleem, zwarte tekstblokken en actiekleuren rechtsonder;
de auto zelf wordt via rembg beschermd en LaMa vult in. Auto onder een hoes of zonder echte foto: de eerste
galerijfoto wordt de hoofdfoto. De schone foto's en de cache staan op de branch `gh-pages`.

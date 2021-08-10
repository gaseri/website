---
author: Vedran Miletić
---

# Pokretanje PHP web aplikacija u web poslužitelju Apache HTTP Server

- `LoadModule`

!!! admonition "Zadatak"
    - Instalirajte PHP (paket `php`).
    - Provjerite je li se mijenjao sadržaj datoteke `/etc/httpd/conf/httpd.conf`.
    - Provjerite je li se mijenjao sadržaj direktorija `/etc/httpd/conf.d` i `/etc/httpd/modules`.
    - Unutar `DocumentRoot`-a stvorite datoteku `index.php` koja ispisuje `phpinfo()` na ekran.

- `ServerName`
- `ServerAdmin`
- `UseCanonicalName`
- `ServerSignature`

!!! admonition "Zadatak"
    - Promijenite ime poslužitelja u `infuniri-minion<broj>.uniri.hr:80`.
    - Promijenite mail adresu administratora u `infuniri-minion<broj>-admin@uniri.hr`.
    - Uključite korištenje kanonskog imena servera.
    - Provjerite razliku u ispisu između `On`, `Off` i `EMail` vrijednosti kod `ServerSignature` direktive.

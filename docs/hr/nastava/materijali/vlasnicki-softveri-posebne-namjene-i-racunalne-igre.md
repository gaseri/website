---
author: Vedran Miletić, Mladen Miletić
---

# Vlasnički softveri posebne namjene i računalne igre

[Vlasnički softver](https://en.wikipedia.org/wiki/Proprietary_software) je softver kod kojega razvijatelj ili izdavač zadržava prava intelektualnog vlasništva: najčešće se radi o pravu kopiranja, ali može se raditi i o patentnom pravu. Značajan udio softvera posebne namjene (razna područja znanosti i inženjerstva, maloprodaja, upravljanje strojevima u postrojenjima, ...) su vlasnički softveri.

Vlasnički softver se vrlo često distribuira u obliku izvršnog koda, a izvorni kod zadržava razvijatelj ili izdavač. Vlasnički softver se može distribuirati i u obliku izvornog koda, međutim tada se ne smatra softverom otvorenog koda jer licenca ne dozvoljava kopiranje, promjenu i daljnju distribuciju promijenjenih verzija.

## Instalacija vlasničkog softvera

Vlasnički softver za Linux može biti distribuiran kao RPM ili deb paket, izvršna instalacijska datoteka ili kao tarball.

Za primjer vlasničkog softvera uzmimo [DraftSight](https://www.draftsight.com/) tvrtke [Dassault Systèmes](https://www.3ds.com/). DraftSight ima osnovnu i profesionalnu verziju: osnovna verzija se ne naplaćuje, ali zahtijeva registraciju nakon instalacije putem e-maila, dok se profesionalna verzija naplaćuje 99 dolara godišnje.

!!! admonition "Zadatak"
    - Preuzmite DraftSight sa [službenih stranica za preuzimanje osnovne verzije](https://www.draftsight.com/freetrial) i proučite [licencu](https://www.draftsight.com/support/license-agreement) na koju pritom pristajete. Uočite u kojem formatu je pakiran i za koje distribucije Linuxa postoje verzije.
    - Analizirajte popis ovisnosti koje paket koji ste preuzeli ima.
    - Instalirajte paket ili raspakirajte njegov sadržaj te analizirajte izvršne datoteke i biblioteke unutar paketa. Jesu li dinamički ili statički povezane? Objasnite zašto.

## Instalacija računalnih igara putem Steama

Posljednje desetljeće obilježeno je rastom popularnosti raznih vrsta trgovina aplikacija, od kojih je jedna trgovina računalnih igara Steam. Steam omogućuje preuzimanje igara u obliku izvršnog koda zaštićenog [Digital Rights Managementom](https://www.eff.org/issues/drm).

!!! admonition "Zadatak"
    - Preuzmite Steam sa [službenih stranica](https://store.steampowered.com/about/), iz repozitorija distribucije Linuxa ili s [RPM Fusiona](https://rpmfusion.org/).
    - Instalirajte Steam i stvorite korisnički račun.
    - Preuzmite [Dotu 2](https://store.steampowered.com/app/570/Dota_2/) i [Team Fortress 2](https://store.steampowered.com/app/440/Team_Fortress_2/), a zatim unutar direktorija `.steam` pronađite gdje su instalirane te analizirajte izvršne datoteke i biblioteke koje tamo pronađete. Jesu li dinamički ili statički povezane? Ima li instalacijskim direktorijima dodatnih biblioteka koje dupliciraju već postojeće biblioteke prisutne na operacijskom sustavu?

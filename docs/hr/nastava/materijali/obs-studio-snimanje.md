---
author: Vedran Miletić
---

# Snimanje zvuka, videa i dijelova zaslona programom OBS Studio

[OBS Studio](https://obsproject.com/) ([Wikipedia](https://en.wikipedia.org/wiki/OBS_Studio)) je jedan od alata za [snimanje zaslona](https://en.wikipedia.org/wiki/Comparison_of_screencasting_software) i drugih izvora audiovizualnog sadržaja na operacijskim sustavima [Windows](https://obsproject.com/forum/list/windows-support.32/), [macOS](https://obsproject.com/forum/list/mac-support.33/) i [Linux](https://obsproject.com/forum/list/linux-support.34/).

OBS Studio je besplatan softver i otvorenog je koda, a njegov razvoj sponzoriraju brojni mali donatori kao i velikani među kojima su [Logitech](https://obsproject.com/blog/logitech-becomes-open-broadcaster-softwares-first-diamond-sponsor-on-open-collective), [Twitch](https://obsproject.com/blog/twitch-becomes-premiere-sponsor-of-the-obs-project), [NVIDIA](https://obsproject.com/blog/nvidias-diamond-sponsorship-enables-obs-presence-at-twitchcon) i [Facebook](https://obsproject.com/blog/facebook-becomes-a-premiere-sponsor). Iako već godinama suvereno vlada scenom producenata sadržaja na platformama YouTube i Twitch, novi skok u popularnosti dobiva upravo početkom 2020. godine zahvaljujući [povećanoj potrebi za snimanjem i emitranjem audiovizualnog sadržaja slijedom epidemije COVID-19](https://obsproject.com/blog/five-simple-tips-for-new-streamers).

Kako OBS Studio snima sadržaj ekrana, on ne pravi razliku koji alat za prikaz prezentacija se koristi pa to može biti Microsoft Office 2016, 2019 ili [365](https://www.office.com/), [LibreOffice](https://www.libreoffice.org/), [Mozilla Firefox](https://www.mozilla.org/firefox/) koji prikazuje prezentaciju u HTML-u i CSS-u, [Microsoft Edge](https://www.microsoft.com/edge) koji prikazuje prezentaciju u PDF-u ili bilo koji drugi alat koji koristi bilo koji drugi format.

## Preuzimanje i instalacija OBS Studija

OBS Studio moguće je [preuzeti sa službenih stranica](https://obsproject.com/download) za operacijske sustave Windows, macOS i Linux. Specijalno:

- [kod instalacije na operacijskom sustavu Windows](https://obsproject.com/wiki/install-instructions#windows), pokreće se instalacijski čarobnjak u kojem se mogu prihvatiti sve zadane postavke
- [kod instalacija na operacijskom sustavu macOS](https://obsproject.com/wiki/install-instructions#macos), otvara se instalacijska datoteka u obliku [Apple Disk Image](https://en.wikipedia.org/wiki/Apple_Disk_Image) pa potom "odvlači" `OBS` u `Aplikacije`
- [kod instalacije na operacijskom sustavu Linux](https://obsproject.com/wiki/install-instructions#linux), može se koristiti pakete koje nudi distributer (npr. [Arch Linux](https://archlinux.org/packages/community/x86_64/obs-studio/), [Debian](https://tracker.debian.org/pkg/obs-studio) i [RPMFusion za Fedoru i RHEL](https://admin.rpmfusion.org/pkgdb/package/free/obs-studio/)) ili izgraditi program iz izvornog koda

Na svim operacijskim sustavima je preporučljivo koristiti [posljednju stabilnu verziju](https://github.com/obsproject/obs-studio/releases/latest) i redovito nadograđivati na nove verzije kada iste postanu dostupne.

## Konfiguracija OBS Studija

Nakon pokretanja OBS Studija nudi se `Čarobnjak za automatsku konfiguraciju` na kojem ćemo odabrati da ga želimo pokrenuti, odnosno `Da`. Kako zasad nemamo namjeru uživo emitirati na [YouTubeu](https://www.youtube.com/) ili [Twitchu](https://www.twitch.tv/), u idućem ćemo koraku odabrati `Optimiziraj samo za snimanje, neću prenositi uživo`.

Sada možemo odabrati rezoluciju i OBS Studio će ponuditi rezoluciju našeg zaslona. Ako nemamo naročito snažno računalo, to može biti previše; tada možemo odabrati i nešto manju rezoluciju, ali ne bi baš trebalo ići bitno ispod 1280x720. Što se tiče postavke `FPS`, 30 okvira po sekundi je sasvim dovoljno.

Nakon ovog koraka OBS Studio će testirati postavke da vidi je li moguće u realnom vremenu vršiti kodiranje video sadržaja i ponudit će manju rezoluciju ako nije uspio dobiti zadovoljavajuće performanse s odabranom.

U osnovnom prozoru OBS Studija vidimo:

- `Scene` koje nam omogućuju da brzo prelazimo između različitih izvora audiovizualnog sadržaja; koristit ćemo samo jednu scenu pa nam ovo neće trebati
- `Izvori` koji nam omogućuju postavljanje što će se snimati
- `Audio Mixer` koji prikazuje trenutne razine ulaznog (i izlaznog) zvuka na računalu
- `Prijelazi scena` koji omogućuju konfiguraciju glatkih pretvorbi slike iz scene u scenu
- `Kontrole` od kojih ćemo koristiti `Počni snimanje`, `Postavke` i `Izlaz`; `Počni streamanje` služi za emitiranje sadržaja uživo, a `Studijski način rada` omogućuje bolju kontrolu nad prijelazima scena

## Dodavanje izvora na scenu

Klikom na gumb `+` pod `Izvori` nudi se niz izvora audiovizualnog sadržaja koje možemo dodati na scenu. Odabirom opcije `Ulaz sa prozora` dobivamo mogućnost da stvorimo novi takav ulaz. (**Napomena:** na različitim operacijskim sustavima postoje izvori za snimanje prozora koji se različito zovu.)

Preporučuje se uključivanje opcije `Snimaj kursor`. Alternativno, na novijim verzijama Windowsa 10 moguće je i odabrati `Windows Graphics Capture` pod `Capture Method` koji uvijek snima kursor i uglavnom daje bolju kvalitetu snimke.

Moguće je snimati alat za prezentaciju koji vrši prikaz u istom prozoru ili u posebnom "prozoru" veličine čitavog ekrana. U oba slučaja potrebno je odabrati odgovarajući prozor u popisu ponuđenih.

Može se dogoditi da je prozor nakon dodavanja veći od platna naše scene. U tom slučaju je potrebno klikom na crvene kvadratiće i povlačenjem smanjiti njegovu veličinu na odgovarajući za platno.

Postoji mogućnost snimanja sadržaja više prozora i uz proizvoljno slaganje na sceni.

Kako bismo snimali zvuk, klikom na gumb `+` pod `Izvori` dodajemo `Zvučni ulaz` ili `Ulaz zvuka`. Odabiremo `Dodaj postojeći` i biramo `Mikrofon/ulaz` ili što već želimo snimati.

## Snimanje sadržaja

Snimanje započinjemo klikom na gumb `Počni snimanje`.

Nakon pokretanja snimanja sekunde se odbrojavaju u `REC:` u statusnoj traci OBS Studija, gdje se prikazuje i zauzeće procesora i broj okvira po sekundi. Snimanje možemo pauzirati klikom na gumb pauze, a zaustaviti klikom na gumb `Zaustavi snimanje`.

## Dodatno podešavanje

Klikom na gumb `Postavke` dobivamo mogućnost konfiguracije mnogih dijelova OBS Studija. Ovdje ćemo samo navesti da u dijelu `Izlaz` možemo pronaći postavku `Putanja za snimanje` gdje će se nalaziti naše datoteke imenovane u obliku datuma i sata, npr. `2020-04-02 14-16-18.mkv`.

Vrijedi spomenuti i postavku `Enkoder` koja omogućuje korištenje hardverskih video enkodera prisutnih u grafičkim procesorima kompanija NVIDIA, AMD i Intel, čime se rasterećuje osnovni procesor. U slučaju da su odgovarajući grafički procesori prepoznati, OBS Studio će u popisu mogućnosti pored softverskih enkodera ponuditi i korištenje hardverskog enkodera, ovisno o grafičkom procesoru:

- NVIDIA -> [NVENC](https://developer.nvidia.com/nvidia-video-codec-sdk) ([dodatne upute](https://www.nvidia.com/en-us/geforce/guides/broadcasting-guide/))
- AMD -> [Advanced Media Framework (AMF)](https://gpuopen.com/advanced-media-framework/)
- Intel -> [Quick Sync Video](https://www.intel.com/content/www/us/en/architecture-and-technology/quick-sync-video/quick-sync-video-general.html)

## Obrada snimljenog sadržaja

Za rezanje, spajanje i uređivanje snimljenog audiovizualnog sadržaja može se koristiti [integrirani video editor u Windows Photos](https://www.microsoft.com/windows/photo-movie-editor), [iMovie](https://www.apple.com/imovie/), [OpenShot](https://www.openshot.org/), [Shotcut](https://shotcut.org/), [Adobe Premiere](https://www.adobe.com/products/premiere.html), [DaVinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve/) ili neki drugi od [uređivača video sadržaja](https://en.wikipedia.org/wiki/List_of_video_editing_software).

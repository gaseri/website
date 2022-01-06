---
author: Vedran Miletić
---

# Konfiguracija čvorova u alatu CORE

Unutar alata CORE Quagga je pokrenuta na svim čvorovima koji su tipa `router`.

Postavljanjem mrežnih elemenata na podlogu oni su automatski konfigurirani i to je dovoljno za pokretanje simulacija, ali ako želimo simulirati neki realni slučaj onda moramo ručno konfigurirati elemente. Podsjetimo se da u CORE-u konfiguriramo element tako da ga prvo označimo te kliknemo na njega desnim klikom pri čemu nam se otvori padajući izbornik; odaberemo Configure ili, alternativno dva puta kliknemo na element lijevim klikom (prije pokretanja emulacije). Nakon pokretanja emulacije moguće je konfigurirati mrežna sučelja korištenjem alata `ifconfig`.

CORE ima četiri vrste konfiguracijskih prozora (ovisno o vrsti mrežnog elementa):

- Konfiguracijski prozor koncentratora i LAN komutatora (promjena naziva elementa),
- Konfiguracijski prozor usmjerivača (promjena naziva, konfiguracija izlazno/ulaznog sučelja za svaku podmrežu na koju je usmjerivač spojen),
- Konfiguracijski prozor PC računala i domaćina (dodavanje vlastitog imena PC-u ili Hostu, dodjela željene IPv4 i IPv6 adrese),
- Konfiguracijski prozor linka (koji smo već koristili u prethodnoj vježbi).

U ovom dijelu bavit ćemo se primarno konfiguracijskim prozorom usmjerivača.

## Konfiguracija daemona alata Quagga unutar alata CORE

!!! note
    [Daemon](https://en.wikipedia.org/wiki/Daemon_(classical_mythology)) (latinska riječ, dolazi od grčke riječi *daimon*) je antički grčki i kasnije rimski pojam koji označava duha voditelja i [nema isto značenje kao pojam demon](https://owencyclops.com/wp-content/uploads/2019/05/web02signed.jpg). Kako su dvije riječi u engleskom jeziku vrlo slične, pojedini projekti slobodnog softvera otvorenog koda koji koriste daemone (na operacijskim sustavima sličnim Unixu daemon je pozadinski proces koji se koristi za implementaciju pojedinih usluga sustava) iskoristili su priliku za šalu s neupućenim korisnicima. Tako i [operacijski sustav FreeBSD](https://www.freebsd.org/) ima svoje daemone koji nude pojedine usluge sustava pa je [odabrao demona kao svoju maskotu](https://www.freebsd.org/art/). Taj odabir je barem jednom kod barem jednog korisnika [izazvao zabrinutost na temu moralne ispravnosti korištenja softvera koji se poistovjećuje s demonom](https://lists.debian.org/debian-project/2017/06/msg00004.html).

[Daemon](https://en.wikipedia.org/wiki/Daemon_(computing)) (engl. *daemon*) je računalni program koji se izvršava kao proces u pozadini bez direktne interaktivne korisničke kontrole. Daemone alata Quagga unutar alata CORE konfiguriramo otvarajem konfiguracijskog prozora usmjerivača, unutar kojeg kliknemo na gumb `Services`. U prvom od ponuđenih stupaca nalazi se popis ponuđenih Quagga daemona; ostali stupci nam za sada nisu bitni. Željeni daemon možemo dovesti u funkciju klikom na njegovo ime, a prozoru za konfiguraciju tog daemona pristupamo klikom na ikonu ključa desno od imena daemona.

Prozor za konfiguraciju svakog od daemona sadrži tri dijela: `Files`, `Directories` i `Startup/shutdown`.

- Pod odjeljkom `Files` možemo odabrati, urediti ili stvoriti potrebne konfiguracijske datoteke pojedinih daemona.
- U `Directories` se nalazi popis direktorija koji su nužni za rad određenog daemon procesa.
- U odjeljku `Startup/shutdown` možemo odrediti vrijeme početka izvršavanja te dodati ili izmjeniti naredbe za pokretanje, prekidanje ili provjeru stanja daemona.

Ukoliko je ikonica za konfiguraciju žute boje, daemon nije ispravno konfiguriran te ga je potrebno dodatno konfigurarati.

## Konfiguracija Quagga daemona pomoću konfiguracijskih datoteka

Unutar alata CORE konfiguracija **svih** daemona alata Quagga dana je u konfiguracijskoj datoteci daemona `zebra`, što je i prikazano kao napomena prilikom otvaranja konfiguracije ostalih daemona (npr. `OSPFv2` ili `RIP`). Mi ćemo ovdje prihvatiti tu konvenciju, međutim svakako treba napomenuti da kada se Quagga koristi van alata CORE najčešće se radi urednosti i lakšeg održavanja konfiguracija svakog pojedinog daemona odvaja u posebnu datoteku.

## Konfiguracija Quagga daemona pomoću ljuske vtysh

U alatu CORE daemone alata Quagga možemo konfigurirati i preko sučelja za konfiguraciju vtysh. Sučelju vtysh pristupamo desnim klikom na usmjerivač, zatim odaberemo Shell window izbornik u kojem se nalazi vtysh. Dobivamo pristup prozoru sa vtysh ljuskom s kojom komuniciramo unošenjem određenih naredbi, poput bash ljuske u Linuxu.

!!! caution
    Ukoliko vam vtysh javi poruku `WARNING: terminal is not fully functional`, pritisnite tipku `q` i zatim tipku `Enter`. Terminal je sasvim dovoljno funkcionalan za naše potrebe.

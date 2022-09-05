---
author: Tina Maršić, Vedran Miletić
---

# Mandantna kontrola pristupa korištenjem sustava SELinux

!!! todo
    Dodaj negdje da je SELinux zaštitio od [VENOM-a](https://venom.crowdstrike.com/) i nađi gdje [Paul Cormier to tvrdi](https://youtu.be/tekg8OjrfDM).

!!! hint
    Za više informacija proučite [SELinux User's and Administrator's Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/selinux_users_and_administrators_guide/index).

- Diskrecijska kontrole pristupa -- ovlasti definirane na razini korisnika

    - implementirana pod Linuxom pomoću dozvola i listi kontrole pristupa datotekama
    - svaka aplikacija koju korisnik pokrene ima sve korisnikove ovlasti, može pristupiti svim datotekama kojima korisnik može pristupiti
    - previše ovlasti je dano aplikacijama, što je evidentno problem kod zlonamjernih aplikacija

- Mandantna kontrola pristupa -- ovlasti definirane na razini korisnika **i procesa**

    - implementirana pomoću pod Linuxom dodavanjem sigurnosnog konteksta datotekama
    - operacijski sustav ograničava korisnicima **i aplikacijama** pristup pojedinim datotekama
    - ako se dobro definiraju ograničenja pojedine aplikacije, zlonamjerne aplikacije ne mogu napraviti puno štete

- [Security-Enhanced Linux](https://en.wikipedia.org/wiki/Security-Enhanced_Linux) (SELinux) je sigurnosni modul Linux jezgre

    - implementira mandantnu kontrolu pristupa u skladu sa zahtjevima Nacionalne agencije za sigurnost (engl. *National Security Agency*, NSA) SAD-a
    - nastoji odvojiti primjenu sigurnosnih pravila od same sigurnosne politike
    - smanjuje količinu koda u pojedinom softveru koja se odnosi na sigurnost
    - dostupan s komercijalnom podrškom u sklopu Red Hat Enterprise Linux (RHEL) verzije 4 i svim kasnijim verzijama
    - dostupan bez komercijalne podrške u Fedori Core 2 i svim kasnijim verzijama

- aplikacije koje koriste SELinux ako je dostupan: `ls`, `ps`, `find`, `login`, `sshd`, `gdm`, `cron`, `chcon`, `setfiles`, `restorecon`, `setenforce`, `getenforce`, `getsebool`, `setsebool`, ...

- SELinux nije ultimativno awesome rješenje za sve sigurnosne probleme

    - nije zamjena za lozinke, SSH ključeve
    - nije zamjena serverske certifikate
    - nije zamjena za vatrozid
    - ...

## Rad sa sigurnosnim kontekstima

- skup atributa koji opisuju razinu ovlasti korisnika (dozvole pristupa datotekama i procesima)
- svakom korisniku i procesu dodjeljen je kontekst zapisan kao znakovni niz oblika `user:role:type:level`, primjerice `system_u:object_r:net_conf_t:s0`

    - SELinux korisnik (`user`) je identitet koji može preuzeti određeni broj uloga; svaki korisnik na operacijskom sustavu pridružen je nekom od SELinux korisnika
    - uloga (`role`) određuje prava korisnika, odnosno procesa, na pristup pojedinim domenama
    - domena (`type`) je atribut koji se dodjeljuje procesima

        - definira popis akcija koje proces smije izvoditi nad određenim tipovima procesa i datoteka s kojima interagira
        - proces se zatim pokreće unutar određene domene koja definira ovlasti tog procesa

    - tip (`type`) je atribut koji se dodjeljuje datotekama

        - definira popis akcija koje procesi mogu izvoditi nad datotekom

    - nivo (`level`) nam nije značajan

- naredbe `ls`, `ps` i `id` primaju parametar `-Z` kojim prikazuju SELinux kontekst
- naredba `stat` prikazuje SELinux kontekst bez dodatnih parametara

!!! admonition "Zadatak"
    - Odredite korisnika, ulogu i tip za datoteke `/etc/hosts`, `/etc/profile` i `/etc/passwd`.
    - Pronađite u popisu procesa dva procesa u domeni `unconfined_t` te po jedan proces u domenama `init_t` i `kernel_t`.
    - Pronađite u popisu procesa jedan proces u domeni koja dosad nije spomenuta i pronađite u datotečnom sustavu barem jednu datoteku koja je pripadnog tipa.

- naredbe `mv` i `cp` zadržavaju sigurnosni kontekst datoteke kod micanja, odnosno kopiranja

!!! admonition "Zadatak"
    Provjerite na koji način `mv` i `cp` zadržavaju sigurnosni kontekst datoteka.

- naredba `chcon` služi za promjenu konteksta

!!! admonition "Zadatak"
    Stvorite datoteku u direktoriju korisnika koji nije `root`. Pronađite njezin kontekst, a zatim ga promijenite u `system_u:object_r:root_t:s0`.

## Načini rada SELinux-a

- SELinux može raditi na tri načina

    - disabled -- stvaranje oznaka sigurnosnog konteksta i primjena sigurnosnih pravila na datotekama i procesima su isključeni
    - permissive -- stvaranje oznaka sigurnosnog konteksta je uključeno, primjena sigurnosnih pravila se nadgleda, ali svaki pristup je dopušten

        - pristupi koji ne bi bili dozvoljeni da se pravila primjenjuju prijavljuju se korisnicima u obliku Access Vector Cache (AVC) poruka

    - enforcing -- stvaranje oznaka sigurnosnog konteksta je uključeno, sigurnosna pravila se primjenjuju

        - pristupi koji nisu dozvoljeni prijavljuju se korisnicima u obliku AVC poruka

- naredbe `sestatus` prikazuje status SELinuxa na sustavu, s parametrom `-v` prikazuje više informacija
- naredbe `getenforce` i `setenforce` dohvaćaju i mijenjaju primjenu sigurnosnih pravila
- datoteka `/etc/selinux/config` služi za konfiguraciju SELinuxa

!!! admonition "Zadatak"
    - Provjerite status SELinuxa na sustavu. Specifično, pronađite je li uključen, koju sigurnosnu politiku koristi, primjenjuje li sigurnosna pravila, koji je kontekst trenutnog procesa i njegovog upravljačkog terminala.
    - Isključite primjenu sigurnosnih pravila, i provjerite da je to uspjelo.
    - Isključite SELinux kompletno, te ponovno pokrenite sustav. Stvorite datoteku proizvoljnog sadržaja i odredite njezin kontekst.

!!! todo
    Ovdje nedostaje neki zadatak koji će eksplicitno prikazati što SELinux blokira.

## Access Vector Cache poruke

- naredba `avcstat` prikazuje statistike za SELinux međuspremnik vektora pristupa (engl. *Access Vector Cache*, AVC), redom

    - lookups -- broj pregleda međuspremnika, prilično dobar indikator opterećenja
    - hits -- broj pogodaka međuspremnika
    - misses -- broj promašaja međuspremnika
    - allocations -- broj alokacija međuspremnika
    - reclaims -- broj povrata međuspremnika
    - frees -- broj oslobađanja međuspremnika

- datoteka `/var/log/audit/audit.log` prikazuje AVC poruke
- naredba `ausearch` omogućuje pretraživanje AVC poruka

    - parametar `-sv` omogućuje pretragu obzirom na uspješnost pristupa

- naredba `audit2allow` omogućuje dozvoljavanje onog što je bilo zabranjeno na temelju pripadne AVC poruke

!!! admonition "Zadatak"
    - Pronađite broj pregleda i pogodaka AVC međuspremnika.
    - Pronađite sve neuspješne pristupe koji se odnose na korisnika `root`.
    - Dozvolite jedan od neuspješnih pristupa. (**Napomena:** ovo radimo samo za vježbu rada s alatom i općenito nije dobra sigurnosna praksa.)

## Boolean vrijednosti značajki

- modifikacije sigurnosne politike moguće je izvesti mijenjanjem boolean vrijednosti značajki
- naredbom `semanage boolean -l` dohvaća se popis svih booleana
- naredbe `getsebool` i `setsebool` dohvaćaju i mijenjaju boolean vrijednosti pojedine značajke

!!! admonition "Zadatak"
    - Uvjerite se da u zadanim postavkama korisnik koji nije `root` može koristiti `ping`.
    - Pronađite značajku kojom korisniku možete onemogućiti korištenje `ping`-a, i iskoristite je da korisniku onemogućite korištenje pinga. Uvjerite se da je onemogućavanje bilo uspješno.
    - Pronađite AVC poruku koja potvrđuje da je korisniku bilo onemogućeno korištenje `ping`-a.

## Sigurnosni kontekst korisnika

- naredbom `semanage login -a -s SELINUX_USER LOGIN_USER` moguće je dodati pravilo po kojemu se korisnik `LOGIN_USER` pridružuje SELinux korisniku `SELINUX_USER`
- naredbom `semanage login -l` vrši se provjera pridruživanja korisnika SELinux korisnicima

!!! admonition "Zadatak"
    Stvorite novog korisnika `gost` i pridružite ga SELinux korisniku `xguest_u`. Prijavite se kao korisnik `gost` i pokušajte pokrenuti naredbe `su` i `sudo`.

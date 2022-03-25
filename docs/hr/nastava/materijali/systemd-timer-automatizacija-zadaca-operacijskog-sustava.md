---
author: Luka Vretenar, Vedran Miletić
---

# Automatizacija zadaća operacijskog sustava

!!! hint
    Za dodatne primjere naredbi proučite [stranicu system/Timers na ArchWikiju](https://wiki.archlinux.org/title/Systemd/Timers) i [Chapter 24. Automating System Tasks](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/ch-automating_system_tasks) u [Red Hat Enterprise Linux 7 System Administrator's Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/index).

## Systemd timer

- za automatizaciju obavljanja pojedinih zadaća u skladu sa određenim vremenskim rasporedom možemo koristeći systemd jedinke vrste mjerač vremena (engl. *timer*)
- sve postojeće mjerače vremena na sustavu i njihova stanja možemo ispisati naredbom `systemctl list-timers`
- svaki mjerač vremena se sastoji od dvije datoteke koje se nalaze u direktoriju `/ect/systemd/system`:
    - datoteka `USLUGA.timer`, definira vremensku ovisnost ili vrijeme pokretanja naše usluge
    - datoteka `USLUGA.service`, definira samu uslugu ili naredbe za pokrenuti
- za definirani mjerač vremena `USLUGA.timer` mora posojati i usluga istog imena `USLUGA.service`, u suprotnom vidjeti savjet na dnu stranice

!!! hint
    Kroz sve primjere koristimo `USLUGA` kao naziv našeg para mjerača vremena i systemd usluge. Za vašu primjenu taj naziv će biti drugačiji.

- omogućavanje i pokretanje mjerača vremena vršimo uz pomoć `systemctl` alata kao i za sve ostale vrste jedinki:
    - `systemctl enable USLUGA.timer`
    - `systemctl disable USLUGA.timer`
    - `systemctl start USLUGA.timer`
    - `systemctl stop USLUGA.timer`

!!! hint
    Predefinirani sistemski mjerač vremenai se nalaze u direktoriju `/usr/lib/systemd/system`. Njih nećemo mijenjati, već ćemo u slučaju potrebe napraviti kopije u direktoriju `/etc/systemd/system` i svoje promjene izvoditi na kopijama.

!!! admonition "Zadatak"
    Ispišite sve aktivne mjerače vremena na sustavu i pronađite njihove definicijske datoteke.

## Struktura systemd.timer datoteka

- datoteka `USLUGA.timer` se mora nalaziti u direktoriju `/etc/systemd/system` i biti sljedećeg oblika:

    ``` ini
    [Unit]
    Description=Moja vremenski ovisna usluga

    [Timer]
    # ...

    [Install]
    WantedBy=timers.target
    ```

- mjerači vremena mogu biti dvije vrste:
    - relativno na neki događaj
    - definirani za specifično vrijeme
- ovisno od koje je vrste definirani mjerač vremena, pod sekcijom `[Timer]` definiramo drugačije stavke, u našem primjeru to je na mjestu `# ...`
- svaka stavka se definira na način da ide ime stavke, znak za jednako, te vrijednost koju dodijeljujemo toj stavci
- ne postoji razmak oko znaka jednakosti
- stavke za mjerač vremena definiran relativno na neki događaj, moguće ih je kombinirati više:
    - `OnActiveSec` -- relativno na vrijeme pokretanja samog mjerača vremena
    - `OnBootSec` -- relativno na vrijeme pokretanja operacijskog sustava
    - `OnStartupSec` -- relativno na pokretanje procesa `systemd`
    - `OnUnitActiveSec` -- relativno na vrijeme zadnjeg pokretanja jedinke koju pokreće ovaj mjerač vremena
    - `OnUnitInactiveSec` -- relativno na vrijeme zadnjeg kraja rada jedinke koju pokreće ovaj mjerač vremena
- relativna vremena definiramo na način da kombiniramo vremenske jedinice, primjer:
    - `1h 20m`
    - `5 s`
- stavke za mjerač vremena definiran za specifično vrijeme:
    - `OnCalendar`
- specifična vremena možemo definirati generički ili specifično vrijednostima:
    - `hourly`, `daily`, `weekly`, `monthly`,
    - `2003-03-05 05:40`

!!! hint
    Za više detalja o načinima specificiranja vremena pogledajte u man stranici `systemd.timer(5)` (naredba: `man 5 systemd.timer`).

!!! admonition "Zadatak"
    Iz datoteka mjerača vremena koje ste pronašli u prethodnom zadatku pročitajte i odredite koje usluge pokreću, koje su vrste ti mjerači vremena i u kojim vremenima pokreću te datoteke.

- primjer:

    ``` ini
    [Unit]
    Description=Primjer mjerača vremena

    [Timer]
    OnCalendar=hourly
    Unit=NEKADRUGAUSLUGA.service

    [Install]
    WantedBy=timers.target

    ```

## Struktura pripadnih systemd.service datoteka

- datoteka `USLUGA.service` se mora nalaziti u `/etc/systemd/system`
- struktura datoteke je opisana u prethodnom poglavlju
- razlika između datoteke `USLUGA.service` za korištenje mjerača vremena od tipične systemd datoteke usluge je u:
    - datoteka `USLUGA.service` za mjerač vremena ne sadrži sekciju `[Install]`
    - uslugu za mjerač vremena nije potrebno omogućiti niti pokretati sa `systemctl`, omogućujemo i pokrećemo jedinku `USLUGA.timer` umjesto toga

- primjer:

    ``` ini
    [Unit]
    # ...

    [Service]
    # ...
    ```

!!! hint
    U slučaju da `USLUGA.timer` i `DRUGAUSLUGA.service` datoteke nemaju isti naziv jedinke, potrebno je u datoteci `USLUGA.timer` pod sekcijom `[Timer]` eksplicitno navesti na koju jedinku se odnosi dodavanjem stavke `Unit=DRUGAUSLUGA.service`.

!!! admonition "Zadatak"
    Izradite vlastiti mjerač vremena koji se sastoji od systemd usluge i mjerača vremena, čija je svrha svakih 2 minute pokrenuti naredbu `date >> /root/datoteka`.

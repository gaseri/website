---
author: Luka Vretenar, Vedran Miletić
---

# Automatizacija zadaća operacijskog sustava

!!! hint
    Za više informacija proučite [Chapter 19. Automating System Tasks](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/ch-automating_system_tasks) u [Red Hat Enterprise Linux 7 System Administrator's Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/index).

## Systemd timer

- za automatizaciju obavljanja pojedinih zadaća u skladu sa određenim vremenskim rasporedom možemo koristeći `systemd` jedinke vrste vremenski brojač -- `timer`
- sve postojeće vremenske brojače na sustavu i njihova stanja možemo ispisati naredbom `systemctl list-timers`
- svaki vremenski brojač se sastoji od dvije datoteke koje se nalaze u direktoriju `/ect/systemd/system`:

    - datoteka `USLUGA.timer`, definira vremensku ovisnost ili vrijeme pokretanja naše usluge
    - datoteka `USLUGA.service`, definira samu uslugu ili naredbe za pokrenuti

- za definirani vremenski brojač `USLUGA.timer` mora posojati i usluga istog imena `USLUGA.service`, u suprotnom vidjeti savjet na dnu stranice

!!! hint
    Kroz sve primjere koristimo `USLUGA` kao naziv našeg para vremensog brojača i `systemd` usluge. Za vašu primjenu taj naziv će biti drugačiji.

- omogućavanje i pokretanje vremenskog brojača vršimo uz pomoć `systemctl` alata kao i za sve ostale vrste jedinki:

    - `systemctl enable USLUGA.timer`
    - `systemctl disable USLUGA.timer`
    - `systemctl start USLUGA.timer`
    - `systemctl stop USLUGA.timer`

!!! hint
    Predefinirani sistemski vremenski brojači se nalaze u direktoriju `/usr/lib/systemd`

!!! admonition "Zadatak"
    Ispišite sve aktivne vremenske brojače na sustavu i pronađite njihove definicijske datoteke.

## Struktura `*.timer` datoteka

- datoteka `USLUGA.timer` se mora nalaziti u `/etc/systemd/system` i biti sljedećeg oblika:

    ``` ini
    [Unit]
    Description=Moja vremenski ovisna usluga

    [Timer]
    ...

    [Install]
    WantedBy=timers.target
    ```

- vremenski brojači mogu biti dvije vrste:

    - relativno na neki događaj
    - definirani za specifično vrijeme

- ovisno od koje je vrste definirani vremenski brojač, pod sekcijom `[Timer]` definiramo drugačije stavke, u našem primjeru to je umjesto `...`
- svaka stavka se definira na način da ide ime stavke, znak za jednako, te vrijednost koju dodijeljujemo toj stavci
- ne postoji razmak oko znaka jednako
- stavke za vremenski brojač definiran relativno na neki događaj, moguće ih je kombinirati više:

    - `OnActiveSec` -- relativno na vrijeme pokretanja samog vremenskog brojača
    - `OnBootSec` -- relativno na vrijeme pokretanja samog linxu sustava
    - `OnStartupSec` -- relativno na pokretanje `systemd` procesas
    - `OnUnitActiveSec` -- relativno na vrijeme zadnjeg pokretanja jedinke koju pokreće ovaj vremenski brojač
    - `OnUnitInactiveSec` -- relativno na vrijeme zadnjeg kraja rada jedinke koju pokreće ovaj vremenski brojač

- relativna vremena definiramo na način da kombiniramo vremenske jedinice, primjer:

    - `1h 20m`
    - `5 s`

- stavke za vremenski brojač definiran za specifično vrijeme:

    - `OnCalendar`

- specifična vremena možemo definirati generički ili specifično vrijednostima:

    - `hourly`, `daily`, `weekly`, `monthly`,
    - `2003-03-05 05:40`

!!! hint
    Za više detalja o načinima specificiranja vremena pogledajte u dokumentaciji za `systemd`:  `man 5 systemd.timer`.

!!! admonition "Zadatak"
    Iz datoteka vremenskih brojača koje ste pronašli u prethodnom zadatku pročitajte i odredite koje usluge pokreću, koje su vrste ti brojači vremena i u kojim vremenima pokreću te datoteke.

- primjer:

    ``` ini
    [Unit]
    Description=Primjer vremenskog brojača

    [Timer]
    OnCalendar=hourly
    Unit=NEKADRUGAUSLUGA.service

    [Install]
    WantedBy=timers.target

    ```

## Struktura `*.service` datoteka

- datoteka `USLUGA.service` se mora nalaziti u `/etc/systemd/system`
- struktura datoteke je opisana u prethodnom poglavlju
- razlika između datoteke `USLUGA.service` za korištenje vremenskog brojača od tipične `systemd` datoteke usluge je u:

    - datoteka `USLUGA.service` za vremenski brojač ne sadrži sekciju `[Install]`
    - uslugu za vremenski brojač nije potrebno omogućiti niti pokretati sa `systemctl`, omogućujemo i pokrećemo jedinku `USLUGA.timer` umjesto toga

- primjer:

    ``` ini
    [Unit]
    ...

    [Service]
    ...
    ```

!!! hint
    U slučaju da `USLUGA.timer` i `DRUGAUSLUGA.service` datoteke nemaju isti naziv jedinke, potrebno je u datoteci `USLUGA.timer` pod sekcijom `[Timer]` dodati stavku `Unit=DRUGAUSLUGA.service`.

!!! admonition "Zadatak"
    Izradite vlastiti vremenski brojač koji se sastoji od `systemd` usluge i vremenskog brojača, čija je svrha svakih 2 minute pokrenuti slijedeću naredbu `date >> /root/datoteka`.

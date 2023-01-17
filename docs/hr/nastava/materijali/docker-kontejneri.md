---
author: Luka Brumnić, Vedran Miletić, Matea Turalija
---

# Kontejnerizacija alatom Docker

Razvoj aplikacija danas zahtijeva mnogo više od samog pisanja koda. Višestruki jezici, okviri, arhitekture i diskontinuirana sučelja između alata stvaraju ogromnu složenost. Docker pojednostavljuje i ubrzava tijek rada i daje razvojnim programerima slobodu za inovacije svojim izborom alata, aplikacija i okruženja za implementaciju za svaki projekt.

Docker je otvorena platforma za razvoj, implementaciju i pokretanje aplikacija. Pruža mogućnost pakiranja i pokretanja aplikacije u izoliranom okruženju koje se naziva spremnik ili kontejner (engl. *container*). Kontejneri sadrže sve što je potrebno za pokretanje aplikacije, tako da se ne morate oslanjati na ono što je trenutno instalirano na nekom računalu. Na taj način se mogu jednostavno dijeliti i možemo biti sigurni da će svi s kojima ih podijelimo dobiti isti kontejner koji radi na isti način. Ukratko, Docker omogućuje da se aplikacija izolira od svog okruženja, što omogućuje brzu implementaciju softvera i rješava problem *ali radi na mom računalu* koji se nekad javlja kod postavljanja aplikacije.

!!! hint
    Za više informacija proučite [službenu dokumentaciju](https://docs.docker.com/).

Formalno, možemo reći da je Docker skup alata za kontejnerizaciju (poznatu još i pod nazivom kontejnerska virtualizacija ili [virtualizacija na razini operacijskog sustava](https://en.wikipedia.org/wiki/Operating-system-level_virtualization), engl. *operating-system-level virtualization*) i dijeljenje aplikacija putem interneta (gdje mu je princip rada vrlo sličan Gitu). Sve je popularniji u svijetu, koristi ga vrlo veliki broj poslovnih subjekata i danas je dostupan kao paket u gotovo svim distribucijama Linuxa.

## Osnovni pojmovi i svojstva

### Arhitektura

Docker koristi arhitekturu klijent-poslužitelj. Docker klijent je alat naredbenog retka koji korisniku omogućuje interakciju s Docker demonom (engl. *daemon*). Docker klijent i demon mogu se izvoditi na istom sustavu ili se Docker klijent može povezati s udaljenim Docker demonom, međusobno komunicirajući putem REST API-ja. Drugi Docker klijent je Docker Compose, koji omogućuje rad s aplikacijama koje se sastoje od skupa kontejnera. O tome će biti više riječi u drugoj lekciji.

Docker demon je pozadinska usluga koja radi na vašem operacijskom sustavu i upravlja izgradnjom, pokretanjem i distribucijom Docker objektata kao što su primjerice slike, kontejneri, mreže i volumeni. Docker demon čeka zahtjeve iz REST API-ja i u skladu s tim izvodi niz operacija. Daemon također može komunicirati s drugim demonima za upravljanje Docker uslugama.

### Objekti

Kada koristite Docker, stvarate i koristite slike, kontejnere, mreže, volumene, dodatke i druge objekte. Ovaj odjeljak je kratak pregled nekih od tih objekata.

Kontejner

:   Kontejneri pružaju mogućnost pakiranja i pokretanja aplikacija u izoliranom okruženju koje sadrži sve što je potrebno na njihovo pokretanje: konfiguracije, skripte, biblioteke itd. Na ovaj način programeri mogu stvoriti predvidljiva okruženja izolirana od drugih aplikacija koja se mogu izvoditi bilo gdje i služiti kao jedinica za distribuciju i testiranje aplikacije.

    Virtualni stroj, često i virtualna mašina (engl. *virtual machine*, kraće VM), je softversko okruženje koje simulira stvarni hardver i u kojem se može pokrenuti određeni operacijski sustav. Svaki VM sadrži kompletnu kopiju operacijskog sustava, aplikacije, potrebne binarne datoteke i biblioteke, koje zauzimaju desetke GB. Stoga su računalni troškovi potrošeni na hardversku virtualizaciju za korištenje gostujućeg OS-a značajni.

    Kontejneri imaju drugačiji pristup jer virtualiziraju operacijski sustav, a ne hardver. Kontejneri su apstrakcija na razini aplikacije koja pakira kod i ovisnosti zajedno. Više kontejnera može se izvoditi na istom računalu i dijeliti jezgru OS-a s drugim kontejnerima, a svaki se izvodi kao izolirani proces u korisničkom prostoru. Kontejneri zauzimaju mnogo manje prostora od virtualnih strojeva, slike kontejnera obično imaju nekoliko desetaka MB, te su prenosivije i učinkovitije.

Slika

:   Kada se kontejner pokrene, on koristi izolirani datotečni sustav. Ovaj prilagođeni datotečni sustav predstavlja sliku kontejnera. Budući da slika sadrži datotečni sustav kontejnera, mora sadržavati sve što je potrebno za pokretanje aplikacije – sve ovisnosti, konfiguracije, skripte, binarne datoteke i sl. Slika također sadrži druge konfiguracije za kontejner, kao što su varijable okruženja, zadana naredba za pokretanje i drugi metapodaci. Dakle, slika je predložak za čitanje koja sadrži skup uputa za stvaranje kntejnera, a kontejner je tada pokrenuta instanca slike.

Registar

:   Docker registar je spremište za Docker slike. Docker klijenti povezuju se s registrima za preuzimanje slika za korištenje ili učitavanje slika koje su izradili. Registri mogu biti javni ili privatni. Glavni javni registar je [Docker Hub](https://hub.docker.com/), web sjedište na kojem se dijele slike kontejnera. Na njemu je moguće pronaći brojne gotove slike, što je i jedan od razloga popularnosti Dockera.

    Posebna vrsta slika na Docker Hubu su [službene slike](https://docs.docker.com/docker-hub/official_images/) (engl. *official images*), čiji razvoj i održavanje financira sam Docker. Osim [slike hello-world](https://hub.docker.com/_/hello-world) kojom se testira ispravnost instalacije Dockera, u službene slike spadaju [httpd](https://hub.docker.com/_/httpd), [python](https://hub.docker.com/_/python), [php](https://hub.docker.com/_/php), [node](https://hub.docker.com/_/node), [haproxy](https://hub.docker.com/_/haproxy), [mariadb](https://hub.docker.com/_/mariadb), [mongo](https://hub.docker.com/_/mongo), [postgres](https://hub.docker.com/_/postgres), [redis](https://hub.docker.com/_/redis), [nextcloud](https://hub.docker.com/_/nextcloud), [memcached](https://hub.docker.com/_/memcached), [docker](https://hub.docker.com/_/docker) (da, Docker može [pokrenuti drugi Docker u kontejneru koji onda može pokretati Docker kontejnere](https://knowyourmeme.com/memes/subcultures/inception)) i [brojne druge](https://hub.docker.com/search?type=image&image_filter=official). Na stranici svake od službenih slika dane su detaljne upute za njezino korištenje.

Volumen

:   Docker volumeni (engl. *volumes*) široko su korišten i koristan alat za osiguravanje postojanosti podataka tijekom rada unutar kontejnera. Volumeni su datotečni sustavi postavljeni na kontejnere za očuvanje podataka koje generira kontejner koji radi. Volumeni se pohranjuju na domaćinu i omogućuju jednostavno sigurnosno kopiranje te dijeljenje datotečnog sustava između kontejnera.

!!! tip
    U nastavku koristimo službene Dockerove alate naredbenog retka, ali moguće je koristiti i Visual Studio Code koji nudi [službeno proširenje za Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) ([pregled značajki](https://code.visualstudio.com/docs/containers/overview)).

    Dodatno, korištenjem [službenih proširenja za udaljeni rad](https://code.visualstudio.com/docs/remote/remote-overview) moguće je [razvijati softver unutar Docker kontejnera](https://code.visualstudio.com/docs/remote/containers). Za one koji žele znati više o razvoju softvera u kontejnerima, dobro mjesto za započeti je [Microsoftov službeni tečaj za Docker početnike](https://docs.microsoft.com/en-us/visualstudio/docker/tutorials/docker-tutorial).

## Rad s kontejnerima aplikacija i usluga

U ovom dijelu naučit ćete kako preuzeti slike i pokrenuti kontejnere, ali i o izlolaciji samih kontejnera. Možete provjeriti je li Docker instaliran i prikupiti neke informacije o trenutnoj verziji pomoću sljedeće naredbe:

``` shell
$ docker version

Client:
 Version:           20.10.17
 API version:       1.41
 Go version:        go1.18.3
 Git commit:        100c70180f
 Built:             Sat Jun 11 23:27:28 2022
 OS/Arch:           linux/amd64
 Context:           default
 Experimental:      true

Server:
 Engine:
  Version:          20.10.17
  API version:      1.41 (minimum version 1.12)
  Go version:       go1.18.3
  Git commit:       a89b84221c
  Built:            Sat Jun 11 23:27:14 2022
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          v1.6.8
  GitCommit:        9cd3357b7fd7218e4aec3eae239db1f68a5a6ec6.m
 runc:
  Version:          1.1.4
  GitCommit:
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0
```

Također postoji brz i jednostavan način da vidite koliko je Docker kontejnera trenutno pokrenuto i vidite neke od Dockerovih konfiguriranih opcija:

``` shell
$ docker info

Client:
 Context:    default
 Debug Mode: false
 Plugins:
  buildx: Docker Buildx (Docker Inc., v0.8.2-docker)

Server:
 Containers: 0
  Running: 0
  Paused: 0
  Stopped: 0
 Images: 0
 Server Version: 20.10.17
 Storage Driver: overlay2
...
```

Korištenje `docker` sastoji se od prosljeđivanja niza opcija, naredbi i praćenih argumentima. Sintaksa ima ovaj oblik:

``` shell
$ docker <opcija> <naredba> <argumenti>
```

Da biste vidjeli sve dostupne podnaredbe, upišite:

``` shell
$ docker

...
Options:
  -v, --version            Print version information and quit
...

Management Commands:
  network     Manage networks
  volume      Manage volumes
...

Commands:
  build       Build an image from a Dockerfile
  create      Create a new container
  history     Show the history of an image
  images      List images
...
```

### Pokretanje kontejnera `hello-world`

Kao i sa svim tehničkim stvarima, *hello world* dobro je mjesto za početak. Upišite donju naredbu da preuzmete sliku iz Docker huba koja će kreirati kontejner `hello-world`:

``` shell
$ docker run hello-world

Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
2db29710123e: Pull complete
Digest: sha256:62af9efd515a25f84961b70f973a798d2eca956b1b2b026d0a4a63a3b0b6a3f2
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.
...
```

Izlaz pokretanja kontejnera temeljenog na slici `hello-world` govori što se upravo dogodilo. Docker koji radi na vašem terminalu pokušao je pronaći sliku pod nazivom `hello-world`. Budući da ste tek započeli, nema slika pohranjenih lokalno (`Unable to find image 'hello-world:latest' locally`) pa Docker odlazi u svoj zadani Docker registar, Docker Hub, i traži sliku pod nazivom `hello-world`. Tamo pronalazi sliku, povlači je i zatim pokreće u kontejneru. Jedina funkcija `hello-world`-a je ispisati tekst koji vidite na vašem terminalu, nakon čega kontejner izlazi.

!!! admonition "Zadatak"
    Pokrenite kontejner zasnovan na [slici Alpine Linuxa](https://hub.docker.com/_/alpine/). [Alpine Linux](https://alpinelinux.org/) je lagana distribucija Linuxa pa se brzo skida i pokreće.

!!! admonition "Zadatak"
    Docker ima mogućnost pokretanja naredbe unutar kontejnera u interaktivnom terminalu korištenjem parametara `-i` i `-t`. Pokrenite naredbu `/bin/sh` unutar kontejnera Alpine Linuxa.

    Kada ste unutar kontejnera koji pokrenuli ljusku, možete isprobati nekoliko naredbi poput `ls -l`, `uname -a` i drugih, isprobajte ih. Imajte na umu da je Alpine Linux malena distribucija pa bi moglo nedostajati nekoliko naredbi. Izađite iz ljuske i kontejnera upisivanjem naredbe `exit`.

### Upravljanje Dockerovim slikama

Sada ste spremni za instaliranje slika s Dockerom. Ako trebate tražiti željeni softver putem Dockera, možete koristiti sljedeću sintaksu naredbe:

``` shell
$ docker search <name>
```

Na primjer, pokušajmo pretražiti python, koji je popularan programski jezik opće namjene:

``` shell
$ docker search python

NAME                                DESCRIPTION                                     STARS     OFFICIAL   AUTOMATED
python                              Python is an interpreted, interactive, objec…   7939      [OK]
pypy                                PyPy is a fast, compliant alternative implem…   335       [OK]
circleci/python                     Python is an interpreted, interactive, objec…   52
hylang                              Hy is a Lisp dialect that translates express…   46        [OK]
bitnami/python                      Bitnami Python Docker Image                     22                   [OK]
clearlinux/python                   Python programming interpreted language with…   7
cimg/python                                                                         5
openwhisk/python3action             Apache OpenWhisk runtime for Python 3 Actions   5
openwhisk/python3aiaction           Apache OpenWhisk runtime for Python 3 Action…   2
openwhisk/python2action             Apache OpenWhisk runtime for Python v2 Actio…   2
pachyderm/python-build                                                              0
appdynamics/python-agent-init       AppDynamics Repository for Python agent inst…   0
bitnami/python-snapshot                                                             0
rapidfort/python-chromedriver                                                       0
mirantis/python-operations-api      https://mirantis.jira.com/browse/IT-40189       0                    [OK]
submitty/python                     Official Repository for Submitty Python Imag…   0
okteto/python-fastapi                                                               0
okteto/python                                                                       0
corpusops/python                    https://github.com/corpusops/docker-images/     0
pipelinecomponents/python-safety    Safety by pyup.io for Python in a container …   0
itisfoundation/python-with-pandas                                                   0
ibmcom/python-sybase-ppc64le        Docker image for python-sybase-ppc64le          0
ibmcom/python-semver-ppc64le        Docker image for python-semver-ppc64leDocker…   0
ibmcom/python-memcached-ppc64le     Docker image for python-memcached-ppc64le       0
ibmcom/python-dropbox-ppc64le       Docker image for python-dropbox-ppc64leDocke…   0
```

Kao što vidite, postoji jedna službena slika za python, jednostavno nazvana python. Dostupna su i druga izdanja. Tada je potrebno pročitati njihove opise da vidite što rade drugačije od službene slike.

Nakon što znate koju sliku želite preuzeti, možete upotrijebiti sljedeću sintaksu kako biste uputili Docker za preuzimanje željenog softvera:

``` shell
$ docker image pull <name>
```

Na primjer, preuzmimo python:

``` shell
$ docker image pull python

Using default tag: latest
latest: Pulling from library/python
23858da423a6: Pull complete
326f452ade5c: Pull complete
a42821cd14fb: Pull complete
8471b75885ef: Pull complete
8ffa7aaef404: Pull complete
15132af73342: Pull complete
aaf3b07565c2: Pull complete
736f7bc16867: Pull complete
94da21e53a5b: Pull complete
Digest: sha256:e9c35537103a2801a30b15a77d4a56b35532c964489b125ec1ff24f3d5b53409
Status: Downloaded newer image for python:latest
docker.io/library/python:latest
```

Izlaz na gornjoj snimci zaslona pokazuje da je Docker uspio pronaći i preuzeti sliku koju smo naveli.

Popis svih instaliranih Docker slika možemo dobiti naredbom:

``` shell
$ docker images

REPOSITORY    TAG       IMAGE ID       CREATED         SIZE
python        latest    e285995a3494   7 days ago      921MB
hello-world   latest    feb5d9fea6a5   12 months ago   13.3kB
```

### Upravljanje Dockerovim kontejnerima

Kada je slika preuzeta Docker kontejner pokrećemo koristeći sljedeću sintaksu naredbe:

``` shell
$ docker container run <name>
```

Pokrenimo Docker kontejner na temelju prethodne slike.

``` shell
$ docker container run python
```

Prethodnoj naredbi možemo dodati nastavke `ls -l`. Obratite tada pozornost na izlaz narebe.

Upravljanje kontejnerima vršimo naredbom `docker container`. Za provjeru kontejnera u stanju rada, koristite sljedeću naredbu:

``` shell
$ docker container list

CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```

Možemo koristiti i naredbu `docker ps` za istu stvar ili `ls` umjesto `list`. Vidimo da nema aktivnih kontejnera. Dodamo li parametar `-a` (`--all`), bit će prikazani svi kontejneri:

``` shell
$ docker container ls -a

CONTAINER ID   IMAGE         COMMAND     CREATED          STATUS                      PORTS     NAMES
761f298e9ecb   python        "python3"   5 minutes ago    Exited (0) 5 minutes ago              quizzical_pare
415b131fe935   hello-world   "/hello"    20 minutes ago   Exited (0) 20 minutes ago             intelligent_babbage
```

Primijetite da `STATUS` stupac pokazuje da su prikazani kontejneri izašli prije nekog vremena.

Za prikaz posljednjeg kontejnera kojeg ste stvorili, dodajte mu parametar -l:

``` shell
$ docker container ls -l

CONTAINER ID   IMAGE     COMMAND     CREATED          STATUS                      PORTS     NAMES
761f298e9ecb   python    "python3"   6 minutes ago   Exited (0) 6 minutes ago             quizzical_pare
```

Ponovo upišimo naredbu:

``` shell
$ docker info

...
Server:
 Containers: 2
  Running: 0
  Paused: 0
  Stopped: 2
 Images: 2
 Server Version: 20.10.17
 Storage Driver: overlay2
...
```

Sada možemo vidjeti da imamo dva kontejnera i dvije slike.

Za pokretanje zaustavljenog kontejnera upotrijebite `docker start`, nakon čega slijedi ID kontejnera ili naziv kontejnera. Pokrenimo kontejner s ID-om 415b131fe935:

``` shell
docker start 415b131fe935
```

Za zaustavljanje kontejnera koji radi upotrijebite `docker stop`, nakon čega slijedi ID ili njegov naziv. Ovaj put ćemo koristiti naziv koji je Docker dodijelio kontejneru, a to je `intelligent_babbage`:

``` shell
$ docker stop intelligent_babbage
```

Nakon što odlučite da vam kontejner više ne treba, uklonite ga pomoću naredbe `docker rm`, opet koristeći ID kontejnera ili ime.

``` shell
$ docker rm intelligent_babbage
```

### Izolacija Docker kontejnera

Pokrenite nekoliko naredbi `docker container run` za Alpine kontejner. Naredba `docker container ls -a` nam pokazuje da je na popisu nekoliko kontejnera. Zašto je navedeno toliko kontejnera ako su svi s `alpine` slike?

Ovo je ključni sigurnosni koncept u svijetu Docker kontejnera! Iako je svaka `docker container run` naredba koristila istu `alpine` sliku, svako izvršenje bilo je zaseban, izoliran kontejner. Svaki kontejner ima zaseban datotečni sustav i radi u drugom prostoru imena; prema zadanim postavkama kontejner nema načina za interakciju s drugim kontejnerima, čak ni s onima iz iste slike. Pokušajmo još jednu vježbu da naučimo više o izolaciji.

``` shell
$ docker container run -it alpine /bin/ash
```

`/bin/ash` je još jedna vrsta ljuske dostupna na `alpine` slici. Nakon što se kontejner pokrene i kada ste u naredbenom retku kontejnera, upišite sljedeće naredbe:

``` shell
$ echo "hello world" > hello.txt
$ ls
```

Prva `echo` naredba stvara datoteku pod nazivom `hello.txt` s riječima `hello world` unutar nje. Druga naredba daje vam popis datoteka u direktoriju i trebala bi prikazati vašu novostvorenu datoteku `hello.txt`. Sada upišite `exit` da napustite ovaj kontejner.

Da pokažete kako izolacija funkcionira, pokrenite sljedeće:

``` shell
$ docker container run alpine ls
```

To je ista `ls` naredba koju smo koristili unutar interaktivne ljuske kontejnera, ali ovaj put, primijetite da nedostaje datoteka `hello.txt`. To je izolacija. Naredba je pokrenuta u novoj i zasebnoj instanci, iako se temelji na istoj slici.

U svakodnevnom radu, korisnici Dockera koriste ovu značajku ne samo za sigurnost, već i za testiranje učinaka promjena aplikacije. Izolacija omogućuje korisnicima da brzo stvore odvojene, izolirane testne kopije aplikacije ili usluge i da se one pokreću usporedno bez ometanja jedna druge.

Sada nam ostaje odgovoriti na pitanje: "Kako se vratiti do kontejnera koji sadrži datoteku `hello.txt`"?

Još jednom pokrenite naredbu `docker container ls` na način:

``` shell
$ docker container ls -a

CONTAINER ID   IMAGE     COMMAND                  CREATED          STATUS                      PORTS     NAMES
36fd29a3e1c0   alpine    "ls"                     4 minutes ago    Exited (0) 4 minutes ago              awesome_wu
091a118b96c4   alpine    "/bin/ash"               5 minutes ago    Exited (0) 5 minutes ago              ecstatic_grothendieck
4c2b93b1596f   alpine    "/bin/sh"                12 minutes ago   Exited (0) 12 minutes ago             naughty_bell
d52407b6b1c3   alpine    "echo 'hello from al…"   2 hours ago      Exited (0) 2 hours ago                quirky_mcclintock
878157a77c7b   alpine    "ls -l"                  3 hours ago      Exited (0) 3 hours ago                boring_ishizaka
```

Kontejner u kojem smo stvorili datoteku `hello.txt` isti je onaj u kojem smo koristili `/bin/ash` ljusku, koju možemo vidjeti navedenu u stupcu `COMMAND`. U ovom slučaju radi se o kontejneru s ID-om `091a118b96c4` naziva `ecstatic_grothendieck`. Prisjetimo se naredbe za pokretanje kontejnera pomoću ID-a ili naziva i pokrenimo navedeni kontejner.

Možemo upotrijebiti nešto drugačiju naredbu kako bismo rekli Dockeru da pokrene ovu specifičnu instancu kontejnera.

``` shell
$ docker container start <kontejner ID>
```

Savjet: Umjesto upotrebe punog ID-a kontejnera, možete upotrijebiti samo prvih nekoliko znakova, sve dok su dovoljni za jedinstveni ID kontejnera. Dakle, mogli bismo jednostavno upotrijebiti `091a` za identifikaciju kontejnera u gornjem primjeru, budući da nijedan drugi kontejner na ovom popisu ne počinje ovim znakovima.

Sada ponovno upotrijebite `docker container ls` naredbu za popis aktivnih kontejnera.

``` shell
$ docker container ls

CONTAINER ID   IMAGE     COMMAND      CREATED          STATUS              PORTS     NAMES
091a118b96c4   alpine    "/bin/ash"   10 minutes ago   Up About a minute             ecstatic_grothendieck
```

Primijetite da ovaj put naš kontejner još uvijek radi. Ovaj put koristili smo `ash shell` tako da umjesto jednostavnog izlaska na način na koji je `/bin/sh` učinio ranije, `ash` čeka naredbu. Možemo poslati naredbu u kontejner da se pokrene pomoću naredbe `exec`:

``` shell
$ docker container exec <kontejner ID> ls
```

Ovaj put dobivamo popis direktorija i prikazuje našu datoteku `hello.txt` jer smo koristili instancu kontejnera u kojoj smo stvorili tu datoteku. Sada počinjete uviđati neke od važnih koncepata kontejnera. U sljedećoj vježbi radit ćemo s aplikacijama koje se sastoje od skupa kontejnera pomoću Docker Composea.

## Druge naredbe za baratanje kontejnerima aplikacija

Baratanje kontejnerima vršimo naredbom `docker container`. Provjerimo ponovno popis pokrenutih kontejnera naredbom `docker container list`:

``` shell
$ docker container list
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```

Vidimo da ih nema. Dodamo li parametar `-a` (`--all`), bit će prikazani svi kontejneri:

``` shell
$ docker container list -a
CONTAINER ID        IMAGE               COMMAND              CREATED             STATUS                     PORTS               NAMES
1e470c81b80a        httpd               "hello"              12 minutes ago      Exited (0) 3 minutes ago                       eager_mcnulty
```

Među njima je i kontejner koji smo stvorili pokretanjem slike `hello-world`. Pokrenimo ga naredbom `docker container start` korištenjem ID-a:

``` shell
$ docker container start 1e470c81b80a
1e470c81b80a
```

ili korištenjem imena:

``` shell
$ docker container start eager_mcnulty
eager_mcnulty
```

Kako god smo izveli pokretanje, uvjerimo se da je pokrenut naredbom `docker ps`:

``` shell
$ docker ps
CONTAINER ID        IMAGE               COMMAND              CREATED             STATUS              PORTS               NAMES
1e470c81b80a        httpd               "hello"              14 minutes ago      Up 6 seconds        80/tcp              eager_mcnulty
```

Naposlijetku, naredbom `docker logs` pročitajmo poruke koje je kontejner ispisao nakon pokretanja:

``` shell
$ docker logs eager_mcnulty
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
1. The Docker client contacted the Docker daemon.
2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
(amd64)
3. The Docker daemon created a new container from that image which runs the
executable that produces the output you are currently reading.
4. The Docker daemon streamed that output to the Docker client, which sent it
to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
$ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
https://hub.docker.com/

For more examples and ideas, visit:
https://docs.docker.com/get-started/
```

Poruke su iste kao i kod pokretanja naredbom `docker run` iznad.

## Izrada vlastitih kontejnera

U sljedećem zadatku stvorite, izradite i pokrenite vlastitu Docker sliku u kontejneru.

!!! adomonition "Zadatak"
    Po uzoru na [zadatak s GitHuba](https://github.com/techno-tim/launchpad/tree/master/docker/custom-image) izradite vlastitu Docker sliku.

U sljedećem zadatku izradit ćete jednostavnu web stranicu za e-trgovinu. Web stranica će koristiti API na uslugu proizvoda kako bi zatražila popis proizvoda za prikaz kupcu.

!!! adomonition "Zadatak"
    Napravite Docker sliku za [aplikaciju koja nudi uslugu proizvoda](https://github.com/jakewright/tutorials/tree/master/docker/02-docker-compose).

U sljedećem zadatku izradit ćete Python aplikaciju koja će vam nuditi izbor filmova slučajnim odabirom.

!!! adomonition "Zadatak"
    Preuzmite [Python skriptu moviepickera](https://github.com/python-engineer/python-fun/tree/master/moviepicker) i napravite `Dockerfile` za Docker sliku koja će pokrenuti skriptu kod pokretanja u kontejneru.

## Primjer primjene

Kao alat, Docker ima vrlo široku primjenu. U ovom projektu iskoristiti ću ga kao alat za dijeljenje kompaktnog OS-a za rad sa JSON-om.

Ideja jest da se napravi mali OS koji ce se moci pokrenuti na svakoj mašini gdje god se nalazio, i da ču na toj mašini imati instalirane sve potrebne alate za rad u JSON-u i podizanje izmjenjenih fileova na GitHub.

### Distribucija prilagođenog OS-a

Proces kreće s instaliravanjem Docker alata na mašinu, što je već i učinjeno. Sljedeći korak jest pokretanje samog dockera i prijavljivanje na online repozitorij. Iskoristite postojeći račun sa github-a za izradu docker repozitorija. Odaberimo `sinatru` za sustav koji ćemo prilagoditi svojim potrebama:

``` shell
$ docker pull training/sinatra
```

Sljedeći korak jest pokretanje preuzetog sustava i izvršavanje željenih izmjena u sustavu, instaliravanje i brisanje paketa i sl. Uzmimo za primjer `json` i još neke dodatne alate:

``` shell
$ docker run -t -i training/sinatra /bin/bash
$ gem install json
$ sudo apt-get install git
```

Dodatno je preuzet repozitorij s Bitbucket-a koji je napravljen na distribuiranim sustavima:

``` shell
$ git clone https://lbrumnic@bitbucket.org/lbrumnic/ds_projekt.git
```

Izlaskom iz trenutne slike, vrši se `commit` napravljenih izmjena prema trenutnom kontejneru koji se kreirao pokretanjem slike:

``` shell
$ docker commit 79de9d702c2e lukabrumnic/sinatraluka
```

Svaki kontejner pri pokretanju dobije svoj individualni `id` koje se može provjeriti naredbom:

``` shell
$ docker ps
```

Na kraju se slika `push-a` na online repozitorij:

``` shell
$ docker push lukabrumnic/sinatraluka
```

U sljedećem koraku sljedi testiranje napravljenih izmjena, a to ćemo napraviti tako da se podigne novi virtualni stroj sa sutavom `Fedora 20`. Na njemu se ulogirajte u svoj Docker repozitorij, preuzmite traženu sliku i provjerite ima li instaliran `json` na njoj.

`Fedora 20` ima Docker u službenom repozitoriju paketa stoga je dovoljno instalirati paket `docker-io`. Nakon instalacije paketa, vrši se `login` na docker repozitorij. Zatim se preuzima željeni `image` te se istog pokreće.

``` shell
$ docker login
$ docker pull lukabrumnic/sinatraluka
$ docker run -t -i lukabrumnic/sinatraluka /bin/bash
```

Provjeru da je riječ o istom `image-u` nalazimo u `home` direktoriju gdje smo preuzeli repozitorij sa Bitbucket-a.

### Ostale primjene

Od ostalih primjena postoji mogućnost pokretanja `daemon` aplikacije u pozadini čime se zapravo pokreće novi kontejner u kojem se neka aplikacije vrti. Još jedna korisna mogućnost je pokretanje web aplikacija u Dockeru. Uzmimo primjer:

``` shell
$ docker run -d -P training/webapp python app.py
```

Aplikacija se pokrene na portu 49153 no može se ručno podesiti da se aplikacije izvršavaju na željenom portu. Na primjer:

``` shell
$ docker run -d -p 5000:5000 training/webapp python app.py
```

Dodatno se još može specificirati i sučelje na kojemu će se pokrenuti aplikacija. Po zadanom će se vezati specificirani port na sva sučelja, no to se može ograničiti. Primjerice:

``` shell
$ docker run -d -p 127.0.0.1:5000:5000 training/webapp python app.py
```

Također se mogu pokrenuti u pozadini i izvršavati određene aktivnosti bez smetnje za klijenta.

``` shell
$ docker run -d -p ubuntu:14.04 /bin/sh -c "while true; do echo hello world; sleep 1; done"
```

## Konfiguracija

U datoteci `/etc/default/docker` mogu se podešavati različiti parametri za rad naredbe `docker`, kao primjerice lokacija spremanja privremenih datoteka i slično. Isto tako moguće je definirati na kojem se zadanom portu pokreću `daemon docker apps`.

Za prikupljanje logova o aktivnostima na pokrenutim kontejneraima postoje mnogi alati. Specijaliziraju se u prikupljanju i centraliziranju logova sa svih pokrenutih kontejnera i šalju ih na centralni poslužitelj gdje se isti spremaju. Neki od alata su [Loggly](https://documentation.solarwinds.com/en/success_center/loggly/content/admin/about-loggly.htm) i [Fluentd](https://www.fluentd.org/guides/recipes/docker-logging).

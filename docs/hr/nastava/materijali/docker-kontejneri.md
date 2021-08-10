---
author: Luka Brumnić, Vedran Miletić
---

# Kontejnerizacija alatom Docker

!!! hint
    Za više informacija proučite [službenu dokumentaciju](https://docs.docker.com/).

Docker je skup alata za kontejnerizaciju (poznatu još i pod nazivom kontejnerska virtualizacija ili [virtualizacija na razini operacijskog sustava](https://en.wikipedia.org/wiki/Operating-system-level_virtualization), engl. *operating-system-level virtualization*) i dijeljenje aplikacija putem interneta (gdje mu je princip rada vrlo sličan Gitu). Sve je popularniji u svijetu, koristi ga vrlo veliki broj poslovnih subjekata i danas je dostupan kao paket u gotovo svim distribucijama Linuxa.

## Primjer primjene

Kao alat, Docker ima vrlo široku primjenu. U ovom projektu iskoristiti ću ga kao alat za dijeljenje kompaktnog OS-a za rad sa JSON-om.

Ideja jest da se napravi mali OS koji ce se moci pokrenuti na svakoj mašini gdje god se nalazio, i da ču na toj mašini imati instalirane sve potrebne alate za rad u JSON-u i podizanje izmjenjenih fileova na GitHub.

### Distribucija prilagođenog OS-a

Proces kreće se instaliravanjem Docker alata na mašinu, što je već i učinjeno. Sljedeći korak jest pokretanje samom dockera i prijavljivanje na online repozitorij. Iskoristio sam postojeći account sa github-a za izradu docker repozitorija te sam se na taj način prijavio. Odabrao sam sinatru za sustav koji ću prilagoditi svojim potrebama pa za početak povlačim na lokalnu mašinu:

``` shell
$ sudo docker pull training/sinatra
```

Sljedeći korak jest pokretanje preuzetog sustava i izvršavanje željenih izmjena u sustavu, instaliravanje i brisanje paketa i sl, u ovom slučaju demonstracije radi sam instalirao json i jos neke dodatne alate:

``` shell
$ sudo docker run -t -i training/sinatra /bin/bash
$ gem install json
$ sudo apt-get install git
```

i dodatno sam preuzeo stari repozitorij sa bitbucketa koji sam napravio na distribuiranim sustavima:

``` shell
$ git clone https://lbrumnic@bitbucket.org/lbrumnic/ds_projekt.git
```

Nakon toga izlaskom iz trenutne slike, vrši se commit napravljenih izmjena prema trenutnom "kontejneru" koji se kreirao pokretanjem slike:

``` shell
$ sudo docker commit 79de9d702c2e lukabrumnic/sinatraluka
```

Svaki kontejner pri pokretanju dobije svoj individualni id koje se može provjeriti naredbom:

``` shell
$ sudo docker ps
```

Na kraju ta ista slika se "push-a" na online repozitorij:

``` shell
$ sudo docker push lukabrumnic/sinatraluka
```

U sljedećem koraku sljedi testiranje napravljenih izmjena a to ću napraviti tako da podignem novi virtualni stroj sa sutavom Fedora 20, te se na njemu ulogiram u svoj Docker repozitorij, preuzmem
traženu sliku i provjerim imam li instaliran `json` na njoj.

Fedora 20 ima Docker u službenom repozitoriju paketa stoga je dovoljno instalirati paket `docker-io`. Nakon instalacije paketa, vrši se loginna docker repozitorij, zatim se preuzima željeni image te se istog pokreće.

``` shell
$ sudo docker login
$ sudo docker pull lukabrumnic/sinatraluka
$ sudo docker run -t -i lukabrumnic/sinatraluka /bin/bash
```

Provjeru da je riječ o istom imageu nalazim u home folderu gdje sam preuzeo repozitorij sa Bitbucketa.

### Ostale primjene

Od ostalih primjena postoji mogućnost pokretanja daemon aplikacije u pozadini čime se zapravo pokreće novi container u kojem se neka aplikacije vrti. Još jedna korisna mogućnost je pokretanje web aplikacija u Dockeru. Uzmimo primjer:

``` shell
$ sudo docker run -d -P training/webapp python app.py
```

Aplikacije se pokrene na portu 49153 no može se ručno podesiti da aplikacije se izvršava na željenom portu. Primjer prema gornjem primjeru:

``` shell
$ sudo docker run -d -p 5000:5000 training/webapp python app.py
```

Dodatno se još može specificirati i interface na kojemu će se pokrenuti aplikacije, po defaultu će vezati specificirani port na sva sučelja no to se može ograničiti. Primjer za gonji primjer:

``` shell
$ sudo docker run -d -p 127.0.0.1:5000:5000 training/webapp python app.py
```

Također se mogu pokrenuti u pozadini i izvršavati određene aktivnosti bez smetnje za klijenta.

``` shell
$ sudo docker run -d -p ubuntu:14.04 /bin/sh -c "while true; do echo hello world; sleep 1; done"
```

## Baratanje kontejnerima aplikacija

Baratanje kontejnerima vršimo naredbom `docker container`. Provjerimo popis pokrenutih kontejnera naredbom `docker container list`:

``` shell
$ sudo docker container list
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```

Vidimo da ih nema. Dodamo li parametar `-a` (`--all`), bit će prikazani svi kontejneri:

``` shell
$ sudo docker container list -a
CONTAINER ID        IMAGE               COMMAND              CREATED             STATUS                     PORTS               NAMES
1e470c81b80a        httpd               "httpd-foreground"   12 minutes ago      Exited (0) 3 minutes ago                       eager_mcnulty
```

Među njima je i kontejner koji smo stvorili pokretanjem slike `httpd`. Pokrenimo ga naredbom `docker container start` korištenjem ID-a:

``` shell
$ sudo docker container start 1e470c81b80a
1e470c81b80a
```

ili korištenjem imena:

``` shell
$ sudo docker container start eager_mcnulty
eager_mcnulty
```

Kako god smo izveli pokretanje, uvjerimo se da je pokrenut naredbom `docker ps`:

``` shell
$ sudo docker ps
CONTAINER ID        IMAGE               COMMAND              CREATED             STATUS              PORTS               NAMES
1e470c81b80a        httpd               "httpd-foreground"   14 minutes ago      Up 6 seconds        80/tcp              eager_mcnulty
```

Naposlijetku, naredbom `docker logs` pročitajmo poruke koje je kontejner ispisao nakon pokretanja:

``` shell
$ sudo docker logs eager_mcnulty
AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.17.0.2. Set the 'ServerName' directive globally to suppress this message
AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.17.0.2. Set the 'ServerName' directive globally to suppress this message
[Thu May 07 23:25:00.371329 2020] [mpm_event:notice] [pid 1:tid 139777378702464] AH00489: Apache/2.4.43 (Unix) configured -- resuming normal operations
[Thu May 07 23:25:00.371673 2020] [core:notice] [pid 1:tid 139777378702464] AH00094: Command line: 'httpd -D FOREGROUND'
[Thu May 07 23:33:30.363896 2020] [mpm_event:notice] [pid 1:tid 139777378702464] AH00491: caught SIGTERM, shutting down
AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.17.0.2. Set the 'ServerName' directive globally to suppress this message
AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.17.0.2. Set the 'ServerName' directive globally to suppress this message
[Thu May 07 23:39:07.310914 2020] [mpm_event:notice] [pid 1:tid 140007882990720] AH00489: Apache/2.4.43 (Unix) configured -- resuming normal operations
[Thu May 07 23:39:07.311262 2020] [core:notice] [pid 1:tid 140007882990720] AH00094: Command line: 'httpd -D FOREGROUND'
```

Poruke su iste kao i kod pokretanja naredbom `docker run` iznad.

## Konfiguracija

U datoteci `/etc/default/docker` mogu se podešavati različiti parametri za rad naredbe `docker`, kao primjerice lokacija spremanja privremenih fileova i slično. Isto tako moguće je definirati na kojem se defaultnom portu pokreću daemon docker apps.

Za prikupljanje logova o aktivnostima na pokrenutim containerima postoje mnogi alati. Specijaliziraju se u prikupljanju i centraliziranju logova sa svih pokrenutih kontejnera i šalju ih na centralni poslužitelj gdje se isti premaju. Neki od alata su [Loggly](https://www.loggly.com/docs/docker-syslog/) i [Fluentd](https://www.fluentd.org/guides/recipes/docker-logging).

---
author: Vedran Miletić, Matea Turalija
---

# Komponiranje kontejnera alatom Docker Compose

Do sada smo istraživali korištenje pojedinačnih instanci kontejnera koji se izvode na jednom glavnom računalu, slično kao što programer radi na jednoj usluzi neke aplikacije. Međutim, proizvodne aplikacije obično su mnogo složenije i sastoje se od više usluga, a ovaj model s jednim poslužiteljem neće funkcionirati za koordinaciju 10 ili 100 kontejnera i mrežnih veza između njih, a da ne spominjemo potrebu za osiguravanjem dostupnosti i skalabilnosti.

[Docker Compose](https://docs.docker.com/compose/) ([GitHub](https://github.com/docker/compose)) je alat za definiranje i pokretanje višekontejnerskih aplikacija korištenjem Dockera, tzv. kompozicija (engl. *composition*). Usluge koje će se pokrenuti definiraju se korištenjem konfiguracije u obliku [YAML](https://yaml.org/). YAML je donekle srodan [JSON](https://www.json.org/)-u, ali je dizajniran kako bi bio čitljiv ljudima (slično kao [TOML](https://toml.io/)).

!!! tip
    [Službeno proširenje za Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) za Visual Studio Code, između ostalog, podržava i [korištenje Docker Composea](https://code.visualstudio.com/docs/containers/docker-compose).

## Struktura konfiguracije

!!! warning
    Ubuntu 20.04 LTS ima paket [docker-compose](https://packages.ubuntu.com/focal/docker-compose) koji sadrži Docker Compose verziju 1.25.0. Ta verzija podržava sve značajke koje u nastavku koristimo, ali potrebno je pripaziti da polje `version` u YAML datoteci u nastavku bude postavljeno na 3.7 (umjesto 3.9) jer je [verzija konfiguracije 3.8 postala dostupna tek u Docker Composeu 1.25.5, a 3.9 u 1.27.1](https://github.com/docker/compose/blob/master/CHANGELOG.md).

Korištenje Docker Composea u osnovi je proces od tri koraka:

- Definiranje okruženja aplikacije s `Dockerfile` tako da se može reproducirati bilo gdje.
- Definiranje usluge koje čine aplikaciju u `docker-compose.yml` kako bi se mogle zajedno izvoditi u izoliranom okruženju.
- Pokretanje naredbe `docker-compose up` koja pokreće cijelu aplikaciju.

!!! note
    Docker Compose krenuo je kao samostalni alat `docker-compose`, ali je u posljednjim verzijama postao dio osnovnog Dockera kao podnaredba `compose` koja ima svoje podnaredbe. Stoga, naredbu za pokretanje `docker-compose up` moguće je pisati i kao `docker compose up`, a analogno vrijedi i za ostale naredbe.

Datoteka `docker-compose.yml` koju Docker Compose koristi može biti oblika:

``` yaml
version: "3.8"
services:
  mojhttpd:
    image: httpd
  mojredis:
    image: redis
```

Uočimo vrijednost ključa `version`. na stranici [Compose file versions and upgrading](https://docs.docker.com/compose/compose-file/compose-versioning/) u tablici možete vidjeti koje verzije datoteke `docker-compose.yml` podržavaju određena izdanja Dockera. Koristit ćemo posljednju verziju Docker Enginea, a ona podržava datotečni format Composea verzije `3.8`.

Promatrajući vrijednost ključa `services,` uočavamo da ćemo u ovom slučaju pokrenut ćemo dva kontejnera, prvi naziva `mojhttpd` temeljen na službenoj slici [httpd](https://hub.docker.com/_/httpd), a drugi `mojredis` temeljen na službenoj slici [redis](https://hub.docker.com/_/redis).

## Pokretanje, nadzor i zaustavljanje kontejnera

Povucimo slike argumentom `pull` ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_pull/)):

``` shell
$ docker-compose pull
[+] Running 12/12
 ⠿ mojredis Pulled                                                         21.1s
   ⠿ a2abf6c4d29d Pull complete                                            17.3s
   ⠿ c7a4e4382001 Pull complete                                            17.5s
   ⠿ 4044b9ba67c9 Pull complete                                            17.6s
   ⠿ c8388a79482f Pull complete                                            17.9s
   ⠿ 413c8bb60be2 Pull complete                                            18.1s
   ⠿ 1abfd3011519 Pull complete                                            18.1s
 ⠿ mojhttpd Pulled                                                         21.6s
   ⠿ dcc4698797c8 Pull complete                                            17.6s
   ⠿ 41c22baa66ec Pull complete                                            17.6s
   ⠿ 67283bbdd4a0 Pull complete                                            18.5s
   ⠿ d982c879c57e Pull complete                                            18.7s
```

Stvaranje kontejnera na temelju preuzetih slika i njihovo pokretanje ćemo izvesti argumentom `up` ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_up/)):

``` shell
$ docker-compose up
[+] Running 2/0
⠿ Container docker-mojhttpd-1  Created                                   0.0s
⠿ Container docker-mojredis-1  Created                                   0.0s
Attaching to docker-mojhttpd-1, docker-mojredis-1
docker-mojhttpd-1  | AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.18.0.2. Set the 'ServerName' directive globally to suppress this message
docker-mojhttpd-1  | AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.18.0.2. Set the 'ServerName' directive globally to suppress this message
docker-mojredis-1  | 1:C 03 Jan 2022 22:26:37.086 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
docker-mojredis-1  | 1:C 03 Jan 2022 22:26:37.086 # Redis version=6.2.6, bits=64, commit=00000000, modified=0, pid=1, just started
docker-mojredis-1  | 1:C 03 Jan 2022 22:26:37.086 # Warning: no config file specified, using the default config. In order to specify a config file use redis-server /path/to/redis.conf
docker-mojredis-1  | 1:M 03 Jan 2022 22:26:37.087 * monotonic clock: POSIX clock_gettime
docker-mojhttpd-1  | [Mon Jan 03 22:26:37.087474 2022] [mpm_event:notice] [pid 1:tid 140323163139392] AH00489: Apache/2.4.51 (Unix) configured -- resuming normal operations
docker-mojhttpd-1  | [Mon Jan 03 22:26:37.087533 2022] [core:notice] [pid 1:tid 140323163139392] AH00094: Command line: 'httpd -D FOREGROUND'
docker-mojredis-1  | 1:M 03 Jan 2022 22:26:37.087 * Running mode=standalone, port=6379.
docker-mojredis-1  | 1:M 03 Jan 2022 22:26:37.087 # Server initialized
docker-mojredis-1  | 1:M 03 Jan 2022 22:26:37.087 # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
docker-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 * Loading RDB produced by version 6.2.6
docker-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 * RDB age 55 seconds
docker-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 * RDB memory usage when created 0.77 Mb
docker-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 # Done loading RDB, keys loaded: 0, keys expired: 0.
docker-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 * DB loaded from disk: 0.000 seconds
docker-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 * Ready to accept connections
```

Izvođenje, kao i inače, prekidamo kombinacijom tipki ++control+c++:

``` shell
^CGracefully stopping... (press Ctrl+C again to force)
[+] Running 2/2
 ⠿ Container docker-mojhttpd-1  Stopped                                     1.2s
 ⠿ Container docker-mojredis-1  Stopped                                     0.2s
canceled
```

Želimo li pokrenuti kompoziciju kontejnera u pozadini, dodat ćemo parametar `--detach`, odnosno `-d`:

``` shell
$ docker-compose up -d
[+] Running 2/2
 ⠿ Container docker-mojredis-1  Started                                   0.3s
 ⠿ Container docker-mojhttpd-1  Started                                   0.3s
```

Argumentom `ps` dobit ćemo popis pokrenutih procesa u pojedinim kontejnerima ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_ps/)):

``` shell
$ docker-compose ps
NAME                 COMMAND                  SERVICE             STATUS              PORTS
docker-mojhttpd-1   "httpd-foreground"       mojhttpd            running             80/tcp
docker-mojredis-1   "docker-entrypoint.s…"   mojredis            running             6379/tcp
```

Želimo li više informacija, primjerice koliko je točno procesa pokrenuo web poslužitelj, koliko se dugo izvode i koji su im identifikatori, ispisat ćemo ih argumentom `top`, ponovno razdvojene po pojedinim kontejnerima ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_top/)):

``` shell
$ docker-compose top
docker-mojhttpd-1
UID    PID      PPID     C    STIME   TTY   TIME       CMD
root   870026   869946   0    23:34   ?     00:00:00   httpd -DFOREGROUND
bin    870323   870026   0    23:34   ?     00:00:00   httpd -DFOREGROUND
bin    870324   870026   0    23:34   ?     00:00:00   httpd -DFOREGROUND
bin    870325   870026   0    23:34   ?     00:00:00   httpd -DFOREGROUND

docker-mojredis-1
UID   PID      PPID     C    STIME   TTY   TIME       CMD
999   870038   869998   0    23:34   ?     00:00:00   redis-server *:6379
```

Argumentom `logs` dobit ćemo uvid u zapisnike pojedinih kontejnera ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_logs/)):

``` shell
$ docker-compose logs
docker-mojhttpd-1  | AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.18.0.2. Set the 'ServerName' directive globally to suppress this message
docker-mojhttpd-1  | AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.18.0.2. Set the 'ServerName' directive globally to suppress this message
docker-mojhttpd-1  | [Sun Oct 16 17:19:59.639291 2022] [mpm_event:notice] [pid 1:tid 140373418044736] AH00489: Apache/2.4.54 (Unix) configured -- resuming normal operations
docker-mojhttpd-1  | [Sun Oct 16 17:19:59.639568 2022] [core:notice] [pid 1:tid 140373418044736] AH00094: Command line: 'httpd -D FOREGROUND'
docker-mojhttpd-1  | [Sun Oct 16 17:20:49.178083 2022] [mpm_event:notice] [pid 1:tid 140373418044736] AH00492: caught SIGWINCH, shutting down gracefully
docker-mojhttpd-1  | AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.18.0.3. Set the 'ServerName' directive globally to suppress this message
docker-mojhttpd-1  | AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.18.0.3. Set the 'ServerName' directive globally to suppress this message
docker-mojhttpd-1  | [Sun Oct 16 17:22:00.688161 2022] [mpm_event:notice] [pid 1:tid 140264025701696] AH00489: Apache/2.4.54 (Unix) configured -- resuming normal operations
docker-mojhttpd-1  | [Sun Oct 16 17:22:00.688314 2022] [core:notice] [pid 1:tid 140264025701696] AH00094: Command line: 'httpd -D FOREGROUND'
docker-mojredis-1  | 1:C 16 Oct 2022 17:19:59.669 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
docker-mojredis-1  | 1:C 16 Oct 2022 17:19:59.669 # Redis version=7.0.5, bits=64, commit=00000000, modified=0, pid=1, just started
docker-mojredis-1  | 1:C 16 Oct 2022 17:19:59.669 # Warning: no config file specified, using the default config. In order to specify a config file use redis-server /path/to/redis.conf
docker-mojredis-1  | 1:M 16 Oct 2022 17:19:59.670 * monotonic clock: POSIX clock_gettime
docker-mojredis-1  | 1:M 16 Oct 2022 17:19:59.670 * Running mode=standalone, port=6379.
docker-mojredis-1  | 1:M 16 Oct 2022 17:19:59.671 # Server initialized
docker-mojredis-1  | 1:M 16 Oct 2022 17:19:59.671 # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
docker-mojredis-1  | 1:M 16 Oct 2022 17:19:59.671 * Ready to accept connections
docker-mojredis-1  | 1:signal-handler (1665940849) Received SIGTERM scheduling shutdown...
docker-mojredis-1  | 1:M 16 Oct 2022 17:20:49.207 # User requested shutdown...
docker-mojredis-1  | 1:M 16 Oct 2022 17:20:49.207 * Saving the final RDB snapshot before exiting.
docker-mojredis-1  | 1:M 16 Oct 2022 17:20:49.209 * DB saved on disk
docker-mojredis-1  | 1:M 16 Oct 2022 17:20:49.209 # Redis is now ready to exit, bye bye...
docker-mojredis-1  | 1:C 16 Oct 2022 17:22:00.645 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
docker-mojredis-1  | 1:C 16 Oct 2022 17:22:00.645 # Redis version=7.0.5, bits=64, commit=00000000, modified=0, pid=1, just started
docker-mojredis-1  | 1:C 16 Oct 2022 17:22:00.645 # Warning: no config file specified, using the default config. In order to specify a config file use redis-server /path/to/redis.conf
docker-mojredis-1  | 1:M 16 Oct 2022 17:22:00.646 * monotonic clock: POSIX clock_gettime
docker-mojredis-1  | 1:M 16 Oct 2022 17:22:00.646 * Running mode=standalone, port=6379.
docker-mojredis-1  | 1:M 16 Oct 2022 17:22:00.646 # Server initialized
docker-mojredis-1  | 1:M 16 Oct 2022 17:22:00.646 # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
docker-mojredis-1  | 1:M 16 Oct 2022 17:22:00.647 * Loading RDB produced by version 7.0.5
docker-mojredis-1  | 1:M 16 Oct 2022 17:22:00.647 * RDB age 71 seconds
docker-mojredis-1  | 1:M 16 Oct 2022 17:22:00.647 * RDB memory usage when created 0.82 Mb
docker-mojredis-1  | 1:M 16 Oct 2022 17:22:00.647 * Done loading RDB, keys loaded: 0, keys expired: 0.
docker-mojredis-1  | 1:M 16 Oct 2022 17:22:00.647 * DB loaded from disk: 0.000 seconds
docker-mojredis-1  | 1:M 16 Oct 2022 17:22:00.647 * Ready to accept connections
```

Zustavljanje izvođenja izvodimo argumentom `stop` ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_stop/))

``` shell
$ docker-compose stop
[+] Running 2/2
 ⠿ Container docker-mojredis-1  Stopped                                     0.2s
 ⠿ Container docker-mojhttpd-1  Stopped                                     1.2s
```

Argumentom `down` ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_down/)) zaustavljamo i uklanjamo kontejnere, mreže, volumene i slike koje je stvorio `up`.

``` shell
$ docker-compose down
[+] Running 3/3
 ⠿ Container docker-mojredis-1  Removed                                     0.3s
 ⠿ Container docker-mojhttpd-1  Removed                                     1.2s
 ⠿ Network docker_default       Removed                                     0.1s
```

Naredba podržava još argumenata, o kojima možemo saznati više proučavanjem [službene dokumentacije](https://docs.docker.com/engine/reference/commandline/compose/) ili korištenjem parametra `--help`:

``` shell
$ docker-compose --help

Usage:  docker compose [OPTIONS] COMMAND

Docker Compose

Options:
      --ansi string                Control when to print ANSI control characters
                                   ("never"|"always"|"auto") (default "auto")
      --compatibility              Run compose in backward compatibility mode
      --env-file string            Specify an alternate environment file.
  -f, --file stringArray           Compose configuration files
      --profile stringArray        Specify a profile to enable
      --project-directory string   Specify an alternate working directory
                                   (default: the path of the Compose file)
  -p, --project-name string        Project name

Commands:
  build       Build or rebuild services
  convert     Converts the compose file to platform's canonical format
  cp          Copy files/folders between a service container and the local filesystem
  create      Creates containers for a service.
  down        Stop and remove containers, networks
  events      Receive real time events from containers.
  exec        Execute a command in a running container.
  images      List images used by the created containers
  kill        Force stop service containers.
  logs        View output from containers
  ls          List running compose projects
  pause       Pause services
  port        Print the public port for a port binding.
  ps          List containers
  pull        Pull service images
  push        Push service images
  restart     Restart containers
  rm          Removes stopped service containers
  run         Run a one-off command on a service.
  start       Start services
  stop        Stop services
  top         Display the running processes
  unpause     Unpause services
  up          Create and start containers
  version     Show the Docker Compose version information

Run 'docker compose COMMAND --help' for more information on a command.
```

!!! adomonition "Zadatak"
    Napravite "kompoziciju" koja se sastoji od jednog kontejnera `website` koristeći [sliku web poslužitelja nginx](https://hub.docker.com/_/nginx/) i povežite ga mrežom s domaćinom na kojem se Docker izvodi tako da njegova vrata 80 budu dostupna na vratima koristeći se vratima 8080 na domaćinu. U pregledniku otvorite korijensku web stranicu upravo pokrenutog web poslužitelja.

!!! admonition "Zadatak"
    Napravite kompoziciju dva kontejnera imena po želji od kojih su oba temeljena na slici nginx, neka jedan bude dostupan na vratima 8081, a drugi na vratima 8082. Uvjerite se da možete otvoriti oba web sjedišta.

!!! admonition "Zadatak"
    Napravite kompoziciju dva kontejnera temeljena na slici nginx tako da su u zajedničkoj vlastitoj podmreži i postavite im IP adrese po želji. Van kompozicije pokrenite kontejner temeljen na [slici curlimages/curl](https://hub.docker.com/r/curlimages/curl) u istoj mreži i njime pristupite jednom, a zatim drugom web sjedištu.

## Pokretanje WordPressa korištenjem Docker Composea

Postavit ćemo usluge koje zahtijeva web sjediše u WordPressu: web poslužitelj, interpreter programskog jezika PHP i bazu podataka MariaDB. Na Docker Hubu već postoji [službena slika WordPressa](https://hub.docker.com/_/wordpress) koji, uz programski kod WordPressa, uključuje i web poslužitelj i interpreter programskog jezika PHP. Tako da nam treba samo još [službena slika MariaDB-a](https://hub.docker.com/_/mariadb).

Napravimo novi direktorij `wordpress`. U njemu kreirajmo novu `docker-compose.yml` datoteku koja će sadržavati dva kontejnera potrebna za WordPress:

``` yaml
version: "3.8"
services:
  wordpress:
    image: wordpress
    ports:
      - "5050:80"
  mariadb:
    image: mariadb
```

Sada je potrebno definirati odjeljak s parametrima okruženja `environment:` koje sadrži parametre specifične za WordPress. Specifično, navesti ćemo sve informacije o bazi podataka na koju će se povezati (varijable `WORDPRESS_DB_HOST`, `WORDPRESS_DB_USER` i dr.). Također ćemo definirati varijable okruženja i za bazu podataka MariaDB na analogni način:

``` yaml
version: "3.8"
services:
  wordpress:
    image: wordpress
    ports:
      - "5050:80"
    environment:
      WORDPRESS_DB_HOST: mariadb
      WORDPRESS_DB_USER: root
      WORDPRESS_DB_PASSWORD: "ide-gas!123"
      WORDPRESS_DB_NAME: wordpress
  mariadb:
    image: mariadb
    environment:
      MARIADB_DATABASE: wordpress
      MARIADB_ROOT_PASSWORD: "ide-gas!123"
```

Sada vidimo da se instanca WordPressa koju pokrećemo povezuje na uslugu MariaDB i koristi je kao svoj sustav za upravljanje bazom podataka. Korisnik je u ovom jednostavnom primjeru `root`, zaporka je `ide-gas!123`, a baza podataka koja se koristi je `wordpress`.

!!! warning
    U nastavku (na složenijim primjerima i u zadacima) ćemo izbjegavati konfigurirati web aplikacije da pristupaju sustavu za upravljanje bazom podataka korištenjem njegovog korijenskog korisnika jer je to loša sigurnosna praksa. Naime, korijenski korisnik ima sve ovlasti.

Dodajmo još zavisnost WordPressu o MariaDB-u u ključu `depends_on`. Docker Compose će zbog toga napraviti i pokrenuti kontejner `mariadb` prije pokretanja kontejnera `wordpress`.

Također ćemo dodati odjeljak `volumes`, koji će nam omogućiti da mapiramo Docker direktorij u direktorij u našem sustavu. Na taj način ako se Docker spremink sruši ili obriše, bez obzira na direktorij, podaci će i dalje biti na domaćinu. Stvorit ćemo volumen imena `wordpress_db` i preslikati to unutar kontejnera.

Dodatno možemo eksplicitno navesti mrežu `wpnet` koja koristi skup adresa 172.19.0.0/16, a u njoj MariaDB ima adresu 172.19.0.3, dok WordPressov poslužitelj ima 172.19.0.4.

``` yaml
version: "3.8"
services:
  wordpress:
    image: wordpress
    ports:
      - "5050:80"
    depends_on:
      - mariadb
    environment:
      WORDPRESS_DB_HOST: mariadb
      WORDPRESS_DB_USER: root
      WORDPRESS_DB_PASSWORD: "ide-gas!123"
      WORDPRESS_DB_NAME: wordpress
    networks:
      mreza:
        ipv4_address: 172.19.0.3
  mariadb:
    image: mariadb
    environment:
      MARIADB_DATABASE: wordpress
      MARIADB_ROOT_PASSWORD: "ide-gas!123"
    volumes:
      - ./wordpress_db:/var/lib/mariadb
    networks:
      mreza:
        ipv4_address: 172.19.0.4
networks:
  mreza:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.19.0.0/16
```

## Korištenje izgradnje kontejnera u kompoziciji

U kompoziciji je moguće izgraditi neke od kontejnera na temelju datoteka `Dockerfile`. Za ilustraciju možemo iskoristiti primjer dan u [Get started with Docker Compose](https://docs.docker.com/compose/gettingstarted/), web aplikaciju napisanu u Pythonu korištenjem [Flaska](https://flask.palletsprojects.com/) koja dohvaća podatke iz [Redisa](https://redis.io/).

Prvo napravite direktorij `projekt` i u njemu kreirajte datoteku `app.py` sadržaja:

``` python
import time

import redis
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host='mojredis', port=6379)

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

@app.route('/')
def hello():
    count = get_hit_count()
    return 'Hello World! I have been seen {} times.\n'.format(count)
```

Uočimo kako se povezivanje na Redis događa korištenjem imena domaćina `mojredis`. To ćemo ime iskoristiti niže kao ime kontejnera.

Jasno je vidljivo iz naredbi `import` da ova aplikacija za svoj rad zahtijeva Python module `flask` i `redis` (podsjetimo se da je modul `time` [dio](https://docs.python.org/3/library/time.html) [standardne biblioteke](https://docs.python.org/3/library/index.html)). Stvorimo datoteku `requirements.txt` koju će [pip](https://pip.pypa.io/) koristiti za preuzimanje potrebnih modula:

```
flask
redis
```

Stvorimo `Dockerfile` koji će povući sliku kontejnera za Python, dodati u nju datoteku `requirements.txt`, instalirati potrebne module (`pip install -r requirements.txt`), pokrenuti aplikaciju (`flask run`) i otvoriti potrebna vrata (u zadanim postavkama Flask koristi vrata 5000):

``` dockerfile
FROM python:3.7-alpine
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run"]
```

Ovako definiran `Dockerfile` govori Dockeru da redom:

- izgradi sliku počevši od [slike Python, verzije 3.7-alpine](https://hub.docker.com/_/python/),
- postavi radni direktorij na `/code`,
- postavi varijable okruženja koje koristi naredba `flask`,
- instalira potrebne zavisnosti `gcc`, `musl-dev` i `linux-headers` naredbom `apk` koju Alpine Linux [koristi za instalaciju paketa](https://docs.alpinelinux.org/user-handbook/0.1a/Working/apk.html), slično kao što Debian GNU/LInux koristi `apt`, a Arch Linux `pacman`,
- kopira `requirements.txt` i instalira Pythonove zavisnosti,
- doda metapodatke na sliku kako bi opisao da kontejner sluša na portu 5000,
- kopira trenutni direktorij `.` u projektu u radni direktorij `.` na slici te
- postavi zadanu naredbu za kontejner na `flask run`.

Ovaj kontejner bismo mogli izgraditi naredbom `docker build` ([dokumentacija](https://docs.docker.com/engine/reference/commandline/build/)), ali tada bismo se morali pobrinuti za Redis. Stoga ćemo napraviti `docker-compose.yml` koji će izgraditi kontejner na temelju `Dockerfile`-a i pokrenuti ga, izraditi i pokrenuti kontejner s Redisom pod nazivom `mojredis` te povezati ta dva kontejnera mrežom:

``` yaml
version: "3.9"
services:
  web:
    build: .
    ports:
      - "5000:5000"
  mojredis:
    image: "redis:alpine"
```

!!! adomonition "Zadatak"
    Promijenite kontejner tako da koristi Python 3.10 kao osnovnu sliku.

!!! adomonition "Zadatak"
    Promijenite korištena vrata na 8080. (*Uputa:* proučite [dokumentaciju sučelja naredbenog retka Flaska](https://flask.palletsprojects.com/en/2.0.x/cli/).)

!!! adomonition "Zadatak"
    Po uzoru na ovu kompoziciju, složite kompoziciju koja koristi službenu sliku za [php](https://hub.docker.com/_/php) i povezuje se na Redis.

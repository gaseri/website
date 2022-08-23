---
author: Vedran Miletić
---

# Komponiranje kontejnera alatom Docker Compose

[Docker Compose](https://docs.docker.com/compose/) ([GitHub](https://github.com/docker/compose)) je alat za definiranje i pokretanje višekontejnerskih aplikacija korištenjem Dockera, tzv. kompozicija (engl. *composition*). Usluge koje će se pokrenuti definiraju se korištenjem konfiguracije u obliku [YAML](https://yaml.org/). YAML je donekle srodan [JSON](https://www.json.org/)-u, ali je dizajniran kako bi bio čitljiv ljudima (slično kao [TOML](https://toml.io/)).

!!! tip
    [Službeno proširenje za Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) za Visual Studio Code, između ostalog, podržava i [korištenje Docker Composea](https://code.visualstudio.com/docs/containers/docker-compose).

## Struktura konfiguracije

!!! warning
    Ubuntu 20.04 LTS ima paket [docker-compose](https://packages.ubuntu.com/focal/docker-compose) koji sadrži Docker Compose verziju 1.25.0. Ta verzija podržava sve značajke koje u nastavku koristimo, ali potrebno je pripaziti da polje `version` u YAML datoteci u nastavku bude postavljeno na 3.7 (umjesto 3.9) jer je [verzija konfiguracije 3.8 postala dostupna tek u Docker Composeu 1.25.5, a 3.9 u 1.27.1](https://github.com/docker/compose/blob/master/CHANGELOG.md).

Datoteka koju Docker Compose koristi ima ime `docker-compose.yml` i ona može biti oblika:

``` yaml
version: "3.9"
services:
  mojhttpd:
    image: httpd
  mojredis:
    image: redis
```

U ovom slučaju pokrenut ćemo dva kontejnera, prvi naziva `mojhttpd` temeljen na službenoj slici [httpd](https://hub.docker.com/_/httpd), a drugi `mojredis` temeljen na službenoj slici [redis](https://hub.docker.com/_/redis).

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
⠿ Container korisnik-mojhttpd-1  Created                                   0.0s
⠿ Container korisnik-mojredis-1  Created                                   0.0s
Attaching to korisnik-mojhttpd-1, korisnik-mojredis-1
korisnik-mojhttpd-1  | AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.18.0.2. Set the 'ServerName' directive globally to suppress this message
korisnik-mojhttpd-1  | AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.18.0.2. Set the 'ServerName' directive globally to suppress this message
korisnik-mojredis-1  | 1:C 03 Jan 2022 22:26:37.086 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
korisnik-mojredis-1  | 1:C 03 Jan 2022 22:26:37.086 # Redis version=6.2.6, bits=64, commit=00000000, modified=0, pid=1, just started
korisnik-mojredis-1  | 1:C 03 Jan 2022 22:26:37.086 # Warning: no config file specified, using the default config. In order to specify a config file use redis-server /path/to/redis.conf
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:26:37.087 * monotonic clock: POSIX clock_gettime
korisnik-mojhttpd-1  | [Mon Jan 03 22:26:37.087474 2022] [mpm_event:notice] [pid 1:tid 140323163139392] AH00489: Apache/2.4.51 (Unix) configured -- resuming normal operations
korisnik-mojhttpd-1  | [Mon Jan 03 22:26:37.087533 2022] [core:notice] [pid 1:tid 140323163139392] AH00094: Command line: 'httpd -D FOREGROUND'
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:26:37.087 * Running mode=standalone, port=6379.
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:26:37.087 # Server initialized
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:26:37.087 # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 * Loading RDB produced by version 6.2.6
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 * RDB age 55 seconds
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 * RDB memory usage when created 0.77 Mb
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 # Done loading RDB, keys loaded: 0, keys expired: 0.
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 * DB loaded from disk: 0.000 seconds
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:26:37.088 * Ready to accept connections
```

Izvođenje, kao i inače, prekidamo kombinacijom tipki ++control+c++:

``` shell
^CGracefully stopping... (press Ctrl+C again to force)
[+] Running 2/2
⠿ Container korisnik-mojhttpd-1  Stopped                                   1.1s
⠿ Container korisnik-mojredis-1  Stopped                                   0.2s
canceled
```

Želimo li pokrenuti kompoziciju kontejnera u pozadini, dodat ćemo parametar `--detach`, odnosno `-d`:

``` shell
$ docker-compose up -d
[+] Running 2/2
⠿ Container korisnik-mojredis-1  Started                                   0.3s
⠿ Container korisnik-mojhttpd-1  Started                                   0.3s
```

Argumentom `ps` dobit ćemo popis pokrenutih procesa u pojedinim kontejnerima ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_ps/)):

``` shell
$ docker-compose ps
NAME                 COMMAND                  SERVICE             STATUS              PORTS
korisnik-mojhttpd-1   "httpd-foreground"       mojhttpd            running             80/tcp
korisnik-mojredis-1   "docker-entrypoint.s…"   mojredis            running             6379/tcp
```

Želimo li više informacija, primjerice koliko je točno procesa pokrenuo web poslužitelj, koliko se dugo izvode i koji su im identifikatori, ispisat ćemo ih argumentom `top`, ponovno razdvojene po pojedinim kontejnerima ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_top/)):

``` shell
$ docker-compose top
korisnik-mojhttpd-1
UID    PID      PPID     C    STIME   TTY   TIME       CMD
root   870026   869946   0    23:34   ?     00:00:00   httpd -DFOREGROUND
bin    870323   870026   0    23:34   ?     00:00:00   httpd -DFOREGROUND
bin    870324   870026   0    23:34   ?     00:00:00   httpd -DFOREGROUND
bin    870325   870026   0    23:34   ?     00:00:00   httpd -DFOREGROUND

korisnik-mojredis-1
UID   PID      PPID     C    STIME   TTY   TIME       CMD
999   870038   869998   0    23:34   ?     00:00:00   redis-server *:6379
```

Argumentom `logs` dobit ćemo uvid u zapisnike pojedinih kontejnera ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_logs/)):

``` shell
$ docker-compose logs
korisnik-mojhttpd-1  | AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.19.0.2. Set the 'ServerName' directive globally to suppress this message
korisnik-mojhttpd-1  | AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.19.0.2. Set the 'ServerName' directive globally to suppress this message
korisnik-mojhttpd-1  | [Mon Jan 03 22:34:07.376872 2022] [mpm_event:notice] [pid 1:tid 140419205336384] AH00489: Apache/2.4.51 (Unix) configured -- resuming normal operations
korisnik-mojhttpd-1  | [Mon Jan 03 22:34:07.377246 2022] [core:notice] [pid 1:tid 140419205336384] AH00094: Command line: 'httpd -D FOREGROUND'
korisnik-mojredis-1  | 1:C 03 Jan 2022 22:34:07.376 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
korisnik-mojredis-1  | 1:C 03 Jan 2022 22:34:07.376 # Redis version=6.2.6, bits=64, commit=00000000, modified=0, pid=1, just started
korisnik-mojredis-1  | 1:C 03 Jan 2022 22:34:07.376 # Warning: no config file specified, using the default config. In order to specify a config file use redis-server /path/to/redis.conf
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:34:07.376 * monotonic clock: POSIX clock_gettime
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:34:07.377 * Running mode=standalone, port=6379.
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:34:07.377 # Server initialized
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:34:07.377 # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
korisnik-mojredis-1  | 1:M 03 Jan 2022 22:34:07.377 * Ready to accept connections
```

Zaustavljanje izvođenja izvodimo argumentom `down` ([dokumentacija](https://docs.docker.com/engine/reference/commandline/compose_down/))

``` shell
$ docker-compose down
[+] Running 3/3
⠿ Container korisnik-mojhttpd-1  Removed                                   1.3s
⠿ Container korisnik-mojredis-1  Removed                                   0.3s
⠿ Network korisnik_default       Removed                                   0.1s
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

## Korištenje izgradnje kontejnera u kompoziciji

U kompoziciji je moguće izgraditi neke od kontejnera na temelju datoteka `Dockerfile`. Za ilustraciju možemo iskoristiti primjer dan u [Get started with Docker Compose](https://docs.docker.com/compose/gettingstarted/), web aplikaciju napisanu u Pythonu korištenjem [Flaska](https://flask.palletsprojects.com/) koja dohvaća podatke iz [Redisa](https://redis.io/). Datoteka `app.py` je sadržaja:

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

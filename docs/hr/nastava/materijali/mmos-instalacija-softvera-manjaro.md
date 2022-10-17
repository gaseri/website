---
author: Matea Turalija
---

# Instalacija i konfiguracija softvera za vježbe iz kolegija Mrežni i mobilni operacijski sustavi te Infrastruktura za podatke velikog obujma

Upute u nastavku su pisane za [Manjaro](https://manjaro.org/).

Alternativno, Docker i Docker Compose je moguće koristiti na Windowsima 10 i 11 prema [službenim uputama](https://docs.docker.com/desktop/install/windows-install/).

## Docker

Provjerite je li sustav ažuran pomoću sljedeće naredbe:

``` shell
$ pamac upgrade -a
(...)
```

Nakon instalacije nadogradnji ponovno pokrenite sustav ako je potrebno (tj. ako je nadograđena jezgra).

Instalirajte Docker sljedećom naredbom:

``` shell
$ pamac install docker
(...)
```

Zatim ćete dobit upit želite li nastaviti s instalacijom, odaberite potvrdno.

Nakon što je instalacija dovršena, uključite pokretanje usluge Dockera korištenjem mrežne utičnice:

``` shell
$ sudo systemctl enable --now docker.socket
```

Pokušajte pokrenuti kontejner temeljen na slici `hello-world` naredbom `docker run` i uočite grešku:

``` shell
$ docker run hello-world
docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/create": dial unix /var/run/docker.sock: connect: permission denied.
See 'docker run --help'.
```

Zatim dodajte svog korisnika u grupu `docker` koja ima pravo pokretanja kontejnera:

``` shell
$ sudo usermod -aG docker $USER
```

Kontejnere ćete moći pokretati nakon ponovne prijave. Najjednostavniji način da se odjavite je:

``` shell
$ loginctl kill-user $USER
```

## Docker Compose

Instalirajte Docker Compose sljedećom naredbom:

``` shell
$ pamac install docker-compose
(...)
```

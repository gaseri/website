---
marp: true
author: Vedran Miletić
title: Kontejnerizacija web aplikacija
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Kontejnerizacija web aplikacija

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Kontejnerizacija

Prema [Wikipediji](https://en.wikipedia.org/wiki/OS-level_virtualization):

* virtualizacija na razini operacijskog sustava (engl. *OS-level virtualization*) je paradigma operacijskog sustava u kojoj jezgra omogućuje postojanje više izoliranih instanci korisničkog prostora
* s gledišta programa koji se u njima izvode mogu izgledati kao prava računala
    - računalni program pokrenut na uobičajenom operativnom sustavu može vidjeti sve resurse (povezane uređaje, datoteke i mape, mrežne dionice, snagu procesora, mjerljive hardverske mogućnosti) tog računala
    - programi koji se izvode u kontejneru mogu vidjeti samo sadržaj i uređaje dodijeljene kontejneru

---

## Kontejnerizacijske tehnologije

* kontejneri (engl. *containers*) ([LXC](https://linuxcontainers.org/), [Docker](https://www.docker.com/))
* zone (engl. *zones*) ([Solaris](https://www.oracle.com/solaris/)/[illumos](https://illumos.org/))
* virtualni privatni poslužitelji (engl. *virtual private servers*) ([OpenVZ](https://openvz.org/))
* particije (engl. *partitions*)
* virtualna okruženja (engl. *virtual enviroments*) ([virtualenv](https://virtualenv.pypa.io/))
* virtualne jezgre (engl. *virtual kernels*) ([DragonFly BSD](https://www.dragonflybsd.org/))
* zatvori (engl. *jails*) ([FreeBSD](https://www.freebsd.org/), [chroot jail](https://en.wikipedia.org/wiki/Chroot) pod Linuxom)

---

## Docker

Prema [Wikipediji](https://en.wikipedia.org/wiki/Docker_(software)):

* skup proizvoda platforme kao usluge (PaaS) koji koriste virtualizaciju na razini OS-a za isporuku softvera u kontejnerima
* kontejneri su međusobno izolirani i donose vlastiti softver, knjižnice i konfiguracijske datoteke; mogu međusobno komunicirati kroz precizno definirane kanale
* svi kontejneri dijele usluge jedne jezgra OS-a pa koriste manje resursa od virtualnih strojeva

![Docker Linux interfaces bg right:45% 95%](https://upload.wikimedia.org/wikipedia/commons/0/09/Docker-linux-interfaces.svg)

---

## Primjeri Docker kontejnera

Svi navedeni su službeno podržane slike dostupne na [Docker Hubu](https://hub.docker.com/):

* [Ubuntu](https://hub.docker.com/_/ubuntu)
* [Node.js](https://hub.docker.com/_/node)
* [Python](https://hub.docker.com/_/python)
* [MongoDB](https://hub.docker.com/_/mongo)
* [PyPy](https://hub.docker.com/_/pypy)
* [Memcached](https://hub.docker.com/_/memcached)
* [Redmine](https://hub.docker.com/_/redmine)
* [PostgreSQL](https://hub.docker.com/_/postgres)
* [Elasticsearch](https://hub.docker.com/_/elasticsearch)
* [Rust](https://hub.docker.com/_/rust)

---

## Kompozicije kontejnera

[Docker Compose](https://docs.docker.com/compose/) omogućuje konfiguriranje i pokretanje grupe kontejnera,  njihovo međusobno povezivanje mrežom itd. Primjerice, možemo komponirati:

* Web poslužitelj
* Interpreter skriptnog jezika
* Balanser opterećenja
* Sustav za upravljanje (relacijskom) bazom podataka
* Sustav za predmemeoriju

![Docker Linux interfaces bg right 60%](https://www.unixmen.com/wp-content/uploads/2017/06/docker-compose-logo.png)

---

## Primjer kompozicije kontejnera

``` dockerfile
FROM python:3.10-alpine
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

``` yaml
services:
  web:
    build: .
    ports:
      - "5000:5000"
  redis:
    image: "redis:alpine"
```

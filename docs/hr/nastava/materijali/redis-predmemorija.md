---
author: Vedran Miletić
---

# Konfiguracija sustava za predmemoriju Redis

[Redis](https://redis.io/) je sustav za spremanje struktura podataka u memoriji koji se može koristiti kao baza podataka (alternativa: [MongoDB](https://www.mongodb.com/)), predmemorija (alternativa: [Memcached](https://www.memcached.org/)) i broker za poruke (alternativa: [RabbitMQ](https://rabbitmq.com/)). Mi u nastavku koristimo redis kao predmemoriju, slično kako to rade [Django](https://docs.djangoproject.com/en/4.0/topics/cache/#redis) i [Laravel](https://laravel.com/docs/8.x/redis).

## Instalacija i pokretanje

Na Ubuntuu Redis instaliramo naredbom:

``` shell
$ sudo apt install redis
```

Nakon instalacije uvjerimo se da se Redis poslužitelj pokrenuo naredbom:

``` shell
$ systemctl status redis-server.service
```

## Povezivanje klijentom i osnovne operacije

Povežimo se na Redis poslužitelj klijentom naredbenog retka `redis-cli` ([dokumentacija](https://redis.io/topics/rediscli)):

``` shell
$ redis-cli
```

Uočimo da se Redis klijent povezao na adresu 127.0.0.1 i vrata 6379 (to ćemo kasnije koristiti kad se budemo povezivali iz interpretera programskog jezika).

Predmemorija ima dvije operacije, `SET` ([dokumentacija](https://redis.io/commands/set)) za postavljanje vrijednosti ključa i `GET` ([dokumentacija](https://redis.io/commands/get)) za dohvaćanje vrijednosti ključa. Naredbu SET koristimo navođenjem ključa i vrijednosti na način:

``` redis
127.0.0.1:6379> SET "name:1" "Tomislav"
"OK"
```

Ključ je proizvoljni niz znakova, a mi smo u ovom primjeru odlučili numerirati imena brojkom iza znaka dvotočke. Naredbom `GET` možemo dohvatiti postavljenu vrijednost i uvjeriti se da je uspješno bila postavljena:

``` redis
127.0.0.1:6379> GET "name:1"
"Tomislav"
```

Pokušamo li dohvatiti vrijednost ključa koji ne postoji, Redis će nam vratiti `nil`:

``` redis
127.0.0.1:6379> GET "name:2"
(nil)
```

## Konfiguracija dozvole pristupa iz interpretera programskog jezika

Pristup iz interpretera programskog jezika koji se izvodi na istom računalu kao i Redis poslužitelj je već sada omogućen. Međutim, puno je češća situacija da će sustav za predmemoriju biti na odvojenom virtualnom stroju. Zbog toga isključimo zaštićeni način rada ([dokumentacija](https://redis.io/topics/security#protected-mode)) koji ograničava pristup Redis poslužitelju iz vanjskih mreža. To ćemo učiniti naredbom `CONFIG SET` ([dokumentacija](https://redis.io/commands/config-set)):

``` redis
127.0.0.1:6379> CONFIG SET protected-mode no
OK
```

Kasnije ćemu ovu promjenu postavki trajno zapisati u konfiguracijskoj datoteci.

Uočimo da Redis poslužitelj sluša samo na localhost adresi (127.0.0.1 za IPv4, ::1 za IPv6). Ako nemamo povjerenja u Redis klijent, možemo se u to uvjeriti naredbom ljuske `ss` (alat sličan `netstat`-u i moderna zamjena za isti) na način:

``` shell
$ ss -l | grep 6379
```

Mi bismo htjeli da sluša na svim adresama kako bismo mu mogli pristupiti s drugog virutalnog stroja, dakle trebamo postaviti adresu na 0.0.0.0 za IPv4 i :: za IPv6. Uočimo u konfiguracijskoj datoteci `/etc/redis/redis.conf` konfiguracijsku naredbu `bind`:

```
################################## NETWORK #####################################

# By default, if no "bind" configuration directive is specified, Redis listens
# for connections from all the network interfaces available on the server.
# It is possible to listen to just one or multiple selected interfaces using
# the "bind" configuration directive, followed by one or more IP addresses.
#
# Examples:
#
# bind 192.168.1.100 10.0.0.1
# bind 127.0.0.1 ::1
#
# ~~~ WARNING ~~~ If the computer running Redis is directly exposed to the
# internet, binding to all the interfaces is dangerous and will expose the
# instance to everybody on the internet. So by default we uncomment the
# following bind directive, that will force Redis to listen only into
# the IPv4 loopback interface address (this means Redis will be able to
# accept connections only from clients running into the same computer it
# is running).
#
# IF YOU ARE SURE YOU WANT YOUR INSTANCE TO LISTEN TO ALL THE INTERFACES
# JUST COMMENT THE FOLLOWING LINE.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bind 127.0.0.1 ::1

# Protected mode is a layer of security protection, in order to avoid that
# Redis instances left open on the internet are accessed and exploited.
#
# When protected mode is on and if:
#
# 1) The server is not binding explicitly to a set of addresses using the
#    "bind" directive.
# 2) No password is configured.
#
# The server only accepts connections from clients connecting from the
# IPv4 and IPv6 loopback addresses 127.0.0.1 and ::1, and from Unix domain
# sockets.
#
# By default protected mode is enabled. You should disable it only if
# you are sure you want clients from other hosts to connect to Redis
# even if no authentication is configured, nor a specific set of interfaces
# are explicitly listed using the "bind" directive.
protected-mode yes
```

Redis poslužitelj već predviđa da će korisnicima u nekim slučajevima odgovarati da poslužitelj sluša na svim mrežnim sučeljima i dovoljno je zakomentirati liniju tako da bude oblika:

```
# bind 127.0.0.1 ::1
```

Dodatno ćemo isključiti zaštićeni način rada promjenom linije `protected-mode` iz:

```
protected-mode yes
```

u

```
protected-mode no
```

Ponovno pokrenim poslužitelj:

``` shell
$ sudo systemctl restart redis-server.service
```

## Pristup sa drugog poslužitelja

Na računalu (ili više njih) s web poslužiteljem i interpreterom programskog jezika instalirajmo klijent kojim ćemo provjeriti da je pristup omogućen:

``` shell
$ sudo apt install redis-tools
```

Povežimo se na poslužitelj navođenjem njegove IP adrese korištenjem parametra `-h` ([dokumentacija](https://redis.io/topics/rediscli#host-port-password-and-database)). Ako je Redis poslužitelj na adresi 192.168.122.34, naredba je oblika:

``` shell
$ redis-cli -h 192.168.122.34
```

Nakon povezivanja uvjerimo se da možemo dohvatiti ranije postavljenu vrijednost:

``` redis
192.168.122.34:6379> GET "name:1"
"Tomislav"
```

Klijentske biblioteke za pristup Redisu iz PHP-a su [Predis](https://github.com/predis/predis/wiki) ([GitHub](https://github.com/predis/predis), [Packagist](https://packagist.org/packages/predis/predis)) i [PhpRedis](https://github.com/phpredis/phpredis). Možemo se lako uvjeriti da Ubuntu nudi obje za instalaciju:

``` shell
$ sudo apt search predis
Sorting... Done
Full Text Search... Done
golang-github-stvp-tempredis-dev/focal 0.0~git20160122.0.83f7aae-2 all
  Go package to start and stop temporary redis-server processes

libphp-predis/focal 0.8.3-1ubuntu1 amd64
  Flexible and feature-complete PHP client library for the Redis key-value store

php-nrk-predis/focal 1.0.0-1 amd64
  Flexible and feature-complete PHP client library for the Redis key-value store
```

Instalirat ćemo Predis jer je nešto jednostavniji za korištenje:

``` shell
$ sudo apt install php-nrk-predis
```

Uredimo `index.php` tako da bude oblika:

``` php
<?php

// Prepend a base path if Predis is not available in your "include_path".
// require 'Predis/Autoloader.php';
// Ubuntu ima Predis u direktoriju php-nrk-predis pa prilagodimo
require 'php-nrk-predis/Autoloader.php';

Predis\Autoloader::register();

// navodi se protokol (tcp://), IP adresa i vrata poslužitelja
$client = new Predis\Client('tcp://192.168.122.34:6379');

$value = $client->get('name:2');
if (isset($value)) {
    // uspješno smo iz predmemorije dohvatili vrijednost
    echo $value . ' (dohvaćen iz cachea)';
} else {
    // u predmemoriji nema te vrijednosti pa ćemo je dohvatiti iz baze pdoataka
    $mysqli = new mysqli('localhost', 'my_user', 'my_password', 'my_database');

    $result = $mysqli->query('SELECT name FROM people WHERE id = 2');
    $row = $result->fetch_array();
    $value = $row['name'];

    echo $value . ' (dohvaćen iz baze)';

    // spremimo vrijednost u predmemoriju za buduću upotrebu
    $client->set('name:2', $value);
}
```

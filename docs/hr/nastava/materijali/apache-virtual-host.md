---
author: Vedran Miletić
---

# Konfiguracija virtualnih domaćina u web poslužitelju Apache HTTP Server

Konfiguracijska naredba `<VirtualHost>` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/core.html#virtualhost)) omogućuje Apacheju da poslužuje više različitih web sjedišta koja se nalaze na više različitih domena putem jedne IP adrese i vrata.

## Primjer primjene virtualnih domaćina

Ilustracije radi, promotrimo web sjedišta `www.math.uniri.hr`, `www.phy.uniri.hr` i `www.biotech.uniri.hr`.

``` shell
$ curl -v -I http://www.phy.uniri.hr/hr/
*   Trying 193.198.209.33:80...
* Connected to www.phy.uniri.hr (193.198.209.33) port 80 (#0)
> HEAD /hr/ HTTP/1.1
> Host: www.phy.uniri.hr
> User-Agent: curl/7.70.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
(...)

$ curl -v -I http://www.math.uniri.hr/hr/
*   Trying 193.198.209.33:80...
* Connected to www.math.uniri.hr (193.198.209.33) port 80 (#0)
> HEAD /hr/ HTTP/1.1
> Host: www.math.uniri.hr
> User-Agent: curl/7.70.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
(...)

$ curl -v -I http://www.biotech.uniri.hr/hr/
*   Trying 193.198.209.33:80...
* Connected to www.biotech.uniri.hr (193.198.209.33) port 80 (#0)
> HEAD /hr/ HTTP/1.1
> Host: www.biotech.uniri.hr
> User-Agent: curl/7.70.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
(...)
```

Uočimo da se ova tri web sjedišta nalaze na istoj IP adresi (193.198.209.33) i istim vratima (80), ali poslužitelj zna točno koji sadržaj mora isporučiti u odgovoru zahvaljujući vrijednosti zaglavlja `Host` ([više detalja o HTTP zaglavlju Host na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Host)) u svakom od HTTP zahtjeva. Vrijednost tog zaglavlja se može postaviti u cURL-u kod slanja HTTP zahtjeva:

``` shell
$ curl --header "Host: www.math.uniri.hr" http://www.biotech.uniri.hr/hr/
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="hr-hr" lang="hr-hr" >
<head>
  <base href="http://www.math.uniri.hr/hr/" />
  <meta http-equiv="content-type" content="text/html; charset=utf-8" />
  <meta name="generator" content="Joomla! - Open Source Content Management" />
  <title>Odjel za matematiku - Vijesti</title>
(...)
```

## Konfiguracija virtualnih domaćina za HTTP

Recimo da imamo dvije domene, `prometej.rm.miletic.net` i `epimetej.rm.miletic.net` te da želimo na njih postaviti dva različita web sjedišta. Stvorimo sadržaj tih web sjedišta:

``` shell
$ mkdir -p www/{epimetej,prometej}/html
$ echo '<html><body><h1>Epimetej</h1></body></html>' > www/epimetej/html/index.html
$ echo '<html><body><h1>Prometej</h1></body></html>' > www/prometej/html/index.html
```

Apache koji koristimo već dolazi s konfiguracijskom datotekom za virtualne domaćine koju možemo prilagoditi i uključiti. Dohvatimo tu datoteku:

``` shell
$ docker run --rm httpd:2.4 cat /usr/local/apache2/conf/extra/httpd-vhosts.conf > my-httpd-vhosts.conf
```

Uredimo tu datoteku; uočimo da su nam konfiguracijske naredbe `ServerAdmin`, `ServerName` i `DocumentRoot` već poznate, samo se nalaze unutar bloka naredbi `<VirtualHost>`:

``` apacheconf
# Virtual Hosts
#
# Required modules: mod_log_config

# If you want to maintain multiple domains/hostnames on your
# machine you can setup VirtualHost containers for them. Most configurations
# use only name-based virtual hosts so the server doesn't need to worry about
# IP addresses. This is indicated by the asterisks in the directives below.
#
# Please see the documentation at
# <URL:http://httpd.apache.org/docs/2.4/vhosts/>
# for further details before you try to setup virtual hosts.
#
#...
<VirtualHost *:80>
    ServerAdmin webmaster@dummy-host.example.com
    DocumentRoot "/usr/local/apache2/docs/dummy-host.example.com"
    ServerName dummy-host.example.com
    ServerAlias www.dummy-host.example.com
    ErrorLog "logs/dummy-host.example.com-error_log"
    CustomLog "logs/dummy-host.example.com-access_log" common
</VirtualHost>

<VirtualHost *:80>
    ServerAdmin webmaster@dummy-host2.example.com
    DocumentRoot "/usr/local/apache2/docs/dummy-host2.example.com"
    ServerName dummy-host2.example.com
    ErrorLog "logs/dummy-host2.example.com-error_log"
    CustomLog "logs/dummy-host2.example.com-access_log" common
</VirtualHost>
```

Maknemo li konfiguracijsku naredbu `ServerAdmin`, za taj virtualni domaćin koristit će se ona vrijednost koju smo već ranije postavili u datoteci `my-httpd.conf`, što nam odgovara. Naredbu `ServerAlias` ne trebamo jer imamo samo jednu domenu po virtualnom domaćinu. Također, logging nam ovdje nije bitan pa ćemo i te dvije naredbe maknuti, čime smo eliminirali i potrebu za modulom [mod_log_config](https://httpd.apache.org/docs/2.4/mod/mod_log_config.html) koja se navodi u zaglavlju konfiguracijske datoteke u komentarima. Naposlijetku, moramo dodati dozvolu pristupa (`Require all granted`) za svaki od direktorija koji se koristi kao `DocumentRoot`. Konfiguracija (zasad samo za HTTP vrata 80) je oblika:

``` apacheconf
<VirtualHost *:80>
    DocumentRoot "/var/www/epimetej/html"
    <Directory "/var/www/epimetej/html">
        Require all granted
    </Directory>
    ServerName epimetej.rm.miletic.net
</VirtualHost>

<VirtualHost *:80>
    DocumentRoot "/var/www/prometej/html"
    <Directory "/var/www/prometej/html">
        Require all granted
    </Directory>
    ServerName prometej.rm.miletic.net
</VirtualHost>
```

Kad se koriste virtualni domaćini, tada prvi navedeni u konfiguracijskoj datoteci postaje zadani virtualni domaćin i koristi se kod odgovora na zahtjeve kod kojih vrijednost u zaglavlju `Host` primljenog HTTP zahtjeva ne odgovara ni jednoj vrijednosti konfiguracijske naredbe `ServerName`. Zato je dobra praksa eksplicitno prvo navesti zadani virtualni domaćin:

``` apacheconf
<VirtualHost *:80>
    DocumentRoot "/var/www/html"
</VirtualHost>

<VirtualHost *:80>
    DocumentRoot "/var/www/epimetej/html"
    <Directory "/var/www/epimetej/html">
        Require all granted
    </Directory>
    ServerName epimetej.rm.miletic.net
</VirtualHost>

<VirtualHost *:80>
    DocumentRoot "/var/www/prometej/html"
    <Directory "/var/www/prometej/html">
        Require all granted
    </Directory>
    ServerName prometej.rm.miletic.net
</VirtualHost>
```

U datoteci `my-httpd.conf` odkomentirajmo liniju koja naredbom `Include` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/core.html#include)) uključuje konfiguracijsku datoteku koju smo upravo uredili. Promijenimo:

``` apacheconf
#...
# Virtual hosts
#Include conf/extra/httpd-vhosts.conf
#...
```

tako da bude:

``` apacheconf
#...
# Virtual hosts
Include conf/extra/httpd-vhosts.conf
#...
```

Promijenimo `Dockerfile` tako da uključuje novu datoteku, dodajmo još jednu naredbu `COPY` koja će kopirati datoteku `my-httpd-vhosts.conf` u Docker kontejner na mjesto `conf/extra/httpd-vhosts.conf`:

``` dockerfile hl_lines="4"
FROM httpd:2.4
COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
COPY ./www /var/www
COPY ./my-httpd-vhosts.conf /usr/local/apache2/conf/extra/httpd-vhosts.conf
```

Izgradimo sliku i pokrenimo Docker kontejner:

``` shell
$ docker build -t "my-httpd:2.4-3" .
Sending build context to Docker daemon  54.78kB
Step 1/4 : FROM httpd:2.4
---> b2c2ab6dcf2e
Step 2/4 : COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
---> e475515ada38
Step 3/4 : COPY ./www /var/www
---> c764360e0255
Step 4/4 : COPY ./my-httpd-vhosts.conf /usr/local/apache2/conf/extra/httpd-vhosts.conf
---> 7839a9247066
Successfully built 7839a9247066
Successfully tagged my-httpd:2.4-3

$ docker run my-httpd:2.4-3
[Sun May 10 22:09:41.239474 2020] [mpm_event:notice] [pid 1:tid 139624574960768] AH00489: Apache/2.4.43 (Unix) configured -- resuming normal operations
[Sun May 10 22:09:41.239504 2020] [core:notice] [pid 1:tid 139624574960768] AH00094: Command line: 'httpd -D FOREGROUND'
```

Uvjerimo se da nam virtualni domaćini rade:

``` shell
$ curl --header "Host: prometej.rm.miletic.net" http://172.17.0.2/
<html><body><h1>Prometej</h1></body></html>
$ curl --header "Host: epimetej.rm.miletic.net" http://172.17.0.2/
<html><body><h1>Epimetej</h1></body></html>
```

Uvjerimo se da nam zadani virtualni domaćin hvata sve zahtjeve koji ne pašu na ova dva iznad:

``` shell
$ curl http://172.17.0.2/
<html><body><h1>Radi!</h1></body></html>
$ curl --header "Host: atlas.rm.miletic.net" http://172.17.0.2/
<html><body><h1>Radi!</h1></body></html>
```

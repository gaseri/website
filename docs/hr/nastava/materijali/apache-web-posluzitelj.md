---
author: Vedran Miletić
---

# Osnovna konfiguracija web poslužitelja Apache HTTP Server

!!! hint
    Uputa za Apache HTTP Server na internetu [ima](https://www.linode.com/docs/guides/web-servers/apache-tips-and-tricks/) [na](https://opensource.com/article/18/2/how-configure-apache-web-server) [pretek](https://blog.apnic.net/2020/04/07/the-wrong-certificate-apache-lets-encrypt-and-openssl/). Za temeljit uvod povrh ovih vježbi preporučam službenu dokumentaciju posljednje dvije verzije [Red Hat Enterprise Linuxa](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux), i to:

    - za verziju 8: [Chapter 1. Setting up the Apache HTTP web server](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/deploying_different_types_of_servers/setting-apache-http-server_deploying-different-types-of-servers) u [Deploying different types of servers](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/deploying_different_types_of_servers/index) ili
    - za verziju 7: [Chapter 14. Web Servers](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/ch-web_servers) u [Red Hat Enterprise Linux 7 System Administrator's Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/index).

    Obje verzije Red Hat Enterprise Linuxa imaju Apache 2.4 koji i koristimo u nastavku.

[Apache HTTP Server](https://httpd.apache.org/), kolokvijalno samo Apache, je HTTP poslužitelj namijenjen za moderne operacijske sustave slične Unixu i Windowse. Cilj projekta koji ga razvija je ponuditi siguran, efikasan i proširiv poslužitelj koji poštuje aktualne standarde HTTP-a:

- [RFC 1945: Hypertext Transfer Protocol -- HTTP/1.0](https://datatracker.ietf.org/doc/html/rfc1945)
- [RFC 2616: Hypertext Transfer Protocol -- HTTP/1.1](https://datatracker.ietf.org/doc/html/rfc2616)

    - [RFC 7230: Hypertext Transfer Protocol (HTTP/1.1): Message Syntax and Routing](https://datatracker.ietf.org/doc/html/rfc7230)
    - [RFC 7231: Hypertext Transfer Protocol (HTTP/1.1): Semantics and Content](https://datatracker.ietf.org/doc/html/rfc7231)
    - [RFC 7232: Hypertext Transfer Protocol (HTTP/1.1): Conditional Requests](https://datatracker.ietf.org/doc/html/rfc7232)
    - [RFC 7233: Hypertext Transfer Protocol (HTTP/1.1): Range Requests](https://datatracker.ietf.org/doc/html/rfc7233)
    - [RFC 7234: Hypertext Transfer Protocol (HTTP/1.1): Caching](https://datatracker.ietf.org/doc/html/rfc7234)
    - [RFC 7235: Hypertext Transfer Protocol (HTTP/1.1): Authentication](https://datatracker.ietf.org/doc/html/rfc7235)

- [RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2)](https://datatracker.ietf.org/doc/html/rfc7540)

Razvoj projekta podržava [zaklada Apache Software Foundation](https://www.apache.org/), najveća zaklada za razvoj slobodnog softvera otvorenog koda na svijetu. Projekt započinje 1995. godine i inicijalno je zasnovan na izvornom kodu poslužitelju [NCSA HTTPd](https://en.wikipedia.org/wiki/NCSA_HTTPd), ali vrlo brzo taj poslužitelj zastaje s razvojem i korisnici prelaze na Apache. Ime `httpd` (HTTP daemon) se održalo i do današnjih dana kao ime naredbe kojom se poslužitelj pokreće. Od travnja 1996. je [najpopularniji web poslužitelj na internetu](https://news.netcraft.com/archives/2019/04/22/april-2019-web-server-survey.html), a u odgovoru na pitanje je li to još uvijek ili ga je prestigao [nginx](https://www.nginx.com/) ne slažu se statistike od [W3Techs koji tvrdi da je još uvijek najpopularniji Apache](https://w3techs.com/technologies/overview/web_server) i [Netcrafta koji tvrdi da je najpopularniji nginx](https://news.netcraft.com/archives/2020/04/08/april-2020-web-server-survey.html); treći po popularnosti je [Microsoft Internet Information Services](https://en.wikipedia.org/wiki/Internet_Information_Services).

!!! tip
    Značenje pojma web poslužitelj je dvojako: može se raditi o softveru kao što je Apache HTTP Server ili nginx, a može u pitanju biti i hardver koji taj softver izvodi. Za ilustraciju kako u praksi takav hardver uglavnom izgleda, možete pogledati [recenziju Dell EMC PowerEdge R640 na ServeTheHome](https://www.servethehome.com/dell-emc-poweredge-r640-review-a-study-in-1u-design-excellence/), a za ilustraciju kako izgleda podatkovni centar prosječnog popularnog web sjedišta koji koristi više web poslužitelja [fotografiju iz podatkovnog centra Wikimedia Foundationa](https://en.wikipedia.org/wiki/Web_server#/media/File:Wikimedia_Foundation_Servers-8055_35.jpg).

Aktualna stabilna verzija Apache HTTP Servera je 2.4, a prethodna verzija 2.2 više nije podržana od kraja 2017. godine. Verzija 2.4 je prvi put postala dostupna u veljači 2012. godine i [od tada redovno dobiva nove značajke u novim izdanjima](https://httpd.apache.org/docs/2.4/new_features_2_4.html). Primjerice, od verzije 2.4.17 postoji podrška za protokol HTTP/2 ([modul mod_http2](https://httpd.apache.org/docs/2.4/mod/mod_http2.html)), a od verzije 2.4.30 podrška za automatsko potpisivanje SSL/TLS certifikata korištenjem [protokola ACME](https://letsencrypt.org/how-it-works/) ([modul mod_md](https://httpd.apache.org/docs/2.4/mod/mod_md.html)). Apache ima vrlo detaljnu [službenu dokumentaciju](https://httpd.apache.org/docs/2.4/).

## Slika `httpd` na Docker Hubu

Apache možemo instalirati kao i svaki drugi program i pokretati kao i svaki drugi daemon. Ipak, kako ćemo u nastavku eksperimentirati s različitim postavkama i kopirati datoteke u sustavske direktorije gdje ih Apache čita, iskoristit ćemo [Docker](https://www.docker.com/) za pokretanje da se ne igramo s vlastitim operacijskim sustavom. Docker omogućuje pokretanje tzv. [kontejnera](https://www.docker.com/resources/what-container), standardiziranih jedinica softvera većih od softverskih paketa i manjih od virtualnih mašina. Svaki Dockerov kontejner uz aplikaciju sadrži i sve biblioteke potrebne za pokretanje te aplikacije. [Docker Hub](https://hub.docker.com/) je pretraživa kolekcija Dockerovih kontejnera.

Apache se na Docker Hubu naziva [httpd](https://hub.docker.com/_/httpd), a zbog svoje popularnosti spada među [službene slike](https://hub.docker.com/search?type=image&image_filter=official) za koje se [garantira redovitost sigurnosnih nadogradnji](https://docs.docker.com/docker-hub/official_images/). Pokretanje kontejnera `httpd` izvodimo naredbom `docker run`:

``` shell
$ docker run httpd:2.4
Unable to find image 'httpd:2.4' locally
2.4: Pulling from library/httpd
54fec2fa59d0: Pull complete
8219e18ac429: Pull complete
3ae1b816f5e1: Pull complete
a5aa59ad8b5e: Pull complete
4f6febfae8db: Pull complete
Digest: sha256:c9e4386ebcdf0583204e7a54d7a827577b5ff98b932c498e9ee603f7050db1c1
Status: Downloaded newer image for httpd:2.4
```

Vidimo da je Docker javio da lokalno nemamo sliku kontejnera i zatim povukao nekoliko slojeva (kontejneri su višeslojni kako bi se lakše višestruko iskorištavalo pojedine dijelove kontejnera). Ako želimo samo povlačenje slike na temelju koje se može stvarati kontejnere, izvest ćemo ga naredbom `docker pull`. Pokretanje Apacheja vidimo u porukama:

```
AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.17.0.2. Set the 'ServerName' directive globally to suppress this message
AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.17.0.2. Set the 'ServerName' directive globally to suppress this message
[Thu May 07 23:25:00.371329 2020] [mpm_event:notice] [pid 1:tid 139777378702464] AH00489: Apache/2.4.43 (Unix) configured -- resuming normal operations
[Thu May 07 23:25:00.371673 2020] [core:notice] [pid 1:tid 139777378702464] AH00094: Command line: 'httpd -D FOREGROUND'
```

Prekid izvođenja, kao i u većini drugih softvera na operacijskim sustavima sličnim Unixu, vršimo kombinacijom tipki Control + C (`^C`). Vidjet ćemo poruku:

```
[Thu May 07 23:33:30.363896 2020] [mpm_event:notice] [pid 1:tid 139777378702464] AH00491: caught SIGTERM, shutting down
```

Ponovimo pokretanje koje smo izveli iznad naredbom `docker run` i uočimo da se slika kontejnera ne mora ponovno preuzeti. Nakon pokretanja uočimo da posljednji redak kaže da je Apache pokrenut, a u retcima prije nam javlja i na kojoj adresi.

Ostavimo sada kontejner pokrenutim i napravimo u drugom terminalu cURL-om HTTP zahtjeve GET i HEAD na tu adresu kako bismo se uvjerili da poslužitelj radi i da je zaista u pitanju Apache 2.4:

``` shell
$ curl http://172.17.0.2/
<html><body><h1>It works!</h1></body></html>
$ curl -I http://172.17.0.2/
HTTP/1.1 200 OK
Date: Fri, 08 May 2020 15:18:10 GMT
Server: Apache/2.4.43 (Unix)
Last-Modified: Mon, 11 Jun 2007 18:53:14 GMT
ETag: "2d-432a5e4a73a80"
Accept-Ranges: bytes
Content-Length: 45
Content-Type: text/html
```

Umjesto stranice `It works!` stavit ćemo kasnije vlastiti sadržaj u kontejner. Uočimo da na strani poslužitelja Apache ispisuje zahtjeve u obliku:

```
172.17.0.1 - - [08/May/2020:15:18:10 +0000] "GET / HTTP/1.1" 200 45
```

U ovom zapisu oblika Common Log Format polja su redom, odvojena razmakom: ime ili adresa domaćina, ime kojim se korisnik prijavio (ako postoji), ime korisnika na udaljenom računalu, vrijeme, prva linija HTTP zahtjeva, HTTP kod statusa i veličina odgovora. Više informacija moguće je pronaći u dokumentaciji modula [mod_log_config](https://httpd.apache.org/docs/2.4/mod/mod_log_config.html).

## Konfiguracija poslužitelja Apache

Poslužitelj sad možemo zaustaviti jer želimo prije ponovnog pokretanja učiniti što od nas traži upozorenje `AH00558: httpd: Could not reliably determine the server's fully qualified domain name, using 172.17.0.2. Set the 'ServerName' directive globally to suppress this message`. Dohvatimo Apachejevu konfiguracijsku datoteku iz kontejnera:

``` shell
$ docker run --rm httpd:2.4 cat /usr/local/apache2/conf/httpd.conf > my-httpd.conf
```

Uočimo da ovdje imamo novi parametar `--rm` koji Dockeru kaže da izbriše stvoreni kontejner nakon što završi izvođenje. Također, uočimo da uz ime kontejnera navodimo i naredbu koju želimo pokrenuti unutar kontejnera (umjesto zadane naredbe `httpd`), a to je ovdje `cat /usr/local/apache2/conf/httpd.conf`. Izlaz te naredbe spremamo u `my-httpd.conf`. Proučimo tu datoteku (možemo je ispisati korištenjem naredbe `cat` ili otvoriti u uređivaču teksta po želji):

``` apacheconf
#
# This is the main Apache HTTP server configuration file.  It contains the
# configuration directives that give the server its instructions.
# See <URL:http://httpd.apache.org/docs/2.4/> for detailed information.
# In particular, see
# <URL:http://httpd.apache.org/docs/2.4/mod/directives.html>
# for a discussion of each configuration directive.
#
#...
#
# Listen: Allows you to bind Apache to specific IP addresses and/or
# ports, instead of the default. See also the <VirtualHost>
# directive.
#
# Change this to Listen on specific IP addresses as shown below to
# prevent Apache from glomming onto all bound IP addresses.
#
#Listen 12.34.56.78:80
Listen 80

#...
#
# ServerAdmin: Your address, where problems with the server should be
# e-mailed.  This address appears on some server-generated pages, such
# as error documents.  e.g. admin@your-domain.com
#
ServerAdmin you@example.com

#
# ServerName gives the name and port that the server uses to identify itself.
# This can often be determined automatically, but we recommend you specify
# it explicitly to prevent problems during startup.
#
# If your host doesn't have a registered DNS name, enter its IP address here.
#
#ServerName www.example.com:80
#...
```

Iako većinu naredbi nećemo koristiti, vrijedi preletiti [popis svih konfiguracijskih naredbi](https://httpd.apache.org/docs/2.4/mod/directives.html) čisto da steknemo dojam koliko je moderan web poslužitelj kompleksan softver. Osnovna konfiguracijska naredba s kojom ćemo početi proučavanje konfiguracijske datoteke je `Listen` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/mpm_common.html#listen)). Ta naredba kaže poslužitelju na kojim adresama i vratima će primati zahtjeve:

``` apacheconf
#Listen 12.34.56.78:80
Listen 80
```

Ako su navedena samo vrata (kao što su u našem slučaju vrata 80), onda poslužitelj sluša na svim adresama putem kojih ga je moguće doseći.

Pod konfiguracijskom naredbom `ServerAdmin` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/core.html#serveradmin)) navest ćemo vlastitu e-mail adresu umjesto `you@example.com`. Apache nam neće slati nikakve e-mailove jer nema konfiguriran SMTP poslužitelj koji može koristiti.

Konfiguracijsku naredbu `ServerName` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/core.html#servername)) ćemo odkomentirati i postaviti na domenu koju želimo da poslužitelj poslužuje. Teoretski možemo koristiti bilo koju domenu koja ne postoji na internetu (npr. `mreze.rijeka` ili `internet.mars`), ali to je loša praksa jer redovito dolaze [nove vršne domene](https://ntldstats.com/tld) i moguće je da nekom padne na pamet registrirati vršne domene `.rijeka` i `.mars` pa nastane problem kolizije naše lokalne razvojne okoline i interneta kao što je već bio [slučaj s domenom .dev](https://anti-pattern.com/dev-domains-not-resolving/). Zbog toga ćemo ovdje koristiti poddomene na `.rm.miletic.net` nad kojima imamo kontrolu, npr. `apache-primjer.rm.miletic.net`:

``` apacheconf
ServerName apache-primjer.rm.miletic.net:80
```

Vrata 80 ne moramo navoditi, ali se to smatra dobrom praksom.

## Izrada slika

Izgradimo sada Docker kontejner koji sadrži našu konfiguracijsku datoteku umjesto zadane. Uređivačem teksta po želji stvorimo datoteku imena `Dockerfile` i sadržaja:

``` dockerfile
FROM httpd:2.4
COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
```

Ova datoteka kaže da će novi Docker kontejneri nastajati iz slike `httpd:2.4` u koju je dodatno kopirana datoteka `./my-httpd.conf` na mjesto `/usr/local/apache2/conf/httpd.conf`. Izgradimo novu sliku naredbom `docker build` na način:

``` shell
$ docker build -t "my-httpd:2.4-1" .
Sending build context to Docker daemon  27.14kB
Step 1/2 : FROM httpd:2.4
---> b2c2ab6dcf2e
Step 2/2 : COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
---> 38dbb60affd2
Successfully built 38dbb60affd2
Successfully tagged my-httpd:2.4-1
```

Ovdje su ime `my-httpd` i verzija `2.4-1` proizvoljni, ali `.` nakon imena i verzije je vrlo važna jer označava trenutni direktorij u kojem se nalaze `Dockerfile` i ostale datoteke. Sada pokrenimo kontejner na temelju stvorene slike:

``` shell
$ docker run my-httpd:2.4-1
[Fri May 08 23:33:45.732132 2020] [mpm_event:notice] [pid 1:tid 139777349747840] AH00489: Apache/2.4.43 (Unix) configured -- resuming normal operations
[Fri May 08 23:33:45.732472 2020] [core:notice] [pid 1:tid 139777349747840] AH00094: Command line: 'httpd -D FOREGROUND'
```

Uočimo kako upozorenja da `ServerName` nije postavljen više nema u izlazu. Kontejner možemo zaustaviti kao i ranije..

Samo ćemo spomenuti da postoji konfiguracijska naredba `ServerRoot` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/core.html#serverroot)) koja navodi putanju do konfiguracijskih datoteka (od kojih jednu upravo uređujemo) i log datoteka u kojima Apache bilježi zahtjeve koje je dobio i odgovore na njih. Tu putanju nećemo mijenjati.

!!! hint
    Više informacija o osnovnim poslužiteljskim konfiguracijskim naredbama postoji u [službenoj dokumentaciji Apacheja](https://httpd.apache.org/docs/2.4/) u dijelu [Server-Wide Configuration](https://httpd.apache.org/docs/2.4/server-wide.html).

## Dodavanje datoteka za posluživanje

Konfiguracijska naredba `DocumentRoot` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/core.html#documentroot)) ima zadanu vrijednost `/usr/local/apache2/htdocs`. U tom direktoriju unutar Docker kontejnera nalazi se datoteka `index.html` sadržaja `It works!`. Mi želimo da poslužitelj poslužuje naše datoteke pa ćemo postaviti tu vrijednost na direktorij `/var/www/html`:

``` apacheconf
DocumentRoot "/var/www/html"
```

Apache ima mogućnost konfiguracije dozvola pristupa pojedinim direktorija korištenjem konfiguracijskog odjeljka `<Directory>` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/core.html#directory)). Svaki otvoreni konfiguracijski odjeljak `<Directory>` zatvoren je s `</Directory>`. Vidimo u konfiguracijskoj datoteci odmah ispod naredbe `DocumentRoot` konfiguraciju pravila pristupa direktoriju `/usr/local/apache2/htdocs` od strane Apacheja; obratimo posebnu pozornost na naredbu `Require` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/mod_authz_core.html#require)) koja ovdje ima vrijednost `all granted`, odnosno dozvoljava bezuvjetan pristup direktoriju:

``` apacheconf hl_lines="26"
<Directory "/usr/local/apache2/htdocs">
    #
    # Possible values for the Options directive are "None", "All",
    # or any combination of:
    #   Indexes Includes FollowSymLinks SymLinksifOwnerMatch ExecCGI MultiViews
    #
    # Note that "MultiViews" must be named *explicitly* --- "Options All"
    # doesn't give it to you.
    #
    # The Options directive is both complicated and important.  Please see
    # http://httpd.apache.org/docs/2.4/mod/core.html#options
    # for more information.
    #
    Options Indexes FollowSymLinks

    #
    # AllowOverride controls what directives may be placed in .htaccess files.
    # It can be "All", "None", or any combination of the keywords:
    #   AllowOverride FileInfo AuthConfig Limit
    #
    AllowOverride None

    #
    # Controls who can get stuff from this server.
    #
    Require all granted
</Directory>
```

Kako nam više neće trebati pristup direktoriju `/usr/local/apache2/htdocs`, već `/var/www/html`, promijenit ćemo samo direktorij koji je naveden tako da konfiguracija da bude oblika:

``` apacheconf
<Directory "/var/www/html">
    #...
    Require all granted
</Directory>
```

Direktorij `/var/www/html` trenutno ne postoji u kontejneru. Stvorimo ga prvo van kontejnera i napunimo sadržajem:

``` shell
$ mkdir -p www/html
$ echo '<html><body><h1>Radi!</h1></body></html>' > www/html/index.html
```

Kopiranje direktorija u kontejner izvest ćemo isto kao kopiranje konfiguracijske datoteke, dodavanjem naredbe `COPY` u `Dockerfile`:

``` dockerfile hl_lines="3"
FROM httpd:2.4
COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
COPY ./www /var/www
```

Izgradimo novu sliku naredbom `docker build` na način:

``` shell
$ docker build -t "my-httpd:2.4-2" .
Sending build context to Docker daemon   25.6kB
Step 1/3 : FROM httpd:2.4
---> b2c2ab6dcf2e
Step 2/3 : COPY ./my-httpd.conf /usr/local/apache2/conf/httpd.conf
---> 567d597ca35e
Step 3/3 : COPY ./www /var/www
---> beaf5a8401fc
Successfully built beaf5a8401fc
Successfully tagged my-httpd:2.4-2
```

Verziju smo postavili na `2.4-2` čisto da bude različita od prethodne i da imamo povijest promjena za kasnije pregledavanje. Sada pokrenimo kontejner na temelju stvorene slike:

``` shell
$ docker run my-httpd:2.4-2
[Sun May 10 15:39:27.908202 2020] [mpm_event:notice] [pid 1:tid 140585480324224] AH00489: Apache/2.4.43 (Unix) configured -- resuming normal operations
[Sun May 10 15:39:27.908547 2020] [core:notice] [pid 1:tid 140585480324224] AH00094: Command line: 'httpd -D FOREGROUND
```

Uvjerimo se da radi:

``` shell
$ curl http://172.17.0.2/
<html><body><h1>Radi!</h1></body></html>
```

!!! hint
    Osim `DocumentRoot`-a, Apache na mnogim instalacijama poslužuje i korisničke direktorije (putanja `/~korisnik/`, primjerice [/~natasah/ na www.inf.uniri.hr](https://www.inf.uniri.hr/~natasah/)). Time se ovdje nećemo baviti, ali ćemo spomenuti da se posluživanje korisničkih direktorija konfigurira pomoću naredbe `UserDir` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/mod_userdir.html#userdir)) iz modula [mod_userdir](https://httpd.apache.org/docs/2.4/mod/mod_userdir.html).

## Uključivanje modula

U konfiguracijskoj datoteci `my-httpd.conf` možemo uočiti veliki broj uglavnom zakomentiranih linija konfiguracijskih naredbi `LoadModule`  ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/mod_so.html#loadmodule)):

``` apacheconf
#
# Dynamic Shared Object (DSO) Support
#
# To be able to use the functionality of a module which was built as a DSO you
# have to place corresponding `LoadModule' lines at this location so the
# directives contained in it are actually available _before_ they are used.
# Statically compiled modules (those listed by `httpd -l') do not need
# to be loaded here.
#
# Example:
# LoadModule foo_module modules/mod_foo.so
#
LoadModule mpm_event_module modules/mod_mpm_event.so
#LoadModule mpm_prefork_module modules/mod_mpm_prefork.so
#LoadModule mpm_worker_module modules/mod_mpm_worker.so
LoadModule authn_file_module modules/mod_authn_file.so
#LoadModule authn_dbm_module modules/mod_authn_dbm.so
#LoadModule authn_anon_module modules/mod_authn_anon.so
#LoadModule authn_dbd_module modules/mod_authn_dbd.so
#LoadModule authn_socache_module modules/mod_authn_socache.so
LoadModule authn_core_module modules/mod_authn_core.so
LoadModule authz_host_module modules/mod_authz_host.so
LoadModule authz_groupfile_module modules/mod_authz_groupfile.so
LoadModule authz_user_module modules/mod_authz_user.so
#LoadModule authz_dbm_module modules/mod_authz_dbm.so
#LoadModule authz_owner_module modules/mod_authz_owner.so
#LoadModule authz_dbd_module modules/mod_authz_dbd.so
LoadModule authz_core_module modules/mod_authz_core.so
#LoadModule authnz_ldap_module modules/mod_authnz_ldap.so
#LoadModule authnz_fcgi_module modules/mod_authnz_fcgi.so
LoadModule access_compat_module modules/mod_access_compat.so
LoadModule auth_basic_module modules/mod_auth_basic.so
#LoadModule auth_form_module modules/mod_auth_form.so
#LoadModule auth_digest_module modules/mod_auth_digest.so
#LoadModule allowmethods_module modules/mod_allowmethods.so
#LoadModule isapi_module modules/mod_isapi.so
#LoadModule file_cache_module modules/mod_file_cache.so
#LoadModule cache_module modules/mod_cache.so
#LoadModule cache_disk_module modules/mod_cache_disk.so
#LoadModule cache_socache_module modules/mod_cache_socache.so
#...
```

Očekivano, te naredbe služe za učitavanje modula čiji opis možemo naći [u službenoj dokumentaciji](https://httpd.apache.org/docs/2.4/mod/).

Primjerice, ako želimo uključiti podršku za protokol HTTP/2 ([više detalja o verzijama HTTP-a na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Evolution_of_HTTP)), iskoristit ćemo već ranije spomenuti modul [mod_http2](https://httpd.apache.org/docs/2.4/mod/mod_http2.html). Prvo ćemo odkomentirati liniju koja ga učitava:

``` apacheconf
#LoadModule http2_module modules/mod_http2.so
```

tako da bude oblika:

``` apacheconf
LoadModule http2_module modules/mod_http2.so
```

i zatim na kraju datoteke dodati konfiguracijsku naredbu `Protocols` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/core.html#protocols)) s odgovarajućim parametrima:

``` apacheconf hl_lines="26"
#
# This is the main Apache HTTP server configuration file.  It contains the
# configuration directives that give the server its instructions.
#...

# Various default settings
#Include conf/extra/httpd-default.conf

# Configure mod_proxy_html to understand HTML4/XHTML1
<IfModule proxy_html_module>
Include conf/extra/proxy-html.conf
</IfModule>

# Secure (SSL/TLS) connections
#Include conf/extra/httpd-ssl.conf
#
# Note: The following must must be present to support
#       starting without SSL on platforms with no /dev/random equivalent
#       but a statically compiled-in mod_ssl.
#
<IfModule ssl_module>
SSLRandomSeed startup builtin
SSLRandomSeed connect builtin
</IfModule>

Protocols h2 h2c http/1.1
```

kojom ćemo omogućiti korištenje protokola HTTP/2 za sadržaj šifriran TLS-om/SSL-om koji se šalje korištenjem HTTPS-a (`h2`) i čisti tekst koji se šalje korištenjem HTTP-a (`h2c`). Protokol HTTP/1.1 (`http/1.1`) nastavljamo koristiti u situacijama kada klijent ne podržava protokol HTTP/2.

Nakon što izgradimo kontejner s novom konfiguracijskom datotekom, možemo se cURL-om uvjeriti da HTTP/2 radi:

``` shell
$ curl --http2 -I http://172.17.0.2/
HTTP/1.1 101 Switching Protocols
Upgrade: h2c
Connection: Upgrade

HTTP/2 200
date: Thu, 01 Jan 1970 00:00:00 GMT
server: Apache/2.4.46 (Debian)
last-modified: Mon, 11 Jun 2007 18:53:14 GMT
etag: W/"2d-432a5e4a73a80"
accept-ranges: bytes
content-length: 45
content-type: text/html
```

Promotrimo li konfiguracijsku datoteku, vidjet ćemo da se konfiguracijske naredbe ovisne o modulu često navode unutar konfiguracijskog odjeljka `<IfModule>` ([dokumentacija](https://httpd.apache.org/docs/2.4/mod/core.html#ifmodule)). U našem slučaju, obzirom da su parametri `h2` i `h2c` naredbe `Protocols` ovisni o učitanom modulu `http2_module`, ispravno navođenje konfiguracijske naredbe bilo bi oblika:

``` apacheconf hl_lines="7-9"
#...
<IfModule ssl_module>
SSLRandomSeed startup builtin
SSLRandomSeed connect builtin
</IfModule>

<IfModule http2_module>
    Protocols h2 h2c http/1.1
</IfModule>
```

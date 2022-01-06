---
author: Viktor Jurčić, Vedran Miletić
---

# Web poslužitelj nginx

[Nginx](https://nginx.org/) je HTTP poslužitelj otvorenog koda čiji je autor ruski haker [Igor Sysoev](https://sysoev.ru/en/). Neke od mogućnosti su mu posluživanje statičnih datoteka, reverse proxy sa keširanjem, balansiranje opterećenja, tolerancija na greške, spajanje sa drugim poslužiteljima te modularna arhitektura i podrška za SSL i TLS protokole. Osim navedenih postoje i brojne druge. Pregled svih mogućnosti te upute za početnike i administratore su dostupni u [službenoj dokumentaciji](https://nginx.org/en/docs/). Osim osnovne verzije nginxa, postoji i [NGINX Plus](https://www.nginx.com/products/nginx/) koja uz godišnju pretplatu nudi [dodatne značajke i podršku](https://www.nginx.com/blog/whats-difference-nginx-foss-nginx-plus/).

Nginx je HTTP poslužitelj koji uspješno rješava glavni problem s kojim se susreću svi poslužitelji današnjice a to je velik broj istovremenih zahtjeva. Nginx je jednostavan za konfiguraciju, ima kvalitetnu dokumentaciju i sretno radi na svim platformama (iako je inačica za Windows [ograničena u performansama](https://nginx.org/en/docs/windows.html)). Uz mogućnost podizanja kao samostalan poslužitelj nudi i mogućnost povezivanja sa drugim (npr. Apache) poslužiteljima kojima služi za keširanje i balansiranje opterećenja. [Od 2014. godine nginx je najkorišteniji web poslužuitelj među 10000 najposjećenijih web sjedišta](https://www.nginx.com/blog/sites-using-nginx-june-2014/) i, [prema podacima iz travnja 2017. koje je objavio W3Techs](https://w3techs.com/blog/entry/nginx_reaches_33_3_percent_web_server_market_share_while_apache_falls_below_50_percent), vidi se da [uzima tržišni udio Apache HTTP Serveru](https://www.servethehome.com/apache-drops-below-50-percent-market-share-nginx-claims-over-33-percent/).

## Postavljanje

U ovom dijelu izneseni su opisi instalacije te konfiguriranja nginx poslužitelja. U radu je provedeno postavljanje funkcionalnosti HTTP poslužitelja koje se mogu isprobati lokalno. Dakle, funkcionalnosti kao što su balansiranje opterećenja, koje zahtijeva veći broj fizičkih poslužitelja, nisu napravljene.

### Instalacija

Na Ubuntu sustavima instalacija se vrši u jednoj naredbi:

``` shell
$ sudo apt-get install nginx
```

Nakon potvrde i instaliranja, nginx poslužitelj se automatski pokrene na adresi `http://localhost`. Ako je instalacija bila uspješna, na toj adresi je dostupna demo nginx stranica.

Zbog potrebe za prikazom načina kako se nginx spaja sa PHP interpreterom potrebno je imati instaliran PHP FastCGI Process Manager (kraće, PHP FPM). Na Ubuntu sustavima to se radi sa:

``` shell
$ sudo apt-get install php5-fpm
```

### Upravljanje

Pokretanje nginx poslužitelja vrši se naredbom

``` shell
$ nginx
```

Signali se mogu davati nginx poslužitelju koristeći naredbu:

``` shell
$ nginx -s signal
```

Parametar signal može biti:

- `stop` -- brzo gašenje
- `quit` -- pravilno gašenje
- `reopen` -- ponovno otvaranje log datoteka
- `reload` -- ponovno učitavanje konfiguracijske datoteke

Nakon mijenjanja nginx konfiguracije potrebno je poslati signal `reload` kako bi poslužitelj učitao nove konfiguracijske podatke.

### Glavna konfiguracija

Sve datoteke potrebne za konfiguriranje nginx poslužitelja se nalaze u direktoriju `/etc/nginx`. Glavna konfiguracija je u datoteci `nginx.conf`. Ta datoteka je mjesto za definiranje globalnih konfiguracijskih postavki koje se primjenjuju na sve, niže konfigurirane, poslužitelje. Postavke HTTP poslužitelja se pišu unutar oznake `http{}`.

Ispod se nalazi sadržaj datoteke `nginx.conf` sa komentarima koji objašnjavaju pojedine definicije.

``` nginx
user www-data;
worker_processes 4;
pid /run/nginx.pid;
events {
  worker_connections 768;
}

http {
  ##
  # Basic Settings
  ##

  sendfile on;

  # Enables the use of the TCP_CORK socket option.
  # Use only when sendfile is used. Enabling the option allows
  # - sending the response header and the beginning of a file in one packet;
  # - sending a file in full packets
  tcp_nopush on;

  # Enables the use of the TCP_NODELAY option
  tcp_nodelay on;

  # Time for keep-alive client connection to stay open on the server side
  keepalive_timeout 65;

  # maximum size of the types hash tables
  types_hash_max_size 2048;
  include /etc/nginx/mime.types;
  default_type application/octet-stream;

  ##
  # Logging Settings
  ##
  access_log /var/log/nginx/access.log;
  error_log /var/log/nginx/error.log;

  ##
  # Virtual Host Configs
  ##
  include /etc/nginx/conf.d/*.conf;
  include /etc/nginx/sites-enabled/*;
}
```

### Konfiguriranje poslužitelja

U direktoriju `sites-enabled` nalazi se datoteka `default` u kojoj se zadaju konfiguracije poslužitelja. Ta datoteka je uključena u datoteku `nginx.conf`. Ispod se nalazi sadržaj datoteke `default` sa objašnjenjima i prikazima rada pojedinih poslužitelja.

#### PHP i IPv6 poslužitelj

Konfiguracija u nastavku definira poslužitelj koji radi na `localhost` vratima 80 i povezan je sa PHP FastCGI Process Managerom koji radi na Unix socketu `/var/run/php5-fpm.soc`. Poslužitelj poslužuje HTML datoteke iz direktorija `/usr/share/nginx/html` a slike iz direktorija `/usr/share/nginx/images`. Datoteke koje završavaju sa `.php` se prosljeđuju PHP interpreteru.

Za prepoznavanje tipa datoteke koriste se regularni izrazi. Regularni izraz `\.(gif|jpg|png)$` odgovara svim datotekama tipa GIF, JPG ili PNG te takve datoteke poslužuje iz direktorija `image`. Regularni izraz `[^/]\.php(/|$)` odgovara svim PHP datotekama te zahtjeve za takvim datotekama prosljeđuje PHP FastCGI Process Manageru.

``` nginx
#
# Virtual host running on localhost with PHP interpreter binding
#
server {
  listen 80 default_server;

  # Enable server access using IPv6 adress
  listen [::]:80 default_server ipv6only=on;

  # Set the root file root directory
  root /usr/share/nginx/html;
  index index.html index.htm;

  # Make site accessible from http://localhost/
  server_name localhost;

  # The '=' modifier stops the search if its an exact match
  location = / {
    # First attempt to serve request as file, then
    # as directory, then fall back to displaying a 404.
    try_files $uri $uri/ =404;
  }

  # If requested file is a GIF, JPF or PNG search for it in the
  # 'images' directory
    location ~ \.(gif|jpg|png)$ {
    root /usr/share/nginx/images;
  }

  # pass the PHP scripts to FastCGI Process Manager listening
  # on socket /var/run/php5-fpm.sock
  location ~ [^/]\.php(/|$) {
    # Check to see if file exists and throw
    # 404 if file is missing.
    fastcgi_split_path_info ^(.+?\.php)(/.*)$;
    if (!-f $document_root$fastcgi_script_name) {
      return 404;
    }
    fastcgi_pass unix:/var/run/php5-fpm.sock;
    fastcgi_index index.php;
    include fastcgi_params;
  }

  error_page 404 /404.html;

  # redirect server error pages to the static page /50x.html
  error_page 500 502 503 504 /50x.html;
  location = /50x.html {
    root /usr/share/nginx/html;
  }
}
```

#### Proxy poslužitelj

Donja konfiguracija postavlja poslužitelj koji radi na localhost vratima 8001. To je poslužitelj koji radi kao proxy za poslužitelj sa localhost porta 8080 čija se konfiguracija nalazi niže.

``` nginx
#
# Virtual proxy server on localhost:8001
#

server {
  listen 8001;
  listen 127.0.0.1:8001;
  server_name proxy.localhost;
  gzip_proxied no-store no-cache private expired auth;

  location = / {
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;

    # NGINX buffers reponses and doesn't send them to the client
    # until the whole response is received from the proxied server.
    proxy_buffers 16 4k;
    proxy_buffer_size 2k;

    # To disable buffering use
    # proxy_buffering off;

    proxy_pass http://127.0.0.1:8080/;
  }
}
```

#### Poslužitelj sa komprimiranjem

Donja konfiguracija definira poslužitelj koji radi na localhost vratima 8080. Poslužitelj poslužuje HTTP datoteke iz direktorija `/data/www` i slike iz direktorija `/data/images`. Ovom poslužitelju uključeno je komprimiranje tekstualnih i PNG datoteka kako bi se smanjila količina podataka koja se prenosi i time ubrzalo dostavljanje odgovora klijentu.

``` nginx
#
# Virtual host for serving gziped files from /data/www/ on localhost:8080
#
server {
  listen 8080;
  listen 127.0.0.1:8080;
  server_name proxied.localhost;
  root /data/www/;
  index proxy.html proxy.htm;

  gzip on;
  gzip_types text/plain image/png;

  location / {
    try_files $uri $uri/ =404;
  }

  location ~ \.(gif|jpg|png)$ {
    root /data/images;
  }
}
```

#### Poslužitelj sa autorizacijom

Donja konfiguracija definira poslužitelj na localhost vratima 8000 kojemu se može pristupiti samo lokalno (sa adrese 127.0.0.1) i upotrebom točnog korisničkog imena i lozinke. Imena korisnika i njihove lozinke nalaze se u datoteci `htpasswd` u direktoriju `/data/conf/` u obliku `username:{ENCRYPTION}password`.

``` nginx
#
# Virtual auth host for serving files from /data/www/ on localhost:8000
#
server {
  listen 8000;
  listen [::]:8000;
  server_name auth.localhost;
  root /data/www/;
  index index.html index.htm;

  satisfy all;
  allow 127.0.0.1;
  deny 192.168.1.2;
  deny all;

  auth_basic "user, password";
  auth_basic_user_file /data/conf/htpasswd;

  location / {
    try_files $uri $uri/ =404;
  }

  location ~ \.(gif|jpg|png)$ {
    root /data/images;
  }
}
```

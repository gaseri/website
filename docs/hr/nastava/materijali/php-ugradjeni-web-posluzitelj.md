---
author: Vedran Miletić
---

# Ugrađeni web poslužitelj u interpreteru jezika PHP

[PHP](https://www.php.net/) je popularan skriptni jezik koji se primarno koristi u izradi web aplikacija. Razvoj PHP-a započeo je [Rasmus Lerdorf](https://toys.lerdorf.com/) 1994. godine, a [godinu kasnije javno je objavio izvorni kod](https://www.php.net/manual/en/history.php.php) što je omogućilo svim zainteresiranima da ga koriste i prilagođavaju svojim potrebama. Od 1997. nadalje važnu ulogu u razvoju PHP-a imaju [Zeev Suraski](https://en.wikipedia.org/wiki/Zeev_Suraski) i [Andi Gutmans](https://en.wikipedia.org/wiki/Andi_Gutmans), koji su 1999. godine i osnovali tvrtku [Zend Technologies](https://en.wikipedia.org/wiki/Zend_(company)) (danas samo [Zend](https://www.zend.com/)) s ciljem pružanja na njemu temeljenih poslovnih rješenja.

Maskota PHP-a je slon [ElePHPant](https://www.php.net/elephpant.php) čije se različite vrste mogu pronaći na [ElePHPant.me](https://elephpant.me/), a više detalja o njima se može pročitati u [A Field Guide to Elephpants](https://afieldguidetoelephpants.net/).

## Korisnici i značaj

PHP je izvorno postao široko korišten kao dio četvorke [Linux, Apache, MySQL i PHP](https://www.ibm.com/cloud/learn/lamp-stack-explained) (kraće [stog LAMP](https://en.wikipedia.org/wiki/LAMP_%28software_bundle%29)) koja je predstavljala *de facto* standard za razvoj web aplikacija krajem 1990-ih. PHP danas (često upravo kao dio stoga LAMP) pogoni web sjedišta svih veličina, od osobnih i projektnih sjedišta kao što je [FreeBSD Live Streaming](https://live.freebsd.org/), preko profesionalnih blogova i portala temeljenih na [Wordpressu](https://wordpress.com/) kao što su [PlayStation.Blog](https://blog.playstation.com/), [TechCrunch](https://techcrunch.com/) i [Microsoft Stories](https://news.microsoft.com/), nacionalnih sustava za e-učenje temeljenih na [Moodleu](https://moodle.org/) kao što je [Merlin](https://moodle.srce.hr/), stranica s vijestima kao što su [GamingOnLinux](https://www.gamingonlinux.com/), [Novi List](https://www.novilist.hr/) i , [Srednja.hr](https://www.srednja.hr/), web trgovina kao što su [Pevex](https://pevex.hr/) i [Emmezeta](https://www.emmezeta.hr/), web sjedišta tvrtki i drugih organizacija kao što su [Tesla](https://www.tesla.com/), [Agrokor](https://www.agrokor.hr/), [Public Internet Registry (domena .org)](https://thenew.org/), [Verisign (domene .com i .net)](https://www.verisign.com/), [Hrvatska zaklada za znanost](https://hrzz.hr/), [IBM](https://www.ibm.com/), [Sveučilište Stanford](https://www.stanford.edu/), [Odjel za informatiku Sveučilišta u Rijeci](https://www.inf.uniri.hr/), [CARNET](https://www.carnet.hr/) i [NASA](https://www.nasa.gov/), pa do tehnoloških divova kao što su [Facebook](https://www.facebook.com/), [Wikipedia](https://www.wikipedia.org/) ([ilustracija načina rada](https://meta.wikimedia.org/wiki/Wikimedia_servers#/media/File:Wikipedia_webrequest_flow_2020.png)), [Flickr](https://www.flickr.com/), [Indeed](https://www.indeed.com/), [Dailymotion](https://www.dailymotion.com/), [VK (nekad VKontakte)](https://vk.com/) i [Slack](https://slack.com/).

Vremenom je LAMP (i time PHP) prestao biti *de facto* standard za razvoj web aplikacija jer su postepeno popularnost stekli i [drugi alati za razvoj web aplikacija](https://insights.stackoverflow.com/survey/2020#technology-most-loved-dreaded-and-wanted-web-frameworks), npr. [ASP.NET (C#)](https://dotnet.microsoft.com/apps/aspnet), [Express.js (Node.js)](https://expressjs.com/), [Spring (Java)](https://spring.io/), [Ruby on Rails (Ruby)](https://rubyonrails.org/), [Django (Python)](https://www.djangoproject.com/) i [Flask (Python)](https://flask.palletsprojects.com/) ([dobar pregled objavila je tvrtka Space-O Technologies](https://www.spaceo.ca/web-development-frameworks/)). (Specijalno, sve popularniji okvir [Express.js](https://expressjs.com/) i njegovo izvršno okruženje [Node.js](https://nodejs.org/) naglašavaju kako stog LAMP zamjenjuju [stogom MEAN](https://en.wikipedia.org/wiki/MEAN_(solution_stack)): [MongoDB, Express.js, AngularJS i Node.js](https://www.ibm.com/cloud/learn/mean-stack-explained).)

Bez obzira na sve brojniju konkurenciju, PHP se i danas široko koristi, a u nekim segmentima čak bilježi i porast upotrebe. Prema podacima dostupnim u studenom 2020. godine, W3Techs [navodi kako PHP pogoni 79% web sjedišta za koja je poznat jezik na poslužiteljskoj strani](https://w3techs.com/technologies/overview/programming_language), a BuiltWith [tvrdi da preko 50%, preko 55% i gotovo 60% web sjedišta među milijun, 100 tisuća i 10 tisuća najposjećenijih (respektivno) koriste PHP](https://trends.builtwith.com/framework/PHP). Osim toga, u studenom 2020. godine Indeed je objavio da je [broj otvorenih natječaja za posao "Entry level PHP developer" porastao 834% u odnosu na siječanj 2020. godine](https://www.indeed.com/career-advice/finding-a-job/rising-tech-jobs-and-skills). Suvremena upotreba PHP-a uglavnom podrazumijeva korištenje okvira za razvoj web aplikacija [Laravel](https://laravel.com/), [Symfony](https://symfony.com/), [CodeIgniter](https://codeigniter.com/), [CakePHP](https://cakephp.org/) i [brojnih drugih](https://kinsta.com/blog/php-frameworks/). Pritom se grupa [PHP Framework Interop Group (PHP-FIG)](https://www.php-fig.org/) bavi standardizacijom načina rada okvira kako bi se njihove pojedine komponente lakše kombinirale kod korištenja i za tu svrhu donosi [preporuke](https://www.php-fig.org/psr/) koje razvijatelji okvira mogu slijediti.

## Upotreba sučelja naredbenog retka

Pokretanje datoteka s izvornim kodom iz naredbenog retka vršimo naredbom `php`. Za početak se uvjerimo da imamo instalirano sučelje naredbenog retka interpretera jezika PHP (naredba `php`) i saznajmo njegovu verziju parametrom `--version`, odnosno `-v`:

``` shell
$ php -v
PHP 7.4.11 (cli) (built: Oct  6 2020 10:34:39) ( NTS )
Copyright (c) The PHP Group
Zend Engine v3.4.0, Copyright (c) Zend Technologies
    with Zend OPcache v7.4.11, Copyright (c), by Zend Technologies
```

!!! note
    Vidimo da koristimo verziju 7.4.11 koja nije [posljednja izdana](https://www.php.net/downloads.php) i zbog toga je [službeno nepodržana](https://www.php.net/releases/index.php), ali je dovoljno nova za potrebe učenja. U praktičnoj primjeni za posluživanje web aplikacija je dobro koristiti posljednju verziju s neke od [aktivno podržanih grana izdanja](https://www.php.net/supported-versions.php).

Mi ćemo u nastavku koristiti tek neke od značajki ovog sučelja i zbog toga ćemo spomenuti kako se koristi tek nekolicina parametara. Pregled svih parametara i njihovog načina korištenja moguće je pronaći u man stranici `php(1)` (naredba `man 1 php`) ili u [PHP-ovom priručniku](https://www.php.net/manual/en/index.php) u [odjeljku Using PHP from the command line](https://www.php.net/manual/en/features.commandline.php).

Korištenjem uređivača teksta po želji stvorimo datoteku `hello.php` sadržaja:

``` php
<?php

echo "Hello, world!\n";
```

pa je pokrenimo korištenjem naredbe `php` na toj datoteci:

``` shell
$ php hello.php
Hello, world!
```

Oznaka `<?php` označava interpreteru da počne interpretirati kod koji slijedi ([dokumentacija](https://www.php.net/manual/en/language.basic-syntax.phptags.php)), a naredba `echo` ([dokumentacija](https://www.php.net/manual/en/function.echo.php)) ispisuje znakovni niz koji joj je dan (u našem slučaju `"Hello, world!\n"`).

Sučelje naredbenog retka interpretera jezika PHP od [verzije 5.4.0](https://www.php.net/releases/5_4_0.php) nadalje nudi za korištenje [ugrađeni web poslužitelj](https://www.php.net/manual/en/features.commandline.webserver.php) koji je namijenjen za isprobavanje web aplikacija u procesu njihovog razvoja, a vrlo je koristan i za učenje načina rada web poslužitelja.

## Pokretanje ugrađenog web poslužitelja

Stvorimo direktorij `public` i u njemu datoteku `index.php`:

``` shell
$ mkdir public
$ touch public/index.php
```

Korištenjem uređivača teksta po želji upišimo u datoteku `public/index.php` poziv funkcije `phpinfo()` ([dokumentacija](https://www.php.net/manual/en/function.phpinfo.php)) tako da sadržaj datoteke bude oblika:

``` php
<?php

phpinfo();
```

Naredbom `php` vršimo pokretanje web poslužitelja na adresi lokalnog domaćina `localhost` i vratima `8000` (parametar `--server`, odnosno `-S`) koji će posluživati sadržaj direktorija `public` (parametar `--docroot`, odnosno `-t`) na način:

``` shell
$ php -S localhost:8000 -t public
[Mon Nov  2 11:17:46 2020] PHP 7.4.11 Development Server (http://localhost:8000) started
```

Adresu `http://localhost:8000` možemo otvoriti u web pregledniku po želji i uvjeriti se da poslužitelj ispravno radi te usput saznati neke informacije o instaliranom interpreteru PHP-a.

!!! warning
    Web preglednici [FireDragon](https://forum.garudalinux.org/t/firedragon-librewolf-fork/5018) i [LibreWolf](https://librewolf-community.gitlab.io/) u zadanim postavkama [isključuju podršku za pronalaženje IPv6 adresa putem DNS-a](https://gitlab.com/librewolf-community/browser/common/-/issues/15) (postavka `network.dns.disableIPv6` u `about:config`). Specijalno, to znači da interpretiraju `localhost` kao IPv4 adresu `127.0.0.1`, a ne kao IPv6 adresu `::1` koju PHP-ov ugrađeni web poslužitelj koristi kad je IPv6 dostupan. Ako se koristi neki od tih preglednika, potrebno je eksplicitno pristupiti adresi `http://[::1]:8000` (u prikazu IPv6 adrese, pojedini su njeni dijelovi odvojeni znakom dvotočke pa se ta adresa dodatno stavlja u uglate zagrade kako bi se dvotočka koja odvaja broj vrata od adrese mogla razlikovati od dvotočki koje odvajaju dijelove adrese).

Preglednik zatvorimo, a poslužitelj ostavimo pokrenutim. Naime, izmjene u datoteci `index.php` su odmah aktivne po spremanju pa nema potrebe za ponovnim pokretanjem poslužitelja.

## Slanje zahtjeva web klijentom

Podsjetimo se da protokol HTTP poznaje dva tipa poruka, zahtjeve koje klijent šalje poslužitelju i odgovore koje poslužitelj šalje klijentu nakon obrade primljenog zahtjeva ([više o HTTP-u na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP); [više o HTTP porukama na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages)). U nastavku ćemo umjesto web preglednika kao HTTP klijent koristiti [cURL](https://curl.se/). Kako bismo lakše u terminalu pregledali odgovor na naš zahtjev, izmijenimo sadržaj datoteke `public/index.php` tako da umjesto velike količine informacija o interpreteru PHP-a odgovor bude jednostavan `Hello, world!` u HTML-u oblika:

``` php
<?php

echo "<p>Hello, world!</p>\n";
```

U drugom terminalu od onog u kojem je poslužitelj pokrenut možemo korištenjem naredbe `curl` saznati kako poslužitelj odgovara na klijentov HTTP zahtjev:

``` shell
$ curl -v http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Mon, 02 Nov 2020 10:41:35 GMT
< Connection: close
< X-Powered-By: PHP/7.4.11
< Content-Type: text/html; charset=UTF-8
<
<p>Hello, world!</p>
* Closing connection 0
```

Kod slanja zahtjeva, nismo naveli metodu pa cURL koristi zadanu metodu GET ([više detalja o metodama HTTP zahtjeva na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods); [više detalja o HTTP metodi GET na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/GET)). Podsjetimo se da parametar `--verbose`, odnosno `-v` čini cURL rječitim u radu pa vidimo poruke o radu HTTP klijenta (znak `*`), zaglavlja i tijelo zahtjeva (znak `>`) te zaglavlja i tijelo odgovora (znak `<`).

Uočimo da su se u terminalu gdje je poslužitelj pokrenut pojavile linije:

```
[Mon Nov  2 11:41:35 2020] [::1]:52708 Accepted
[Mon Nov  2 11:41:35 2020] [::1]:52708 [200]: GET /
[Mon Nov  2 11:41:35 2020] [::1]:52708 Closing
```

koje nam redom kažu:

- da je poslužitelj primio i prihvatio HTTP zahtjev
- da zahtjev metodom GET dohvaća putanju `/` i da odgovor na taj zahtjev poslan od strane poslužitelja ima [HTTP statusni kod](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes) [200 OK](https://http.cat/200) ([više detalja o HTTP statusnim kodovima na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status); [više detalja o HTTP statusnom kodu 200 OK na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/200)).
- da su klijent i poslužitelj zatvorili vezu nakon razmjene podataka; možemo uočiti i u izlazu cURL-a iznad da je u odgovoru bilo postavljeno zaglavlje `Connection` na vrijednost `close`, što povlači zatvaranje veze nakon razmjene podataka ([više detalja o HTTP zaglavlju Connection na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Connection)).

## Posluživanje statičnog sadržaja

Zatražimo na isti način neku datoteku koja ne postoji, npr. `moja-stranica.html`:

``` shell
$ curl -v http://localhost:8000/moja-stranica.html
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /moja-stranica.html HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 404 Not Found
< Host: localhost:8000
< Date: Mon, 02 Nov 2020 10:46:35 GMT
< Connection: close
< Content-Type: text/html; charset=UTF-8
< Content-Length: 551
<
<!doctype html><html><head><title>404 Not Found</title><style>
body { background-color: #fcfcfc; color: #333333; margin: 0; padding:0; }
h1 { font-size: 1.5em; font-weight: normal; background-color: #9999cc; min-height:2em; line-height:2em; border-bottom: 1px inset black; margin: 0; }
h1, p { padding-left: 10px; }
code.url { background-color: #eeeeee; font-family:monospace; padding:0 2px;}
</style>
* Closing connection 0
</head><body><h1>Not Found</h1><p>The requested resource <code class="url">/moja-stranica.html</code> was not found on this server.</p></body></html>
```

Uočimo da smo dobili odgovor koji sadrži HTML datoteku s porukom da stranica nije pronađena i da taj odgovor ima HTTP statusni kod [404 Not Found](https://http.cat/404) ([više detalja o HTTP statusnom kodu 404 Not Found na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404)). Na strani poslužitelja su poruke sad oblika:

```
[Mon Nov  2 11:46:35 2020] [::1]:52710 Accepted
[Mon Nov  2 11:46:35 2020] [::1]:52710 [404]: (null) /moja-stranica.html - No such file or directory
[Mon Nov  2 11:46:35 2020] [::1]:52710 Closing
```

Stvorimo datoteku `public/moja-stranica.html` sadržaja:

``` html
<h1>Moja stranica</h1>
<p>Dobrodošli na moju stranicu!</p>
```

Ponovimo isti zahtjev:

``` shell
$ curl -v http://localhost:8000/moja-stranica.html
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /moja-stranica.html HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Mon, 02 Nov 2020 21:43:48 GMT
< Connection: close
< Content-Type: text/html; charset=UTF-8
< Content-Length: 60
<
<h1>Moja stranica</h1>
<p>Dobrodošli na moju stranicu!</p>
* Closing connection 0
```

Vidimo da smo uspješno primili statičnu datoteku s HTML-om. Na strani poslužitelja poruke su oblika:

```
[Mon Nov  2 22:43:48 2020] [::1]:52772 Accepted
[Mon Nov  2 22:43:48 2020] [::1]:52772 [200]: (null) /moja-stranica.html
[Mon Nov  2 22:43:48 2020] [::1]:52772 Closing
```

Na sličan način možemo koristiti i datoteke drugih vrsta. Za ilustraciju, preuzmimo cURL-om s [Wikimedia Commons logotip Odjela za informatiku Sveučilišta u Rijeci u formatu SVG](https://commons.wikimedia.org/wiki/File:Inf-uniri-hr-logo.svg) i spremimo ga u datoteku `public/inf-logo.svg` (parametar `--output`, odnosno `-o`). To možemo učiniti naredbom:

``` shell
$ curl -o public/inf-logo.svg https://upload.wikimedia.org/wikipedia/commons/5/59/Inf-uniri-hr-logo.svg
```

Sad možemo i tu datoteku zatražiti od poslužitelja:

``` shell
$ curl -v http://localhost:8000/inf-logo.svg
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /inf-logo.svg HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Mon, 02 Nov 2020 21:53:04 GMT
< Connection: close
< Content-Type: image/svg+xml
< Content-Length: 2619
<
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   id="svg2"
   version="1.1"
   inkscape:version="0.47 r22583"
   width="123.28125"
   height="100.47875"
   xml:space="preserve"
   sodipodi:docname="SOILogo.pdf"><metadata
     id="metadata8"><rdf:RDF><cc:Work
         rdf:about=""><dc:format>image/svg+xml</dc:format><dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" /><dc:title></dc:title></cc:Work></rdf:RDF></metadata><defs
     id="defs6"><inkscape:perspective
       sodipodi:type="inkscape:persp3d"
       inkscape:vp_x="0 : 0.5 : 1"
       inkscape:vp_y="0 : 1000 : 0"
       inkscape:vp_z="1 : 0.5 : 1"
       inkscape:persp3d-origin="0.5 : 0.33333333 : 1"
       id="perspective10" /></defs><sodipodi:namedview
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1"
     objecttolerance="10"
     gridtolerance="10"
     guidetolerance="10"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:window-width="1280"
     inkscape:window-height="743"
     id="namedview4"
     showgrid="false"
     inkscape:zoom="1.432"
     inkscape:cx="187.2025"
     inkscape:cy="48.369602"
     inkscape:window-x="0"
     inkscape:window-y="0"
     inkscape:window-maximized="1"
     inkscape:current-layer="g14" /><g
     id="g12"
     inkscape:groupmode="layer"
     inkscape:label="SOILogo"
     transform="matrix(1.25,0,0,-1.25,-0.2975,101.7125)"><g
       id="g14"><path
         d="m 22.738,58.182 c 6.746,0 12.219,-5.469 12.219,-12.219 0,-6.746 -5.473,-12.215 -12.219,-12.215 -6.746,0 -12.215,5.469 -12.215,12.215 0,6.75 5.469,12.219 12.215,12.219 z"
         style="fill:#009ee0;fill-opacity:1;fill-rule:nonzero;stroke:none"
         id="path20" /><path
         d="m 66.313,1.006 -12.454,0 0,15.074 12.454,0 0,50.211 -9.934,0 0,15.079 29.926,0 0,-65.29 12.558,0 0,-15.093 -12.558,0.011 0,-0.011 -19.992,0 0,0.019 z"
         style="fill:#383838;fill-opacity:1;fill-rule:nonzero;stroke:none"
         id="path22" /><path
         d="m 25.25,1.006 -25.012,0 0,15.074 45.004,0 0,-15.093 -19.992,0.019 z"
         style="fill:#383838;fill-opacity:1;fill-rule:nonzero;stroke:none"
* Closing connection 0
         id="path24" /></g></g></svg>
```

Uočimo promjenu u zaglavlju `Content-Type` koje sadrži MIME tip ([više detalja o MIME tipovima na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types)) resursa poslanog u odgovoru; ranije je resurs bio HTML dokument pa je njegov MIME tip bio `text/html`, a sad je to SVG pa je `image/svg+xml` ([više detalja o zaglavlju Content-Type na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Type)).

SVG je tekstualni format pa njegov sadržaj možemo čitati slično kao što je slučaj s HTML-om. Da se radilo o binarnom formatu kao što su JPEG ili PNG, to ne bi bio slučaj, u što se možete uvjeriti postavljanjem takvih datoteka unutar direktorija `public` te zahtjevom za odgovarajućom datotekom korištenjem cURL-a. Vidjet ćete i da se MIME tipovi mijenjaju ovisno o ekstenziji koju datoteka ima, a potpun popis podržanih MIME tipova od strane PHP-ovog ugrađenog web poslužitelja možete pronaći u [tablici Supported MIME Types (file extensions) u službenoj dokumentaciji](https://www.php.net/manual/en/features.commandline.webserver.php).

## Dohvaćanje različitih putanja korištenjem različitih metoda

Za usporedbu isprobajmo dohvaćanje putanja `/proba` i `/proba/`:

``` shell
$ curl -v http://localhost:8000/proba
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /proba HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Mon, 02 Nov 2020 21:57:35 GMT
< Connection: close
< X-Powered-By: PHP/7.4.11
< Content-Type: text/html; charset=UTF-8
<
<p>Hello, world!</p>
* Closing connection 0

$ curl -v http://localhost:8000/proba/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> GET /proba/ HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Mon, 02 Nov 2020 21:57:41 GMT
< Connection: close
< X-Powered-By: PHP/7.4.11
< Content-Type: text/html; charset=UTF-8
<
<p>Hello, world!</p>
* Closing connection 0
```

Uočimo da web poslužitelj odgovara na istim sadržajem kao kad smo poslali zahtjev na putanju `/` metodom GET. Osim toga, kada ostavimo putanju istom, a promijenimo metodu GET u POST ([više detalja o HTTP metodi POST na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST)), poslužitelj ponovno odgovara istim sadržajem:

``` shell
$ curl -v -X POST http://localhost:8000/
*   Trying ::1:8000...
* Connected to localhost (::1) port 8000 (#0)
> POST / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.72.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Host: localhost:8000
< Date: Mon, 02 Nov 2020 22:01:54 GMT
< Connection: close
< X-Powered-By: PHP/7.4.11
< Content-Type: text/html; charset=UTF-8
<
<p>Hello, world!</p>
* Closing connection 0
```

Na strani web poslužitelja također vidimo da odgovori na zahtjeve imaju statusni kod [200 OK](https://http.cat/200):

```
[Mon Nov  2 22:57:35 2020] [::1]:52786 Accepted
[Mon Nov  2 22:57:35 2020] [::1]:52786 [200]: GET /proba
[Mon Nov  2 22:57:35 2020] [::1]:52786 Closing
[Mon Nov  2 22:57:41 2020] [::1]:52788 Accepted
[Mon Nov  2 22:57:41 2020] [::1]:52788 [200]: GET /proba/
[Mon Nov  2 22:57:41 2020] [::1]:52788 Closing
[Mon Nov  2 23:01:54 2020] [::1]:52790 Accepted
[Mon Nov  2 23:01:54 2020] [::1]:52790 [200]: POST /
[Mon Nov  2 23:01:54 2020] [::1]:52790 Closing
```

Naime, svi zahtjevi za nekom putanjom koja nije direktorij koji postoji ili statični sadržaj (datoteka s poznatim nastavkom i MIME tipom) serviraju se pokretanjem skripte `index.php`. Uočimo da u trenutnoj varijanti naša skripta vraća u odgovoru `<p>Hello, world!</p>` bez obzira na metodu zahtjeva i putanju na koju je upućen. U nastavku ćemo prvo naučiti različito obrađivati zahtjeve ovisno o metodi i putanji, a zatim i po sadržaju.

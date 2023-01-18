---
marp: true
author: Vedran Miletić
title: Mjerenje i tehnike poboljšanja performansi web aplikacija
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Mjerenje i tehnike poboljšanja performansi web aplikacija

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Korisnička percepcija brzine rada web sjedišta

[Nielsen Norman Group](https://www.nngroup.com/), istraživači u domeni korisničkog iskustva, su u [članku iz 1997. godine](https://www.nngroup.com/articles/the-need-for-speed-1997/) kao najveći razlog za sporost web sjedišta istaknuli **velike slike**.

13 godina kasnije, nakon prodora *broadband* pristupa kod velikog broja korisnika interneta, ponovno su razmotrili problem i objavili [članak o vremenima odgovora na zahtjeve od strane web sjedišta](https://www.nngroup.com/articles/website-response-times/) u kojemu kažu:

> Slow page rendering today is typically caused by **server delays** or **overly fancy page widgets**, not by big images. **Users still hate slow sites** and don't hesitate telling us.

---

## Ograničenja vremena odgovora na zahtjev (1/2)

Isti autori su u [starijem članku iz 1993. godine](https://www.nngroup.com/articles/response-times-3-important-limits/) zaključili na temelju **istraživanja ljudskih faktora u korištenju tehnologije** da postoje ograničenja vremena odgovora na zahtjev:

- **0,1 sekunda daje osjećaj trenutnog odgovora** -- to jest, ishod se čini kao da ga je uzrokovao korisnik, a ne računalo. Ova je razina odzivnosti ključna za podršku osjećaju izravne manipulacije (izravna manipulacija jedna je od ključnih GUI tehnika za povećanje angažmana i kontrole korisnika).
- **1 sekunda održava tijek misli korisnika besprijekornim**. Korisnici mogu osjetiti kašnjenje i tako znaju da računalo generira ishod, ali i dalje osjećaju kontrolu nad ukupnim iskustvom i da se slobodno kreću, a ne čekaju na računalo. Ovaj stupanj odziva potreban je za dobru navigaciju.

---

## Ograničenja vremena odgovora na zahtjev (2/2)

- **10 sekundi zadržava pažnju korisnika**. Nakon 1–10 sekundi korisnici definitivno osjećaju da su prepušteni na milost i nemilost računala i priželjkuju da je brže, ali mogu se nositi s tim. Nakon 10 sekundi počinju razmišljati o drugim stvarima, što otežava vraćanje na razmišljanje o problemu koji rješavaju nakon što računalo napokon reagira. Dodatno o tome kažu:
  > A 10-second delay will often make users **leave a site** immediately. And even if they stay, it's harder for them to understand what's going on, making it less likely that they'll succeed in any difficult tasks.
  > Even a few seconds' delay is enough to create an **unpleasant** user experience. Users are no longer in control, and they're consciously annoyed by having to wait for the computer. Thus, with repeated short delays, users will give up unless they're extremely committed to completing the task.

---

## Jednostavno mjerenje vremena odgovora na zahtjev

Iskoristimo [Network Monitor (Firefox Developer Tools)](https://developer.mozilla.org/en-US/docs/Tools/Network_Monitor) za mjerenje brzine prikaza kućne stranice nekoliko web sjedišta:

- [webglsamples.org](https://webglsamples.org/)
- [razmjena.org](https://razmjena.org/)
- [reviews.llvm.org](https://reviews.llvm.org/)
- [www.inf.uniri.hr](https://www.inf.uniri.hr/)

---

## Stvaranje opterećenja i mjerenje vremena odgovora na zahtjev

Dosad smo mjerili vrijeme odgovora na zahtjev jednog korisnika. Moguće je simulirati opterećenje većeg broja istovremenih korisnika i mjeriti vrijeme odgovora na zahtjev. Alati koji se pritom koriste su:

- [ab (Apache HTTP server benchmarking tool)](https://httpd.apache.org/docs/current/programs/ab.html)
- [Siege](https://www.joedog.org/siege-home/) (izlaz u obliku JSON)
- [wrk](https://github.com/wg/wrk) (podržava skriptiranje u jeziku [Lua](https://www.lua.org/))
- [Apache JMeter](https://jmeter.apache.org/) (složeniji alat, vrlo moćan), [primjeri izvještaja koje generira](https://jmeter.apache.org/usermanual/generating-dashboard.html)
- [Flood](https://www.flood.io/) (rješenje u oblaku)

---

## Druga motivacija osim korisničkog iskustva

> When real users have a slow experience on mobile, they're much less likely to find what they are looking for or purchase from you in the future. For many sites this equates to a huge missed opportunity, especially when more than half of visits are abandoned if a mobile page takes over 3 seconds to load.
>
> Users want to find answers to their questions quickly and data shows that people really care about how quickly their pages load. The Search team announced speed would be a ranking signal for desktop searches in 2010 and as of this month (July 2018), page speed will be a ranking factor for mobile searches too.

Izvor: [Speed is now a landing page factor for Google Search and Ads (Addy Osmani and Ilya Grigorik, Google Developers, July 2018)](https://developers.google.com/web/updates/2018/07/search-ads-speed)

Googleovi alati za developere: [Make the Web Faster](https://developers.google.com/speed/), [PageSpeed Insights](https://pagespeed.web.dev/)

---

## Brzina prikaza stranice kao faktor u rangiranju

> Google has indicated that site speed (and as a result, page speed) is one of the signals used by its algorithm to rank pages. And research has shown that Google might be specifically measuring time to first byte as when it considers page speed. In addition, a slow page speed means that search engines can crawl fewer pages using their allocated crawl budget, and this could negatively affect your indexation.

Izvor: [Page Speed (Moz)](https://moz.com/learn/seo/page-speed)

Drugi članci na Moz blogu u kojima govori o brzini prikaza web stranica: [Winning the Page Speed Race: How to Turn Your Clunker of a Website Into a Race Car](https://moz.com/blog/winning-page-speed), [Better Site Speed: 4 Outside-the-Box Ideas](https://moz.com/blog/site-speed-4-ideas)

---

## Web Vitals

[Web Vitals](https://web.dev/vitals/) inicijativa je Googlea za pružanje jedinstvenih smjernica za kvalitetne signale koji su neophodni za pružanje izvrsnog korisničkog iskustva na webu.

- [detaljne upute](https://web.dev/learn-web-vitals/) ([sekvencijalni pregled za lakše učenje](https://developers.google.com/learn/pathways/web-vitals))
- [opis tipičnih uzoraka: carousel, oblici pisama, slike, beskonačno pomicanje itd.](https://web.dev/patterns/web-vitals-patterns/)
- drugi ih navode, npr. [Cloudflare](https://www.cloudflare.com/learning/performance/what-are-core-web-vitals/)

---

## Smanjenje broja HTTP zahtjeva

- Inline JavaScript koda (ako nije velik) u HTML datotekama
- Korištenje [CSS Sprites](https://css-tricks.com/css-sprites/) za prikaz slika (naročito ikona, simbola)
- Korištenje manjeg broja CSS/JavaScript biblioteka i okvira
    - npr. [GitHubova eliminacija jQueryja](https://github.blog/2018-09-06-removing-jquery-from-github-frontend/)
- Spajanje CSS/JavaScript datoteka
    - npr. [Font Awesome](https://fontawesome.com/) (izvorno 6 CSS i 7 JavaScript datoteka se distribuira kao 1 JavaScript i 1 CSS datoteka)
- Korištenje manjih CSS/JavaScript biblioteka i okvira
    - npr. [Ionicons](https://ionicons.com/) (1 JavaScript datoteka, 1300 ikona) umjesto [Font Awesome](https://fontawesome.com/) (1 JavaScript i 1 CSS datoteka, 7800 ikona)

---

## Smanjenje veličine CSS i JavaScript datoteka

Alati kao [minify](https://www.npmjs.com/package/minify) s npm-a mogu iz CSS i JavaScript datoteka bez promjene značenja napisanog koda maknuti:

- Neke razmake
- Znakove novog retka
- Komentare
- Neke znakove za blok koda

> Yet Gmail downloads about three megabytes worth of JavaScript to render its inbox. HEY downloads less than sixty kilobytes. If you can build a Gmail rival with this stack, and have it be met with broad applaud by tens of thousands of paying customers, you can probably build just about anything with it.

Izvor: [Rails 7 will have three great answers to JavaScript in 2021+ (DHH, HEY World)](https://world.hey.com/dhh/rails-7-will-have-three-great-answers-to-javascript-in-2021-8d68191b)

---

## Resursi koji blokiraju prikaz stranice

CSS:

- CSS atribut media omogućuje [označavanje pojedinih CSS datoteka kao neblokirajućih za prikaz stranice](https://developers.google.com/web/fundamentals/performance/critical-rendering-path/render-blocking-css)

JavaScript:

- Moguće je postaviti `<script></script>` na dnu stranice da izvođenje skripte ne prekida nastavak učitavanja HTML-a i resursa koje HTML navodi
- `<script async src="..."></script>` asinkrono učitava skriptu
- `<script defer src="..."></script>` odgađa izvođenje skripte dok se stranica ne učita

---

## Korištenje mreže za distribuciju sadržaja

Mreža za isporuku sadržaja ili mreža za distribuciju sadržaja (engl. *content delivery network*, kraće CDN) je geografski distribuirana mreža proxy poslužitelja i njihovih podatkovnih centara. Cilj je pružiti visoku dostupnost i performanse distribucijom usluge prostorno blizu krajnjih korisnika. ([Wikipedia](https://en.wikipedia.org/wiki/Content_delivery_network))

Rad CDN-ova omogućen je korištenjem [Cross-Origin Resource Sharinga (CORS-a)](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS).

Popularni CDN-ovi:

- [Cloudflare](https://www.cloudflare.com/learning/cdn/what-is-a-cdn/)
- [Microsoft Azure CDN](https://docs.microsoft.com/en-us/azure/cdn/cdn-overview), [Amazon CloudFront](https://aws.amazon.com/cloudfront/)
- [Akamai](https://www.akamai.com/us/en/cdn/what-is-a-cdn.jsp#what-is-a-cdn), [KeyCDN](https://www.keycdn.com/what-is-a-cdn), [Imperva CDN](https://www.imperva.com/learn/performance/what-is-cdn-how-it-works/)
- specifične namjene: [jsDelivr](https://www.jsdelivr.com/about), [cdnjs](https://cdnjs.com/about)

---

## Smanjenje korištenja HTTP 301 Moved Permanently

- Preusmjeravanje uvodi barem jedno dodatno povratno vrijeme slanja zahtjeva i primanja odgovora na zahtjev
- Ilustracija rješenja: [github.com/fidit-rijeka/](https://github.com/fidit-rijeka/) i [github.com/fidit-rijeka](https://github.com/fidit-rijeka) imaju isti sadržaj i nema preusmjeravanja
    - Tražilice kao Google penaliziraju duplicirani sadržaj, ali [postoji rješenje korištenjem kanonskih oznaka](https://moz.com/learn/seo/canonicalization) `<link rel="canonical" href="..." />`

---

## Eliminacija polomljenih poveznica (HTTP 404 Not Found)

Web poslužitelji i web aplikacije izvode usmjeravanje na način:

- potraži statičku datoteku (web poslužitelj izvodi usmjeravanje),
- ako ne postoji datoteka, potraži dinamički resurs (web aplikacija izvodi usmjeravanje, npr. [Django URL dispatcher](https://docs.djangoproject.com/en/3.2/topics/http/urls/), [Rails router](https://guides.rubyonrails.org/routing.html), [Laravel router](https://laravel.com/docs/8.x/routing)),
- ako ne postoji putanja koju web aplikacija poznaje, vrati HTTP statusni kod 404 Not Found i pripadnu stranicu
    - web aplikacija nakon neuspjelog usmjeravanja izgrađuje stranicu pa su greške tipa 404 Not Found vrlo zahtjevne za resursima
    - dobro je periodički provjeravati stranicu za polomljenim poveznicama, naročito na statičke datoteke, npr. [BrokenLinkCheck.com](https://www.brokenlinkcheck.com/), [Ahrefs Broken Link Checker](https://ahrefs.com/broken-link-checker), [dead link checker](https://www.deadlinkchecker.com/) i [Dr. Link Check](https://www.drlinkcheck.com/)

---

## Keširanje

Web aplikacije koje uključuju [Memcached kao potpornu uslugu na način kako to propisuje dvanaest faktora](https://12factor.net/backing-services) se zasigurno nekad pitaju:

> Muči me pitanje  
> O da li bi me volela i da nemam keš?

Izvor: [GAZDA PAJA x MARLON BRUTAL -- DIZEL (OFFICIAL VIDEO)](https://youtu.be/B3IV9lBacyY)

**Keširanje je jedna od najvažnijih, ako ne i najvažnija tehnika za postizanje visokih performansi web aplikacije**. Razlikujemo nekoliko vrsta keširanja:

- keširanje resursa u pregledniku (kod korisnika)
- keširanje odgovora ispred web poslužitelja
- keširanje odgovora u web aplikaciji
- keširanje operacija u interpreteru programskog jezika

---

## Keširanje u pregledniku (1/3)

HTTP zaglavlje `Pragma` ([dokumentacija na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Pragma)) je HTTP/1.0 ekvivalent zaglavlja `Cache-Control` i koristi se samo za kompatibilnost sa starijim sustavima.

HTTP zaglavlje `Cache-Control` ([dokumentacija na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control)):

- `public`, `private`, `no-cache`, `no-store`
- `max-age`, `s-maxage` (vrijeme u sekundama)
- `must-revalidate`, `proxy-revalidate`, `immutable`

Provjerimo zaglavlja (naredba `curl -I`) na:

- [example.group.miletic.net](http://example.group.miletic.net/)
- [www.inf.uniri.hr](https://www.inf.uniri.hr/)

---

## Keširanje u pregledniku (2/3)

Primjer u [Apache HTTP Serveru](https://httpd.apache.org/) s uključenim [mod_headers](https://httpd.apache.org/docs/2.4/mod/mod_headers.html):

``` apache
<IfModule mod_headers.c>
    Header set Cache-Control "max-age=84600, public"
</IfModule>
```

Ilustrativan primjer u čistom PHP-u (Laravel ima klasu [`SetCacheHeaders`](https://laravel.com/api/8.x/Illuminate/Http/Middleware/SetCacheHeaders.html)):

``` php
<?php

header('Cache-Control: max-age=86400, public');
```

Okviri za razvoj aplikacija brinu o keširanju na višoj razini i pritom u odgovorima postavljaju adekvatna HTTP zaglavlja (o tome programeri uglavnom ne brinu).

---

## Keširanje u pregledniku (3/3)

HTTP zaglavlje `Expires` ([dokumentacija na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Expires)) navodi vrijeme isteka resursa (standardna specifikacija dozvoljava do godinu dana); **ako `Cache-Control` navodi `max-age` ili `s-maxage`, sadržaj zaglavlja `Expires` se ignorira**.

Primjer postavki koje generiraju zaglavlje `Expires` u web poslužitelju [Apache HTTP Server](https://httpd.apache.org/) koji koristi [mod_expires](https://httpd.apache.org/docs/2.4/mod/mod_expires.html):

``` apache
ExpiresByType text/html "access plus 1 seconds"
ExpiresByType image/x-icon "access plus 2592000 seconds"
ExpiresByType image/gif "access plus 2592000 seconds"
ExpiresByType image/jpeg "access plus 2592000 seconds"
ExpiresByType image/png "access plus 2592000 seconds"
ExpiresByType text/css "access plus 604800 seconds"
ExpiresByType application/x-javascript "access plus 604800 seconds"
ExpiresByType application/javascript "access plus 604800 seconds"
ExpiresByType application/ecmascript "access plus 604800 seconds"
ExpiresByType application/font-woff "access plus 604800 seconds"
ExpiresByType font/woff2 "access plus 604800 seconds"
```

---

## Keširanje ispred web poslužitelja

Obrnuti HTTP proxy poslužitelj s keširanjem nalazi se ispred web poslužitelja tako da korisnik koji pristupa web poslužitelju prvo pristupa tom proxyju.

Taj proxy služi da korisnicima koji zahtijevaju isto što su oni ili drugi već ranije zahtijevali vrati stvoreni keširani odgovor (npr. rezultat pretrage za artiklima u web trgovini).

Primjer takvog poslužitelja je [Varnish](https://varnish-cache.org/).

![Wikipedia webrequest flow 2020 bg right 90%](https://upload.wikimedia.org/wikipedia/commons/b/b3/Wikipedia_webrequest_flow_2020.png)

---

## Keširanje u web aplikaciji

Kako vidimo na primjeru Wikipedije, aplikacija može i sama spremiti generirani odgovor na primljeni zahtjev u sustav za keširanje i dohvatiti ga kad ponovno dobije isti zahtjev.

Popularni okviri za razvoj aplikacija imaju gotove implementacije sustava za keširanje koji onda umjesto programera brinu o spremanju i dohvaćanju sadržaja te postavljanju odgovarajućih zaglavlja u HTTP odgovorima. Primjeri takvih sustava su:

- [Django's cache framework](https://docs.djangoproject.com/en/3.2/topics/cache/), koristi Memcached ili bazu podataka
- [Laravel Cache](https://laravel.com/docs/8.x/cache), koristi Memcached ili Redis
- [Caching with Rails](https://guides.rubyonrails.org/caching_with_rails.html), koristi Memcached ili Redis

---

## Keširanje operacija u interpreteru programskog jezika

Interpretirani programski jezici pretvaraju kod koji je programer napisao u vlastiti interni kod ([bytecode u Pythonu](https://opensource.com/article/18/4/introduction-python-bytecode), [opcode u PHP-u](https://php.watch/articles/php-dump-opcodes)) koji onda izvode njihove implementacije (CPython u Pythonu, Zend Engine u PHP-u). Te kodove je moguće keširati:

- [How do I create a .pyc file?](https://docs.python.org/3/faq/programming.html?highlight=pyc#how-do-i-create-a-pyc-file)
- [PHP: OPcache](https://www.php.net/manual/en/book.opcache.php)

---

## Korak dalje: just-in-time (JIT) prevođenje u strojni kod

Just-in-time (JIT) prevoditelj unutar interpretera omogućuje prevođenje (dijela) koda dok se interpretirani program izvodi u strojni kod za izravno izvođenje bez interpretera ([više detalja na Wikipediji](https://en.wikipedia.org/wiki/Just-in-time_compilation)).

Implementacije JIT prevoditelja:

- Pythonova službena implementacija [CPython](https://github.com/python/cpython) nema JIT prevoditelj i vjerojatno ga neće imati, implementacije [PyPy](https://www.pypy.org/), [Cinder](https://github.com/facebookincubator/cinder), [Pyston](https://www.pyston.org/) (fokus: web i druge aplikacije opće namjene) i [Numba](https://numba.pydata.org/) (fokus: numerički izračuni) imaju JIT prevoditelj
- PHP [sadrži JIT prevoditelj od verzije 8.0](https://www.php.net/releases/8.0/en.php)
- Ruby [sadrži JIT prevoditelj od verzije 2.6 nadalje](https://blog.heroku.com/ruby-just-in-time-compilation)
- V8 koji Node.js koristi za izvođenje JavaScripta [sadrži JIT prevoditelj](https://v8.dev/blog/jitless)

---

## Brzina rada interpretera programskog jezika

> It's important to be realistic: most people don't care about program performance most of the time.

Izvor: [Laurence Tratt](https://tratt.net/laurie/), [What Challenges and Trade-Offs do Optimising Compilers Face?](https://tratt.net/laurie/blog/entries/what_challenges_and_trade_offs_do_optimising_compilers_face.html)

- [PHP vs JavaScript](https://benchmarksgame-team.pages.debian.net/benchmarksgame/fastest/php.html)
- [PHP vs Python](https://benchmarksgame-team.pages.debian.net/benchmarksgame/fastest/php-python3.html)
- [PHP vs Ruby](https://benchmarksgame-team.pages.debian.net/benchmarksgame/fastest/php-ruby.html)

[Django overview](https://www.djangoproject.com/start/overview/): "Ridiculously fast. Django was designed to help developers take applications from concept to completion as quickly as possible."

[The Rails Doctrine](https://rubyonrails.org/doctrine/): 'One of the early productivity mottos of Rails went: "You're not a beautiful and unique snowflake". It postulated that by giving up vain individuality, you can leapfrog the toils of mundane decisions, and make faster progress in areas that really matter.'

---

## Problemi mjerenja brzine interpretera programskog jezika (1/2)

> Our results show that real web applications behave very differently from the benchmarks and that there are definite ways in which the benchmark behavior might mislead a designer.

Izvor: Paruj Ratanaworabhan, Benjamin Livshits, Benjamin G. Zorn: [JSMeter: Comparing the Behavior of JavaScript Benchmark swith Real Web Applications](https://dl.acm.org/doi/10.5555/1863166.1863169) (2010 USENIX conference on Web application development)
Više informacija: [JSMeter: Measuring JavaScript Web Applications (Microsoft Research)](https://www.microsoft.com/en-us/research/project/jsmeter-measuring-javascript-web-applications/)

---

## Problemi mjerenja brzine interpretera programskog jezika (2/2)

> Our results indicate that just-in-time compilation often increases the execution time for web applications, and that there are large differences in the execution behavior between benchmarks and web applications at the bytecode level.

Izvor: Jan Kasper Martinsen, Håkan Grahn, Anders Isberg: [A Comparative Evaluation of JavaScript Execution Behavior](https://link.springer.com/chapter/10.1007/978-3-642-22233-7_35) (11th International Conference on Web Engineering -- ICWE 2011)

---

## Dohvaćanje imena domene unaprijed (`dns-prefetch`)

Dohvaćanje imena domene unaprijed dobro je rješenje za pronalaženje IP adrese povezane s imenom domene prije nego što korisnik odabere slijediti ("klikne na") poveznicu na tu domenu. Primjer koda u HTML-u:

``` html
<head>
  <!-- ... -->
  <link rel="dns-prefetch" href="https://www.example.com/">
</head>
```

([Više detalja o dns-prefetchu na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTML/Link_types/dns-prefetch) i [njegova primjena u ostvarivanju boljih performansi](https://developer.mozilla.org/en-US/docs/Web/Performance/dns-prefetch), [specifikacija u Resource Hints na W3C-u](https://www.w3.org/TR/resource-hints/#dfn-dns-prefetch).)

---

## Povezivanje unaprijed (`preconnect`)

Povezivanje unaprijed omogućuje pregledniku da uspostavi vezu s poslužiteljem prije nego će doći do potrebe za slanjem HTTP zahtjeva na taj poslužitelj. Pronalaženje IP adrese za domenu putem DNS-a, TCP rukovanje i dogovor oko šifrarnika u TLS-u mogu se pokrenuti unaprijed, čime se eliminra povratno kašnjenje za te veze i štedi vrijeme iz perspektive korisnika.

``` html
<head>
  <!-- ... -->
  <link rel="preconnect" href="https://www.example.com/"
        crossorigin="anonymous">
</head>
```

([Više detalja o preconnectu na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTML/Link_types/preconnect), [specifikacija u Resource Hints na W3C-u](https://www.w3.org/TR/resource-hints/#dfn-preconnect).)

---

## Kompresija

[Gzip](https://www.gzip.org/) ([Wikipedia](https://en.wikipedia.org/wiki/Gzip)) je oblik kompresije koji komprimira HTML, CSS i JavaScript datoteke na poslužitelju prije slanja klijentu. Ovo je jednostavno uključiti u popularnim web poslužiteljima, a može drastično smanjiti količinu podataka koja se prenosi.

Više detalja na MDN-u:

- [HTTP kompresija](https://developer.mozilla.org/en-US/docs/Web/HTTP/Compression)
- [HTTP zaglavlje Accept-Encoding](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Encoding)
- [HTTP zaglavlje Content-Encoding](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Encoding)

---

## Kompresija (Apache HTTP Server)

Primjer u [Apache HTTP Serveru](https://httpd.apache.org/) s uključenim [mod_deflate](https://httpd.apache.org/docs/2.4/mod/mod_deflate.html):

``` apache
<IfModule mod_deflate.c>
    AddOutputFilterByType DEFLATE application/javascript
    AddOutputFilterByType DEFLATE application/x-javascript
    AddOutputFilterByType DEFLATE application/xhtml+xml
    AddOutputFilterByType DEFLATE application/xml
    AddOutputFilterByType DEFLATE font/opentype
    AddOutputFilterByType DEFLATE font/otf
    AddOutputFilterByType DEFLATE font/ttf
    AddOutputFilterByType DEFLATE image/svg+xml
    AddOutputFilterByType DEFLATE image/x-icon
    AddOutputFilterByType DEFLATE text/css
    AddOutputFilterByType DEFLATE text/html
    AddOutputFilterByType DEFLATE text/javascript
    AddOutputFilterByType DEFLATE text/plain
    AddOutputFilterByType DEFLATE text/xml
</IfModule>
```

---

## Kompresija (PHP)

``` php
<?php

$contents = '...'

if (isset($_SERVER['HTTP_ACCEPT_ENCODING']) &&
    str_contains($_SERVER['HTTP_ACCEPT_ENCODING'], 'gzip')) {
  header('Content-Encoding: gzip');
  $compressed_contents = gzencode($contents);
  echo $compressed_contents;
}
else {
  echo $contents;
}
```

---

## Web fontovi

Nedostatak web fontova (npr. [Google Fonts](https://fonts.google.com/) i [Adobe Fonts](https://fonts.adobe.com/)) je u tome što uvode dodatne HTTP zahtjeve za vanjskim resursima. Učitavanje web fontova također blokira prikaz sadržaja (stranice koriste "ružne" fontove dok nisu učitani).

Bolje performanse mogu se ostvariti:

1. Smanjenjem broja stilova (npr. težina 400 (regular) uspravni i zakošeni, težina 700 (bold) samo uspravni)
1. Smanjenjem skupa korištenih znakova (npr. na [latin](https://en.wikipedia.org/wiki/ISO/IEC_8859) i [latin extended](https://en.wikipedia.org/wiki/Latin_Extended-A))
1. Dohvaćanjem fontova unaprijed (npr. prefetch, preconnect)

---

## Prevencija hotlinkanja

Prevencija hotlinkanja odnosi se na ograničavanje skupa prihvatljivih [HTTP referrera](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Referer) kako bi se spriječilo da drugi ugrađuju vaše datoteke (npr. slike i snimke) na web stranice osim vaše, što štedi širinu pojasa zabranom prikazivanja datoteka na drugim web lokacijama.

Primjer korištenjem Apache HTTP Servera i modula [mod_rewrite](https://httpd.apache.org/docs/2.4/mod/mod_rewrite.html):

``` apache
<IfModule mod_rewrite.c>
    RewriteEngine On
    RewriteCond %{HTTP_REFERER} !^http://(.+\.)?example\.com/ [NC]
    RewriteCond %{HTTP_REFERER} !^$
    RewriteRule .*\.(jpe?g|gif|png|mp4)$ http://example.com/nolink.png [L]
</IfModule>
```

---

## HTTP/2

HTTP/2 značajno poboljšava performanse učitavanja web stranice koja ima mnogo slika (za ilustraciju: [Go + HTTP/2 demo poslužitelj](https://http2.golang.org/), [Akamai HTTP/2 poslužitelj](https://http2.akamai.com/demo)).

> HTTP/2 works by making a single connection to the server, and then "multiplexing" multiple requests over that connection to receive multiple responses at the same time.

Izvor: [HTTP/2: A Fast, Secure Bedrock for the Future of SEO (Moz Blog)](https://moz.com/blog/http2-a-fast-secure-bedrock-for-the-future-of-seo)

---

## HTTP/2 u Apache HTTP Serveru

Apache HTTP Server [podržava HTTP/2](https://httpd.apache.org/docs/2.4/howto/http2.html) korištenjem [mod_http2](https://httpd.apache.org/docs/2.4/mod/mod_http2.html). Za HTTP/2 preko TLS-a ("HTTPS") konfiguracija je oblika:

``` apache
Protocols h2 http/1.1
```

Sve varijante HTTP/2 ("HTTPS" i "HTTP"):

``` apache
Protocols h2 h2c http/1.1
```

---

## HTTP/2 demo na example.group.miletic.net

``` shell
$ curl -I http://example.group.miletic.net/
```

``` http
HTTP/1.1 200 OK
Date: Mon, 31 Jan 2022 20:58:34 GMT
Server: Apache/2.4.52 (Debian)
Upgrade: h2,h2c
Connection: Upgrade
Last-Modified: Wed, 25 Aug 2021 14:52:01 GMT
```

``` shell
$ curl -I --http2 http://example.group.miletic.net/
```

``` http
HTTP/1.1 101 Switching Protocols
Upgrade: h2c
Connection: Upgrade

HTTP/2 200
last-modified: Wed, 25 Aug 2021 14:52:01 GMT
```

---

## Utjecaj HTTP/2 na web

> With HTTP2, you no longer pay a large penalty for sending many small files instead of one big file. A single connection can multiplex all the responses you need. No more managing multiple connections, paying for multiple SSL handshakes. This means that bundling all your JavaScript into a single file loses many of its performance benefits (yes, yes, [tree-shaking](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking/) \[link added\] is still one).

Izvor: [Modern web apps without JavaScript bundling or transpiling (DHH, HEY World)](https://world.hey.com/dhh/modern-web-apps-without-javascript-bundling-or-transpiling-a20f2755)

---

## HTTP/3

![the naive engineer geek comic bg 95% left:55%](https://turnoff.us/image/en/http3.png)

## Mazohizam

Izvor: [the naive engineer](https://turnoff.us/geek/the-naive-engineer/) ({turnoff.us})

---

## Dodatne optimizacije

- Optimizacija baze podataka (npr. [upute za MariaDB](https://mariadb.com/kb/en/optimization-and-tuning/): [korištenje indeksa](https://mariadb.com/kb/en/optimization-and-indexes/), [optimizacija upita](https://mariadb.com/kb/en/query-optimizations/); [upute za PostgreSQL](https://www.enterprisedb.com/postgresql-tutorials-resources/performance-tuning-optimization); [optimizacija upita u MongoDB-u](https://docs.mongodb.com/manual/core/query-optimization/))
- Povećanje dostupne memorije sustavu za keširanje (npr. [upute za Memcached](https://github.com/memcached/memcached/wiki/ConfiguringServer))
- Skaliranje aplikacije ([konkurentnost kod dvanestofaktorske aplikacije](https://12factor.net/concurrency)) nije optimizacija

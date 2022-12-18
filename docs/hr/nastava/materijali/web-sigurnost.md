---
marp: true
author: Vedran Miletić
title: Sigurnost web aplikacija
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Sigurnost web aplikacija

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Uvod i motivacija

U svom inboxu pronađete e-mail prikazan na slici s desne strane. Vjerujete li mu i zašto?

![Phishing bg right 95%](https://upload.wikimedia.org/wikipedia/commons/d/d0/PhishingTrustedBank.png)

---

## TLS (SSL) i HTTPS

Naučili smo na *Računalnim mrežama* i *Sigurnosti informacijskih i komunikacijskih sustava*, odnosno *Računalnim mrežama 2*:

- [Transport Layer Security](https://en.wikipedia.org/wiki/Transport_Layer_Security) (TLS), suvremena zamjena za Secure Sockets Layer (SSL)
    - [grupa šifrarnika](https://en.wikipedia.org/wiki/Cipher_suite), npr. `TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256`: ECDHE za razmjenu ključeva, RSA za autentifikaciju, 128-bitni AES (varijanta GCM) za šifriranje podataka prenesenih unutar sesije, SHA256 za hashiranje poruka
    - podržani TLS 1.3 i TLS 1.2, starije verzije se smatraju nesigurnima ([RFC 8996](https://datatracker.ietf.org/doc/html/rfc8996), [detaljna analiza](https://www.michalspacek.com/disable-tls-1.0-and-1.1-today))
- [Hypertext Transfer Protocol Secure](https://en.wikipedia.org/wiki/HTTPS) (HTTPS)
- Simple Mail Transfer Protocol (SMTP) [može koristiti TLS](https://en.wikipedia.org/wiki/Simple_Mail_Transfer_Protocol#Security_extensions)

---

## Evolucija TLS-a

TODO

- Qualys SSL Labs [SSL Server Test](https://www.ssllabs.com/ssltest/)
- Mozillin [SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [Korištenje HTTPS-a može biti faktor u rangiranju u Google rezultatima](https://moz.com/blog/https-is-table-stakes-for-2020-seo)

---

## Napadači ciljaju na web aplikacije

Prema [Impervi](https://www.imperva.com/learn/application-security/application-security/), tvrtci specijaliziranoj za kibernetičku sigurnost:

- Kompleksne aplikacije imaju velik izvorni kod i time velik potencijal za postojanje ranjivosti i zlonamjernih promjena izvornog koda.
- Potencijalne nagrade u slučaju uspješnog napada su visoke vrijednosti, uključujući osjetljive privatne podatke (npr. brojeve kreditnih kartica) prikupljene uspješnom manipulacijom izvornim kodom.
- Napadi se jednostavno izvršavaju jer se većina napada odjednom može lako automatizirati i pokrenuti neselektivno protiv tisuća, desetaka tisuća ili stotina tisuća ciljeva (web aplikacija).

Nedavni primjeri: [Log4j](https://blog.cloudflare.com/inside-the-log4j2-vulnerability-cve-2021-44228/), [JWT biblioteke](https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/)

---

![Security bg 80% left](https://www.monkeyuser.com/assets/images/2017/24-security.png)

## Sigurnost

Izvor: [Security](https://www.monkeyuser.com/2017/security/) (MonkeyUser, 17th January 2017)

---

## Ubrizganje u SQL upit

Ubrizganje u SQL upit (engl. *SQL injection*) događa se kada počinitelj koristi zlonamjerni SQL kod za manipulaciju pozadinskom bazom podataka kako bi otkrio informacije. Posljedice uključuju neovlašteno pregledavanje popisa, brisanje tablica i neovlašteni administrativni pristup ([Imperva](https://www.imperva.com/learn/application-security/sql-injection-sqli/)).

![xkcd 327: Exploits of a Mom](https://imgs.xkcd.com/comics/exploits_of_a_mom.png)

---

## Zaštita od ubrizganja u SQL upit

- [Django](https://docs.djangoproject.com/en/3.2/topics/security/#sql-injection-protection)
- [Laravel](https://laravel.com/docs/8.x/queries)
- [Ruby on Rails](https://guides.rubyonrails.org/security.html#injection)

---

## Cross-site Scripting (XSS)

Skriptiranje na više web sjedišta (engl. *cross-site scripting*, kraće XSS) je injekcijski napad koji cilja korisnike kako bi ostvario pristup računima, aktivirao trojanske programe ili izmijenio sadržaj stranice. Pohranjeni XSS događa se kada se zlonamjerni kôd ubrizga izravno u aplikaciju ([Imperva](https://www.imperva.com/learn/application-security/cross-site-scripting-xss-attacks/)). Reflektirani XSS se javlja kada se zlonamjerna skripta odrazi iz aplikacije u korisnikovu pregledniku ([Imperva](https://www.imperva.com/learn/application-security/reflected-xss-attacks/)).

---

## Zaštita od XSS-a

- [Django](https://docs.djangoproject.com/en/3.2/topics/security/#cross-site-scripting-xss-protection)
- [Laravel](https://laravel.com/docs/8.x/blade)
- [Ruby on Rails](https://guides.rubyonrails.org/security.html#cross-site-scripting-xss)

---

## Cross-site Request Forgery (CSRF)

Krivotvorenje zahtjeva za više web sjedišta (engl. *Cross-site Request Forgery*, kraće CSRF) je napad koji može rezultirati neželjenim prijenosom sredstava, promijenjenim lozinkama ili krađom podataka. Uzrokuje se kada zlonamjerna web aplikacija tjera korisnikov preglednik da izvrši neželjenu radnju na web mjestu na koje je korisnik prijavljen ([Imperva](https://www.imperva.com/learn/application-security/csrf-cross-site-request-forgery/)).

---

## Zaštita od CSRF-a

- [Django](https://docs.djangoproject.com/en/3.2/topics/security/#cross-site-request-forgery-csrf-protection)
- [Laravel](https://laravel.com/docs/8.x/csrf)
- [Ruby on Rails](https://guides.rubyonrails.org/security.html#csrf-countermeasures)

---

## Uključivanje udaljene datoteke

Uključivanje udaljene datoteke (engl. *remote file inclusion*) ubacuje datoteku na poslužitelj web aplikacija i uključuje je kod izvođenja. To može rezultirati izvršavanjem zlonamjernih skripti ili koda unutar aplikacije, kao i krađom podataka ili manipulacijom ([Imperva](https://www.imperva.com/learn/application-security/rfi-remote-file-inclusion/)).

Uključivanje udaljene datoteke sprječava se sanitizacijom korisničkog unosa, provjerom izlaza na poslužitelju i ograničavanjem vrsta datoteka s kojima aplikacija radi na poznat popis (npr. PDF, PNG, JPEG i GIF).

---

## Cross-Origin Resource Sharing (CORS)

Prema [Wikipediji](https://en.wikipedia.org/wiki/Cross-origin_resource_sharing):

Dijeljenje resursa s više podrijetla (engl. *Cross-Origin Resource Sharing*, kraće CORS) je mehanizam koji omogućuje da se resursi na web stranici **kojima je u zadanim postavkama ograničen pristup** zatraže s druge domene izvan domene u odnosu na onu s koje je prvi resurs poslužen.

- [CORS na MDN-u](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)
- [Access-Control-Allow-Origin](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin)
- [Content-Security-Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Security-Policy)

---

## Web application firewall

Prema [Wikipediji](https://en.wikipedia.org/wiki/Web_application_firewall):

Vatrozid web aplikacije (engl. *web application firewall*, kraće WAF) je vatrozid koji filtrira, nadgleda i blokira HTTP promet prema i od web servisa ili aplikacije. Pregledom HTTP prometa može spriječiti napade koji iskorištavaju poznate ranjivosti web aplikacija, poput ubrizganja u SQL upit, skriptiranja na više web sjedišta (XSS), uključivanja udaljene datoteka i nepravilne konfiguracija sustava ([Cloudflareova ilustracija rada](https://www.cloudflare.com/learning/ddos/glossary/web-application-firewall-waf/)).

Implementacije u oblaku:

- [AWS Web Application Firewall](https://aws.amazon.com/waf/)
- [Azure Web Application Firewall](https://azure.microsoft.com/en-us/services/web-application-firewall/)
- [Cloudflare Web Application Firewall](https://www.cloudflare.com/waf/)

Implementacije otvorenog koda: [Shadow Daemon](https://shadowd.zecure.org/), [ModSecurity](https://www.modsecurity.org/) i druge.

---

## Open Web Application Security Project (OWASP)

[Open Web Application Security Project](https://owasp.org/) (OWASP) je neprofitna zaklada koja radi na poboljšanju sigurnosti programske podrške na webu kroz softverske projekte otvorenog koda, stotine lokalnih organizacija širom svijeta, desetke tisuća članova i obrazovne konferencije.

Najzanimljiviji među [projektima koje provode](https://owasp.org/projects/):

- [OWASP Top Ten](https://owasp.org/www-project-top-ten/) ([novi web u izradi](https://www.owasptopten.org/))
- [OWASP Mobile Security Testing Guide](https://owasp.org/www-project-mobile-security-testing-guide/)
- [OWASP ModSecurity Core Rule Set](https://owasp.org/www-project-modsecurity-core-rule-set/)

---

## TODO Dodati

- API key, JWT token, OAuth

---
marp: true
author: Vedran Miletić
title: Programiranje za web / Dinamičke web aplikacije 2
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Uvodno predavanje predmeta Programiranje za web

Studij: Preddiplomski studij informatike  
Modul: Razvoj programske potpore (RPP)  
Godina i semestar: 3. godina, 5. semestar

Nositelj: **doc. dr. sc. Vedran Miletić**, vmiletic@inf.uniri.hr, vedran.miletic.net

Asistent: **Milan Petrović, mag. inf.**, milan.petrovic@inf.uniri.hr

Fiksni termin za konzultacije: nakon predavanja, odnosno prije vježbi uživo u učionici ili na [sveučilišnom BBB-u: bigbluebutton.uniri.hr/b/ved-h7x-6vq](https://bbb.uniri.hr/b/ved-h7x-6vq)

Naziv predmeta na starom jednopredmetnom i dvopredmetnom studiju informatike: *Dinamičke web aplikacije 2*, 3. godina, 6. semestar  
📝 **Napomena:** Dio studenata polaže predmet u okviru [Veleri-OI IoT School](https://iot-school.veleri.hr/)

---

## Razvoj aplikacija u praksi: web, mobilne i stolne (1/2)

> (...) since the early 2000s, **the number of web development jobs (and within the past decade, mobile jobs) has increased exponentially** (...)
> (...) it's clear that a number of the most-used languages are utilized in many different contexts, including web, mobile, and desktop; however, it's pretty clear that **most usage targets mobile and the web**; just look at how Apple has been trying to desperately convince developers to build for macOS (...)
> (...) JetBrains survey of Python development in 2018 showed data analysis as the most popular (58 percent) with **web development at 52 percent**, machine learning at 38 percent and **explicit desktop development at just 19 percent** (...)

Izvor: [Is Desktop Development Dead? Or Still Worth It?](https://insights.dice.com/2020/03/04/desktop-development-dead-still-worth-it/) (Dice Insights, 4th March 2020)

---

## Razvoj aplikacija u praksi: web, mobilne i stolne (2/2)

> (...) I noticed my wife was using Google Maps on her PC to map something. I asked her why she was using Google Maps instead of our old pal \[Microsoft\] Streets and Trips, and she said, quite matter of factly, "Because it's faster."
> (...) \[Microsoft\] Streets and Trips seems to be completely stuck in the old world mentality of toolbars, menus, and right-clicking. **All the innovation in user interface seems to be taking place on the web**, and desktop applications just aren't keeping up. **Web applications are evolving online at a frenetic pace**, while most desktop applications are mired in circa-1999 desktop user interface conventions, plopping out yearly releases with barely noticeable new features (...)

Izvor: [Who Killed the Desktop Application?](https://blog.codinghorror.com/who-killed-the-desktop-application/) (Coding Horror, 7th June 2007)

---

## Opće informacije o predmetu

- Usvajanje složenijih koncepta iz područja razvoja web aplikacija i usluga
- Otvara vrata za rad na razvoju web (i mobilnih) aplikacija i usluga u Hrvatskoj i šire: [FIVE](https://five.agency/our-work/), [Infobip](https://www.infobip.com/), [Mono](https://mono.software/products/), [Infinum](https://infinum.com/work), [Factory](https://factory.hr/our-work), [Hexis](https://hexis.hr/nasi-radovi/), [Bornfight](https://bornfight.com/work/), [Neuralab](https://www.neuralab.net/portfolio-work/), [Aplitap](https://www.aplitap.hr/radovi), [Lumos](https://lumos.hr/portfolio/), [Perpetuum](https://www.perpetuum.hr/projekti), [Init](https://init.hr/#ourwork), [Q agency](https://q.agency/projects), [Own.Solutions](https://own.solutions/), [Papar](https://papar.hr/portfolio/), [Async Labs](https://www.asynclabs.co/work) itd.
    - Primjeri poslova: [Web Developer (Indeed)](https://www.indeed.com/q-Web-Developer-jobs.html)
- Znanja primjenjiva na brojnim predmetima diplomskog studija informatike: *Elektroničko gospodarstvo*, *Programsko inženjerstvo*, *Multimedijski i hipermedijski sustavi*, *Upravljanje mrežnim sustavima* te *Informacijska tehnologija i društvo*

---

## Ciljevi predmeta

- *Naučiti izraditi čitavu web aplikaciju*; UPW/DWA1 započinje:
    - Odabir programskih alata i postavljanje razvojnog okruženja
    - Razvoj prednjeg dijela web aplikacije (korisničko sučelje vrši upite na nerelacijskoj bazi podataka i/ili šalje HTTP zahtjeve putem REST API-ja)
- PW/DWA2 nastavlja:
    - Razvoj stražnjeg dijela web aplikacije (monolit, mikrousluge)
    - Korištenje objektno orijentiranog modeliranja i programiranja na webu
    - Povezivanje s bazom podataka, pretvorba objektnog u relacijski model
    - Faktori razvoja koji olakšavaju postavljanje i održavanje aplikacije
    - Testiranje i automatizacija testiranja u sustavu koninuirane integracije
    - Poboljšanje performansi i izazovi sigurnosti aplikacije

---

![Now you can explain your grandma what is that you are doing bg 95% left:55%](https://www.monkeyuser.com/assets/images/2017/61-web-app-visualized.png)

## Vizualizacija web aplikacije

Izvor: [Web App - Visualized](https://www.monkeyuser.com/2017/web-app-visualized/) (MonkeyUser, 19th September 2017)

---

## Sintetski predmet

- Na PW se koriste znanja iz predmeta:
    - *Uvod u programsko inženjerstvo*, *Objektno programiranje*, *Baze podataka*
    - *Računalne mreže*, *Uvod u programiranje za web* (vjerovali ili ne)
- Također su korisna znanja iz predmetima:
    - *Upravljanje informatičkim projektima*, *Razvoj informacijskih sustava*
    - *Sigurnost informacijskih i komunikacijskih sustava*
- Na DWA2 se koriste znanja iz predmeta *Dinamičke web aplikacije 1*, *Objektno orijentirano programiranje*, *Uvod u baze podataka*, *Operacijski sustavi 1*, *Operacijski sustavi 2* i *Računalne mreže 2*, a također su korisna znanja iz predmeta *Objektno orijentirano modeliranje* i *Uvod u programsko inženjerstvo*
- Ukupna koristi koju ćete imati od pohađanja predmeta ovisi o vašem predznanju

---

## "You know what I like about *courses*? They stack so well."

- **Programske paradigme i jezici**, **Deklarativni programski jezici**:
    - funkcijska paradigma, objektna paradigma, generičko programiranje
    - skriptni jezici, dinamički i statički tipovi podataka
- **Mrežni i mobilni operacijski sustavi**, **Upravljanje računalnim sustavima**
    - postavljanje poslužitelja, odnosno postavljanje poslužiteljskog dijela web aplikacije korištenjem usluga oblaka
    - postavljanje klijentskog dijela web aplikacije na mobilni uređaj
- **Komunikacijske mreže**: umrežavanje poslužitelja u podatkovnom centru, oblaku ili na rubu mreže (engl. *edge computing*)
- **Administriranje i sigurnost baza podataka**, **Dizajn korisničkog sučelja i iskustva**: *nomen est omen*

---

## Developer Roadmaps

Isprepletenost predmeta na studiju informatike nije slučajnost.

Za ilustraciju, slične uzorke vidimo na web stranici [roadmap.sh](https://roadmap.sh/) (izvorni kod na [GitHubu](https://github.com/kamranahmedse/developer-roadmap)) koja nudi tri putanje za učenje:

- [Frontend](https://roadmap.sh/frontend) ([slika](https://roadmap.sh/roadmaps/frontend.png))
- [Backend](https://roadmap.sh/backend) ([slika](https://roadmap.sh/roadmaps/backend.png))
- [DevOps](https://roadmap.sh/devops) ([slika](https://roadmap.sh/roadmaps/devops.png))

![A3 greece road map bg 95% right](https://upload.wikimedia.org/wikipedia/commons/d/db/A3_greece_road_map.jpg)

---

## Literatura za predavanja

📝 **Napomena:** Literatura nije fokusirana na neki jezik i okvir za razvoj web aplikacija jer se oni mijenjaju, dok temeljna načela programskog inženjerstva na webu ostaju.

- [MDN Web Docs](https://developer.mozilla.org/) (slučajno pokriva i vježbe: [/Learn/Server-side/Django](https://developer.mozilla.org/en-US/docs/Learn/Server-side/Django))
- Prezentacije (izrađene u [Marp](https://marp.app/)-u) i *vaše bilješke* s predavanja
- Fowler, M., Rice, D., Foemmel, M., Hieatt, E., Mee, R., Stafford, R. [Patterns of Enterprise Application Architecture](https://www.informit.com/store/patterns-of-enterprise-application-architecture-9780321127426). (Addison-Wesley, 2002).
- McConnel, S. [Code Complete: A Practical Handbook of Software Construction](https://www.microsoftpressstore.com/store/code-complete-9780735619678). (Microsoft Press, 2004.)
- Swartz, A. A Programmable Web: An Unfinished Work. (Morgan & Claypool Publishers, 2013.) [doi:10.2200/S00481ED1V01Y201302WBE005](https://doi.org/10.2200%2FS00481ED1V01Y201302WBE005)
- Wiggins, A. [The Twelve-Factor App](https://12factor.net/). (Heroku, 2017).

---

![Grgo Šipek bg 95% left:55%](https://upload.wikimedia.org/wikipedia/commons/9/94/Grgo_%C5%A0ipek_-_Gr%C5%A1e.jpg)

## Službena PW/DWA2 glazbena tema za vježbe

> Plešeš po metcima tango, wild west -- **Django**

Izvor: [GRŠE -- HIGHLIFE (OFFICIAL VIDEO)](https://youtu.be/Psqu7lQkWUw)

Prerade tipa "Grše tvrđi nego bilo koji [devops](https://www.ryadel.com/en/devops-methodology-lifecycle-best-practices-explained/)", "Grše voli JS, Grše nije [Peter Wayner](https://www.infoworld.com/article/3072163/7-programming-languages-we-love-to-hate-but-cant-live-without.html)" i "Grše GitHuber veći nego [Ilya Kantor](https://github.com/iliakan)" su više nego poželjne.

---

## Okviri i programski jezici

U popisu su *zakošeni jezici u kojima su primjeri koda na predavanjima*, **podebljani okviri korišteni na vježbama iz UPW/DWA1 i PW/DWA2, odnosno Veleri-OI IoT School**:

- **[Express.js](https://expressjs.com/)** (*[Node.js](https://nodejs.org/)*), [Laravel](https://laravel.com/) i [Lumen](https://lumen.laravel.com/) (*[PHP](https://www.php.net/)*)
- **[Django](https://www.djangoproject.com/)** i [Flask](https://flask.palletsprojects.com/) (*[Python](https://www.python.org/)*), [Ruby on Rails](https://rubyonrails.org/) (*[Ruby](https://www.ruby-lang.org/)*)
- [ASP.NET](https://dotnet.microsoft.com/apps/aspnet) ([C#](https://docs.microsoft.com/en-us/dotnet/csharp/)), [Spring](https://spring.io/) ([Java](https://www.java.com/))
- brojni drugi; neki od popisa po popularnosti su [Statistics & Data](https://statisticsanddata.org/data/most-popular-backend-frameworks-2012-2021/), [Statista](https://www.statista.com/statistics/1124699/worldwide-developer-survey-most-used-frameworks-web/) i [StackOverflow Developer Survey](https://insights.stackoverflow.com/survey/2021), [dio Technology](https://insights.stackoverflow.com/survey/2021#technology)
- zanimljivi primjeri: [Phoenix Framework](https://www.phoenixframework.org/) ([Elixir](https://elixir-lang.org/)), [Rocket](https://rocket.rs/) ([Rust](https://www.rust-lang.org/))

---

## Node.js

``` javascript
const http = require('http');

const server = http.createServer((req, res) => {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'text/html');
  res.end('<p>Hello, world!</p>\n');
});

const host = 'localhost';
const port = 8000;

server.listen(port, host, () => {
  console.log('Web server running at http://%s:%s', host, port);
});
```

---

## PHP

Recimo da je datoteka `public/index.php` sadržaja:

``` php
<?php

http_response_code(200);
header('Content-Type: text/html');
echo '<p>Hello, world!</p>\n';
```

Uočite kako HTTP poslužitelj nije dio koda. Pokreće se na način:

``` shell
$ php -S localhost:8000 -t public
[...] PHP 8.0.8 Development Server (http://localhost:8000) started
```

---

![It's all about making the wrong choice at the wrong time bg 95% left:55%](https://www.monkeyuser.com/assets/images/2021/208-masochism.png)

## Mazohizam

Izvor: [Masochism](https://www.monkeyuser.com/2021/masochism/) (MonkeyUser, 9th March 2021)

Pripadna glazbena tema:

> Ak' si glup, kako kažu "Nemoj izlaziti iz mečke"  
> Da ne bi upoznao naše zločeste plave dečke

Izvor: [TRAM 11 - Kužiš spiku (Remix) feat. Phat Phillie](https://youtu.be/wrwkn9qQYFM)

---

## Python

``` python
import http.server
import socketserver
from http import HTTPStatus

class Handler(http.server.SimpleHTTPRequestHandler):
  def do_GET(self):
    self.send_response(HTTPStatus.OK)
    self.send_header('Content-Type', 'text/html')
    self.end_headers()
    self.wfile.write(b'<p>Hello, world!</p>\n')

server = socketserver.TCPServer(('localhost', 8000), Handler)
server.serve_forever()
```

---

## Ruby

``` ruby
require 'socket'

server = TCPServer.new('localhost', 8000)

loop do
  socket = server.accept
  request = socket.gets
  STDERR.puts request

  response = "<p>Hello, world!</p>\n"
  socket.print "HTTP/1.1 200 OK\r\n" +
               "Content-Type: text/html\r\n" +
               "Content-Length: #{response.bytesize}\r\n" +
               "Connection: close\r\n"
  socket.print "\r\n"
  socket.print response
  socket.close
end
```

---

## Uređivači koda

- [Atom](https://atom.io/)
- [Brackets](https://brackets.io/)
- [GNU Emacs](https://www.gnu.org/software/emacs/)
- [Sublime Text](https://www.sublimetext.com/)
- [Vim](https://www.vim.org/about.php)

![Screenshot of GNU Emacs 26.2 bg 90% right](https://upload.wikimedia.org/wikipedia/commons/b/b3/GNU_Emacs_26.2_screenshot.png)

---

## Razvojna okruženja

- [Adobe Dreamweaver](https://www.adobe.com/products/dreamweaver.html)
- [Apache NetBeans](https://netbeans.apache.org/)
- [CodeLite](https://codelite.org/)
- [Eclipse](https://www.eclipse.org/downloads/packages/), [Eclipse Che](https://www.eclipse.org/che/)
- [JetBrains](https://www.jetbrains.com/products/)
- [Komodo IDE](https://www.activestate.com/products/komodo-ide/)
- [Visual Studio Code](https://code.visualstudio.com/) ⬅️

![Screenshot of PyCharm Community Edition 2021.1 bg 95% right:60%](https://upload.wikimedia.org/wikipedia/commons/8/83/PyCharm_2021.1_Community_Edition_screenshot.png)

---

## Platorme tipa low-code i no-code

- [Microsoft PowerApps](https://powerapps.microsoft.com/), [Oracle Application Express (APEX)](https://apex.oracle.com/)
- [Mendix](https://www.mendix.com/) (za ilustraciju: [A Tour of the Mendix Platform](https://mendix.hubs.vidyard.com/watch/cdX4pAssyUkHiZDm2AyfbS))
- [Appian](https://www.appian.com/), [Appy Pie](https://www.appypie.com/), [Zoho Creator](https://www.zoho.com/creator/), [OutSystems](https://outsystems.com/):

    > Low-code and no-code tools are increasingly playing a crucial role in speeding up the delivery of applications. Gartner predicts that by 2023, over 50% of medium to large enterprises will have adopted a low-code or no-code as one of their strategic application platforms and that low-code will be responsible for more than 65% of application development activity by 2024.

    Izvor: [Low-Code and No-Code: What's the Difference and When to Use What?](https://www.outsystems.com/blog/posts/low-code-vs-no-code/) (Forsyth Alexander, [OutSystems](https://outsystems.com/), 8th January 2021)

---

## Izgled web sjedišta

Korišteni programski jezik i okvir ne određuje izgled stranice.

Primjerice, u Ruby on Railsu su razvijene (prema [BuiltWith](https://builtwith.com/)):

- [gitlab.com](https://about.gitlab.com/), [github.com](https://github.com/)
- [www.ssmb.hr](http://www.ssmb.hr/), [bbb.uniri.hr](https://bbb.uniri.hr/)
- [rabljenavozila.dacia.hr](https://rabljenavozila.dacia.hr/), [rabljenavozila.renault.hr](https://rabljenavozila.renault.hr/)

![Cat reading web page bg 95% right:55%](https://live.staticflickr.com/3160/3092987032_fe2f2d5fda_b.jpg)

---

## Izgled i korisničko iskustvo web sjedišta

Izgled web sjedišta i korisničko iskustvo određuju HTML, CSS i JS, danas uglavnom na prednjem dijelu web aplikacija: [Bootstrap](https://getbootstrap.com/) (MMS), [Vue.js](https://vuejs.org/) i [Quasar](https://quasar.dev/) (UPW/DWA1), [Headless UI](https://headlessui.dev/) i [Tailwind CSS](https://tailwindcss.com/) (PW/DWA2) ([ostale](https://www.keycdn.com/blog/frontend-frameworks) [mogućnosti](https://athemes.com/collections/best-css-frameworks/)).

> To me it's obvious that the days of low level JavaScript for front end applications for the average developer are numbered and **baseline frameworks [React, Vue.js, Angular, Svelte etc.] are going to be the future for the majority of Web developers** that need to get stuff done. **The productivity gain** and the fact that the frameworks encapsulate hard-won knowledge and experience about quirks and performance in the DOM and JavaScript, **make it impractical to ‘roll your own' any longer** and stay competitive in the process.

Izvor: [The Rise of JS Frameworks -- Part 1: Today](https://weblog.west-wind.com/posts/2015/Jul/18/The-Rise-of-JavaScript-Frameworks-Part-1-Today) ([Rick Strahl](https://weblog.west-wind.com/), 18th July 2015)

---

## Aktivnosti i ocjenjivanje studenata

- **Pohađanje nastave**: obavezno, kao i praćenje obavijesti na [Merlinu](https://moodle.srce.hr/2022-2023/)
    - predavanja u učionici, iznimno online (sveučilišni BigBlueButton): predmet je sintetski i zbog toga su *predavanja zamišljena kao interaktivna*
    - vježbe u učionici, iznimno online (sveučilišni BigBlueButton)
- **Test na Merlinu (30 bodova)**: pokriva gradivo predavanja
    - pišete ga u računalnoj učionici i možete za vrijeme pisanja testa koristiti *svoje bilješke* s predavanja; pitanja slična primjerima na predavanjima
- **Praktični kolokvij (20 bodova)**: pokriva osnovni dio gradiva vježbi
    - prag za pravo pristupa završnom ispitu iznosi 50% ostvarenih bodova; postoji mogućnost ispravka kolokvija u posljednjem tjednu
- **Izrada modela i predložaka (pogleda) za web aplikaciju ili uslugu (20 bodova)**
- **Završni ispit (30 bodova)**: timski razvoj web aplikacije ili usluge na temelju izrađenih modela i predložaka (pogleda) te prezentacija projekta

---

## Projekt

- Grupni razvoj čitave web aplikacije na određenu temu (ponuđenu ili vaš prijedlog)
    - npr. razviti web aplikaciju za karting klub, booking treninga u dvorani ili nastave u učionicama, praćenje rezultata nogometnih utakmica itd.
    - moguće je proširiti i dopuniti projekte s drugih predmeta, npr. UPW/DWA1
- Programski jezik po želji (osim dosad spomenutih, tu su i C, Perl, C++, Go, Scala, Erlang, Swift itd.), okvir(i) i biblioteke po želji; preporuke:
    - za monolit: [Django](https://www.djangoproject.com/), [Ruby on Rails](https://rubyonrails.org/) ili [Laravel](https://laravel.com/)
    - za mikrouslugu i sučelje: [Flask](https://flask.palletsprojects.com/) i [PonyORM](https://ponyorm.org/) ili [Express](https://expressjs.com/) + [Vue.js](https://vuejs.org/) ili [React](https://reactjs.org/)
- Dokumentacija unutar programskog koda obavezna ([Docstring](https://www.python.org/dev/peps/pep-0257/) u Pythonu i Elixiru, [JSDoc](https://jsdoc.app/) u JavaScriptu, [Doxygen](https://www.doxygen.nl/) u C/C++-u i slično u drugim jezicima)
- Izgled: potrudite se (preporuka: [Tailwind CSS](https://tailwindcss.com/) i [Headless UI](https://headlessui.dev/), mnogo gotovih besplatno dostupnih komponenata postoji na [Tailwind UI](https://tailwindui.com))

---
marp: true
author: Vedran Miletić
title: "Arhitekture web aplikacija: monolitna i uslužno orijentirana"
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Arhitekture web aplikacija: monolitna i uslužno orijentirana

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Monolitna arhitektura (1/4)

🙋 **Pitanje:** Je li ovo monolitna web aplikacija?

``` html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>title</title>
    <link rel="stylesheet" href="style.css">
    <script src="script.js"></script>
  </head>
  <body>
    <p>Hello, world!</p>
  </body>
</html>
```

---

## Monolitna arhitektura (2/4)

🙋 **Pitanje:** Je li ovo monolitna web aplikacija?

``` php
<?php

echo '<p>Hello, world!</p>\n';
```

---

## Monolitna arhitektura (3/4)

Prema [Wikipediji](https://en.wikipedia.org/wiki/Monolithic_application):

- aplikacija kojoj su korisničko sučelje i kod za pristup podacima su *dio istog programa* na istoj platformi (platforma je ovdje OS, programski jezik i biblioteke)
- aplikacija izvodi svaki korak potreban da bi se određena radnja izvela
    - npr. restoran ima ponudu dnevnih specijaliteta dostupnih za van, online narudžbu, dostavu i knjigu žalbe te je sva ta funkcionalnost implementirana unutar jedne aplikacije
- izvorno je termin označavao aplikacije namijenjene za izvođenje na mainframeu pisane bez mogućnosti ponovnog iskorištavanja koda
- može biti *višeslojna* (engl. *multitier*, *multilayer*) kada odvaja pojedine funkcije u zasebne dijelove aplikacije

---

## Monolitna arhitektura (4/4)

🙋 **Pitanje:** Je li ovo monolitna web aplikacija?

``` html
<html>
  <head>
    <title>Templating in Flask</title>
  </head>
  <body>
    <h1>Hello {{ user }}!</h1>
    <p>Welcome to the world of Flask!</p>
  </body>
</html>
```

``` python
from flask import Flask, render_template

app = Flask(__name__)
@app.route('/hello/<user>')
def hello_world(user=None):
  return render_template('index.html', user=user)
```

---

## Višeslojna arhitektura

Prema [Wikipediji](https://en.wikipedia.org/wiki/Multitier_architecture):

- odvojeni su prezentacijski sloj (korisničko sučelje), poslovna logika (obrada podataka) i sučelje prema pohrani podataka (najčešće relacijskoj bazi)
- programeri mogu dodavati ili mijenati funkcionalnost unutar određenog sloja umjesto rada odjednom na čitavoj aplikaciji, npr. [Djangov model-pogled-predložak](https://docs.djangoproject.com/en/3.2/faq/general/#django-appears-to-be-a-mvc-framework-but-you-call-the-controller-the-view-and-the-view-the-template-how-come-you-don-t-use-the-standard-names)

![Overview of a three-tier application bg 95% right](https://upload.wikimedia.org/wikipedia/commons/5/51/Overview_of_a_three-tier_application_vectorVersion.svg)

---

![Conway's Law bg 95% left:60%](https://i0.wp.com/www.comicagile.net/wp-content/uploads/2021/05/761C8B2B-C2F7-4100-99A2-7FD502CB1E78.jpeg)

## Conwayev zakon

Izvor: [Conway's Law](https://www.comicagile.net/comic/conways-law/) (Comic Agilé #85)

Prema [Wikipediji](https://en.wikipedia.org/wiki/Conway%27s_law), američki računalni znanstvenik [Melvin Conway](https://en.wikipedia.org/wiki/Melvin_Conway) je rekao:

> Svaka organizacija koja dizajnira sustav (široko definiran) proizvest će dizajn čija je struktura kopija komunikacijske strukture organizacije.

---

## Monolitna i mikrouslužna arhitektura

![Monolithic architecture vs microservices architecture](https://devopedia.org/images/article/129/3581.1541934780.png)

---

## Mikrouslužna arhitektura (1/3)

Prema [Wikipediji](https://en.wikipedia.org/wiki/Microservices):

- usluge (engl. *services*) su procesi koji komuniciraju putem mreže korištenjem protokola neovisnog o tehnologiji (programski jezici, biblioteke i okviri)
    - najčešće se koristi HTTP ([REST](https://restfulapi.net/), [GraphQL](https://graphql.org/)), osobito u komunikaciji prednjeg dijela web aplikacije s mikrouslugama na stražnjem dijelu
    - mogu se koristiti i drugi protokoli, npr. [AMQP](https://www.amqp.org/) ([RabbitMQ](https://www.rabbitmq.com/)), [Enduro/X](https://www.mavimax.com/products/endurox), [MQTT](https://mqtt.org/) i [ZMTP](https://rfc.zeromq.org/spec/23/) ([ZeroMQ](https://zeromq.org/)) i osobito su prikladni u situacijama kada je potrebno implementirati [složenije komunikacijske uzorke](https://zguide.zeromq.org/docs/preface/)

---

## Mikrouslužna arhitektura (2/3)

Prema [Wikipediji](https://en.wikipedia.org/wiki/Microservices) (nastavak):

- usluge su organizirane ovisno o poslovnim potrebama (npr. servis za upis studenata u višu godinu, servis za rad s ispitnim rokovima i servis za ispis svjedodžbi mogu biti različite aplikacije koje rade nad istom relacijskom bazom)
- usluge mogu biti organizirane u više slojeva (npr. [glina i kamenje](https://philcalcado.com/2018/09/24/services_layers.html))
- usluge mogu po potrebi biti implementirane u različitim programskim jezicima, koristiti iste ili različite baze podataka, izvoditi se na različitom hardveru na odvojenim fizičkim lokacijama
- usluge su malene, komuniciraju porukama, neovisno razvijane i postavljene, kontinuirano isporučene, decentralizirane i jednostavne za skaliranje

---

## Mikrouslužna arhitektura (3/3)

Mikrousluge u suštini slijede [Unixovu filozofiju](https://en.wikipedia.org/wiki/Unix_philosophy):

> Make each program **do one thing well**. To do a new job, build afresh rather than complicate old programs by adding new "features".

[Martin Fowler](https://martinfowler.com/) kaže za [arhitekturu temeljenu na mikrouslugama](https://martinfowler.com/articles/microservices.html):

> Lends itself to a continuous delivery software development process. **A change to a small part of the application only requires rebuilding and redeploying only one or a small number of services.**
> Adheres to principles such as fine-grained interfaces (to independently deployable services), business-driven development (e.g. domain-driven design).

---

### Monolitna ili mikrouslužna arhitektura

🙋 **Pitanje:** Imaju li navedene aplikacije monolitnu ili mikroservisnu arhitekturu?

- [www.inf.uniri.hr](https://www.inf.uniri.hr/) (iskoristite [View Source](https://developer.mozilla.org/en-US/docs/Tools/View_source) u pregledniku ili proučite [BuiltWith technology profile za inf.uniri.hr](https://builtwith.com/?https://www.inf.uniri.hr/))
- [www.facebook.com](https://www.facebook.com/) (pogledajte [Building Real Time Infrastructure at Facebook - Facebook - SRECon2017](https://youtu.be/ODkEWsO5I30))
- [www.instagram.com](https://www.instagram.com/) ([Scaling Instagram Infrastructure](https://youtu.be/hnpzNAPiC0E))
- [hr.wikipedia.org](https://hr.wikipedia.org/) ([Wikimedia servers](https://meta.wikimedia.org/wiki/Wikimedia_servers))
- [moodle.srce.hr](https://model.srce.hr/) ([Moodle architecture](https://docs.moodle.org/dev/Moodle_architecture))
- [www.openstreetmap.org](https://www.openstreetmap.org/) ([Component overview](https://wiki.openstreetmap.org/wiki/Component_overview))
- [razmjena.org](https://razmjena.org/) (pronađite korištenu tehnologiju u footeru stranice)

---

## Primjene mikrouslužne arhitekture

- oblaku urođene aplikacije (engl. *cloud-native applications*), odnosno [softver kao usluga](https://en.wikipedia.org/wiki/Software_as_a_service) (engl. *software as a service*)
    - npr. [Amazon, Netflix, Etsy](https://divante.com/blog/10-companies-that-implemented-the-microservice-architecture-and-paved-the-way-for-others/) i [fiktivna e-prodaja](https://microservices.io/patterns/microservices.html)
- neke (web) aplikacije razvijene za tzv. [bezposlužiteljsko računarstvo](https://en.wikipedia.org/wiki/Serverless_computing) (engl. *serverless computing*) (koriste poslužitelje, ali ih vi ne konfigurirate)
    - npr. [Šahovski klub Rječina](https://skrjecina.hr/) (diplomski rad [Antonija Bana](https://www.linkedin.com/in/antonio-ban-ba97b689/)) koji se izvodi na [AWS Lambda](https://aws.amazon.com/lambda/) (slične usluge: [Google Cloud Functions](https://cloud.google.com/functions/) i [Azure Functions](https://azure.microsoft.com/en-us/services/functions/))
- neke (web) aplikacije koje se isporučuju u [kontejnerima](https://en.wikipedia.org/wiki/OS-level_virtualization)
    - npr. [Jenkins](https://www.jenkins.io/doc/book/installing/) (dva kontejnera) i [Alfresco](https://docs.alfresco.com/content-services/6.0/install/containers/) (nekoliko nužnih, deseci opcionalnih kontejnera), ali ne i [Discourse](https://github.com/discourse/discourse/blob/master/docs/INSTALL.md) (monolitna aplikacija u kontejneru)
- velik broj (web) aplikacija koje se danas razvijaju po narudžbi
    - npr. vaš projekt s UPW/DWA1 ili Veleri-OI IoT School

---

## Evolucija računarstva do besposlužiteljskog

![Pre-Cloud, IaaS, PaaS, and Serverless](https://www.instana.com/media/serverless-evolution.jpeg)

Izvor: [Introduction to Serverless Computing](https://www.instana.com/blog/introduction-to-serverless-computing/) (Instana, 18th April 2018)

---

## Usporedba monolitne i mikrouslužne arhitekture

![Monolith, Microservices, FaaS](https://www.instana.com/media/serverless-components.png)

Izvor: [Introduction to Serverless Computing](https://www.instana.com/blog/introduction-to-serverless-computing/) (Instana, 18th April 2018)

---

## Bezposlužiteljsko računarstvo

- komercijalno se sreće pod nazivom funkcija kao usluga (engl. *function as a service*): [AWS Lambda](https://aws.amazon.com/lambda/), [Azure Functions](https://azure.microsoft.com/en-us/services/functions/), [Google Cloud Functions](https://cloud.google.com/functions) itd.
- bezposlužiteljska baza podataka (engl. *serverless database*): [Nutanix Era](https://www.nutanix.com/products/era), [Amazon Aurora](https://aws.amazon.com/rds/aurora/), [Azure Data Lake](https://azure.microsoft.com/en-us/solutions/data-lake/), [Google Firebase](https://firebase.google.com/) itd.
- [elastično](https://en.wikipedia.org/wiki/Elasticity_(cloud_computing)), a ne samo [skalabilno](https://en.wikipedia.org/wiki/Scalability): pružatelj usluga u oblaku skalira broj poslužitelja dodijeljenih aplikaciji i njenoj bazi podataka po potrebi
- gotova rješenja za popularne okvire za razvoj aplikacija, npr.
    - [Laravel Vapor](https://vapor.laravel.com/) za pokretanje Laravela na AWS Lambda
    - [Ruby on Jets](https://rubyonjets.com/) za pokretanje Ruby on Railsa na AWS Lambda

---

## Ukratko

Monolitna i mikrouslužna arhitektura se koriste u praksi prema potrebi:

- Za jednostavne projekte monolitna arhitektura je logičan izbor. Pritom se prikaz stranica može izvoditi na poslužitelju (engl. *server-side rendering*) ili na klijentu (engl. *client-side rendering*) ([detaljni pregled načina prikaza stranica](https://developers.google.com/web/updates/2019/02/rendering-on-the-web)).
- Mikrouslužna arhitektura omogućuje da veliki broj programera neovisno radi promjene na pojedinim uslugama dokle god se zadržava dogovoreno aplikacijsko programsko sučelje.

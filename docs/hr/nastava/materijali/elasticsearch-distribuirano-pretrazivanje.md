---
author: Toni Butković, Vedran Miletić
---

# Distribuirano pretraživanje u realnom vremenu alatom ElasticSearch

[ElasticSearch](https://www.elastic.co/elasticsearch/) je distribuirani sustav otvorenog koda za analizu i pretragu u stvarnom vremenu. Navedena tehnologija napisana je u Java programskom jeziku te se nalazi pod Apache licencom. Jedna je od najpopularnijih i najkorištenijih tehnologija pretraživanja današnjice. Da bi koristili ElasticSearch, potrebno je dodati određene informacije u bazu i indeksirati ih, stvoriti clustere i replikacije. Kako bi sve to imalo smisla po pitanju stalne dostupnosti tehnologije, odnosno search engine-a, potrebno je bilo napraviti failover instance.

ElasticSearch failover instancu može biti ostvarena na nekoliko načina.

Jedan od načina jest da se pokrene više clustera unutar iste instance ElasticSearch-a gdje se jedan od clustera navodi kao master cluster te preko njega idu upiti i ostale radnje na druge clustere. Ostali clusteri mogu dobivati upite i direktno no tada se može raditi samo sa njihovim indeksima, dok glavni klaster indeksira svoj search index ali su mu dostupni i svi ostali indexi podclustera. ([Više o kreiranju grupe clustera.](https://www.elastic.co/blog/tribe-node))

Ono što treba napomenuti i što na neki način ostvaruje cilj ovog rada jest to da u slučaju prestanka rada jednog od clustera, odnosno zadanog master clustera, drugi po redu pokrenuti cluster preuzima ulogu mastera. To je failover unutar jedne instance ElasticSearch-a, no odemo li korak dalje doći ćemo do sljedećih pitanja.

Što ukoliko netko ugasi trenutnu serversku instancu na kojem nam je pokrenuta instanca ElasticSearch-a? Trebamo li onda razmišljati na isti način nasljeđivanja master uloge clustera ili jednostavno trebamo vratiti "error 500" i restartati server prvom prilikom?

To je jedno od pomalo nemarnih rješenja, no sljedeće pitanje je dali će ozbiljni server ikada biti ugašen bez prethodne provjere i preusmjeravanja potrebnih servisa na drugu serversku instancu?

Moje je mišljenje da za to uvijek postoji mogućnost, pogotovo na serverima manje kvalitete, no o tome bi trebali brinuti sistemski administratori. Iz iskustva mi je poznato da je idealna situacija kada je ElasticSearch postavljen na zasebnu mašinu koja služi samo u svrhu pretrage. Poželjno je isto tako da se vrti na SSD diskovima svog brže pretrage indeksiranih podataka. Te je isto tako i više nego poželjno imati replikacijsku bazu na drugoj nezavisnoj mašini sa istim konfiguracijskim postavkama kako bi mogla u trenutku preuzeti posao prve mašine ukoliko dođe do kvara na istoj.

Upravo je to cilj zadatka koji je napravljen pomoću dvije virtualne mašine na kojima se vrti Linux distribucija Debian. Razlika između te dvije mašine jest jedino u IP adresama preko kojih su dostupne unutar localne mreže. Prva mašina ima adresu 10.211.55.13, a druga 10.211.55.14 te su to njihove defaultne adrese.

Nakon pokretana mašina pokrenemo instance ElasticSearcha koje postaju dostupne preko IP adresa mašine i vrata koji se dodjeli toj instanci ElasticSearch-a, a on se nalazi u rasponu većem od 9200. Po defautu u konfiguraciji za ElasticSearch instance ta vrata su 9201.

``` shell
# ./bin/elasticsearch
```

Svaka od instacni dobije random ime (koje je bas cool) te onda ranije pokrenuta instanca postaje master i prilikom pokretanja svake iduce dobije informaciju o toj drugo pokrenutoj, kao i novo pokrenuta o master instanci.

Pristup instanci search-a moguć je preko adrese 10.211.55.13:9201 odnosno 10.211.55.14:9201.

ElasticSearch ima jako puno mogućnosti. Od pregleda clustera, nodova u clusteru, zdravlja i trenutnog stanja i statusa clustera raznih query-a te searcha koji je ostvariv naredbom:

```
10.211.55.13:9201/_search?pretty
```

Riječ pretty na kraju url adrese formatira tekst u nešto čitljiviji format.

Ideja krajnjeg faliovera ostvarena je UNIX cron servisom te skriptom koja se poziva cronom i izvodi ping na zadanu mašinu na kojoj je trenutno aktivna instanca ElasticSearch-a. Ukoliko mašina postane nedostupna, cron koji se svake minute pokreće i provjerava status adresirati će trenutnu mašinu na IP stare mašine na kojoj se vrtio ElasticSearch. Tim pristupom dobiti ćemo maksimalno 1 minutu downtime-a za ElasticSearch servis.

Skripta:

``` bash
#!/bin/bash
# Simple SHELL script for Linux and UNIX system monitoring with
# ping command
# -------------------------------------------------------------------------
# no ping request
COUNT=1
count=$(ping -c $COUNT 10.211.55.13 | grep 'received' | awk -F',' '{ print $2 }' | awk '{ print $1 }')
if [ $count -eq 0 ]; then
  # 100% failed
  /sbin/ifconfig eth0 10.211.55.13;
fi
```

Prva dignuta instanca ElasticSearch-a sa imenom Loa.

Na slici ispod vidljiva je pretraga prve instance ElasticSearch-a. Rezultati su iz postojeće baze te svaki od njih ima definirani index, tip, id, score te source. Index jest ono po čemu instanca ElasticSearch-a indeksira određene podatke, tip opisuje kojeg je tipa podatak, id označava id u bazi, score jest pridodan kako bi se prikazala važnost podatka (1.0 je najveća važnost podatka, a 0 najmanja) te source sadrži podatke o id-u i short id-u koji su potrebni za interne pretrage.

Druga instanca naziva Vengeance nakon preuzimanja IP adrese od prve mašine koja je ugašena.

Ispod se nalazi druga slika na kojoj je vidljivo da je rezultat pretrage potpuno isti kao i na prvoj mašini. To je ostvareno replikacijom baze na dvije mašine te preuzimanjem funkcionalnosti prve mašine nakon njenog gašenja.

Konfiguracija ElasticSearcha nalazi se u folderu config pod imenom `elasticsearch.yml`. U konfiguraciji je moguće definirati ime clustera koje će se dodijeliti prilikom pokretanja instance. Isto tako moguće je podesiti razne opcije za čvorove instance pa tako možemo definirati ime čvora, ulogu čvora -- dali je master ili ne te dali je čvor nositelj podataka ili ne.

Moguće je definirati broj replikacija te broj krhkotina (engl. *shards*) za konkretnu instancu. Otvorena je mogućnost definiranja određenih putanja do raznih log fileova i sl.

Sljedeći korak koji je moguće definirati jest konfiguracija mreže, odnosno konfiguracija hosta na koji će se priljepiti instanca, konfiguracija hosta preko kojeg je dostupna, tcp port, tcp kompresija te http port, http duljinu sadržaja te dostupnost preko http-a.

---
author: Jovan Jokić, Vedran Miletić
---

# IPsec alat strongSwan

!!! todo
    Ovaj dio treba pročistiti i vidjeti gdje dopuniti slikama/tablicama.

IPSec je niz protokola za osiguravanje mrežnih veza, s velikim brojim detalja i varijacija koje vrlo brzo postanu enormne. Ovo je posebice slučaj prilikom međusobne suradnje, odn. funkcioniranja različitih sustava. U nastavku će se obraditi protokoli niže razine koji se koriste u IPv4 kontekstu (bez IPv6 konteksta), iz perspektive "bottom-up" (od dna prema gore). Problematika IPSec-a će se teoretski obraditi prvo, a zatim će ukratko biti popisane mogućnosti jedan od alata otvorenog koda koji implementira IPSec svitu protokola za operacijski sustav GNU/Linux.

Jedna od prvih stvari koja "upada u oko" prilikom podešavanja IPSec-a jest pregršt postavki i mogućnosti koje nudi: čak i par protokola koji potpuno konformiraju standardima, odn. njhove analogne implementacije "osiguravaju" zastrašujući broj načina za onemogućavanje uspješnog uspostavljanja sigurne veze. Jednostavno je posrijedi nevjerovatno kompleksan skup protokola. Jedan od razloga takve kompleksnosti jest činjenica da IPSec omogućava mehanizme, ne pravila: umjesto striktne definicije određenog enkripcijskog algoritma ili određene autentifikacijske funkcije, omogućava okvir (engl. *framework*) koji osigurava implementaciji pružanje bilo kakvog oblika usluge oko koje se usaglase oba kraja koja uspostavljaju sigurnu vezu. U ovom dijelu pobliže ću objasniti uobičajenu terminologiju i njihov međusobni odnos u kontekstu IPSec tehnologije, uz opasku da je skup tehnologija u toj domeni preopširan da bi se obradio u detaljima.

## Protokol

**AH naspram ESP-a.** AH ("Authentification Header") i ESP ("Encapsulating Security Payload") su 2 glavna protokola na "razini žice" koja upošljava IPSec usvom radi, a njihova funkcija je omogućavanje uspješne autentifikacije (AH) i enkripcije + autentifikacije (ESP) podataka koji se razmjenjuju putem takve sigurnosne veze. Uobičajeno se koriste zasebno, no moguće je (iako neučestalo) koristiti ih oboje istovremeno.

**Režim tuneliranja naspram transportnog režima.** Transportni režim osigurava sigurnu vezu između 2 čvora budući da enkapsulira sadržaj IP paketa (engl. *payload*), dok režim enkapsulira čitav IP paket ne bi li omogućio virtualni "sigurni skok" između 2 usmjerivača. Potonje se koristi za kreiranje tradicionalnog VPN-a, gdje uloga tunela je, generalno, stvaranje sigurnog tunela kroz nesigurni i nepovjerljivi Internet.

**IPSec kriptografski algoritmi.** Podešavanje IPSec veze uključuje različita kriptografska rješenja, no, taj postupak je značajno pojednostavljen činjenicom da bilo koja proizvoljna IPSec veza može istovremeno koristiti samo kombinaciju 2 (odn. 3 u nekim okolnostima) takva algoritma. Autentifikacijski proces se svodi na izračun ICV ("Integrity Check Value") vrijednosti nad sadržajem paketa, te se obično temelji na kriptografskom "hashu", poput MD5, odn. SHA-1. Sadržava tajni ključ, poznat objema krajevima, čime se omogućuje primatelju izračun ICV vrijednost na istovjetan način. Ukoliko primatelj dobije izračuna istu vrijednost, pošaljitelj je učinkovito se autentificirao (oslanjajući se na svojstvo da kriptografske "hasheve" praktički nije moguće preokrenuti. AH uvijek omogućava autentifikaciju, dok ESP opcionalno nudi tu mogućnost. U procesu enkripcija, koristi se tajni ključ za potrebe enkriptiranja podataka prije odašiljanja, čime se zaštićuje sadržaj paketa od mogućih presretača podataka, odn. prisluškivanja. Postoji niz izbora za kriptografski algoritam, od kojih su DES, 3DES, Blowfish i AES najčešće korišteni u praksi, no moguće je koristiti i druge.

**IKE naspram ručnih ključeva.** Obzirom da obje strane konverzacije moraju znati tajne vrijednosti korištene u procesu "hashiranja" i enkripcije, postavlja se logično pitanje: kako se ovi podaci razmjenjuju. Ručni ključevi zahtjevaju ručni unos tajnih vrijednosti na oba kraja, s pretpostavkom da se dostavljaju nekakvim izvanpojasnim (engl. *out-of-band*) mehanizmom, a IKE ("Internet Key Exchange") je sofisticirani
mehanizam za taj proces online načinom.

**Glavni režim naspram agresivnog režima.** Ovi režimi kontroliraju kompromis između učinkovitosti naspram sigurnosti tijekom inicijalne IKE razmjene ključeva. "Glavni režim" zahtijeva 6 paketa u oba smjera, no garantira potpunu sigurnost tijekom uspostave IPSec veze, dok "agresivni režim" koristi upola manje izmjena, čime se gubi na sigurnosti izmjene, budući da neki od podataka u procesu izmjene se prenose u čistom (engl. *cleartext*) obliku.

## IP paket ("datagram")

Obzirom na "bottom-up" aspekt pregleda IPSec tehnologije u ovom radu, potrebno je napraviti mali obilazak IPv4 tehnologije i opisati zaglavlje samog IPv4 pekta, koji sadrži sve podatke, tj. promet od značaja. Ovime se neće dati jezgroviti opis i podrobna analiza samog zaglavlja IP paketa, no poslužit će se potrebe bolje ilustracije određenih IPSec mehanizama koje nas zanimaju.

Opis pojedinih dijelova zaglavlja, odn. polja u zaglavlju i ostatka paketa:

- ver: Ovo je verzija IP protokola, koja je u ovom slučaju 4 = IPv4
- hlen: Duljina IP zaglavlja, predstavljena kao 4-bitni broj koji enkodira 32-bitne riječi, raspona od 0 do 15. Standardno IPv4 zaglavlje je uvijek duljine 20 bajtova (5 riječi), a IP opcije, ukoliko su definirane, se indiciraju većom veličinom hlen polja, do maksimalno 60 bajtova. Veličina ovog polja nikad ne uključuje veličinu korisnog dijela paketa (engl. "payload"), ili drugih polja koja slijede.
- TOS: "Type of Service". Ovo polje je bit-maska koja daje neke naznake o tipu usluge koji bi dani paket trebao biti dostavljen: optimizirati za propusnost ? Latenciju ? Pouzdanost ?
- pkt len: Ukupna duljina paketa u bajtovima, do maksimalne veličine od 65535. Ovaj brojač uključuje i veličinu zaglavlja, stoga to praktično znači da je maksimalna veličina korisnog dijela paketa barem 20 bajtova manja od definirane maksimalne veličine, zbog veličine zaglavlja. Velika većina IP paketa su puno, puno manji.
- ID: ID polje je koristi za povezivanje srodnih paketa koji su fragmentirani (veliki paketi podijeljeni u manje)
- flgs: Ovo su male "zastavice" koje uglavnom kontroliraju proces fragmentacije: jedna označava paket nepoželjnim za fragmentaciju, druga označava da slijedi pošiljka ostatka fragmenata.
- frag offset: Kada je paket fragmentiran, ova vrijednost pokazuje kojem dijelu u ukupnom "virtualnom" paketu dotični fragment pripada.
- TTL: Ovo je tzv. "Time to Live" vrijednost, koju umanjuje svaki usmjerivač koji zaprimi paket za 1. Kada ova vrijednost padne na nulu, sugerira obično na neku pogrešku u procesu usmeravanja (najčešće nekakvu petlju), stoga takav paket biva odbačen posljedično, ne bi li se sprječilo njegovo beskonačno usmjeravanje kroz Internet.
- proto: Ovo polje predstavlja protokol koji se sadržan u paketu, koji je vrlo bitan za potrebe diskusije o IPSecu budući da ćemo se često referirati na njega. Iako samim paketom upravlja IP ("Internet Protocol"), IP paket uvijek enkapsulira neki pomoćni protokol (TCP, UDP, ICMP, itd.) u sebi. Možemo ga smatrati kao oznaka tipa zaglavlja koji slijedi dalje u IP paketu.
- header cksum: Ova vrijednost sadrži kontrolni zbroj (engl. "checksum") čitavog IP zaglavlja, koji je dizajniran s ciljem detektiranja grešaka u procesu prijenosa paketa. Ovo nije kriptografski kontrolni zbroj, te ne uključuje nijedan dio paketa koji slijedi nakon IP zaglavlja.
- src IP address: Izvorna 32-bitna IP adresa, koju primatelj koristi za odgovor na odaslani IP paket. Općenito uzevši, moguće je zamaskirati ove adrese (npr. lagati o pravoj adresi s koje se šalje IP datagram).
- dst IP address: Odredišna 32-bitna IP adresa na koju je adresiran IP paket pošiljatelja.
- IP Options: Ovo su opcionalni dijelovi IP zaglavlja koji sadrže aplikacijsko-specifične informacije, iako obično se ne koristi za usmjeravanje paketa. Prisutnost IP opcija je indicirano veličinom hlen polja većim od 5; uključene su u kontrolni zbroj zaglavlja paketa.
- Payload: Svaki tip protokola podrazumijeva svoj format za ono što slijedi nakon IP zaglavlja; koristili smo TCP samo za potrebe davanja primjera.

Slijedeći proto kodovi su definirani od strane IANA ("Internet Assigned Numbers Authority") organizacije, iako ovo svakako nije ekstenzivna i kompletna lista, no za većinu potreba, ovo su svakako najčešće korišteni.

- 1: ICMP -- Internet Control Message Protocol
- 2: IGMP -- Internet Group Management Protocol
- 4: IP within IP (a kind of encapsulation)
- 6: TCP -- Transmission Control Protocol
- 17: UDP -- User Datagram Protocol
- 41: IPv6 -- next-generation TCP/IP
- 47: GRE -- Generic Router Encapsulation (used by PPTP)
- 50: IPSec: ESP -- Encapsulationg Security Payload
- 51: IPSec: AH -- Authentication Header

## AH: Samo autentifikacija

AH se koristi za autentifikaciju, ali ne i enkripciju, IP prometa, čija je svrha osiguravanje autentičnosti primatelja kojem šaljemo poruke, te detekciju izmjena podataka u procesu slanja, te (opcionalno) radi zaštite od "napada ponavljanjem" od strane napadača koji prikupljaju podatke "s žice", te pokušavaju ga injektirati ponovno u medij kasnije.

Autentifikacija se obavlja izračunom kriptografskog "hash"-zasnovanog autentifikacijskog koda za poruke nad skoro svim poljima IP paketa (isključujući one koji se mogu modificirati u prijenosu, poput TTL-a ili kontrolne sume), te pohranom toga u novo-dodano AH zaglavlje, nakon čega se takav paket šalje primatelju. Ovo AH zaglavlje sadrži samo 5 polja od interesa, koji su umetnuti između izvornog IP zaglavlja, te korisnog dijela IP paketa ("payload").

Polja u AH zaglavlju:

- next hdr: Ovime se ustanovljuje tip protokola za korisni dio koji slijedi, te je izvorni tip paketa koji se enkapsulira: na ovaj način su IPSec zaglavlja međusobno povezana.
- AH len: Ovo polje definira duljinu, u 32-bitnim rječima, čitavog AH zaglavlja, umanjenog za 2 rječi (ova klauzula umanjivanja za 2 kodne rječi slijedi iz formata "IPv6 RFC 1883 Extension Headers" specifikacije, od kojih je AH jedan od.
- Reserved: Ovo polje je rezervirano za buduće upotrebe, te mora biti 0.
- Security Parameters Index (SPI): Ovo je 32-bitni identifikator koji pomaže primatelju odabrati kojim od mnogim tekućim "razgovorima" paket pripada. Svaka AH-zaštićena veza sadrži "hash" algoritam (MD5, SHA-1, itd.), nekakav tip tajnih podataka, te niz drugih parametara. SPI možemo smatrati kao indeks u tablicu ovih postavki, koja omogućuje lagano povezivanja paketa s određenim parametrom.
- Sequence Number: Monotono-povećavajući identifikator koji se koristi u asistenciju sprječavanja "napada ponavljanjem". Ova vrijednost se uključuje u autentifikacijske podatke, stoga svaka modifikacija time (namjerna ili slučajna) biva detektirana.
- Authentication Data: ICV ("Integrity Check Value") izračunat pomoću čitavog sadržaja paketa -- uključuje većinu zaglavlja. Primatelj ponovno izračunava istu "hash" vrijednost. U slučaju nepodudarnih vrijednosti, paket se markira kao, bilo oštećen u prijenosu, ili kao paket kojem nedostaje ispravan tajni ključ. Takvi paketi se odbacuju.

### Transportni režim rada

Najjednostavni režim rada IPSec-a za razumjeti je transportni režim, koji se koristi za zaštitu komunikacije tipa točka-točka između 2 čvora. Ova zaštita se provodi kroz autentifikaciju, ili enkripciju (ili oboje), no ne predstavlja tunelirajući protokol. Naime, nema nikakvih srodnih značajki s klasičnim VPN-om: jednostavno je posrijedi zaštićena IP veza.

U AH transportnom režimu, IP paket se modificira samo malo, ne bi sadržao novo AH zaglavlje između IP zaglavlja i korisnog podatkovnog dijela od protokola (TCP, UDP, itd.), te kodovi oznake protokola koji povezuje različita zaglavlja skupa se zamjenjuju.

Promjena mjesta protokola je nužna za omogućavanje rekonstrukcije izvornog IP paketa na drugom kraju: nakon validacije IPSec zaglavlja po primitku, odbacuju se, te izvorni tip protokola (TCP, UDP, ...) se ponovo pohranjuje u IP zaglavlje.

Nakon što paket dosegne svoje odredište, te prođe test autentifikacije, AH zaglavlje se miče, a "Proto=AH" polje u IP zaglavlju se zamjenjuje sa spremljenim "Next Protocol". Ovime se IP datagram vraća u svoje prvotno stanje, te može biti dostavljen čekajućem procesu.

### Tunelirajući režim rada

Tunelirajući režim čini osnovu poznatijoj VPN funkcionalnosti, pri čemu se čitavi IP paketi enkapsuliraju jedan u drugome, te dostavljaju na odredište. oput transportnog režima, paket se osigurava pomoću ICV ("Integrity Check Value") vrijednosti za potrebe autentifikacije pošiljatelja, te sprječavanje modifikacije u prijenosu. No, za razliku od transportnog režima, enkapsulira čitavo IP zaglavlje, uz korisni dio ("payload"), čime se omogućava različita IP adresa pošiljatelja i primatelja od adrese paketa koji ga sadržava, odn.enkapsulira. Navedeno omogućuje implementaciju tunela.

Kada paket u tunelirajućem režimu rada dosegne svoje odredište, prolazi kroz istu autentifikacijsku provjeru kao bilo koji AH-tip paketa, te se onima koji prođu provjeru odbacuju čitava IP i AH zaglavlja. Time se, efektivno, rekonstruira izvorni IP paket, koji se onda ubacuje u uobičajeni proces usmjeravanja.

Većina implementacija tretira točku u tunelirajućem režimu rada kao virtualno mrežno sučelje, poput Ethernet sučelja ili "localhost", a promet koji ulazi, odn. odlazi s tog sučelja je podložan uobičajenim zahvatima usmjeravanja. Rekonstruirani paket može se dostaviti lokalnom računalu, ili biti usmjeren dalje (sukladno odredišnoj IP adresi koja se nalazi u enkapsuliranom paketu), no, u svakom slučaju, više nije pod zaštitom IPSec-a. U tom trenutku postaje običan, generički IP paket. Iako transportni režim rada se koristi striktno za osiguravanje veze tipa točkatočka između 2 računala, tunelirajući režim rada se uobičajeno koristi između mostova (usmjerivači, vatrozidovi, odn. zasebni VPN uređaji) za potrebe stvaranja VPN ("Virtual Private Network") mreže.

### Transportni ili tunelirajući režim?

Zanimljivost je da ne postoji polje za definiranje eksplicitnog režima rada u IPSec-u: ono što razlikuje transportni režim od tunelirajućeg je iduće polje zaglavlja u AH zaglavlju. Kada je vrijednost u idućem zaglavlju IP, to znači da taj paket enkapsulira čitav IP datagram (uključujući nezavisne IP adrese izvora i odredišta koji omogućuju zasebno usmjeravanje nakon de-enkapsulacije). To je tunelirajući režim rada. Bilo koja druga vrijednost (TCP, UDP, ICMP, itd.) znači da je riječ o transportnom režimu koji osigurava vezu tipa točka-točka između krajnjih čvorova. Gornji sloj IP datagrama je strukturiran na isti način, neovisno o režimu, te posredni usmjerivači tretiraju sve varijante IPSec/AH prometa identično bez dublje inspekcije. Čvor, za razliku od mosta, mora podržavati oba režima rada, no u slučaju kreiranja veze tipa čvor-čvor, čini se pomalo suvišnim koristiti tunelirajući režim. Dodatno, most (usmjerivač, vatrozid, itd.) mora podržavati samo tunelirajući režim, iako podržavanje oba režima je korisno u slučaju stvaranja veze na sam most, poput za potrebe mrežno upravljačkih funkcija.

### Autentifikacijski algoritmi

AH provodi ICV ("Integrity Check Value") provjeru integriteta u autentifikacijsko-podatkovnom dijelu zaglavlja, te je tipično zasnovan na standardnim kriptografskim "hash" algoritmima poput MD5 ili SHA-1. Umjesto čiste kontrole sume, koja ne bi omogućila nikakvu sigurnost protiv namjernog uplitanja treće maliciozne strane u komunikaciji, koristi HMAC ("Hashed Message Authentication Code"), koji uključuje tajnu vrijednost u procesu kreiranja ICV-a. Iako napadač može lagano ponovno izračunati "hash", bez tajne vrijednosti ne može rekreirati ispravni ICV. HMAC opisuje dokument RFC 2104, te je idućom slikom ilustriran proces u kojem podaci poruke i tajne sudjeluju u kreiranju krajnje ICV vrijednosti.

Važno je napomenuti da IPSec/AH ne definiraju koja se autentifikacijska funkcija mora koristiti, već osiguravaju okvir ("framework") koji omogućuje korištenje bilo koje prihvatljive implementacije, na koju su pristale obje strane. Moguće je koristiti druge autentifikacijske funkcije, poput digitalnog potpisa ili enkripcijske funkcije, sve dok obje strane omogućuju takvu implementaciju.

## Encapsulating Security Payload (ESP)

Dodavanje enkripcije čini ESP malo kompliciranijim budući da enkapsulacija obuhvaća "payload", umjesto da mu prethodi, kao u slučaju s AH: ESP uključuje zaglavlje i dodatna polja radi podržavanja enkripcije i opcionalne autentifikacije. Također omogućuje tunelirajuće i transportne režime rada koji se koriste na već opisani način. IPSec RFC dokumenti ne sadržavaju eksplicitni naputak za korištenje bilo kojeg specifičnog protokola, no nalazimo DES, 3DES,AES i Blowfish u svakodnevnoj praktičnoj upotrebi za potrebe zaštite od presretanja od strane napadača.

Algoritam korišten za pojedinu vezu određuje SA ("Security Association"), koja uključuje i algoritam i korišteni kripto-ključ. Za razlikuod AH-a, koji uvodi malo zaglavlje ispred "payload"-a, ESP okružuje korisni podatkovni dio, odn. "payload" kojeg štiti. SPI ("Security Parameters Index") i SN ("Sequence Number") imaju istu ulogu kao i u AH, no nalazimo "padding", iduće zaglavlje, te opcionalni "Authentication Data" na kraju, u ESP dodatnim poljima. Moguće je koristiti ESP bez ikakve enkripcije (koristeći NULL algoritam), no struktura paketa se time ne mijenja. Time se ne osigurava povjerljivost komunikacije, te jedino ima smisla u slučaju da se kombinira s ESP autentifikacijom. Besmisleno je koristiti ESP bez enkripcije ili autentifikacije (osim za potrebe testiranja protokola eventualno). "Padding" se omogućuje radi blokovno orijentiranih enkripcijskih algoritama koji trebaju višekratnu veličinu bloka podataka za rad, a njegova veličina je zabilježenau pad len polju. Next hdr polje sadrži tip (IP, TCP, UDP, itd.) "payload"-a.

Osim enkripcije, ESP može opcionalno omogućiti autentifikaciju, s istim HMAC algoritmom kojeg nalazimo u AH. Za razliku od AH, međutim, ova autentifikacija je samo za ESP zaglavlje i enkriptirani "payload": ne "pokriva" čitavi IP paket. Iznenađujuće, ovime se ne slabi značajno sigurnost autentifikacije, no ima nekih praktičnih benefita. Kada 3. strana pregledava IP paket koji sadrži ESP podatke, praktički joj je nemoguće odgonetnuti njegov sadržaj, osim uobičajenih podataka u IP zaglavlju (posebice izvorna i odredišna IP adresa). Napadač će svakako znati da su posrijedi ESP podaci -- ta informacija se također nalazi u zaglavlju -- no tip "payload"-a je enkriptiran skupa s "payload"-om. Čak i prisutnost "Authentication Data" nije moguće ustvrditi samo inspekcijom sadržaja paketa.

### ESP u transportnom režimu rada

Kao i s AH, transportni režim enkapsulira samo korisni dio paketa ("payload"), te je dizajniran striktno za komunikacije tipa točka-točka. Izvorno IP zaglavlje se ne mijenja (osim zamjenjenog "Protocol" polja), što znači, među ostalim, da izvorna i odredišna IP adresa ostaju iste.

### ESP u tunelirajućem režimu rada

ESP u tunelirajućem režimu rada enkapsulira čitav IP paket unutar enkriptiranog omotača.

Omogućavanje ekriptiranog tunelirajućeg režima rada je vrlo blizu tradicionalom VPN-u koji je većini prva asocijacija uz IPSec tehnologiju, no, potrebno je dodati toj slici i autentifikaciju nekog tipa radi njezine potpunosti. Za razliku od AH, u kojem presretač prometa može lako odrediti je li promet u tunelirajućem ili transportnom režimu, takva informacija je u ovom slučaju nedostupna: činjenica da je posrijedi tunelirajući režim (next=IP polje) je dio enkriptiranog "payload"-a, te jednostavni nije vidljiva onome tko pokušava dekriptirati paket.

## strongSwan

[strongSwan](https://strongswan.org/) je potpuna IPSec implentacija za Linux 2.6 i 3.x kernele. Fokusovog projekta je na jakim autentifikacijskim mehanizmima koristeći X.509 certifikate javnog ključa i opcionalnu sigurnosnu pohranu privatnih ključeva na pametne kartice kroz standardizirano PKCS#11 sučelje.

Kao potomak projekta [FreeS/WAN](https://www.freeswan.org/) (uz [Openswan](https://www.openswan.org/)), strongSwan nastavlja biti izdan pod GPL licencom. Podržava liste opozvanih certifikata, kao i OCSP ("Online Certifikate Status Protocol") protokol. Jedinstvena mogućnost je upotreba X.509 atributnih certifikata za implementiranje shema prava pristupa baziranih za grupnom članstvu. StronSwan uredno surađuje s drugim IPSec implementacijama, uključujući različite IPSec VPN klijente koji rade na Microsoft Windows i Mac OS X operacijskim sustavima. StrongSwan 5.0 grana u potpunosti implementira IKEv2 ("Internet Key Exchange") protokol koji je definiran u RFC 5996. StrongSwan podržava IKEv1, te u potpunosti implementira IKEv2.

!!! todo
    Ovaj dio treba napisati.

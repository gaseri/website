# Optimizacija programskog koda

## Opće informacije

Nositelj predmeta: doc. dr. sc. Vedran Miletić  
Naziv predmeta: Optimizacija programskog koda  
Studijski program: Preddiplomski sveučilišni studij Informatika  
Status predmeta: obvezatan/**izborni**  
Godina: 3.  
Bodovna vrijednost i način izvođenja nastave:

- ECTS koeficijent opterećenosti studenata: 5
- Broj sati (P+V+S): 30+30+0

## Opis predmeta

### Ciljevi predmeta

Cilj ovog predmeta je uvesti temeljna načela i metode optimizacije programskog koda na razini apstraktne sintakse, grafa toka programa i izvršnog (strojnog) koda.

### Uvjeti za upis predmeta

Položen predmet Algoritmi i strukture podataka.

### Očekivani ishodi učenja za predmet

Očekuje se da nakon izvršavanja svih programom predviđenih obveza studenti budu sposobni:

I1. Analizirati svojstva koja omogućuju transformaciju programskog koda i prikazati programski kod grafom toka.  
I2. Prikazati razlike između lokalne i globalne optimizacije te identificirati gdje se svaka od njih primjenjuje.  
I3. Provesti klasičnu analizu toka podataka, alokaciju registara bojenjem registara i eliminaciju zajedničkih podizraza.  
I4. Opisati način rada optimizacije višeg nivoa i primijeniti postojeće optimizacije.  
I5. Opisati razlike optimizacija višeg nivoa i optimizacija ovisnih o ciljnoj arhitekturi.  
I6. Provesti odabir instrukcije.  
I7. Analizirati problem redoslijeda faza optimizacije.

### Sadržaj predmeta

- Pregled optimizirajućeg prevoditelja programskog jezika. Optimizacija po dijelovima. Analiza svojstava koja omogućuju transformaciju. Graf toka i reprezentacija programskih koncepata. Problem redoslijeda faza optimizacije.
- Vrste optimizacije. Lokalna optimizacija: optimizacija kroz okance, zakazivanje instrukcija. Globalna optimizacija: zajednički podizrazi, kretanje koda. Interproceduralna optimizacija. Graf poziva.
- Klasična analiza toka podataka. Algoritmi na grafovima, skupovi živih i dostupnih varijabli. Alokacija registara bojenjem registara. Eliminacija zajedničkih podizraza. Prolijevanje u memoriju; baratanje privremenim izrazima uvedenim kod eliminacije zajedničkih podizraza. Anomalije toka podataka. Oblik statičke jednostruke dodjele vrijednosti varijablama.
- Pregled optimizacija višeg nivoa. Analiza memorijskih lokacija na koje varijable pokazuju i analiza pseudonima.
- Optimizacija ovisna o ciljnoj arhitekturi. Odabir instrukcije. Zakazivanje instrukcija i povezani problem redoslijeda faza optimizacije.

### Vrsta izvođenja nastave

- [x] predavanja
- [ ] seminari i radionice
- [x] vježbe
- [x] obrazovanje na daljinu
- [ ] terenska nastava
- [x] samostalni zadaci
- [ ] multimedija i mreža
- [x] laboratorij
- [ ] mentorski rad
- [ ] ostalo ___________________

### Komentari

Nastava se izvodi kombinirajući rad u učionici i računalnom laboratoriju uz primjenu sustava za udaljeno učenje. Studenti će kod upisa kolegija biti upućeni na korištenje sustava za udaljeno učenje. U izvedbenom planu objavit će se detaljan raspored nastave s predavanjima i vježbama.

### Obveze studenata

Obaveze studenata u predmetu su:

- Redovito pratiti aktivnosti predmeta u okviru sustava za udaljeno učenje i pohađati nastavu kada se odvija u obliku predavanja, auditornih i/ili laboratorijskih vježbi.
- Pristupiti kontinuiranim provjerama znanja (teorijskim i praktičnim kolokvijima) i uspješno ih položiti.
- Izraditi individualni ili timski praktični rad na zadanu temu.
- Pristupiti završnom ispitu i na njemu postići barem 50% bodova.

Detaljan način razrade bodovanja na predmetu te pragovi prolaza za pojedine aktivnosti koje se boduju biti će navedeni u izvedbenom planu predmeta

### Praćenje[^1] rada studenata

- Pohađanje nastave: 2
- Aktivnost u nastavi:
- Seminarski rad:
- Eksperimentalni rad:
- Pismeni ispit: 1
- Usmeni ispit:
- Esej:
- Istraživanje:
- Projekt:
- Kontinuirana provjera znanja: 1
- Referat:
- Praktični rad: 1
- Portfolio:

### Postupak i primjeri vrednovanja pojedinog ishoda učenja tijekom nastave i na završnom ispitu

- Praktična provjera znanja na računalu (praktični kolokvij) u kojoj student analizira i transformira kod te koristi i prilagođava postojeće optimizacije (I1, I2, I3, I4, I6).
- Grupni ili individualni praktični rad u kojem studenti prema zadanim uputama implementiraju rješenje s traženim optimizacijama i pišu dokumentaciju vlastite implementacije (I1, I2, I3, I4, I6).
- Pisana ili online provjera znanja u kojoj student pokazuje razumijevanje teorijskih koncepata optimizacije programskog koda, na primjer pomoću pitanja višestrukog izbora, pitanja nadopunjavanja i esejskih pitanja (I1, I2, I4, I5, I7).

### Obvezna literatura (u trenutku prijave prijedloga studijskog programa)

1. Cooper, K. D. & Torczon, L. *Engineering a compiler*. (Elsevier/Morgan Kaufmann, 2011).
2. Holub, A. I. *Compiler design in C*. (Prentice Hall, 1990). (e-knjiga je dostupna za besplatno preuzimanje s autorove stranice [holub.com/compiler/](https://holub.com/compiler/) i može se ispisati po potrebi)
3. Skripte, prezentacije i ostali materijali za učenje dostupni u e-kolegiju.

### Dopunska literatura (u trenutku prijave prijedloga studijskog programa)

1. Fraser, C. W. & Hanson, D. R. *A retargetable C compiler: design and implementation*. (Benjamin-Cummings, 1995).
2. Muchnick, S. S. *Advanced compiler design and implementation*. (Morgan Kaufmann, 1997).
3. Nielson, F., Nielson, H. R. & Hankin, C. *Principles of program analysis*. (Springer, 1999).
4. Appel, A. W. *Modern compiler implementation in C*. (Cambridge University Press, 2004).
5. Aho, A. V., Lam, M. S., Sethi, R. & Ullman, J. D. *Compilers: principles, techniques, & tools*. (Pearson/Addison-Wesley, 2006).
6. Morgensen, T. Ae. *Basics of Compiler Design*. (Lulu, 2010).
7. Wilhelm, R. & Seidl, H. *Compiler design: virtual machines*. (Springer, 2011).
8. Hack, S., Wilhelm, R. & Seidl, H. *Compiler design: code generation and machine-level optimization*. (Springer, 2019).
9. The GNU Compiler Collection. *GCC online documenatation*. (GNU, 2019). (dostupna online: [gcc.gnu.org/onlinedocs/](https://gcc.gnu.org/onlinedocs/))
10. The LLVM Compiler Infrastructure. *LLVM documentation*. (LLVM, 2019). (dostupna online: [llvm.org/docs/](https://llvm.org/docs/))

### Broj primjeraka obavezne literature u odnosu na broj studenata koji trenutno pohađaju nastavu na predmetu

| Naslov | Broj primjeraka | Broj studenata |
| ------ | --------------- | -------------- |
| Engineering a compiler | 1 | 30 |
| Compiler design in C | [Dostupno online](https://holub.com/compiler/) | 30 |

### Načini praćenja kvalitete koji osiguravaju stjecanje izlaznih znanja, vještina i kompetencija

Predviđa se periodičko provođenje evaluacije s ciljem osiguranja i kontinuiranog unapređenja kvalitete nastave i studijskog programa (u okviru aktivnosti Odbora za upravljanje i unapređenje kvalitete Odjela za informatiku). U zadnjem tjednu nastave provodit će se anonimna evaluacija kvalitete održane nastave od strane studenata. Provest će se i analiza uspješnosti studenata na predmetu (postotak studenata koji su položili predmet i prosjek njihovih ocjena).

[^1]: **Važno:** Uz svaki od načina praćenja rada studenata unijeti odgovarajući udio u ECTS bodovima pojedinih aktivnosti tako da ukupni broj ECTS bodova odgovara bodovnoj vrijednosti predmeta. Prazna polja upotrijebiti za dodatne aktivnosti.

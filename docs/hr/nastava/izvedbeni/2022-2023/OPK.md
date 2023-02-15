SVEUČILIŠTE U RIJECI  
FAKULTET INFORMATIKE I DIGITALNIH TEHNOLOGIJA  
Ulica Radmile Matejčić 2, Rijeka  
Akademska godina 2022./2023.

# OPTIMIZACIJA PROGRAMSKOG KODA

## OSNOVNI PODACI O PREDMETU

Naziv predmeta: Optimizacija programskog koda  
Studijski program: Sveučilišni prijediplomski studij Informatika  
Status predmeta: obvezatan/**izborni**  
Semestar: 6.  
Bodovna vrijednost i način izvođenja nastave:

- ECTS koeficijent opterećenosti studenata: 5
- Broj sati (P+V+S): 30+30+0

Nositelj predmeta: doc. dr. sc. Vedran Miletić  
E-mail: vmiletic@inf.uniri.hr  
Ured: O-520  
Vrijeme konzultacija: Srijedom od 12:00 do 14:00 uz prethodni dogovor e-mailom

Asistent:  
E-mail:  
Ured:  
Vrijeme konzultacija:

## DETALJNI OPIS PREDMETA

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

### Načini praćenja kvalitete koji osiguravaju stjecanje izlaznih znanja, vještina i kompetencija

Predviđa se periodičko provođenje evaluacije s ciljem osiguranja i kontinuiranog unapređenja kvalitete nastave i studijskog programa (u okviru aktivnosti Odbora za upravljanje i unapređenje kvalitete Fakulteta informatike i digitalnih tehnologija). U zadnjem tjednu nastave provodit će se anonimna evaluacija kvalitete održane nastave od strane studenata. Provest će se i analiza uspješnosti studenata na predmetu (postotak studenata koji su položili predmet i prosjek njihovih ocjena).

### Mogućnost izvođenja na stranom jeziku

Kolegij se izvodi na engleskom jeziku za studente u mreži YUFE.

## OBVEZE, PRAĆENJE RADA I VREDNOVANJE STUDENATA

| VRSTA AKTIVNOSTI | ECTS | ECTS -- PRAKTIČNI RAD | ISHODI UČENJA | SPECIFIČNA AKTIVNOST | METODA PROCJENJIVANJA | BODOVI MAX. |
| ---------------- | ---- | --------------------- | ------------- | -------------------- | --------------------- | ----------- |
| Pohađanje nastave | 2 | 1 | I1--I7 | Prisutnost studenata i odgovaranje na pitanja nastavnika | Popisivanje (evidencija), Kahoot! | 0 |
| Kontinuirana provjera znanja | 1 | 1 | I1, I2, I3, I4, I6 | Dvije domaće zadaće | Ovisno o stupnju točnosti i potpunosti | 30 |
| Pismeni ispit | 1 | 0,5 | I1, I2, I4, I5, I7 | Dva testa na Merlinu i dva pisana osvrta | Ovisno o stupnju točnosti i potpunosti | 30 |
| Završni ispit | 1 | 1 | I1, I2, I3, I4, I6 | Praktični rad | Vrednovanje potpunosti i točnosti odrađenog zadatka i odgovora na pitanja prema unaprijed definiranim kriterijima | 40 |
| **UKUPNO** | **5** | **3,5** |   |   |   | **100** |

### Obveze i vrednovanje studenata

#### 1. Pohađanje nastave

Nastava se odvija prema mješovitom modelu u kombinaciji klasične nastave u učionici i online nastave uz pomoć sustava za e-učenje prema rasporedu koji je prikazan je tablicom u nastavku. Studenti su dužni koristiti sustav za e-učenje Merlin ([moodle.srce.hr](https://moodle.srce.hr/)) gdje će se objavljivati informacije o predmetu, materijali za učenje, zadaci za vježbu, zadaci za domaće zadaće te obavijesti vezane za izvođenje nastave (putem foruma *Obavijesti*).

Studenti su dužni redovito pohađati nastavu, aktivno sudjelovati tijekom nastave te izvršavati aktivnosti predmeta u okviru sustava Merlin koje će nastavnici najavljivati putem foruma.

#### 2. Kontinuirana provjera znanja

Tijekom semestra bit će zadane dvije domaće zadaće koje će uključivati praktične zadatke. Na svakoj od njih će student moći skupiti maksimalno po 15 bodova, što nosi ukupno maksimalno 30 bodova.

#### 3. Pismeni ispit

Tijekom semestra pisat će se dva testa na Merlinu koji će uključivati pitanja iz gradiva predavanja. Na svakom od njih će student moći skupiti maksimalno po 10 bodova, što nosi ukupno maksimalno 20 bodova.

Tijekom semestra studenti će predati i dva kratka pisana osvrta na dane teme. Na svakom od njih će student moći skupiti maksimalno po 5 bodova, što nosi ukupno maksimalno 10 bodova.

#### 4. Završni ispit

Tijekom semestra student će odabrati jednu od ponuđenih optimizacijskih tehnika, implementirati je korištenjem LLVM-ovih biblioteka, napisasti testove i dokumentirati svoju implementaciju.

Na završnom ispitu će biti organizirana obrana praktičnog rada usmenim putem uz popratno ispitivanje znanja iz gradiva predavanja i vježbi. Na taj način studenti će moći ostvariti do 40 bodova.

### Ocjenjivanje

Kontinuiranim radom tijekom semestra na prethodno opisani način studenti mogu ostvariti najviše 60 ocjenskih bodova, a da bi mogli pristupiti završnom ispitu moraju ostvarili 50% i više bodova (minimalno 30).

Završni ispit nosi udio od maksimalno 40 ocjenskih bodova, a smatra se položenim samo ako na njemu student postigne minimalno 50%-ni uspjeh (ispitni prag je 50% uspješno riješenih zadataka).

Ukoliko je završni ispit prolazan, skupljeni bodovi će se pribrojati prethodnima i prema ukupnom rezultatu formirati će se pripadajuća ocjena. U suprotnom, student ima pravo pristupa završnom ispitu još 2 puta (ukupno do 3 puta).

#### Konačna ocjena

Donosi se na osnovu zbroja svih bodova prikupljenih tijekom izvođenja nastave prema sljedećoj skali:

- A -- 90%--100% (ekvivalent: izvrstan 5)
- B -- 75%--89,9% (ekvivalent: vrlo dobar 4)
- C -- 60%--74,9% (ekvivalent: dobar 3)
- D -- 50%--59,9% (ekvivalent: dovoljan 2)
- F -- 0%--49,9% (ekvivalent: nedovoljan 1)

### Ispitni rokovi

Redoviti:

- 21\. lipnja 2023.
- 5\. srpnja 2023.

Izvanredni:

- 30\. kolovoza 2023.
- 13\. rujna 2023.

## RASPORED NASTAVE -- ljetni (6.) semestar ak. god. 2022./2023.

Nastava će se na predmetu odvijati u zimskom semestru prema sljedećem rasporedu:

- predavanja: konzultativno
- vježbe: konzultativno

| Tj. | Datum | Vrijeme | Prostor | Tema | Nastava | Izvođač |
| --- | ----- | ------- | ------- | ---- | ------- | ------- |
| 2. | 8. 3. 2023. | 18:00--19:30 | O-365 | **1. pisani osvrt** | I | doc. dr. sc. Vedran Miletić |
| 4. | 22. 3. 2023. | 18:00--19:30 | O-365 | **1. domaća zadaća** | I | doc. dr. sc. Vedran Miletić |
| 7. | 12. 4. 2023. | 18:00--19:30 | O-365 | **1. test na Merlinu** | I | doc. dr. sc. Vedran Miletić |
| 8. | 19. 4. 2023. | 18:00--19:30 | O-365 | **2. pisani osvrt** | I | doc. dr. sc. Vedran Miletić |
| 9. | 26. 4. 2023. | 18:00--19:30 | O-365 | **2. domaća zadaća** | I | doc. dr. sc. Vedran Miletić |
| 10. | 3. 5. 2023. | 18:00--19:30 | O-365 | **Odabir teme praktičnog rada** | I | doc. dr. sc. Vedran Miletić |
| 12. | 17. 5. 2023. | 18:00--19:30 | O-365 | **2. test na Merlinu** | I | doc. dr. sc. Vedran Miletić |
| 15. | 7. 6. 2023. | 18:00--19:30 | O-365 | **Predaja praktičnog rada** | I | doc. dr. sc. Vedran Miletić |

P -- predavanja  
V -- vježbe  
I -- pisani ili usmeni ispit, kontinuirana provjera znanja

Napomena: Moguće su izmjene rasporeda nastave. Za nove verzije rasporeda potrebno je pratiti obavijesti u e-kolegiju.

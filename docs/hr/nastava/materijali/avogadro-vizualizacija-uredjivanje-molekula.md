---
author: Matea Turalija, Irena Hartmann, Vedran Miletić
---

# Vizualizacija i uređivanje molekula alatom Avogadro

[Avogadro](https://two.avogadro.cc/) je besplatan alat otvorenog koda namijenjen vizualizaciji, izradi i uređivanju molekula. Dizajniran je za korištenje u računalnoj kemiji, bioinformatici, znanosti materijala i srodnim područjima.

Najnovija verzija alata, [Avogadro 2](https://www.openchemistry.org/projects/avogadro2/), predstavlja značajna poboljšanja u odnosu na [originalni Avogadro](https://avogadro.cc/). Umjesto nadogradnje postojećeg sustava, cijeli je kod nanovo napisan kako bi se riješili problemi vezani uz rad s većim skupovima podataka, nesavršenim korisničkim sučeljem i ograničenom primjenom.

![Avogadro 2](https://www.openchemistry.org/wp-content/uploads/2014/04/Avogadro2_256.png)
Izvor slike: [Avogadro 2 (Open Chemistry)](https://www.openchemistry.org/projects/avogadro2/)

[Modularni dizajn nove verzije](https://www.kitware.com/avogadro-2-and-open-chemistry/) omogućuje lakše korištenje različitih komponenti uz smanjenu međuovisnost s drugim alatima. Iako Avogadro 2 još uvijek ne nudi sve funkcionalnosti originalnog Avogadra, obje verzije mogu se koristiti istovremeno na istom operacijskom sustavu.

Ovaj alat podržava uvoz različitih kemijskih podataka, poput [Open Babela](open-babel-pretvorba-analiza-spremanje-podataka-molekula.md), kao i izravno učitavanje podataka iz [baza podataka proteina](pdb-baza-proteina.md) (engl. _**P**rotein **D**ata **B**ank_, PDB) i [kemijskih molekula](chembl-baza-bioaktivnih-molekula.md).

!!! tip
    Baze kemijskih spojeva, kao što je [ChEMBL](https://www.ebi.ac.uk/chembl/), uglavnom imaju spojeve u 2D-u s nerealno kratkim kemijskim vezama. Kod otvaranja takvih spojeva Avogadro će ponuditi pretvorbu koordinata u 3D.

Program se može preuzeti sa [službenih stranica](https://two.avogadro.cc/install/index.html#install), gdje su dostupne i dodatne upute za rad poput [dokumenata za početnike](https://two.avogadro.cc/docs/getting-started/index.html).

Upute u nastavku napravljene su korištenjem Avogadra verzije 1.99.0 na [operacijskom sustavu Windows](https://www.microsoft.com/windows/), ali bi sučelje trebalo biti vrlo slično i na drugim operacijskim sustavima koje Avogadro podržava, kao što su [macOS](https://www.apple.com/macos/), [Linux](https://archlinux.org/) i [FreeBSD](https://www.freebsd.org/).

## Sučelje alata Avogadro 2

Korisničko sučelje sastoji se od platna za crtanje (crni ekran) pri čemu je u donjem lijevom kutu označen [Kartezijev koordinatni sustav](https://en.wikipedia.org/wiki/Cartesian_coordinate_system). S lijeve strane platna za crtanje nalaze se alatni okviri s proširenim funkcionalnostima alata. Gornji dio prozora sadrži [alatnu traku](https://two.avogadro.cc/docs/tools/index.html) s osnovnim funkcijama:

- navigacije (*engl.* [Navigation Tool](https://two.avogadro.cc/docs/tools/navigation-tool.html))
- crtanja (*engl.* [Draw Tool](https://two.avogadro.cc/docs/tools/draw-tool.html))
- predložaka (*engl.* [Template Tool](https://two.avogadro.cc/docs/tools/template-tool.html))
- oznaka (*engl.* [Label Tool](https://two.avogadro.cc/docs/tools/label-tool.html))
- odabira (*engl.* [Selection Tool](https://two.avogadro.cc/docs/tools/selection-tool.html))
- manipulacije (*engl.* [Manipulation Tool](https://two.avogadro.cc/docs/tools/manipulation-tool.html))
- manipulacije usmjerene na veze (*engl.* [Bond-Centric Manipulation Tool](https://two.avogadro.cc/docs/tools/bond-centric-manipulation-tool.html))
- mjerenja (*engl.* [Measure Tool](https://two.avogadro.cc/docs/tools/measure-tool.html))
- animacije (*engl.* [Animation Tool](https://two.avogadro.cc/docs/tools/animation-tool.html))
- poravnanja (*engl.* [Align Tool](https://two.avogadro.cc/docs/tools/align-tool.html)).

![Avogadro](https://two.avogadro.cc/_images/home_screenshot_1.png)

Izvor slike: [Avogadro](https://two.avogadro.cc/index.html)

## Alat za crtanje atoma i molekula

Alat [`Draw Tool`](https://two.avogadro.cc/docs/tools/draw-tool.html) (++ctrl+2++), u alatnoj traci, omogućuje crtanje i uređivanje molekulskih struktura u trodimenzionalnom prostoru. Lijevo u alatnom okviru za crtanje `Draw` pod `Element` možemo odabrati bilo koje atome iz periodnog sustava, a vrstu veze odabiremo pod `Bond Order`.

![Draw Tool Icon](https://two.avogadro.cc/_images/icon_draw.svg)

Izvor slike: [Draw Tool (Avogadro documentation)](https://two.avogadro.cc/docs/tools/draw-tool.html)

Atomi i veze se crtaju pomoću miša, a osnovne su funkcije:

- lijevi klik: dodavanje atoma
- lijevi klik + povlačenje: dodavanje atoma i veze s postojećim atomom
- lijevi klik na vezu: promjena broja veza između povezanih atoma
- desni klik: brisanje odabranog atoma ili veze.

!!! admonition "Zadatak"
    1. Pomoću Avogadro alata za crtanje molekula nacrtajte [metanol](https://pubchem.ncbi.nlm.nih.gov/compound/887).
    1. Promijenite strukturu molekule tako da između ugljika i kisika dodate dvostruku vezu. Prepoznajte molekulu koja nastaje nakon ove promjene.
    1. Daljnjom izmjenom prethodne molekule oblikujte ju u [kisikov fluorid](https://pubchem.ncbi.nlm.nih.gov/compound/24547).

## Spremanje datoteke

Spremiti molekulu možemo odabirom opcija u alatnoj traci `Save` ili `Save As` ovisno o tome želimo li spremiti postojeću datoteku ili stvoriti novu. Zatim, odaberemo željeni format datoteke (npr. `.cml`) te kliknemo na `Save`.

!!! warning "Datotečni format"
    Prilikom spremanja ili izvoza datoteke, **nužno** je specificirati ekstenziju datoteke u njezinom imenu. Inače, može se pojaviti greška oblika `Unable to find a suitable file reader for the selected file`.

Ako želimo izvesti molekulu u drugi format prikladnima za druge programe, u alatnoj traci odaberemo `Export Molecule...` i izaberemo željeni format izvoza (npr. `.pdb`) te kliknemo na `Export`.

Najčešće korišteni formati:

| Format | Naziv | Opis |
| ------ | ----- | ---- |
| `.cml` | [Chemical Markup Language](https://en.wikipedia.org/wiki/Chemical_Markup_Language) | Opisuje atome, veze, molekule, reakcije, spektre i analitičke podatke te dr. |
| `.gro` | [GROMACS](https://en.wikipedia.org/wiki/GROMACS) file format | Specifičan za GROMACS, softver za molekulsku dinamiku. Pohranjuje pozicije atoma, brzine i veličinu simulacijske kutije. Dizajniran za rad s velikim sustavima (npr. membrane, proteini u vodi). |
| `.mol2` | [Tripos Mol2 format](https://paulbourke.net/dataformats/mol2/) | Sadrži informacije o atomima, vezama, nabojima i interakcijama. Popularan u računalnom dizajnu lijekova i farmakoinformatici. |
| `.pdb` | [Protein Data Bank](https://en.wikipedia.org/wiki/Protein_Data_Bank) [format](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) | Pogodan za velike molekule poput proteina. Često korišten u bioinformatici i molekulskoj dinamici. |
| `.sdf` | [Structure Data File](https://en.wikipedia.org/wiki/Chemical_table_file#SDF) | Standardni format za opis kemijskih struktura. Često se koristi u farmaceutskoj industriji i kemoinformatici za pohranu velikih skupova molekula. Ne sadrži kemijska svojstva, naboje ili druge specifične informacije o molekuli. |
| `.xyz` | [XYZ format](https://en.wikipedia.org/wiki/XYZ_file_format) | Format pohranjuje samo pozicije atoma, ali ne i veze među njima. Softver mora veze izračunavati na temelju udaljenosti između atoma. |

Svaki format ima specifične prednosti ovisno o vrsti projekta i potrebama simulacije ili vizualizacije:

- jednostavne strukture i male biomolekule: `.xyz`, `.sdf`
- velike biomolekule: `.pdb`
- napredne analize i simulacije: `.gro`, `.mol2`
- baze podataka: `.cml`, `.sdf`.

Sliku molekule možemo spremiti na način da u izborniku `File` pod `Export` odaberemo `Graphics...`.

!!! admonition "Zadatak"
    1. Spremite prethodno nacrtanu molekulu kisikovog fluorida u format `.cml`. Ponovo ju otvorite kako biste se uvjerili da je struktura dobro spremljena.
    1. Nacrtajte molekulu [joda](https://pubchem.ncbi.nlm.nih.gov/compound/24345) i izvezite ju u format `.sdf`.
    1. Nacrtajte molekulu [sumporne kiseline](https://pubchem.ncbi.nlm.nih.gov/compound/1118) i spremite ju u slikovnu datoteku formata PNG.

## Alat za navigaciju

Alat za navigaciju [Navigation Tool](https://two.avogadro.cc/docs/tools/navigation-tool.html) (++ctrl+1++) koristi se za rotiranje, pomicanje i povećanje prikaza molekule unutar platna za crtanje. Mijenja se samo perspektiva pogleda, a zadržavaju se položaji u prostoru svih atoma.

![Navigation Tool Icon](https://two.avogadro.cc/_images/icon_navigate.svg)

Izvor slike: [Navigation Tool (Avogadro documentation)](https://two.avogadro.cc/docs/tools/navigation-tool.html)

Osnovne funkcije navigacije pomoću miša su:

- lijevi klik + povlačenje: molekula će se rotirati u smjeru u kojem se pomiče miš
- desni klik + povlačenje: pomicanje molekule po platnu
- srednji klik + povlačenje: rotiranje i uvećavanje molekule
- srednji dvoklik: centrira i optimalno uveća molekulu.

!!! admonition "Zadatak"
    Koristeći Avogadro alat za crtanje molekula nacrtajte [triksan](https://pubchem.ncbi.nlm.nih.gov/compound/8081). Koristeći alat za navigaciju, rotirajte i uvećajte prikaz molekule.

## Alat za odabir

Alat za odabir [Selection Tool](https://two.avogadro.cc/docs/tools/selection-tool.html) (++ctrl+5++) koristi se za odabir atoma, bilo njihovim pojedinačnim odabirom ili korištenjem okvira za odabir.

![Selection Tool Icon](https://two.avogadro.cc/_images/icon_select.svg)

Izvor slike: [Selection Tool (Avogadro documentation)](https://two.avogadro.cc/docs/tools/selection-tool.html)

Osnovne funkcije odabira pomoću miša su:

- lijevi klik na atom: odabir pojedinačnog atoma
- lijevi klik + povlačenje: odabir više atoma crtanjem okvira za odabir
- lijevi dvoklik na atom: odabir svih povezanih atoma, tj. cijele molekule
- Ctrl + lijevi klik/povlačenje: neodabrani atomi unutar molekule su odabrani, a odabrani su poništeni
- ++ctrl+a++: odabir svih atoma na platnu
- ++ctrl+shift+a++: poništenje odabira.

## Alat za manipulaciju

Alat za manipulaciju [Manipulation Tool](https://two.avogadro.cc/docs/tools/manipulation-tool.html) (++ctrl+6++) omogućuje pomicanje atoma, molekula ili fragmenata.

![Manipulation Tool Icon](https://two.avogadro.cc/_images/icon_manipulate.svg)

Izvor slike: [Manipulation Tool (Avogadro documentation)](https://two.avogadro.cc/docs/tools/manipulation-tool.html)

Osnovne funkcije alata za manipulaciju pomoću miša su:

- lijevi klik na atom: pritisnite i povucite atom da biste njime manipulirali
- srednji/desni klik + povlačenje: rotiranje atoma. Na jednom atomu to neće imati vidljiv učinak

Ako želite poništiti svoje prilagodbe, idite na izbornik `Edit` na gornjoj traci i odaberite `Undo Change Atom Position` ili kraće kombinacijom tipki ++ctrl+z++.

!!! admonition "Zadatak"
    1. Alatom za manipulaciju uredite molekulu triksana iz prethodnog primjera tako da udaljite vodikove atome na veće udaljenosti od ugljikovih atoma.
    1. Alatom za odabir označite bilo koji par susjednih atoma kisika i ugljika. Zatim, alatom za manipulacijom, udaljite i rotirajte odabrani par. Ponovite postupak za sve ostale parove kisika i ugljika.

## Optimizacija geometrije molekule

U mnogim slučajevima prilikom crtanja molekula njezina geometrija možda nije ispravna ili ne izgleda potpuno savršeno. Da biste to ispravili potrebno je u izbornik `Extensions` odabrati `Optimize Geometry` ili jednostavno koristiti kombinaciju tipki ++ctrl+alt+o++. Na ovaj način postižemo realističan prikaz molekule.

!!! admonition "Zadatak"
    Optimizirajte prethodnu molekulu triksana i spremite ju u slikovnu datoteku formata PNG.

## Uvoz i generiranje molekula

U Avogadru molekule možemo dobiti na brži i jednostavniji način od crtanja. Navedeno postižemo uvozom putem imena, preuzimanjem iz baza podataka ili generiranjem pomoću formata SMILES.

### Uvoz putem imena molekule

Uvoz molekula putem njihova imena možemo dobiti tako da ood `File` u `Import` odaberemo `Download by Name...` te upišemo željeno ime molekule, npr. `aspirin`.

!!! admonition "Zadatak"
    Koristeći Avogadro uvezite molekule [alanina](https://pubchem.ncbi.nlm.nih.gov/compound/5950) i [kofeina](https://pubchem.ncbi.nlm.nih.gov/compound/2519) putem njihova imena.

### Uvoz iz baze molekula

Avogadro omogućuje jednostavan uvoz molekula iz vanjskih baza podataka poput [PubChema](https://pubchem.ncbi.nlm.nih.gov/). Ovaj način omogućuje brzo preuzimanje molekula u različitim formatima koji se zatim mogu otvoriti i analizirati unutar Avogadra.

!!! admonition "Zadatak"
    1. Na web stranici PubChem pronđite molekulu saharina i preuzmite njezinu `.sdf` datoteku. Otvorite ju i prikažite u programu Avogadro.
    1. Na istoj web stranci pronađite molekulu diklofenak i preuzmite ju na računalo. Otvorite ju i prikažite u programu Avogadro. Istražite koji je popularni naziv molekule.

### Generiranje molekule pomoću formata SMILES

[Format SMILES](smiles-format.md) omogućuje izgradnju 3D molekula putem niza teksta. U izborniku `Build`, pod opciju `Insert`, odaberite `SMILES...`. Unesite željeni SMILES tekst i pritisnite `OK`.

!!! admonition "Zadatak"
    Prisjetite se [formata SMILES](smiles-format.md) te pomoću njega generirajte sliku [ciklooktana](https://pubchem.ncbi.nlm.nih.gov/compound/9266) i [izopropanola](https://pubchem.ncbi.nlm.nih.gov/compound/Isopropyl-Alcohol) u formatu PNG.

## Alat za mjerenje

Koristeći alat za mjerenje [Measure Tool](https://two.avogadro.cc/docs/tools/measure-tool.html) (++ctrl+8++) možemo odrediti duljine veza, kuteve i dihedrale.

![Measure Tool Icon](https://two.avogadro.cc/_images/icon_measure.svg)

Izvor slike: [Measure Tool (Avogadro documentation)](https://two.avogadro.cc/docs/tools/measure-tool.html)

Alat omogućuje odabir do četiri atoma za mjerenje. Klikom na dva atoma, računa se udaljenosti između odabranih atoma. Ako odaberemo najmanje tri atoma, izračunat će se kut između njih, koristeći drugi atom kao vrh. Kod odabira četiri atoma, alat određuje dihedralni kut. Lijevim klikom na prikaz resetira se odabir prethodno odabranih atoma.

!!! admonition "Zadatak"
    Uvezite molekulu [kofeina](https://pubchem.ncbi.nlm.nih.gov/compound/2519) putem njezina imena i odredite duljinu veze, kuteve i dihedrale bilo koja četiri susjedna atoma po izboru.

## Vrste prikaza molekula

Avogadro dolazi opremljen različitim vrstama prikaza molekula koji pomažu u molekulskoj interpretaciji. U okviru `Display Types` s lijeve strane platna za crtanje možemo odabrati različite reprezentacije molekula od kojih su nam najkorisnije `Ball and Stick`, `Van der Waals Spheres` i `Wireframe`.

!!! admonition "Zadatak"
    Uvezite molekulu [kisikovog difluorida](https://pubchem.ncbi.nlm.nih.gov/compound/24547) putem imena i prikažite je pomoću reprezentacije molekula korištenjem Van der Waalsovih sfera. Spremite je u slikovnu datoteku formata PNG.

## Avogadro 1

Avogadro 1, češće samo Avogadro (različit od Avogadra 2 po prepoznatljivoj narančastoj ikoni), originalna je verzija alata za vizualizaciju i uređivanje molekula. U nastavku opisujemo rad u verziji 1.2.0 izvornog Avogadra.

![PyMOL_logo](https://upload.wikimedia.org/wikipedia/commons/c/c1/Avogadro.png)

Izvor slike: [File:Avogadro.png (Wikimedia Commons)](https://commons.wikimedia.org/wiki/File:Avogadro.png)

Iako je funkcionalan i široko korišten, ima ograničenja pri radu s većim molekulama i skupovima podataka. Sučelje je oba programa vrlo slično, olakšavajući korisnicima prelazak na novu verziju.

!!! warning
    Avogadro (1) se više ne održava. Korištenje nove verzije Avogadra (2) preporučuje se u svim slučajevima.

### Pregled osnovne funkcionalnosti

Alatna traka organizirana je u drugačijem rasporedu, s drugačijim stilom ikonica i manjim izborom funkcionalnosti u odnosu na Avogadro 2. Osnovni alati su:

- alat crtanja (*engl.* [Draw Tool](https://avogadro.cc/docs/tools/draw-tool/))
- alat navigacije (*engl.* [Navigation Tool](https://avogadro.cc/docs/tools/navigate-tool/))
- alat manipulacije usmjerene na veze (*engl.* [Bond-Centric Manipulation Tool](https://two.avogadro.cc/docs/tools/bond-centric-manipulation-tool.html))
- alat manipulacije (*engl.* [Manipulation Tool](https://avogadro.cc/docs/tools/manipulate-tool/))
- alat odabira (*engl.* [Selection Tool](https://avogadro.cc/docs/tools/selection-tool/))
- alat automatskog zakretanje (*engl.* [Auto Rotate Tool](https://avogadro.cc/docs/tools/auto-rotate-tool/))
- alat automatskog optimiziranja (*engl.* [Auto Optimize Tool](https://avogadro.cc/docs/tools/auto-optimize-tool/))
- alat mjerenja (*engl.* [Measure Tool](https://avogadro.cc/docs/tools/measure-tool/))
- alat poravnanja (*engl.* [Align Tool](https://avogadro.cc/docs/tools/align-tool/)).

!!! admonition "Zadatak"
    1. Koristeći Avogadro alat za crtanje molekula nacrtajte [butan](https://pubchem.ncbi.nlm.nih.gov/compound/7843).
    1. Alatom za navigaciju, rotirajte i uvećajte prikaz nacrtane molekule.
    1. Alatom za manipulaciju, rotirajte i razmaknite vodikove atome na veće udaljenosti od ugljikovih atoma.
    1. Optimizirajte prethodnu molekulu i spremite ju u slikovu datoteku formata PNG.
    1. Nacrtajte složenije molekulske spojeve [jodne](https://pubchem.ncbi.nlm.nih.gov/compound/24345) i [sumporne kiseline](https://pubchem.ncbi.nlm.nih.gov/compound/1118) te ih spremite redom u formate `.cml` i `.pdb`.
    1. Na web stranici [PubChem](https://pubchem.ncbi.nlm.nih.gov/) pronđite molekulu [saharina](https://en.wikipedia.org/wiki/Saccharin) i preuzmite njezinu `.sdf` datoteku. Otvorite ju i prikažite u programu Avogadro.
    1. Uvezite molekulu [alanina](https://pubchem.ncbi.nlm.nih.gov/compound/5950) putem njezina imena (`File > Import > Fetch by chemical name...`).
    1. Pomoću formata SMILES generirajte sliku [kofeina](https://pubchem.ncbi.nlm.nih.gov/compound/2519).
    1. Uvezite molekulu [glicina](https://pubchem.ncbi.nlm.nih.gov/compound/12176) putem njezina imena i odredite duljinu veze, kuteve i dihedrale bilo koja četiri susjedna atoma po izboru.

### Elektrostatički potencijal

[Elektrostatičke potencijalne sfere](https://avogadro.cc/docs/tutorials/viewing-electrostatic-potential/) pomažu vizualizaciji distribucije naboja i drugih svojstava povezanih s nabojima molekula.

U izborniku `Extensions`, odaberemo `Create Surfaces...`. Pojavit će se dijaloški okvir s različitim opcijama površine. Pod `Color By`, odaberemo `Electrostatic Potential`, a zatim kliknemo na `Calculate`. Nakon što Avogadro izračuna površinu, kliknemo `Close`.

Na stvorenoj elektrostatičkoj površini možemo vidjeti gdje se nalazi najveća gustoća elektrona (područja označena crvenom bojom) i gdje je najmanja gustoća elektrona (tamnoplava područja). Opcije poput prozirnosti i boje površine mogu se mijenjati klikom na ključ pored prikaza `Surfaces`.

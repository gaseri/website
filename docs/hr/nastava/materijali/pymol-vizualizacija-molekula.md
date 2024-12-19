---
author: Matea Turalija, Vedran Miletić
---

# Vizualizacija molekula u alatu PyMOL

[PyMOL](https://pymol.org/) je sustav za molekulsku vizualizaciju otvorenog koda. Omogućuje generiranje visokokvalitetnih 3D slika malih molekula i bioloških makromolekula, kao što su proteini. Prema izvornom autoru, do 2009. godine gotovo četvrtina svih objavljenih 3D slika proteinskih struktura u znanstvenoj literaturi stvorena je pomoću PyMOL-a.

Alat je moguće preuzeti sa službene stranice [PyMOL-a](https://pymol.org/), a detaljne upute za rad s programom mogu se pronaći u [PyMOL tutorialu](https://edisciplinas.usp.br/pluginfile.php/7380340/mod_resource/content/1/PyMOL_tutorial_2022.pdf).

## Sučelje alata PyMOL

Korisničko sučelje sastoji se od platna za prikaz modela koji omogućuje interaktivni prikaz i manipulaciju 3D modelima.

![PyMOL](https://pymol.org/2/img/PyMOL_screenshot.png)

Izvor slike: [PyMOL | pymol.org](https://pymol.org/)

Konzola za unos naredbi `PyMOL>`, iznad platna za prikaz modela, koristi se za unos naredbi i parametara omogućujući manipulaciju i analizu modela.

Na desnoj strani, u izborniku objekata, aktivni objekti se ističu sivom pozadinom i mogu se vizualizirati na glavnom platnu za prikaz modela.

Ispod izbornika objekata nalaze se upute za navigaciju mišem, a ispod njih je okno selekcije koje omogućuje odabir specifičnih elemenata u modelu.

## Učitavanje molekule iz baze podataka

Najjednostavniji način učitavanja molekule je putem njezinog PDB koda. U [RCSB online bazi podataka proteina](https://www.rcsb.org/), možete pronaći strukturu proteina koju želite vizualizirati u PyMOL-u i očitati njezin PDB kod. Zatim, u konzolu za unos naredbi unesemo `fetch <PDB kod>` ili iz izbornika `File` odaberemo `Get PDB ...` te u prozoru unesemo PDB kod i odaberemo `download...`.

!!! example "Zadatak"
    U [RCSB online bazi podataka proteina](https://www.rcsb.org/) pronađite protein [hemoglobin](https://en.wikipedia.org/wiki/Hemoglobin) i njegov četveroznamenkasti PDB kod te ga učitajte u alatu PyMOL.

U PyMOL-u možemo pod `File\Open...` otvarati kemijske spojeve u različitim formatima uključujući SDF te proteine u brojnim formatima uključujući PDB.

## Prikaz modela

Za proučavanje modela koristimo miš za navigaciju:

- lijevi klik: rotacija
- srednji klik: pomicanje
- desni klik: povećavanje (pomicanje po z osi)
- lijevi dvoklik: izbornik

U kombinaciji sa (++ctrl++) i (++shift++) moguće su i druge akcije koje su ukratko opisane u oknu s uputama za navigaciju mišem.

!!! example "Zadatak"
    Isprobajte funkcionalnosti navigacije mišem na prethodnom primjeru hemoglobina.

## Odabir dijelova molekule

Ovisno koja je opcija odabrana u `Mouse\Selection Mode`, klikom na molekulu, možemo odabrati njezine različite elemente. Moguće je odabrati:

- atoms,
- residues,
- segments,
- objects,
- molecules i
- C-alphas.

Jednostavnije, tu opciju možemo mijenjati klikom na određeni element u oknu selekcije pod `Selecting`.

### Odabir ostataka

Odaberemo sekvencu ostataka (engl. *residues*). U izborniku `Display` uključimo prikaz sekvenci `Sequence`. Povlačenjem miša po traci sekvenci biramo ostatke, molekule ili dijelove molekule.

!!! example "Zadatak"
    Učitajte protein PDB koda `2GTL`. Iz liste ostataka odaberite molekulu `HEM` (na kraju liste), uvećajte sliku da vidite njezinu strukturu, a zatim ju obrišite.

## Izbornik objekata

Izbornik objekata nudi korištenje značajki za prikaz objekata. Svaki objekt ima nekoliko opcija:

- `Action`, `A`
- `Show`, `S`
- `Hide`, `H`
- `Label`, `L`
- `Color`, `C`

`Action` (hrv. *akcija*) omogućuje izvođenje raznih radnji, analize i proračuna:

- `preset` nam omogućuje postavljanje atraktivne vizualizacije u jednom koraku, često se koriste `pretty` i `publication`.  `ball and stick` će nam dati klasični prikaz pogodan za male molekule,
- `align` pa `to molecule` nam omogućuje poravnavanje dva proteina,
- `generate` pa `vacuum electrostatics` stvara prikaz pozitivno i negativno nabijenih dijelova proteina koji nam je koristan kod vizualne analize površine,
- `compute` računa svojstva kao što su broj atoma, naboji, površina i težina.

!!! example "Zadatak"
    Učitajte molekulu djelatne tvari lijeka naziva [diklofenak](https://en.wikipedia.org/wiki/Diclofenac). Uključite klasični prikaz pogodan za male molekule. Izračunajte broj atoma, površinu, masu i naboje unutar molekule.

Opcijom `Show` (hrv. *prikaži*) omogućuje primjenu određenog grafičkog efekta na prikaz objekta:

- `as` nudi prikaz proteina na različite načine, često su korišteni `ribbon`, `cartoon` i `surface`

Pod `Hide` (hrv. *sakrij*) imamo suprotne akcije od `Show` gdje skrivamo djelove objekta koji nam nisu bitni.

Korištenjem `Label` (hrv. *oznaka*) možemo uključiti prikaz naziva atoma, reziduma i dr.

Pod `Color` (hrv. *boja*) možemo detaljno konfigurirati način bojanja odabranih objekata.

!!! example "Zadatak"
    U primjeru iz prethodnog zadatka uključite prikaz naziva atoma te obojajte benzene po želji.

Prikaz sekvence proteina uključujemo klikom na slovo `S` u donjem desnom dijelu ekrana. Slovo `F` prebacuje nas u prikaz preko čitavog ekrana.

## Spremanje

PyMOL-ovu sesiju moguće je spremiti korištenjem opcije `File\Save Session`, odnosno `File\Save Session As...`.

!!! example "Zadatak"
    Spremi primjer iz prethodnog zadatka u bilo kojem podržanom formatu.

## Izvoz

Korištenjem stavke izbornika `File\Export Image As` moguće je izvesti prikaz iz PyMOL-a u sliku u formatu PNG. Korištenjem alata `Draw/Ray` moguće je dodatno dobiti sliku proizvoljno visoke rezolucije korištenjem [praćenja zraka svjetlosti](https://en.wikipedia.org/wiki/Ray_tracing_(graphics)) (engl. *ray tracing*). Taj pristup daje bolju kvalitetu prikaza od [renderiranja](https://en.wikipedia.org/wiki/Rendering_(computer_graphics)) (engl. *rendering*) koje se koristi za prikaz u PyMOL-u.

!!! example "Zadatak"
    Učitajte protein SARS-CoV-2 spike glycoprotein (zatvoreno stanje). Označite na njemu ostatke 80 - 143 i obojite ih u boju magen[t](https://www.hrvatskitelekom.hr/magenta1)a. Uvećajte i prikažite taj dio molekule pa prikaz spremite u formatu `.png`.

    Ponovite izvoz, ali tako da koristite praćenje zraka i da je rezolucija najmanje [4K/UHD](https://youtu.be/FFhE2dDm_D0).

## Izgradnja spojeva i proteina

Osim otvaranja možemo kemijske spojeve i proteine samostalno izgraditi alatom [Builder](https://pymolwiki.org/index.php/Builder). Početak spoja mora biti nekakav fragment u kojem se nalazi ugljik i tada nam se nudi opcija `Create As New Object`. Odabirom odgovarajućih fragmenata i mjesta gdje ih želimo postaviti možemo izgraditi po želji veliku molekulu.

Dvije vrste spojeva koje možemo graditi su:

- male molekule, ondnosno kemijski spojevi (kartica `Chemical`) i
- peptidi, odnosno proteini (kartica `Protein`).

!!! example "Zadatak"
    Pomoću Buildera izradite ibuprofen i [stimulans](https://en.wikipedia.org/wiki/Stimulant) po vlastitom izboru.

    Pomoću Buildera izradite [tripeptide](https://en.wikipedia.org/wiki/Tripeptide) Lactotripeptides i Leupeptin te [oligopeptid](https://en.wikipedia.org/wiki/Oligopeptide) po vlastitom izboru.

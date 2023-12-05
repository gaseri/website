---
author: Matea Turalija, Irena Hartmann, Vedran Miletić
---

# Vizualizacija i uređivanje molekula alatom Avogadro

[Avogadro](https://avogadro.cc/) je besplatan digitalni alat otvorenog koda namijenjen vizualizaciji, izradi i uređivanju molekula. Dizajniran je za korištenje na više platformi u računalnoj kemiji, modeliranju molekula, bioinformatici, znanosti materijala i srodnim područjima. Ovaj alat podržava uvoz različitih vrsta podataka specifičnih za kemiju, poput [Open Babela](open-babel-pretvorba-analiza-spremanje-podataka-molekula.md). Također omogućuje izravno učitavanje podataka iz [baza podataka proteina](pdb-baza-proteina.md) (engl. _**P**rotein **D**ata **B**ank_, PDB) i [kemijskih molekula](chembl-baza-bioaktivnih-molekula.md).

Avogadro se može preuzeti ovdje <sourceforge.net/projects/avogadro/>, a detaljne upute za rad s programom mogu se pronaći na službenim stranicama [Avogadra](avogadro.cc/docs/).

## Sučelje alata Avogadro

Korisničko sučelje sastoji se od platna za crtanje (crni ekran), pri čemu je u donjem lijevom kutu označen [Kartezijev koordinatni sustav](https://en.wikipedia.org/wiki/Cartesian_coordinate_system). Gornji dio prozora sadrži alatnu traku s osnovnim funkcijama poput [crtanja](https://avogadro.cc/docs/tools/draw-tool/), [navigacije](https://avogadro.cc/docs/tools/navigate-tool/), [mjerenja](https://avogadro.cc/docs/tools/measure-tool/) itd., a s lijeve strane platna za crtanje nalaze se alatni okviri s proširenim funkcionalnostima alata.

Za brže snalaženje, možemo koristiti miš za navigaciju:

- lijevi klik: dodavanje atoma
- lijevi klik + povlačenje: dodavanje atoma i veze
- lijevi klik na molekulsku vezu: mijenjanje broj veza
- desni klik: brisanje atoma ili veze
- desni klik + povlačenje: pomicanje slike
- srednji klik + povlačenje: rotiranje i uvećavanje slike
- srednji dvoklik: centrira i optimalno uvećanje slike

## Crtanje atoma i molekula

Alat [`Draw Tool`](https://avogadro.cc/docs/tools/draw-tool/), u alatnoj traci, omogućuje crtanje i uređivanje molekulskih struktura u trodimenzionalnom prostoru. Lijevo u alatnom okviru za crtanje `Draw` pod [`Element`](https://avogadro.cc/docs/tools/draw-tool/#4) možemo odabrati bilo koje atome iz periodnog sustava, a vrstu veze odabiremo pod [`Bond Order`](https://avogadro.cc/docs/tools/draw-tool/#7).

Dodatno, alatom [`Manipulation Tool`](https://avogadro.cc/docs/tools/manipulate-tool/) možemo pomicati pojedine atome. Postoji i varijanta [`Bond Centric Manipulate Tool`](https://avogadro.cc/docs/tools/bond-centric-manipulate-tool/), a alatom za navigaciju [`Navigate Tool`](https://avogadro.cc/docs/tools/navigate-tool/) pomićemo, rotiramo i uvećavamo prikaz molekule.

!!! admonition "Zadatak"
    Koristeći Avogadro alat za crtanje molekula nacrtajte: [metan](https://pubchem.ncbi.nlm.nih.gov/compound/297), [etan](https://pubchem.ncbi.nlm.nih.gov/compound/6324), [propan](https://pubchem.ncbi.nlm.nih.gov/compound/6334) i [butan](https://pubchem.ncbi.nlm.nih.gov/compound/7843). Koristeći alat za navigaciju, rotirajte i uvećajte prikaz molekula. Zatim, alatom za manipulaciju, rotirajte i razmaknite vodikove atome na veće udaljenosti od ugljikovih atoma.

## Spremanje datoteke

Nakon što ste izradili molekulu odaberite opciju `Save` ili `Save As` ovisno o tome želite li spremiti postojeću datoteku ili stvoriti novu. Odaberite željeni format datoteke (npr. `.cml`) te kliknite na `Save`.

Ako želite izvesti molekulu u drugi format prikladnima za druge programe, u alatnoj traci odaberite `Export Molecule` i odaberite željeni format izvoza (npr. `.sdf`, `.pdb`, `.xyz` i sl.) i kliknite na `Export`.

!!! warning "Datotečni format"
    Prilikom spremanja ili izvoza datoteke, **nužno** je specificirati ekstenziju datoteke u njezinom imenu. Inače, može se pojaviti greška "Unable to find suitable file writer for the selected format".

!!! admonition "Zadatak"
    Nacrtajte složenije molekulske spojeve [jodne](https://pubchem.ncbi.nlm.nih.gov/compound/24345) i [sumporne kiseline](https://pubchem.ncbi.nlm.nih.gov/compound/1118) te ih spremite redom u formate `.sdf` i `.xyz`.

## Vrste prikaza molekula

Avogadro dolazi opremljen različitim vrstama prikaza molekula koji pomažu u molekulskoj interpretaciji. U okviru [`Display Types`](https://avogadro.cc/docs/display-types/) možemo odabrati različite reprezentacije molekula od kojih su nam najkorisnije [`Ball and Stick`](https://avogadro.cc/docs/display-types/#ball-and-stick) i [`Van der Waals Spheres`](https://avogadro.cc/docs/display-types/#van-der-waals-spheres).

!!! admonition "Zadatak"
    Iz prethodog primjera uvezite nacrtane molekulske spojeve i prikažite ih pomoću `Van der Waals Spheres` reprezentacije molekula.

## Optimizacija geometrije

U mnogim slučajevima prilikom crtanja molekule njezina geometrija možda nije ispravna ili ne izgleda potpuno savršeno. Da biste to ispravili potrebno je pod `Extensions` u `Open Babel` pronaći `Optimize Geometry` ili jednostavno koristiti kombinaciju tipki ++ctrl+alt+o++. Na ovaj način postižemo realističan prikaz molekule.

!!! admonition "Zadatak"
    Koristeći Avogadro nacrtajte [benzen](https://pubchem.ncbi.nlm.nih.gov/compound/241), [glicin](https://pubchem.ncbi.nlm.nih.gov/compound/12176) i [aspirin](https://pubchem.ncbi.nlm.nih.gov/compound/2244) te ih optimizirajte.

## Uvoz putem imena molekule

Molekule možemo dobiti i putem uvoza koristeći njihova imena. Pod `File` u `Import` odaberemo `Download by Name...` te upišemo željeno ime molekule, npr. `diclofenac`.

!!! admonition "Zadatak"
    Koristeći Avogadro uvezite molekule [alanina](https://pubchem.ncbi.nlm.nih.gov/compound/5950) i [kofeina](https://pubchem.ncbi.nlm.nih.gov/compound/2519).

## Uvoz iz baze molekula

!!! admonition "Zadatak"
    Na web stranici [PubChem](https://pubchem.ncbi.nlm.nih.gov/) pronđite molekulu [saharina](https://en.wikipedia.org/wiki/Saccharin) i preuzmite njezinu `.sdf` datoteku. Otvorite ju i prikažite u programu Avogadro.

## Izrada molekula pomoću formata SMILES

[Format SMILES](smiles-format.md) omogućuje izgradnju 3D molekula putem niza teksta. U izborniku `Build`, pod opciju `Insert`, odaberite `SMILES...`. Unesite željeni SMILES tekst i pritisnite `OK`.

!!! admonition "Zadatak"
    Prisjetite se [formata SMILES](smiles-format.md) s prethodnih vježbi te pomoću SMILES teksta generirajte slike [ugljikovog dioskida](https://pubchem.ncbi.nlm.nih.gov/compound/280), [ciklookata](https://pubchem.ncbi.nlm.nih.gov/compound/9266) i [izopropanola](https://pubchem.ncbi.nlm.nih.gov/compound/Isopropyl-Alcohol).

## Mjerenje u Avogadru

Koristeći alat za mjerenje [`Measure Tool`](https://avogadro.cc/docs/tools/measure-tool/) možemo odrediti duljine veza, kuteve i dihedrale.

Alat omogućuje odabir do četiri atoma za mjerenje. Klikom na dva atoma, računa se udaljenosti između odabranih atoma. Ako odaberemo najmanje tri atoma, izračunat će se kut između njih, koristeći drugi atom kao vrh. Kod odabira četiri atoma, alat određuje dihedralni kut. Lijevim klikom na prikaz resetira se odabir prethodno odabranih atoma.

!!! admonition "Zadatak"
    Uvezite molekulu [kofeina](https://pubchem.ncbi.nlm.nih.gov/compound/2519) putem njezina imena i odredite duljinu veze, kuteve i dihedrale bilo koja četiri susjedna atoma po izboru.

## Elektrostatički potencijal

[Elektrostatičke potencijalne sfere](https://avogadro.cc/docs/tutorials/viewing-electrostatic-potential/) pomažu vizualizaciji distribucije naboja i drugih svojstava povezanih s nabojima molekula.

U izborniku `Extensions`, odaberemo `Create Surfaces...`. Pojavit će se dijaloški okvir s različitim opcijama površine. Pod `Color By`, odaberemo `Electrostatic Potential`, a zatim kliknemo na `Calculate`. Nakon što Avogadro izračuna površinu, kliknemo `Close`.

Na stvorenoj elektrostatičkoj površini možemo vidjeti gdje se nalazi najveća gustoća elektrona (područja označena crvenom bojom) i gdje je najmanja gustoća elektrona (tamnoplava područja). Opcije poput prozirnosti i boje površine mogu se mijenjati klikom na ključ pored prikaza `Surfaces`.

## Planarne molekule

Baze kemijskih spojeva kao što je ChEMBL uglavnom imaju spojeve u 2D-u s nerealno kratkim kemijskim vezama. Kod otvaranja takvih spojeva Avogadro će ponuditi pretvorbu koordinata u 3D.

## Avogadro 2

Avogadro je danas robusno, fleksibilno rješenje koje povezuje i koristi snagu [Visualization Toolkit (VTK)](https://www.vtk.org/), program otvorenog koda za 3D grafiku, procesiranje slika i vizualizaciju, uz dodatne mogućnosti analize i vizualizacije.

Avogadro projekt je u završim fazama ponovnog pisanja središnjih struktura podataka, algoritama i sposobnosti vizualizacije kojeg su autori nazvali Avogadro 2, iako se na službenim stranicama Avogadra još uvijek preporuča stara verzija.

[Avogadro 2](https://www.openchemistry.org/projects/avogadro2/) nije samo nova verzija Avogadra -- umjesto ažuriranja i nadogradnje ranijih verzija, cijeli kod je nanovo napisan zbog problema u radu Avogadra sa većim skupovima podataka, nesavršenog sučelja i želje da se proširi područje primjene originalno namijenjeno korisnicima Avogadra. Također, jedna od bitnih promjena u odnosu na originalni Avogadro je [primjena modularnosti u dizajnu](https://www.kitware.com/avogadro-2-and-open-chemistry/) koja dopušta veće korištenje komponenti, kao i manji broj međuovisnosti na druge alate. Autori ističu kako Avogadro 2 još uvijek nema sve funkcionalnosti Avogadra, stoga je moguće imati i koristiti oba na istom sistemu.

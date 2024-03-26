---
author: Matea Turalija
---

# Format za unos formula molekula SMILES

Pojednostavljeni sustav unosa linija s molekularnim ulazom ili kraće [SMILES](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html) (engl. _**S**implified **M**olecular **I**nput **L**ine **E**ntry **S**ystem_) omogućava lako čitljiv i jednostavan zapis molekula pomoću niza znakova. U nastavku ćemo razmotriti nekoliko osnovnih sintaktičkih pravila koji se moraju poštovati.

Poznate spojeve, ali i vlastite nove spojeve, moguće je crtati na platnu koristeći mrežne alate:

- [PubChem SMILES generator](https://pubchem.ncbi.nlm.nih.gov/edit3/index.html)
- [ChemInfo SMILES generator](https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html)

!!! admonition "Zadatak"
    1. Koristeći [PubChem](https://pubchem.ncbi.nlm.nih.gov/edit3/index.html) mrežni alat za crtanje molekula, nacrtajte sljedeći kemijski spoj koji je prikazan na slici:

        ![aspirin](https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=2244&t=l)

        Izvor slike: [PubChem CID 2244](https://pubchem.ncbi.nlm.nih.gov/compound/2244)

    2. Za automatski generirani SMILES zapis prethodnog crteža, pronađite naziv, kemijsku formulu i 3D prikaz molekule u bazi [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

## Atomi i veze

SMILES podržava sve elemente u [periodnom sustavu](https://en.wikipedia.org/wiki/Periodic_table). Svaki se atom označava svojim kemijskim simbolom u uglatim zagradama, na primjer `[Au]` za zlato. Zagrade se mogu izostaviti za organski dio molekule i to za elemente `B`,`C`, `N`, `O`, `P`, `S`, `F`, `Cl`, `Br` i `I`. Svi drugi elementi moraju biti u zagradama. Ako su zagrade izostavljene, pretpostavlja se da se radi o regularnom broju atoma vodika. Naprimjer, za vodu je jednostavno `O` jer se podrazumjeva da su na kisik spojena 2 atoma vodika iako nije greška napisati `[OH2]` ili `[H]O[H]`

Veze su predstavljene pomoću jednog od sljedećih simbola prikazanih u tablici:

| Simbol | Opis veze | Primjer |
| :----: | --------- | ------- |
| `.` | Nepovezane strukture | Natrijev klorid `[Na+].[Cl-]` |
| `-` | Jednostruka veza | Etan `C-C` |
| `=` | Dvostruka veza | Ugljikov dioksid`O=C=O` |
| `#` | Trostruka veza | Cijanovodična kiselina `C#N` |
| `/` | Prikaz jednostrukih veza uz dvostruku vezu | 1,2-difluoretilen `F/C=C/F` |

Iako se pojedinačne veze mogu zapisati s oznakom `-`, to se obično izostavlja. Na primjer, etanol može biti napisan kao `C-C-O`, `CC-O` ili `C-CO`, ali obično se piše `CCO`.

Kombiniranjem simbola atoma i simbola veza možemo prikazati jednostavne lančane strukture pomoću SMILES notacije. Važno je napomenuti da strukture koje se unose pomoću SMILES-a ne uključuju vodikove atome. SMILES softver automatski prepoznaje broj mogućih veza koje svaki atom može imati. Ako korisnik ne specificira dovoljno veza putem SMILES notacije, sustav će automatski pretpostaviti da su preostale veze zadovoljene vodikovim vezama.

!!! admonition "Zadatak"
    Koristeći PubChem, mrežni alat za crtanje molekula, nacrtajte sljedeće spojeve pomoću formata SMILES: [metan](https://en.wikipedia.org/wiki/Methane), [etan](https://en.wikipedia.org/wiki/Ethane), [propan](https://en.wikipedia.org/wiki/Propane) i [butan](https://en.wikipedia.org/wiki/Butane). Prokomentirajte dobivene rezultate.

## Grane

Grane se označavaju zagradama, kao što je prikazano u primjeru `CCC(=O)O` za [propionsku kiselinu](https://en.wikipedia.org/wiki/Propionic_acid) i `FC(F)F` za [fluoroform](https://en.wikipedia.org/wiki/Fluoroform). Prvi atom unutar zagrada i prvi atom nakon zagrada su povezani na isti atom. Simbol veze mora biti unutar zagrada, na primjer izraz izvan zagrada poput `CCC=(O)O` nije valjan.

!!! admonition "Zadatak"
    Koristeći PubChem nacrtajte sljedeće spojeve pomoću formata SMILES: [izopropanol](https://en.wikipedia.org/wiki/Isopropyl_alcohol), [aceton](https://bs.wikipedia.org/wiki/Aceton), [izopentan](https://en.wikipedia.org/wiki/Isopentane) i [2,2-dimetilbutan](https://en.wikipedia.org/wiki/2,2-Dimethylbutane).

## Ioni

Ako je atom u ioniziranom stanju može se dodati znak `+` ili `-`, na primjer [hidroksid](https://en.wikipedia.org/wiki/Hydroxide) je `[OH-]`, [amonij](https://en.wikipedia.org/wiki/Ammonium) `[NH4+]`, a `[Ti+4]` ili `[Ti++++]` je [titanij](https://en.wikipedia.org/wiki/Titanium).

!!! admonition "Zadatak"
    Koristeći PubChem nacrtajte sljedeće spojeve pomoću formata SMILES: [hidronij](https://en.wikipedia.org/wiki/Hydronium) i [kobalt(III)](https://pubchem.ncbi.nlm.nih.gov/compound/Cobaltic-cation).

## Prstenaste strukture

Prstenaste strukture možemo kreirati korištenjem brojeva za identifikaciju otvaranja i zatvaranja atoma prstena. Na primjer, u `C1CCCCC1`, prvi ugljik ima broj `1` koji se povezuje jednostrukom vezom s posljednjim ugljikom koji također ima broj `1`. Dobivena struktura predstavlja [cikloheksan](https://en.wikipedia.org/wiki/Cyclohexane). Spojevi koje imaju višestruke prstenove mogu se identificirati korištenjem različitih brojeva za svaki prsten.

!!! admonition "Zadatak"
    Koristeći PubChem nacrtajte sljedeće spojeve pomoću formata SMILES: [ciklooktan](https://en.wikipedia.org/wiki/Cyclooctane) i [benzen](https://en.wikipedia.org/wiki/Benzene).

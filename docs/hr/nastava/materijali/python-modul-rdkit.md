---
author: Matea Turalija
---

# Python modul RDKit

[RDKit](https://www.rdkit.org/) je otvorena programska biblioteka za računalnu kemiju. Omogućava vizualizaciju molekula, izračun molekulskih opisnika (engl. *molecular descriptors*), modeliranje i manipulaciju kemijskim reakcijama te mnoge druge funkcionalnosti. U nastavku ćemo djelomično pratiti odjeljak [Getting Started with the RDKit in Python](https://www.rdkit.org/docs/GettingStartedInPython.html), koji je dio [službene dokumentacije RDKita](https://www.rdkit.org/docs/index.html) i može poslužiti kao dopuna ovim materijalima.

Za razumijevanje zadataka potrebno je [osnovno poznavanje](python-programiranje.md) [programskog jezika Python](https://www.python.org/) i pojmova iz kemoinformatike. Zadatke ćemo rješavati unutar okruženja [Google Colab](https://colab.research.google.com/), koje nudi mogućnost pokretanja [Jupyterovih](https://jupyter.org/) [bilježnica](https://jupyter-notebook.readthedocs.io/).

U Google Colabu, Python kod se unosi kroz Python ćelije, a u tekstualnim ćelijama primjenjuje se jezik Markdown. O jeziku Markdown već smo raspravljali u temi [Suradnički uređivač teksta HackMD i jezik Markdown](hackmd-markdown.md).

Naredbe u Linux terminalu zapisuju se u bilježnicu dodavanjem znaka `!` ispred same naredbe. Na ovaj način, možemo instalirati RDKit korištenjem naredbe:

``` python
!pip install rdkit
```

U programskom jeziku Python, napisat ćemo programsku skriptu za crtanje spojeva koristeći njihov zapis u [formatu SMILES](smiles-format.md). U nastavku je isječak programskog koda koji koristi module [Chem](https://www.rdkit.org/docs/source/rdkit.Chem.html#module-rdkit.Chem) i [Draw](https://www.rdkit.org/docs/source/rdkit.Chem.Draw.html#module-rdkit.Chem.Draw) za crtanje molekule:

``` python
from rdkit import Chem
from rdkit.Chem import Draw

molekula = Chem.MolFromSmiles('CCCC')
Draw.MolToImage(molekula)
```

!!! example "Zadatak"
    U Python programskom jeziku nacrtajte sljedeće kemijske spojeve: [aceton](https://en.wikipedia.org/wiki/Acetone), [benzen](https://en.wikipedia.org/wiki/Benzene) i [aspirin](https://en.wikipedia.org/wiki/Aspirin).

Želimo li nacrtati dvije molekule jednu do druge, morat ćemo kreirati SMILES listu u kojoj ćemo pohraniti SMILES zapis molekula. Zatim ćemo pomoću petlje `for` kreirati slike molekula. To možemo postići na sljedeći način:

``` python
from rdkit import Chem
from rdkit.Chem import Draw

smiles_list = ['CC(=O)C', 'C1=CC=CC=C1']
m_list = []

for smiles in smiles_list:
  m = Chem.MolFromSmiles(smiles)
  m_list.append(m)

Draw.MolsToGridImage(m_list, molsPerRow=2)
```

!!! example "Zadatak"
    1. Nacrtajte četiri aminokiseline [glicin](https://en.wikipedia.org/wiki/Glycine), [fenilalanin](https://en.wikipedia.org/wiki/Phenylalanine), [histidin](https://en.wikipedia.org/wiki/Histidine) i [cistein](https://en.wikipedia.org/wiki/Cysteine) tako da se nalaze na istoj slici poredane jedna ispod druge.
    2. Prilagodite prethodni zadatak tako da imate dva retka, svaki s dvije slike molekula.

## Ispis SMILES zapisa

Tip objekta pohranjenog u varijabli `molekula` možemo provjeriti funkcijom `type()`.

``` python hl_lines="5"
from rdkit import Chem
from rdkit.Chem import Draw

molekula = Chem.MolFromSmiles('CCCC')
type(molekula)
```

``` shell
rdkit.Chem.rdchem.Mol
```

`rdkit.Chem.rdchem.Mol` je klasa koja predstavlja molekulu u RDKitu. Ova klasa ima različite metode i atribute koji omogućuju pristup raznim informacijama o molekuli, uključujući atomske podatke, veze između atoma i druge relevantne podatke o molekularnoj strukturi.

Želimo li pretvoriti objekt `molekula` u znakovni niz u formatu SMILES, napisat ćemo sljedeće:

``` python hl_lines="5"
from rdkit import Chem
from rdkit.Chem import Draw

molekula = Chem.MolFromSmiles('CCCC')
smiles = Chem.MolToSmiles(molekula)
```

Ako provjerimo kojeg je tipa objekt `smiles` vidjet ćemo da je u zapisu `<class 'str'>`, dakle niz znakova. Ispišimo na ekran objekt `smiles` da dobijemo SMILES zapis molekule:

``` python hl_lines="6"
from rdkit import Chem
from rdkit.Chem import Draw

molekula = Chem.MolFromSmiles('CCCC')
smiles = Chem.MolToSmiles(molekula)
print(smiles)
```

``` shell
CCCC
```

## Molekulski opisnici

Molekulski opisnici su kvantitativne reprezentacije određenih karakteristika molekula. Omogućavaju konverziju kompleksnih molekularnih struktura u numeričke podatke, čime se olakšava analiza i modeliranje.

Iz modula `Chem` ćemo uvesti pripadni modul [Descriptors](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html) za pristup molekulskim opisnicima. Primjer ispod koristi nekoliko različitih opisnika:

- `MolWt` izračunava molarnu masu (engl. *molecular weight*),
- `HeavyAtomCount` broji atome (oni koji nisu vodik) u molekuli,
- `NumValenceElectrons` računa ukupan broj valentnih elektrona u molekuli.

``` python hl_lines="8-10"
from rdkit import Chem
from rdkit.Chem import Descriptors

# Definira molekulu iz SMILES zapisa
molekula = Chem.MolFromSmiles('CCCC')

# Izračun i ispis opisnika
molarna_masa = Descriptors.MolWt(molekula)
broj_atoma_u_molekuli = Descriptors.HeavyAtomCount(molekula)
valentni_elektroni = Descriptors.NumValenceElectrons(molekula)

print("Molarna masa:", molarna_masa, "g/mol")
print("Broj atoma u molekuli:", broj_atoma_u_molekuli)
print("Broj valentnih elektrona:", valentni_elektroni)
```

!!! example "Zadatak"
    Nacrtajte [mravlju kiselinu](https://en.wikipedia.org/wiki/Formic_acid) (metanska kiselina) te ispišite na ekran molarnu masu i broj atoma u molekuli te broj valentnih elektrona.

## Pohrana slikovne datoteke

Slike generiranih molekula mogu se pohraniti na dva načina:

``` python hl_lines="6"
from rdkit import Chem
from rdkit.Chem import Draw

smiles = 'C(=O)O'
molekula = Chem.MolFromSmiles(smiles)
Draw.MolToFile(molekula, "slika1.png", size=(500,500))
```

``` python hl_lines="6 7"
from rdkit import Chem
from rdkit.Chem import Draw

smiles = 'C(=O)O'
molekula = Chem.MolFromSmiles(smiles)
slika = Draw.MolToImage(molekula)
slika.save("slika2.png", size=(500,500))
```

!!! example "Zadatak"
    Nacrtajte molekulu djelatne tvari lijeka naziva [diklofenak](https://en.wikipedia.org/wiki/Diclofenac). Pronađite o kojoj je djelatnoj tvari riječ i njezin popularni naziv. Sliku molekule, veličine 600x400 točaka, pohranite tako da se na samoj slici ispod nacrtane molekule nalazi i naziv lijeka. Koristite argument `legend` za ispis naziva lijeka ispod slike.

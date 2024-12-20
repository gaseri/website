---
author: Ema Penezić, Matea Turalija Reščić
---

# Praktični vodič za Python modul RDKit: moduli i funkcije

## Biblioteka `RDKit`

`RDKit` ([web sjedište](https://www.rdkit.org/), [stranica na Wikipediji](https://en.wikipedia.org/wiki/RDKit)) je [programska biblioteka](https://en.wikipedia.org/wiki/Library_(computing)) pisana u jezicima [C++](https://isocpp.org/) i [Python](python-programiranje.md) za [računalnu kemiju](kemoinformatika-povijest-osnove.md) koja omogućuje analizu, vizualizaciju i [modeliranje molekula](kemoinformatika-povijest-osnove.md#molekulsko-modeliranje). Koristi se za istraživanje kemijskih svojstava i procesa.

### Modul biblioteke `RDKit`

| Modul | Opis |
| ----- | ---- |
| `rdkit.Chem` | Pruža osnovne alate za kreiranje, manipulaciju i analizu molekulskih struktura |

### Funkcije unutar `Chem` modula

| Funkcija | Opis |
| -------- | ---- |
| `MolFromSmiles` | Pretvaranje SMILES zapisa kemijskih spojeva u RDKit-ov objekt molekule |
| `MolToSmiles` | Pretvaranje objekta u znakovni niz u format SMILES |

### Funkcije unutar `Draw` modula

| Funkcija | Opis |
| -------- | ---- |
| `MolToImage` | Crtanje jedne molekule na ekranu |
| `MolToFile` | Pohrana slikovne datoteke |
| `MolsToGridImage` | Omogućuje istovremeni prikaz više molekula |

### Funkcije unutar `Descriptors` modula

| Funkcija | Opis |
| -------- | ---- |
| `MolWt` | Izračunava molarnu masu (g/mol) |
| `HeavyAtomCount` | Broji atome u molekuli (osim vodika) |
| `NumValenceElectrons` | Računa ukupan broj valentnih elektrona u molekuli |
| `RingCount` | Izračunava broj prstenova unutar molekule |
| `MolLogP` | Izračunava hidrofobnost molekule, izraženu kao logaritamski omjer raspodjele između vode i oktanola |

### Primjer

``` python
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

# Definira varijablu molekula iz SMILES zapisa
molekula = Chem.MolFromSmiles("(=O)O")

# Izračun i ispis opisnika
molarna_masa = Descriptors.MolWt(molekula)
print("Molarna masa:", molarna_masa, "g/mol")

# Pohrana slikovne datoteke
Draw.MolToFile(molekula, "slika.png", size=(500,500), legend="Mravlja kiselina")

# Crtanje molekule na ekranu
Draw.MolToImage(molekula)
```

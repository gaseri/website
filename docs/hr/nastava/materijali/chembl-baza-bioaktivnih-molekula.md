---
author: Vedran Miletić
---

# Baza bioaktivnih molekula ChEMBL

[ChEMBL](https://www.ebi.ac.uk/chembl/) je vrlo značajna baza kemijskih spojeva i drugih podataka (mete, eseji, dokumenti, stanice i tkiva).

## Pretraga

Pretragom dobivamo rezultate koje možemo preuzeti kao CSV i TSV (datoteke u kojem su elementi retka odvojeni točkom i zarezom, odnosno znakom tabulatora, prikladne za obradu u tabličnom kalkulatoru ili alatu kao [pandas](https://pandas.pydata.org/)). Osim toga, možemo preuzeti SDF datoteku s rezultatima pretraživanja koju ćemo koristiti u alatima za vizualizaciju molekula (dobit ćemo datoteku komprimiranu GZIP-om, što na Windowsima možemo raspakirati korištenjem [7-Zipa](https://www.7-zip.org/)).

[Format SDF](https://en.wikipedia.org/wiki/Chemical_table_file#SDF) je vrlo popularan među softverima za molekularni docking, a nastao je kao proširenje [formata MDL Molfile](https://en.wikipedia.org/wiki/Chemical_table_file) koje omogućuje dodavanje metapodataka o molekuli i pohranjivanje više molekula unutar jedne datoteke.

## Podaci o spoju

Odabirom bilo kojeg rezultata pretrage dobit ćemo detaljan izvještaj o tom spoju (sliku, naziv(e), svojstva i dr.).

Nama je zanimljiv dio `Representations` gdje možemo preuzeti spoj u formatima MDL Molfile (SDF sa samo jednim spojem u njemu) i [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system). Zapis u formatu [InChI](https://en.wikipedia.org/wiki/International_Chemical_Identifier) je također dostupan, ali on nam rjeđe treba.

## Ostale baze slične namjene

- [ChemSpider](https://www.chemspider.com/)
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/)

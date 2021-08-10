---
author: Vedran Miletić
---

# Pretvorba formata alatom Open Babel

Često nam u praksi trebaju različiti formati zapisa istih molekula za korištenje u različitim softverima za računalnu kemiju. Open Babel nam omogućuje pretvorbu između različitih formata.

Na lijevoj strani biramo ulazni format i ulaznu datoteku ili klikom na `Input below` izravni unos zapisa molekule.

Na desnoj strani biramo izlazni format i izlaznu datoteku ili klikom na `Output below` izravni ispis zapisa molekule.

Ilustracije radi, možemo pretvoriti `smi -- SMILES format` neke molekule sa ChEMBL-a ili Wikipedije u `sdf -- MDL MOL format`. Ukoliko u ulazu unesemo više molekula, dobit ćemo isto to u izlazu, jer format SDF podržava više molekula i odvaja ih znakovima `$$$$`.

## Opcije

Bez uključivanja dodatnih opcija dobit ćemo molekulu bez vodika tamo gdje bi trebali biti. Vodike možemo dobiti uključivanjem opcije `Add hydrogens`.

Također molekula neće imati koordinate, odnosno imat će sve tri koordinate svih atoma jednake `0.00`. Uključimo li u opcijama `Generate 2D coordinates`, dobit ćemo približne koordinate molekule u 2D-u.

Opcijom `Generate 3D coordinates` dobit ćemo koordinate u tri dimenzije. Takvu molekulu sad možemo koristiti za molekularni docking ili kvantnu kemiju.

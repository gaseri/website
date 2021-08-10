---
author: Vedran Miletić
---

# Dokumentiranje Python koda

## Dokumentiranje korištenjem Docstringova

Kod se dokumentira korištenjem Pythonovih [Docstringova](https://www.python.org/dev/peps/pep-0257/). Konvencija koju ćemo mi koristiti je oblika:

``` python
def pphs_podaci_na_studiju(studij):
    """
    Vraća podatke o kolegiju PPHS na određenom studiju (broj sati
    predavanja i vježbi, broj ECTS-a).

    Argumenti:
    studij -- "Jednopredmetna informatika" ili "Dvopredmetna informatika"

    Vraća:
    Podatke o kolegiju PPHS na danom studiju tipa tuple, oblika
    (sati_predavanja, sati_vježbi, ects_bodovi) ili None ako kolegij
    ne postoji na danom studiju.
    """
    if studij == "Jednopredmetna informatika":
        return (30, 30, 5)
    else:
        return None

def pphs_podaci_na_studiju2(studij):
    podaci = pphs_podaci_na_studiju(studij)
    if podaci is not None:
       return "PPHS ima %d sati predavanja, %d sati vježbi i %d ECTS." % podaci
    else:
       return None
```

Važne značajke naše konvencije:

- Svaki redak neka ima **najviše** 72 stupca (80 je širina terminala, a 8 stupaca je uvlaka). Cilj je da ne dođe do automatskog prelamanja redaka.
- Prvi redak je općeniti opis funkcije koji može biti relativno kratak, ali **mora dati ideju čemu funkcija služi**.
- Redak `Argumenti:` odmaknut je jednim praznim retkom od općenitog opisa funkcije.
- Ispod retka `Argumenti:` navedeni su argumenti funkcije **onim redom kojim su navedeni u definciji funkcije**, i svaki je opisan. Opis je od argumenta odvojen dvjema crticama, razmaknutim od teksta s obje strane.
- Redak `Vraća:` odmaknut je jednim praznim retkom od opisa posljednjeg argumenta.
- Ispod retka `Vraća:` detaljno je opisano što funkcija vraća i kojeg tipa je to što vraća u kojem od mogućih slučajeva.

!!! admonition "Zadatak"
    Dokumentirajte funkciju `pphs_podaci_na_studiju2()` po uzoru na funkciju `pphs_podaci_na_studiju()`, obavezno slijedeći našu konvenciju.

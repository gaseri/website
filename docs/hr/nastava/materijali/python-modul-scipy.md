---
author: Domagoj Margan, Vedran Miletić
---

# Rad s Python modulom scipy

Uključivanje modula `scipy` najčešće se vrši na način:

``` python
import scipy as sp
```

Uključivanje specifičnih podmodula vrši se na način:

``` python
import scipy.sparse.csgraph as csg
```

## Python modul scipy: podmodul sparse.csgraph

Podmodul `sparse.csgraph` nudi niz funkcija za izvršavanje algoritama nad grafovima definiranim matricom incidencije. Matricu incidencije implementiramo funkcijama za rad s poljima `numpy` modula. Rezultate dobivene u vrijednosti `numpy` matrica možemo prikazsti i pretvoriti u polja funkcijom `toarray()`. Primjerice, jednostavan potpuni graf s četiri vrha možemo definirati kao iduće polje:

```
0 1 1 1
1 0 1 1
1 1 0 1
1 1 1 0
```

Nad grafovima definiranim matricama incidencija možemo provesti niz algoritama podmodula `csgraph`:

- `csg.connected_components()` -- izračun broja komponenti povezanosti grafa
- `csg.dijkstra()` -- izračun najkraćih puteva grafa po Dijkstrinom algoritmu
- `csg.floyd_warshall()` -- izračun najkraćih puteva grafa po Floyd-Warshall algoritmu
- `csg.bellman_ford()` -- izračun najkraćih puteva grafa po Bellman-Ford algoritmu
- `csg.johnson()` -- izračun najkraćih puteva grafa po Johnsonovu algoritmu
- `csg.breadth_first_order()` -- izračun poredaka čvorova po širini
- `csg.depth_first_order()` -- izračun poredaka čvorova po širini
- `csg.breadth_first_tree()` -- pretraživanje grafa u širinu
- `csg.depth_first_tree()` -- pretraživanje grafa u dubinu
- `csg.minimum_spanning_tree()` -- izračun minimalnog razapinjućeg stabla grafa

!!! admonition "Zadatak"
    Definirajte `numpy` poljem idući graf:

    ```
         (0)--------(3)
        /   \
       /     \
      /       \
    (1)-------(2)
      \
       \            (6)
        \
        (4)
    ```

    Zatim:

    - saznajte broj komponenti povezanosti grafa,
    - izračunajte minimalno razapinjuće stablo i prikažite ga u obliku matrice incidencije,
    - usporedite vrijeme izvođenja izračuna najkraćih puteva grafa Bellman-Fordovim algoritmom s izračunom Dijkstrinim algoritmom,
    - izvedite pretraživanje grafa u dubinu te dobiveno stablo prikažite u obliku matrice incidencije.

## Python modul scipy: podmodul linalg

Podmodul `linalg` nudi niz funkcija za izvršavanje izračuna iz područja linearne algebre. Funkcije podmodula `linalg` možemo kombinirati s objektima modula `numpy`. Svi objekti za rad s funkcijama podmodula `linalg` moraju biti dvodimenzionalna polja.

Nad matricama (dvodimenzionalnim poljima oblika (n,n)) možemo provesti iduće izračune:

- `linalg.inv()` -- inverziranje matrice
- `linalg.solve()` -- izračun jednadžbe `a x = b` za `x`
- `linalg.det()` -- izračun determinante matrice
- `linalg.lstsq()` -- izračun najmanjih kvadrata jednadžbe `Ax = b`

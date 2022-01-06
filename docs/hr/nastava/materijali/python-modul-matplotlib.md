---
author: Domagoj Margan, Vedran Miletić
---

# Rad s Python modulom matplotlib i sučeljem pyplot

Modul `matplotlib` pruža niz funkcija i metoda za grafičke prikaze podataka u obliku histograma, grafikona, dijagrama, mapa, itd. Podmodulom `pyplot` omogućen je rad s korisničkim sučeljem koje omogućava rad s funkcijama u stilu MATLAB-a. Izrađeni grafikoni mogu se pohraniti u različite formate, poput `png` ili `pdf`.

Uključivanje modula `matplotlib` sa sučeljem `pyplot` najčešće se vrši na način:

``` python
import matplotlib.pyplot as plt
```

Osnovne funkcija za crtanje:

- `plt.figure()` -- inicijalizacija crteža
- `plt.plot()` -- temeljna funkcija za crtanje zadanih podataka
- `plt.title()` -- naslov crteža
- `plt.xlabel()` -- naziv x osi
- `plt.ylabel()` -- naziv y osi
- `plt.axis()` -- određivanje raspona osi
- `plt.show()` -- prikaz nacrtanog
- `plt.savefig()` -- spremanje crteža

Dodatne funkcije za crtanje:

- `plt.grid()` -- prikaz mreže kordinatnog sustva
- `plt.fill()` -- punjenje nacrtanog poligona bojom
- `plt.arrow()` -- dodavanje strelice osima
- `plt.xlim()` -- ograničavanje raspona x osi
- `plt.ylim()` -- ograničavanje raspona y osi
- `plt.legend()` -- crtanje legende

Specifični tipovi grafičkih prikaza podataka sa pripadajućim funkcijama za crtanje:

- histogram -- `plt.hist()`
- stupčasti grafikon -- `plt.bar()`
- horizontalni stupčasti grafikon -- `plt.hbar()`
- grafikon grešaka -- `plt.errorbar()`
- loglog dijagram -- `plt.loglog()`
- pita grafikon -- `plt.pie()`

Argumentima funkcije `plt.plot()` možemo odrediti boje objekata na crtežu:

- `b` -- plava boja
- `g` -- zelena boja
- `r` -- crvena boja
- `k` -- crna boja

Također, moguće je odrediti i oblik linija korištenih na crtežu:

- `-` -- neprekidna linija
- `--` -- isprekidana linija
- `-.` -- linija u obliku crta-točka-crta
- `:` -- točkasta linija

Ostale oblike i boje linija možete pogledati u [službenoj dokumentaciji](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html) funkcije `plt.plot()`.

``` python
import matplotlib.pyplot as plt

labels = 'PPHS', 'DS', 'OS1', 'OS2'
sizes = [22, 11, 75, 63]
colors = ['green', 'yellow', 'blue', 'red']
plt.pie(sizes, labels=labels, colors=colors)
plt.show()
```

!!! todo
    Nedostaje zadatak.

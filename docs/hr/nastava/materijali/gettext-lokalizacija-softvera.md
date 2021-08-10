---
author: Maja Grubišić, Vedran Miletić
---

# Lokalizacija softvera korištenjem GNU gettexta

Uz sve veći broj korisnika javlja se i potreba za lokalizacijom softvera na različite jezike kako bi se poboljšalo korisničko iskustvo.

Proces prevodenja softvera koji koriste [GNU gettext](https://www.gnu.org/software/gettext/) ([Wikipedia](https://en.wikipedia.org/wiki/Gettext)) vrši se pomoću PO datoteka. PO datoteke su tekstualni dokumenti koji sadrže originalni tekst i prijevod a koriste GNU gettext format. Koriste se u tri različita konteksta:

- izvorni dinamički prijevodi -- KDE, Gnome desktop environments, GNU tools i mnogi drugi koriste PO kao izvorni (nativni) format za njihov tekst korisničkog sučelja. Prevedeni PO dokumenti su kompajlirani u binarne MO datoteke i instalirani na prave lokacije. Program dohvaća prijevode pri pokretanju.
- posredni dinamički prijevodi -- neki programi drže tekst svog korisničkog sučelja u svom custom formatu (npr. Mozilla i OpenOffice programi) i takve je potrebno prvo pretvoriti u PO datoteke, prevesti tekst a potom pretvoriti natrag u izvorni format.
- posredni statični prijevodi -- statični tekst, poput programske dokumentacije, je pretvoren iz svog izvornog formata u PO format, preveden i potom pretvoren natrag u izvorni format.

## Struktura PO datoteke

Jedna PO datoteka se odnosi na jedan jezik i struktura joj je sljedeća:

```
prazan prostor
# komentari prevoditelja
#. komentari programera
#: reference na programski kod
#, zastavica...
#| msgid prethodni-neprevedeni-string
msgid neprevedeni-string
msgstr prevedeni-string
```

pri čemu su komentari neobavezni. Enkodiranje je specificirano unutar PO datoteke, i defaultno je UTF-8. Ako ga želite promijeniti morate tako naznačiti u zaglavlju PO datoteke. Postoje i različiti specijalizirani uredivači PO datoteka od kojih su neki Gtranslator, Lokalize, Poedit, Virtaal. Da bi mogli prevoditi Anacondu uz pomoć PO datoteke potrebno je prvo preuzeti kod Anaconde na računalo s repozitorija, što činimo pomoću Git-a naredbom:

```
git clone git://git.fedorahosted.org/git/anaconda.git
```

Preuzeta datoteka će najčešće biti POT formata koji je u principu isti kao PO format jedina razlika je u tome da POT dokument sadrži samo originalni tekst koji je potrebno prevesti, dok PO dokumenti sadrže i prijevode.

Specifikacija PO dokumenta se nalazi na vrhu dokumenta i sadrži podatke o vlasničkim pravima, popis prevodioca s njihovim e-mail adresama, naziv projekta, datumu nastanka projekta i zadnje revizije, podatke o zadnjem prevodiocu, timu koji prevodi te naznačava na koji jezik se prevodi.

Nakon lokalnih izmjena na POT dokumentu dokument vraćamo u repozitorij pomoću git push naredbe. Bilo kakve promjene trebaju proći pregled/recenziju što činimo slanjem izmjenjenog dokumenta na anaconda-devel-list:

```
git send-email --to anaconda-patches@lists.fedorahosted.org --suppress-from --compose <patch files here>
```

## Primjeri prevodenja

``` po
#: pyanaconda/constants.py:73
msgid "Start VNC"
msgstr "Pokreni VNC"
```

Iznad vidimo najjednostavniji primjer prijevoda koji se sastoji od reference na programski kod, neprevedenog i prevedenog stringa. Ispod ćemo pokazati malo složeniji prijevod.

``` po
#: pyanaconda/vnc.py:132
#, python-format
msgid "%(productName)s %(productVersion)s installing on host %(name)s"
msgstr "%(productName)s %(productVersion)s instalacija na domaćinu %(name)s"
```

Drugi primjer sadrži ispise imena i verzije proizvoda koji se instalira na odredenom domaćinu. Ti podaci se dohvaćaju te potom ispisuju na ekran i te stringove nije potrebno prevoditi. Da ih se prevede došlo bi do greške jer ne bi mogli dohvatiti tražene podatke.

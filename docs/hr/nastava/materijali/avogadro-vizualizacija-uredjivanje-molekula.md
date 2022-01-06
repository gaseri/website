---
author: Irena Hartmann, Vedran Miletić
---

# Vizualizacija i uređivanje molekula alatom Avogadro

[Avogadro](https://avogadro.cc/) je je napredni vizualizator i editor molekula, dizajniran za korištenje na više platformi u računalnoj kemiji, modeliranju molekula, bioinformatici, znanosti materijala i srodnim područjima.

Projekt je nastao 2006. kao odgovor na potrebe i nedostatke koji su bili primjećivani u ranije postojećim rješenjima komercijalnog softvera i programa otvorenog koda.

## Osnovno korištenje

Avogadro koristi koordinatni sustav koji vidimo u donjem lijevom kutu.

Alatom `Draw Tool` u traci s alatima možemo crtati molekule. Kako bismo odabrali atome koje želimo i automatsko dodavanje vodika, možemo otvoriti `Postavke alata`.

U `Postavkama prikaza` možemo odabrati različite reprezentacije molekula, od kojih su nam korisne `Ball and Stick`, `Stick`, `Van der Waals Spheres` te `Force` koja prikazuje silnice na pojedinim atomima.

Alatom `Manipulation Tool` možemo pomicati pojedine atome. Postoji i varijanta `Bond Centric Manipulation Tool`.

Uključimo li `Auto Optimization Tool`, sve molekule će prijeći u stanje minimalne energije. Kasnijim potezanjem pojedinih atoma možemo steći uvid u karakteristike molekule.

## Planarne molekule

Baze kemijskih spojeva kao što je ChEMBL uglavnom imaju spojeve u 2D-u s nerealno kratkim kemijskim vezama. Kod otvaranja takvih spojeva Avogadro će ponuditi pretvorbu koordinata u 3D.

## Avogadro 2

Avogadro je danas robusno, fleksibilno rješenje koje povezuje i koristi snagu [Visualization Toolkit (VTK)](https://www.vtk.org/), program otvorenog koda za 3D grafiku, procesiranje slika i vizualizaciju) uz dodatne mogućnosti analize i vizualizacije.

Avogadro projekt je u završim fazama ponovnog pisanja središnjih struktura podataka, algoritama i sposobnosti vizualizacije kojeg su autori nazvali Avogadro 2, iako se na službenim stranicama Avogadra još uvijek preporuča stara verzija.

[Avogadro 2](https://www.openchemistry.org/projects/avogadro2/) nije samo nova verzija Avogadra -- umjesto ažuriranja i nadogradnje ranijih verzija, cijeli kod je nanovo napisan zbog problema u radu Avogadra sa većim skupovima podataka, nesavršenog sučelja i želje da se proširi područje primjene originalno namijenjeno korisnicima Avogadra. Također, jedna od bitnih promjena u odnosu na originalni Avogadro je [primjena modularnosti u dizajnu](https://www.kitware.com/avogadro-2-and-open-chemistry/) koja dopušta veće korištenje komponenti, kao i manji broj međuovisnosti na druge alate. Autori ističu kako Avogadro 2 još uvijek nema sve funkcionalnosti Avogadra, stoga je moguće imati i koristiti oba na istom sistemu.

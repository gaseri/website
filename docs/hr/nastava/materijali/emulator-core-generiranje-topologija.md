---
author: Vedran Miletić
---

# Generiranje topologija

Mrežne topologije su u stvarnim mrežama uglavnom pravilne na neki način. Zbog toga kod izučavanja mreža možemo uočiti [prstenastu topologiju, isprepletenu topologiju, zvijezdastu topologiju, linearnu topologiju i druge](https://en.wikipedia.org/wiki/Computer_network#Network_topology).

Kako bi CORE omogućio da se vjerno reprezentiraju mreže kakve postoje u stvarnosti, podržano je generiranje topologija mreže po različitim pravilima. U CORE-ovom izborniku pod `Tools` moguće je pronaći `Topology generator` koji nam omogućuje brzo stvaranje različitih topologija mreže. Čvorovi mogu biti nasumično postavljeni, poredani u rešetku, zvijezdu ili neki drugi od ponuđenih topoloških obrazaca; odabirom stavke izbornika `Tools/Topology generator` vidljiv je podizbornik s popisom topologija koje je moguće generirati.

Kako bismo generirali topologiju po želji, najprije je potrebno odabrati vrstu čvora od kojeg se treba sastojati topologija (u zadanim postavkama nakon pokretanja CORE-a to su čvorovi tipa `router`). Zatim je potrebno odabrati uzorak topologije koji želimo generirati; svi podržani uzorci opisani su u tablici ispod.

| Uzorak | Opis |
| ------ | ---- |
| Slučajni (`Random`) | Čvorovi su nasumično postavljeni na platno, ali nisu međusobno povezani. Ovaj uzorak se može koristiti u kombinaciji s čvorom tipa `wireless LAN` za brzo stvaranje bežične mreže. |
| Rešetka (`Grid`) | Čvorovi su smješteni u vodoravnim redovima koji počinju u  gornjem lijevom kutu, ravnomjerno razmaknuti kod razmještanja prema desno. Kao i kod slučajnog uzorka, čvorovi nisu međusobno povezani. |
| Povezana rešetka (`Connected Grid`) | Čvorovi su smješteni u pravokutnu mrežu širine N i visine M i svaki je čvor povezan s čvorom iznad, dolje, lijevo i desno od sebe. |
| Lanac (`Chain`) | Čvorovi su povezani jedan za drugim u lancu. |
| Zvijezda (`Star`) | Jedan je čvor postavljen u središte s N čvorova koji ga okružuju i svaki je čvor povezan sa središnjim čvorom. |
| Ciklus (`Cycle`) | Čvorovi su raspoređeni u krug, pri čemu je svaki čvor spojen sa susjedom i zajedno tvore zatvorenu kružnu stazu. |
| Kotač (`Wheel`) | Povezuje čvorove u kombinaciji uzorka zvijezde i ciklusa. |
| Kocka (`Cube`) | Povezuje čvorove kako su povezani vrhovi (hiper)kocke. |
| Klika (`Clique`) | Povezuje čvorove u kliku (potpuni graf) u kojoj je svaki čvor povezan sa svim ostalim čvorovima. |
| Bipartitni (`Bipartite`) | Povezuje čvorove u bipartitni graf koji ima dva odvojena skupa čvorova i svaki je čvor povezan sa svim čvorovima iz drugog skupa. |

U posljednjem koraku biramo veličinu topologije koju želimo koristiti. Tu je jedino ograničenje snaga računala na kojem pokrećemo emulaciju, ali mi ćemo se iz praktičnih razloga ograničiti na topologije veličine do nekoliko desetaka čvorova.

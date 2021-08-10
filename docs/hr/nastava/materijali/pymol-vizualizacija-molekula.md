---
author: Vedran Miletić
---

# Vizualizacija molekula u alatu PyMOL

U PyMOL-u možemo pod `File\Open...` otvarati kemijske spojeve u brojnim formatima uključujući SDF te proteine u brojnim formatima uključujući PDB.

Osim toga možemo pod `File\Get PDB...` dohvatiti strukturu proteina izravno s RCSB PDB-a.

## Značajke

Nakon otvaranja proteina nude nam se akcije (`Action`, slovo `A` pored naziva proteina):

- `preset` nam omogućuje postavljanje atraktivne vizualizacije u jednom koraku, često se koriste `pretty` i `publication`
- `align` pa `to molecule` nam omogućuje poravnavanje dva proteina
- `generate` pa `vacuum electrostatics` stvara prikaz pozitivno i negativno nabijenih dijelova proteina koji nam je koristan kod vizualne analize površine
- `compute` računa svojstva kao što su broj atoma, naboji, površina i težina

Pod prikazom (`Show`, slovo `S` pored naziva proteina):

- `as` nudi prikaz proteina na različite načine, često su korišteni `ribbon`, `cartoon` i `surface`

Pod skrivanjem (`Hide`, slovo `H` pored naziva proteina) imamo suprotne akcije od `Show`.

Pod nazivima (`Label`, slovo `L` pored naziva proteina) možemo uključiti prikaz naziva atoma, reziduma i dr.

Pod bojanjem (`Color`, slovo `C` pored naziva proteina) možemo detaljno konfigurirati način bojanja ako nam trenutno ne odgovara.

Prikaz sekvence proteina uključujemo klikom na slovo `S` u donjem desnom dijelu ekrana. Slovo `F` prebacuje nas u prikaz preko čitavog ekrana.

## Spremanje

PyMOL-ovu sesiju je moguće spremiti korištenjem opcije `File\Save Session`, odnosno `File\Save Session As...`. Otvaranje se zatim izvodi kao otvaranje bilo kojeg drugog podržanog formata.

## Izvoz

Korištenjem `Draw/Ray` moguće je izvesti prikaz iz PyMOL-a u sliku u formatu PNG proizvoljno visoke rezolucije. Raytracing daje bolju kvalitetu prikaza.

## Izgradnja spojeva i proteina

Osim otvaranja možemo kemijske spojeve i proteine samostalno izgraditi alatom Builder. Početak spoja mora biti nekakav fragment u kojem se nalazi ugljik i tada nam se nudi opcija `Create As New Object`. Odabirom odgovarajućih fragmenata i mjesta gdje ih želimo postaviti možemo izgraditi po želji veliku molekulu.

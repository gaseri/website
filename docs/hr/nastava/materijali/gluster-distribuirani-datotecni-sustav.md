---
author: Mario Zadravec, Vedran Miletić
---

# Distribuirani datotečni sustav Gluster

!!! hint
    Za više informacija proučite [službenu dokumentaciju](https://docs.gluster.org/).

[GlusterFS](https://www.gluster.org/) je distribuirani datotečni sustav otvorenog koda koji se može nadograđivati za pohranjivanje nekoliko petabajta podataka. Sastoji se od dvije komponente: serverske i klijentske. Server sadrži datoteke u xfs formatu (ili nekom sličnom formatu) te time pruža velike brzine pristupa datotekama i veliku mogućnost paralelizacije. Datoteke se automatski raspodjeljuju po raznim serverima i time olakšavaju pristup različitim podacima od strane više klijenata. Klijenti samo zatraže datoteke te sam sustav pronalazi datoteku na najbližem i najmanje opterećenom serveru i preusmjerava klijenta na taj server. Time se izbjegavaju redovi čekanja u slučaju da veliki broj klijenata traži različite datoteke sa servera.

GlusterFS radi preko TCT/IP ili InfiniBand mreže. Velika prednost GlusterFS-a prema većini ostalih distribuiranih datotečnih sustava je što ne postoji središnji server koji bi upravljao cijelim sustavom. Možemo započeti rad sa samo jednim serverom te s vremenom nadograđivati sustav sa više servera kako bismo povećali podatkovni prostor ili poboljšali protok podataka. Ukoliko jedan od servera prestane funkcionirati, klijenti i dalje imaju pristup svim podacima jer se često rade kopije kako bi se ovakav problem izbjegao.Kad server ponovno postane funkcionalan, novo traženje podataka može bez problema nastaviti tražiti podatke na tom donedavno nefunkcionalnom serveru.

GlusterFS je moćan distribuirani datotečni sustav koji nije vezan uz određenu vrstu hardvera ili određene količine diskovnog prostora. Može funkcionirati sa jednim serverom ili sa velikim brojem servera. Što se performansi tiče, što veći broj servera nam omogućava brži rad sustava jer se smanjuje vjerojatnost da više korisnika istovremeno traži pristup datotekama koje se nalaze na istom serveru. Također, pomoću redundantnih datoteka se može brzina povećati ukoliko su česti pristupi klijenata na istu datoteku. Kod nadogradnje sustava novim serverima, sustav prepoznaje novonastali diskovni prostor te nove datoteke prosljeđuje na njega kako bi smanjio opterećenost ostalih servera.

Najveća prednost GlusterFS-a je njegova otpornost na greške. Ukoliko jedan od servera prestane funkcionirati, drugi serveri preuzimaju njegovu ulogu ukoliko imaju jednake datoteke kao i server koji više ne funkcionira. Kad se server popravi, javlja sustavu da je opet u funkcionalnom stanju i svi daljnji zahtjevi mogu se prosljeđivati i na njega.

## Temelji GlusterFS sustava

Kako bi server mogao željene datoteke učiniti dostupnima, potrebno je te datoteke organizirati u određenu mapu na server, tzv. ciglu (engl. *brick*). Tih mapa možemo imati više, čime nadograđujemo kapacitet servera bez da mijenjamo mjesto pristupa datotekama koje se već nalaze u drugim mapama. Kada klijent zatraži neku datoteku, nezna na kojem serveru i u kojoj mapi se nalazi željena datoteka, sam sustav ga preusmjerava na pravu lokaciju za pristup toj datoteci.

Ovakvim načinom rada možemo izgraditi veliki datotečni sustav (do nekoliko petabajta) od mnogo manjih sustava. Osim što imamo mogućnost kreiranja velikih datotečnih sustava, ti sustavi su razdjeljeni na razna računala i time na različite djelove mreže što smanjuje opterećenost koji bi jedan veći server imao. Kako različiti korisnici istovremeno uglavnom ne traže jednaku datoteku, cijeli promet se raspodjeljuje na pojedinačne servere i time omogućava brži protok podataka i izbjegava zastoje u radu servera.

## Rad s jednim serverom

Ukoliko radimo samo sa jednim računalom koje preuzima ulogu servera, pristup datotekama je nešto jasniji. Dakako, na jednom serveru možemo imati više djeljenih mapa koje mogu biti na različitim diskovima te time ubrzati vrijeme pristupa različitim podacima jer smanjujemo opterećenost koji se vrši na jednom disku. Na serveru se definiraju datoteke koje se žele podijeliti sa ostalim korisnicima pa tada možemo te datoteke raspodijeliti po našim djeljenim mapama. Nebitno je koja datoteka se tada nalazi u kojoj mapi jer sam sustav rješava pristupanje i usmjeravanje na prave datoteke. Ukoliko radimo sa većim datotekama, te velike datoteke je također moguće podijeliti na manje i raspodjeliti ih po dijeljenim mapama. Sustav sam pronalazi najbolje mjesto na koje postavlja datoteke prema opterećenosti dijeljene mape i raspoloživom diskovnom prostoru.

Slika prikazuje osnovni način rada GlusterFS sustava. Sustav komunicira sa svojim djeljenim mapama (koje su u ovom slučaju zasebni diskovi) te dijeli datoteke među njima. Na korisnikov zahtjev pristupa podacima na jednoj od ove dvije lokacije i vraća korisniku tražene datoteke. Datoteke se također mogu i kopirati među diskovima što nam osigurava protočnost prometa ako jedan od diskova prestane funkcionirati.

``` dot
digraph G {
   node [shape = box];
   GlusterFS -> "Brick #1";
   "Brick #1" -> GlusterFS;
   GlusterFS -> "Brick #2";
   "Brick #2" -> GlusterFS;
   "Brick #1" -> "Brick #2";
   "Brick #2" -> "Brick #1";
}
```

## Rad s više servera

Rad sa više servera ne razlikuje se drastično od rada sa jednim serverom. Razlika kod pristupa podacima je u tome što sada moramo i znati na kojem serveru i u kojoj mapi se nalazi tražena datoteka. Kako jedan server može imati veći broj mapa i isto tako može postojati veći broj servera, način pristupa podacima postaje kompleksniji. Datoteke se raspršuju po serverima ovisno o opterećenosti i dostupnom prostoru. Prema istom algoritmu za raspršivanje datoteka među serverima, klijentska strana izračuna na kojem serveru se tražena datoteka nalazi.

``` dot
digraph G {
   node [shape = box];
   "Storage node 1" -> "Storage node 2";
   "Storage node 2" -> "Storage node 1";
   "Storage node 1" -> "Storage node 3";
   "Storage node 3" -> "Storage node 1";
   "Storage node 2" -> "Storage node 3";
   "Storage node 3" -> "Storage node 2";
   node [shape = circle];
   "Storage node 1" -> {"/export1-1", "/export1-2", "/export1-3"};
   "Storage node 2" -> {"/export2-1", "/export2-2", "/export2-3", "/export2-4", "/export2-5"};
   "Storage node 3" -> {"/export3-1", "/export3-2", "/export3-3", "/export3-4"};
}
```

Kako bismo mogli najbolje iskoristiti mogućnost redundancije datoteka u svrhu neometanog rada, preporuka je da dijeljene mape budu iste veličine. Tako se može cijela mapa bez poteškoća prebaciti na drugi server te tamo čuvati u slučaju pada nekog drugog servera.

## Praktična primjena

GlusterFS ćemo postaviti na tri računala, od čega će 2 računala biti serveri sa repozitorijem podataka dok će treće računalo biti klijentsko računalo. Ako bismo imali tri računala, mogli bismo ih mrežno povezati te vidjeti kako GlusterFS radi u realnom okruženju. Budući da nemamo tri računala, koristiti ćemo tri virtualne mašine sa operativnim sustavom Fedora 20. Nakon što sva tri računala uključimo, svakome dodijelimo IP adresu te im pomoću DNS-a dodijelimo smislena imena. U ovom slučaju su nam IP adrese i imena domaćina koja ćemo unijeti u datoteku `/etc/hosts`:

```
192.168.255.243 HOST1
192.168.255.247 HOST2
192.168.255.242 CLIENT
```

Potrebno je pripaziti na postavke vatrozida kako bi nam računala mogla neometano komunicirati međusobno. Na svakom računalu omogućimo neometani pristup sa ostalih računala pomoću naredbe:

``` shell
# iptables -I INPUT -p all -s 192.168.255.247 -j ACCEPT
```

Naravno, mijenjamo IP adrese ostalih dva računala kako bismo omogućili pristup sa pravih IP adresa. Nakon što smo dozvolili vanjski pristup, potrebno je instalirati potrebne pakete programa. Kao prvo, potrebno je instalirati paket FUSE koji nam omogućava kreiranje sustava za dijeljenje podataka. Nakon toga, na serveru je potrebno instalirati sljedeće pakete: `glusterfs`, `glusterfs-cli`, `glusterfs-fuse`, `glusterfs-libs` i `glusterfs-server`.

Slijedi nam pokretanje glusterd servisa na oba servera kako bismo mogli upravljati kreiranim particijama. Pokrećemo ih naredbom:

``` shell
# /etc/init.d/glusterd start
```

na oba servera. Nakon što su servisi pokrenuti, potrebno je povezati servere kako bi oni međusobno znali komunicirati. Na prvom serveru pokrenemo naredbu

``` shell
# gluster peer probe 192.168.255.247
peer probe: success
# gluster peer status
Number of Peers: 1
```

i time mu pridružujemo drugi server da bi radili zajedno. Ako pokušamo dodati taj server na kojem radimo (IP adresa 192.168.255.243), dobit ćemo poruku da server ne mora dodavati svoju IP adresu u listu.

``` shell
# gluster peer probe 192.168.255.243
peer probe: success: on localhost not needed
```

Nakon što su serveri povezani u bazen pohrane (engl. *storage pool*), možemo upravljati cijelim klasterom servera sa jednog računala. Kako bismo kreirali dvoje particije veličine 100 MB, svaku na jednom serveru, unosimo sljedeću naredbu:

``` shell
# gluster volume create test-volume HOST1:/exp3 HOST2:/exp4 force
volume create: test-volume: success: please start the volume to access data
# gluster volume info

Volume Name: test-volume
Type: Distribute
Volume ID: 6662735e-7463-4513-822b-70eb7988075c
Status: Created
Number of Bricks: 2
Transport-type: tcp
Bricks:
Brick1: HOST1:/exp3
Brick2: HOST2:/exp4
```

Iz slike je vidljivo da smo na prvom serveru kreirali mapu `/exp3` i na drugom serveru mapu `/exp4` te ih spojili i jednu koja će se prikazivati na klijentskom računalu i imati veličinu 200 MB. Prije toga smo pokrenuli i naredbu:

``` shell
# gluster volume create test-volume-replika replica 2 HOST1:/exp1 HOST2:/exp2 force
```

i kreirali nove dvije mape, svaka kapaciteta 100MB koje rade paralelno te se klijentu prikazuje mapa veličine 100MB. Svi podaci koju stavljamo u tu mapu se kopiraju na oba servera te time držimo rezervnu kopiju svih podataka. Ako jedan od servera prestane funkcionirati, klijent će i dalje moći pristupiti svojim podacima. Sljedi nam samo pokretanje tih particija sljedećim naredbama:

``` shell
# gluster volume start test-volume
volume start: test-volume: success
# gluster volume start test-volume-replika
volume start: test-volume: success
```

Postoje još tri načina kreiranja particija: striped volume, distributed striped volume i distributed replicated volume. Striped volume omogućava podacima da se razdjele na manje djelove te tako postave na servere. Distributed striped volume isto tako dijeli podatke na manje dijelove ali ih može postaviti na više brickova na serverima. Jedini uvjet je da broj brickova bude umnožak broja strijepva (manjih dijelova). Distributed replicated volume je sličan repliciranom, ali može imati kombinacije brickova na koje se pohranjuju podaci i onih brickova na kojima se čuvaju kopije. Naredbom

``` shell
# gluster volume create test-volume replica 2 server1:/exp1 server1:/exp3 server2:/exp2 server2:/exp4"
```

kreiramo dvije kombinacije brickova, `server1:/exp1` i `server1:/exp3` te `server2:/exp2` i `server2:/exp4`. Tako svi podaci koji se postavljaju u mapu exp1 budu replicirani u mapu exp3 na istom serveru.

U daljnjem primjeru pokazati ćemo samo prva dva načina, distribuirani i replicirani. Kako bismo kreirane particije mogli vidjeti na klijentovom računalu, moramo na klijentovo računalo instalirati sljedeće pakete: `fuse`, `fuse-libs`, `glusterfs-core`, `glusterfs-rdma` i `glusterfs-fuse`. Nakon instalacije paketa, na klijentovom računalu naredbom mount označavamo naziv particije na serveru te naziv mape koju ćemo i sami vidjeti na klijentovom računalu.

``` shell
# mount -t glusterfs HOST1:/test-volume-replika /replika
# mount -t glusterfs HOST1:/test-volume /bricked
```

Ovime smo na klijentovom računalu kreirali mape `/bricked` i `/replika` koje nam predstavljaju skup svih podataka koji su razdijeljeni po serverima. Ako u mapi `/bricked` kreiramo npr. mapu `testni`, ona će se kreirati na jednom od servera, u ovom slučaju na prvom serveru u mapi `/exp3`.

Ako pak u mapi `/replika` kreiramo datoteku `test.txt`, ona će se kopirati na oba servera te time taj podatak držimo na dva odvojena računala pa u slučaju pada jednog od servera i dalje imamo pristup podacima na drugom serveru.

Ovo je prikaz dva osnovna načina rada GlusterFS-a, no što ako želimo mijenjati broj brickova na nekom serveru kako bismo povećali dodijeljeni diskovni prostor. To možemo uraditi sljedećom naredbom na prvom serveru:

``` shell
# gluster volume add-brick test-volume HOST1:/exp5 force
volume add-brick: success
```

Ovime smo dodali dodatni brick veličine 100 MB na prvom serveru te time klijentov dostupan prostor povećali na 300 MB. Naredba `gluster volume info` nam prikazuje novi dodani brick.

``` shell
# gluster volume info

Volume Name: test-volume
Type: Distribute
Volume ID: 6662735e-7463-4513-822b-70eb7988075c
Status: Started
Number of Bricks: 3
Transport-type: tcp
Bricks:
Brick1: HOST1:/exp3
Brick2: HOST2:/exp4
Brick1: HOST1:/exp5

Volume Name: test-volume-replika
Type: Replicate
Volume ID: f570a81d-d784-4aa7-a6f7-29778164c2fb
Status: Started
Number of Bricks: 1 x 2 = 2
Transport-type: tcp
Bricks:
Brick1: HOST1:/exp1
Brick2: HOST2:/exp2
```

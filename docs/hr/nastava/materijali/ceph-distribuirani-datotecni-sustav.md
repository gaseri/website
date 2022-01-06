---
author: Petar Živković, Vedran Miletić
---

# Distribuirani datotečni sustav Ceph

!!! hint
    Za više informacija proučite [službenu dokumentaciju](https://docs.ceph.com/).

[Ceph](https://ceph.com/) je platforma otvorenog koda za distribuirano spremanje podataka koju odlikuje otpornost na kvarove i koja je izuetno skalabilna ("na nivo eksabajta"). Ceph se sastoji od više razina.

``` dot
digraph G {
   node [shape = box];
   RBD -> RADOS;
   CephFS -> RADOS;
   RADOS -> pool;
   pool -> "CRUSH map";
   "CRUSH map" -> "PG 1";
   "CRUSH map" -> "PG 2";
   "CRUSH map" -> "PG 3";
   "CRUSH map" -> "PG n";
   "PG 1" -> "cluster node 1";
   "PG 1" -> "cluster node 2";
   "PG 2" -> "cluster node 2";
   "PG 2" -> "cluster node 3";
   "PG 3" -> "cluster node 1";
   "PG 3" -> "cluster node 3";
   "cluster node 1" -> "OSDs 1 (proizvoljan broj diskova)";
   "cluster node 2" -> "OSDs 2 (proizvoljan broj diskova)";
   "cluster node 3" -> "OSDs 3 (proizvoljan broj diskova)";
}
```

## Razine Cepha

### Prva razina: OSD

Object store daemon (OSD) obavlja spremanje podataka, replikaciju, oporavljanje podataka te pruža neke podatke Monitorima tako što provjerava druge OSD-eove da li su u funkciji. Možemo reći da oni otprilike predstavljaju konkretni hard-disk na kojemu će se spremati podaci.

### Druga razina: placement groups

Grupe za razmještaj (engl. placement groups) prikupljaju objekte sa sloja iznad i upravljaju njima kao grupom te ih mapiraju u OSD-ove ispod sebe. Na ovom sloju se vrši kopiranje podataka. Broj kopija određuje se u sloju iznad, točnije, Pool razina. PGovi su zaslužni za hash mapiranje objekata u OSD-ove zato da OSD-ovi budu ravnomjerno iskorišteni prilikom dodavanja novih objekata u Pool.

### Treća razina: pool

Na ovoj razini se događa većina korisničke interakcije sa sustavom. Tu se unose naredbe poput get, put i delete za objekte u bazenu. Pool se sastoji od nekoliko PGova i njihov broj se određuje kada se on kreira.

CRUSH mape -- specificiraju se za po jedan Pool i služe tome da prepravljaju distribuciju objekata u OSD-eove prema pravilima koje definira administrator. To se radi iz razloga da dvije kopije ne završe na istom disku. Crush mape se ručno rade te se kompajliraju i proslijede klasteru.

### Četvrta razina: usluge

Reliable Autonomic Distributed Object Store (RADOS) predstavlja dijeljeno spremanje podataka visokih performansi, odnosno prva tri sloja čine RADOS uslugu. RADOS je baza na koju se grade malo složeniji sustavi poput RBD (Rados Block Device) i CephFS.

RADOS block device (RBD) je, kao što samo ime sugerira, blok uređaj spremljen u RADOSU. RBD-i su razgranati preko nekoliko PGova radi boljih performasi te su promjenjivih veličina.

CephFS je POSIX datotečni sustav implementiran nad RADOSom. CephFS je još u izradi dok su RADOS i RBD usluge dostupne.

## Implementacija

Da bi implementirali osnovni Ceph klaster za spremanje podataka moramo imati najmanje 2 OSD-a, Monitor te MDS (Metadata Server).

Monitor -- održava mape stanja klastera što uključuje, mapu monitora, mapu OSD-a, mapu PGova i CRUSH mapu.

MDS -- čuva meta podatke u ime Ceph datotečnog sustava. Oni omogučavaju da korisnici unose naredbe poput ls, find i slično.

Sama implementacija sastoji se od dvije faze, ali ja nažalost nisam uspio do kraja implementirati sustav. Točnije greška se javlja kada bi admin čvor morao putem naredbe `ceph-deploy` instalirat ceph repozitorije na ostale čvorove i pripremiti ih za instaliranje ceph monitora.

Za implementaciju koristio sam Vmware Workstation te četiri virtualne mašine u mreži sa CentOS oeprativnim sustavom.

### Faza preflight

Ova faza uključuje postavljanje admin čvora sa kojega zovemo ceph-deploy i 3 čvora koja će predstavljati naš ceph klaster.

Instaliramo ssh ako je potrebno:

``` shell
$ sudo yum install openssh-server
```

Prvo moramo instalirati ceph repozitorij na admin čvor. To radimo tako da napravimo YUM entry:

``` ini
[ceph-noarch]
name=Ceph noarch packages
baseurl=https://download.ceph.com/rpm-{ceph-release}/{distro}/noarch
enabled=1
gpgcheck=1
type=rpm-md
gpgkey=https://download.ceph.com/keys/release.asc
```

Zatim radimo update repozitorija i instaliramo ceph-deploy:

``` shell
$ sudo yum update && sudo yum install ceph-deploy
```

Sada moram za svaki čvor dodati hostname u `/etc/hosts`.

```
(...)
192.168.72.129  node1
192.168.72.130  node2
192.168.72.131  node3
```

Na sve čvorove dodajem ceph korisnike kojima moram staviti root ovlasti tako da se ceph-deploy može spojiti bez šifre i instalirati ceph:

``` shell
$ sudo useradd -d /home/ceph -m ceph
$ sudo passwd ceph
```

Te im dododajem root ovlasti:

``` shell
$ echo "ceph ALL = (root) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/ceph
$ sudo chmod 0440 /etc/sudoers.d/ceph
```

Idući korak je konfiguriati sve čvorove sa SSH pristupom bez šifre:

``` shell
$ ssh-keygen
Generating public/private key pair.
Enter file in which to save the key (/ceph-client/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /ceph-client/.ssh/id_rsa.
Your public key has been saved in /ceph-client/.ssh/id_rsa.pub.
```

Za svaki čvor kopiram ssh key do svakog drugog čvora:

``` shell
$ ssh-copy-id ceph@node1
$ ssh-copy-id ceph@node2
$ ssh-copy-id ceph@node3
```

Te na kraju modificiram datoteku `~/.ssh/config` admin čvora tako da se logira na korisnike `ceph` ostalih čvorova:

```
Host node1
  User ceph
Host node2
  User ceph
Host node3
  User ceph
```

### Faza storage cluster

U ovoj fazi postavljamo Monitor i dva OSD-a. Za početak možemo napravit direktorij na admin čvoru gdje će nam biti konfiguracijske datoteke koje kreira ceph-deploy prilikom generiranja klastera.

``` shell
$ mkdir my-cluster
$ cd my-cluster
```

Stvaramo klaster:

``` shell
$ ceph-deploy new node1
```

Idući korak bi bio instaliranje Cepha na čvorove putem ceph-deploya:

``` shell
$ ceph-deploy install node1 node2 node3
```

Ovako bi izgledala uspješna instalacija, ali meni prilikom pokretanja naredbe javi grešku da ceph korisnik nema root ovlasti kada ceph-deploy treba pokrenuti naredbu za dohvaćanje cepha. Nije pomoglo ni dodavanje cepha kao roota u datoteci `/etc/sudoers`.

Nakon toga slijedi dodavanje Monitora i dohvaćanje ključeva:

``` shell
$ ceph-deploy mon create-initial
$ ceph-deploy gatherkeys node1
```

Dodajemo 2 OSD-a:

``` shell
$ ssh node2
$ sudo mkdir /var/local/osd0
$ exit
$ ssh node3
$ sudo mkdir /var/local/osd1
$ exit
```

Pomoću ceph-deploya pripremimo OSD-ove:

``` shell
$ ceph-deploy osd prepare node2:/var/local/osd0 node3:/var/local/osd1
```

Te ih aktiviramo:

``` shell
$ ceph-deploy osd activate node2:/var/local/osd0 node3:/var/local/osd1
```

Osiguramo da imamo pristup datoteci `ceph.client.admin.keyring`:

``` shell
$ sudo chmod +r /etc/ceph/ceph.client.admin.keyring
```

Zatim putem ceph-deploya kopiramo konfiguracijsku datoteku i admin ključ admin čvoru i ostalim čvorovima tako da možemo koristiti cephov CLI bez da moramo unositi adresu Monitora i ceph.client.admin.keyring prilikom izvođenja naredbi.

``` shell
$ ceph-deploy admin node1 node2 node3 admin-node
```

Naredbom:

``` shell
$ ceph health
```

provjerimo da li nam je klaster aktivan, a MDS (Metadata server) dodajemo pomoću naredbe:

``` shell
$ ceph-deploy mds create node1
```

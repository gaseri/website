---
author: Ema Matijević, Vedran Miletić
---

# Kriptografija javnog ključa alatom GnuPG

U današnje vrijeme mrežna razmjena dokumenata postala je glavna vrsta razmjene u privatnim i poslovnim sferama. Poslovni, a i privatni dokumenti vrlo su primamljivi pojedincima ili organizacijama koje razmišljaju o njihovoj neovlaštenoj upotrebi i korištenju u neplanirane svrhe. U samim počecima ovakve razmjene, kako bi se riješio problem sigurne mrežne komunikacije, bilo je potrebno stvoriti protokol za enkripciju poruka koji bi bio dovoljno siguran da se počne na veliko koristiti.

Najviše korišten način enkripcije podataka je [standard OpenPGP](https://www.openpgp.org/) i njegove najpoznatije implementacije [Pretty Good Privacy (PGP)](https://www.pgp.com/) i [GNU Privacy Guard (GnuPG)](https://gnupg.org/). U nastavku će biti opisan GnuPG, njegova konfiguracija i povezivanje sa poslužiteljima ključeva.

## Kriptografija javnog ključa

Kriptografija javnog ključa (engl. *Public Key Infrastructure*, kraće PKI) otkrivena je 70-ih godina, temelji se na paru komplementarnih ključeva koji obavljaju operacije enkripcije ili dekripcije, poznata je i kao asimetrična kriptografija. Prvi ključ je javni (engl. *public key*), a drugi tajni (engl. *private key*). Javni ključ slobodno se distribuira.

Enkripcija i dekripcija obavljaju se asimetričnim algoritmima koji su definirani tako da koriste par ključeva od kojih se bilo koji može koristiti za kriptiranje. Ako se koristi jedan ključ iz para za kriptiranje, onda se drugi ključ iz para koristi za dekriptiranje. Obično se kriptiranje obavlja javnim ključem, a dekriptiranje tajnim, tako poruku može dekriptirati samo vlasnik tajnog ključa. Javni ključevi su javno dostupni. Bitno je napomenuti da je jedna od osnovnih sigurnosnih pretpostavki kriptografije javnog ključa ta da je (praktički) nemoguće izvesti tajni privatni ključ iz poznatog javnog ključa.

Postoji mnogo praktičnih primjena kriptografije javnog ključa, među kojima su najbitnije:

- Šifriranje poruka (javnim ključem) -- sadržaj poruke je tajan za svakoga tko ne posjeduje odgovarajući privatni ključ; ostvaruje se povjerljivost poruke.
- Digitalno potpisivanje (privatnim ključem) -- svakome se omogućuje provjera autentičnosti poruke korištenjem pošiljaočevog javnog ključa; ostvaruje se autentičnost poruke.
- Dogovaranje ključa -- omogućuje se siguran dogovor oko tajnog ključa, koji će se kasnije koristiti u nekom simetričnom kriptosustavu.

Digitalno potpisati poruku znači kreirati tekstualnu reprezentaciju poruke (sažetak) i kriptirati je tajnim ključem. Digitalni potpis autentificira pošiljatelja i čuva integritet poruke. Hash algoritam je skup matematičkih operacija kojima se iz dokumenta generira jedinstven sažetak, hash kod ili otisak poruke (engl. message digest) iz kojeg je nemoguće ponovno kreirati izvorni dokument. Postoji više hash funkcija (MD5, SHA-1, SHA-256, ...), one za svaki ulazni niz bilo koje duljine daju izlazni niz iste fiksne duljine.

Kreirani digitalni potpis prilaže se izvornoj poruci. Samo potpisnik, vlasnik tajnog ključa, može kreirati takav digitalni potpis. Također, svatko uz posjedovanje javnog ključa potpisnika može provjeriti valjanost digitalnog potpisa. Dovoljno je usporediti dva sažetka: prvi koji se dobije dekriptiranjem digitalnog potpisa i drugi koji se dobije primjenom hash funkcije na izvornu poruku. Ako se ta dva sažetka poklapaju, digitalni potpis je valjan.

## GnuPG

Sve implementacije OpenPGP-a koriste kombinaciju simetričnog i asimetričnog algoritma za enkripciju te algoritma za stvaranje hash otiska poruke. Princip rada je ostao isti od početka postojanja standarda, mijenjali su se samo korišteni algoritmi. Programi PGP i GnuPG omogućavaju šifriranje i digitalni potpis dokumenata. Princip rada oba programa je sličan i bazira se na kombinaciji raznih enkripcijskih algoritama kako bi se postigla što veća razina sigurnosti.

Prva verzija programa GnuPG je razvijena 1999. godine unutar open source zajednice kao odgovor na brojne patentne i zakonske probleme koji su se javljali kod distribucije PGP programa. GnuPG posjeduje tekstualno korisničko sučelje, ali i brojne dodatke koji omogućuju njegovo korištenje putem grafičkog sučelja.

Datoteke se kriptiraju pomoću asimetričnog para enkripcijskih para ključeva kojega pojedinačno stvara svaki korisnik. Par ključeva sastoji se od privatnog ključa, koji je poznat samo korisniku, i javnog ključa kojega je moguće distribuirati na različite načine (npr. putem Interneta pomoću različitih poslužitelja).

### Lokalni rad s ključevima

Postupak stvaranja para enkripcijskih ključeva započinje naredbom:

``` shell
# gpg --gen-key
```

Ovom naredbom stvara se direktorij `.gnupg` unutar kojega se pohranjuju ključevi, konfiguracijske datoteke i slično. Nakon što odaberemo vrstu ključa, duljinu u bitovima, valjanost ključa, ime, e-mail adresu i lozinku, stvoren je par ključeva. Ključ se generira uz pomoć slučajnih vrijednosti koje program skuplja iz ugrađenog generatora slučajnih brojeva. Sigurnost ključa može se povećati unošenjem slučajnih znakova preko tipkovnice.

```
public and secret key created and signed.
gpg: checking the trustdb
gpg: 3 marginal(s) needed, 1 complete(s) needed, PGP trust model
gpg: depth: 0 valid: 5 signed: 0 trust: 0-, 0q, 0n, 0m, 0f, 5u
pub   2048R/7FABF75D 2013-01-29
      Key fingerprint = 31A4 CA08 000B 0B2E 6C41 A8F3 BED7 1A44 7FAB F75D
uid   Ema Matijevic <ema@example.com>
sub   2048R/A3787563 2013-01-29
```

Vidimo redom: javni ključ, 2048-bitni RSA ključ, ID ključa, otisak ključa koji služi za autentifikaciju, ID korisnika ključa.

Sljedeća naredba omogućava nam ispis svih generiranih ključeva, ovdje provjeravamo je li ključ ispravno unesen u bazu:

```
pub 2048D/E2A80315 2013-01-15 [expired: 2013-01-16]
uid Ema Matijevic (Enki) <astaroth.sf@gmail.com>

pub 2048R/05837DBE 2013-01-24
uid Rijeka (Rijeka Kljuc) <rijeka.foo@gmail.com>
sub 2048R/A617C8F7 2013-01-24

pub 2048R/C80BE6DA 2013-01-24
uid Test Key (Test Proba) <proba@foo.com>
sub 2048R/B1242DF7 2013-01-24

pub 2048R/74C4949A 2013-01-29
uid Ema Matijevic (N/A) <astaroth.sf@gmail.com>
sub 2048R/2A464346 2013-01-29

pub 2048R/05F06AD8 2013-01-29
uid Ivan Ivic (Proba) <ivan@ivan.com>
sub 2048R/52C6FF36 2013-01-29

pub 2048R/72BAFE22 2013-01-29
uid Marina Miler <marina@gmail.com>
sub 2048R/53A7BCF4 2013-01-29

pub 2048R/7FABF75D 2013-01-29
uid Ema Matijevic <ema@gmail.com>
sub 2048R/A3787563 2013-01-29

pub 2048R/2BDF2221 2013-01-29
uid Viktor V <viktor@email.com>
sub 2048R/E02CC32F 2013-01-29
```

Generirani ključ može se ispisati naredbom:

``` shell
# gpg --armor --export user_id
```

što u našem slučaju postaje:

``` shell
# gpg --armor --export 7FABF75D
-----BEGIN PGP PUBLIC KEY BLOCK-----
Version: GnuPG v1.4.12 (GNU/Linux)
mQENBFEINLkBCACmzOJSmN8l5yieUPs5xf6OVS9y692ZaLxPl/rXgZR+BQV5zG1f
FhLxw/KbNl1xh5KJzQiddbKRqdDYtrLiXu5iC+0GLwz3LPluDeiGGOcDoLJn8IOA
YQLYe2lKcv7tk55aEn1nqNZsiM6DJMDAuedVNrB2YeEnj0T7igKbKD2e0DNrtWbQ
BXFlyGzEqh+Bioj7Sj0pK07fHAcq3CQstRf3VbiCkpk/02su+fbtYI2LtjfWsHHw
K+D2bKsVBhJ1UUfzZqYWSItc7Unld+M5h3SZx3Uu4uch5JLljALPJzNoRgQtY0D8
y+BtpgStOG27DgFr0OTwVaSeIb7Y9mj33X7fABEBAAG0HUVtYSBNYXRpamV2aWMg
PGVtYUBnbWFpbC5jb20+iQE4BBMBAgAiBQJRCDS5AhsDBgsJCAcDAgYVCAIJCgsE
FgIDAQIeAQIXgAAKCRC+1xpEf6v3XcVvCACRcvcLzZrexshcHqqbVbZMzaF0CH/K
GkJu2wQhyXy9QWsZB7X2TY6NlHNy0RlTnlWspq0dtIoeJquwLA1pf6w1Dm9PmZAz
Jqfdw8M8GAlVYg5iRvMoO4mTzWEnCCEUKATTS2Ejy41alwwpnOdo/3mN/IILpR1F
LIgcNO9NjgbyTIhnUXjsykQVyDAtWuR+2PfwvzzFHWnNK9FAgh9VWOY10j29zHy6
h1lzVn/zAQe1qIjOhSvKJNwFsxCs82Yyp2sIqOFQAqj6/tbzqbk+6Oh5eXukV+Wg
a7M3nuXm58w1QMGAWfCSTRFqFEaXdck4kbySvRwu5waGjl8JbJZ2vtlWuQENBFEI
NLkBCADsNM4rvJf6xiuYQNZTxviQtxmGLv3p5DtCVeiXZfTQHFphqt1ok90egLeM
36KRyAC8VUuisG6ZK7UmtuShEGFABD/LNW/++0bcsojovGHQeZDbpNk1EQ6T+la5
BFHEmuOqQbVk9NQaI1hS4zEWHA+6HUBu6Ds0mzeHoNn+hBw5vXzPwYbQ8OzJS1/z
jQZ3vfbAVtPFddk7DK4LyKc50mVUUzZkVd1ttoh2jLtPyhUYKhW1L04GRqGJzGXi
E3Wn8+1pnBlhq9fiD3P/SjEGziz4KM8Mx7LSsC5B+OwEfnJR4pNEqdxae7fbC9iD
Jygtw24O2U1TMM5Jl+qcqIohyoWBABEBAAGJAR8EGAECAAkFAlEINLkCGwwACgkQ
vtcaRH+r911Kjgf+ICcj6VC7kYkKKWSyUjBEiJ9Af2+frpw78xdlJyRcAsZeBwd9
viwJt7qsVFus7Iemf2XEBP4ORirGfmS54m7VeW2J7yJMN/P7PANYQSU8LsGnavqy
tWeH5zpSTthgglh1/yjwUc/HjnUiULYSUXZ8N2aohcJ43ZaVuf45p0KfiwSYVlyw
zPgqCcVYhDrFoD+VvlMAP+8DeWZTYmibMJo8s3uiDeiC2ZXSa2bJJ00nyr7UFcqL
P2IWorkcGeAxxynrWkoEq5qcbFe2k1hRrCv7S062cmnApZ1NSiUR5oRxbwgFUGRD
bgaQ9IxvXiUJMiMPAd4p65NdocqVyZ5iE1//7Q==
=Lq6K
-----END PGP PUBLIC KEY BLOCK-----
```

Naredba --armor (ili kraće --a) određuje hoće li ključ biti zapisan u tekstualnom ili binarnom obliku. Još jedan slučaj je eksportiranje u datoteku, npr.:

``` shell
# gpg --export Ema > ema_javni.gpg
```

Za učitavanje ključeva u program (tj. u bazu podataka javnih ključeva) iz datoteke koristimo naredbu:

``` shell
# gpg --import ime_datoteke
```

gdje `ime_datoteke` označava datoteku u kojoj se nalaze željeni ključevi. Ako se izostavi, program će ključ čitati sa standardnog ulaza.

Fingerprint naredba ispisuje otisak pohranjenih ključeva. To je jedan od načina kako se može provjeriti autentičnost primljenog ključa.

``` shell
# gpg --fingerprint
```

Prije korištenja uvezenog javnog ključa, potrebno ga je potpisati:

``` shell
# gpg --sign-key user_id
```

Datoteku je moguće kriptirati naredbom (u ovom slučaju Viktor je primatelj datoteke):

``` shell
# gpg --recipient Viktor --armor --encrypt ema_test.asc
```

Viktor tada može dekriptirati datoteku naredbom:

``` shell
# gpg --decrypt ema_test.asc
You need a passphrase to unlock the secret key for
user: "ema (test) "
2048-bit RSA key, ID 48B4D021, created 2013-01-15 (main key ID 95860D54)
Enter passphrase:
```

Sljedećom naredbom možemo deekriptirati poruku u .txt datoteku.

``` shell
# gpg --decrypt ema_test.asc > ema_file.txt
```

### Rad s poslužiteljem ključeva

Naredbe koje koristimo kako bi komunicirali sa poslužiteljem ključeva su sljedeće; opciju `--keyserver` koristimo ako u datoteci `gpg.conf` imamo navedeno više poslužitelja:

- Slanje ključeva na poslužitelj:

    ``` shell
    # gpg --send-keys key/user_id --keyserver hkp://mykeyserver
    ```

- Primanje ključeva sa poslužitelja:

    ``` shell
    # gpg --recv-keys key/user_id --keyserver hkp://mykeyserver
    ```

- Pretraga ključeva na poslužitelju:

    ``` shell
    # gpg --search-keys key/user_id --keyserver hkp://mykeyserver
    ```

## Poslužitelji ključeva i način konfiguracije

Poslužitelji ključeva (engl. *keyservers*) koriste se za distribuciju javnih ključeva s drugim poslužiteljima ključeva kako bi drugi korisnici mogli po imenu ili e-mail adresi pronaći javni ključ određene osobe kojoj misle poslati kriptiranu poruku. To eliminira proces fizičke ili nesigurne razmjene javnih ključeva i omogućuje drugima da nas mogu pronaći u bazi podataka na Internetu.

### Instalacija poslužitelja SKS

[Synchronising Key Server (SKS)](https://github.com/SKS-Keyserver/sks-keyserver) je OpenPGP poslužitelj čiji je cilj jednostavnost implementacije, decentraliziranost i vrlo pouzdana sinkronizacija. Slijedi nekoliko osnovnih koraka instalacije i konfiguriranja poslužitelja SKS:

``` shell
# yum install sks -y
```

Prva naredba je jasna, radi se o instalaciji poslužitelja, poslije toga potrebno je konfigurirati/stvoriti datoteku `/etc/sks/sksconf` i unijeti odgovarajuće podatke, za potrebe ovog primjera promijenjen je samo `hostname` koji je postavljen na `localhost`. Sadržaj datoteke je:

``` shell
# Debug Level 4 is default (maximum is 10)
debug level: 4
# Set the hostname of this server
#hostname: keyserver.mydomain.net
hostname: hkp://localhost
# Bind addresses
#hkp_address: 1.2.3.4
#recon_address: 1.2.3.4
# Use sendmail
#sendmail_cmd: / usr / sbin / sendmail-t-oi-fpks-admin @ mydomain.net
# From address in sync emails to PKS
#from_addr: pks-admin@mydomain.net
from_addr: keysync@keyserver.nausch.org
# When to calculate statistics
#stat_hour: 2
stat_hour: 0
# Runs database statistics calculation on boot
initial_stat:
```

Sljedeća konfiguracijska datoteka je `/etc/sks/membership`, ona sadrži sve SKS čvorove (hostname i portove) s kojima želimo uspostaviti komunikaciju.

``` shell
# List of other keyservers to peer with
sks.pkqs.net               11370
keys.keysigning.org        11370 # Jonathan Oxer <jon@oxer.com.au> 0x64011A8B
keyserver.gingerbear.net   11370
keyserver.oeg.com.au       11370
zimmermann.mayfirst.org    11370
# EOF
```

S obzirom da ne možemo početi sa praznom bazom ključeva, još jedan važan korak je kreiranje dump direktorija, gdje će biti sadržani svi ključevi.

``` shell
# mkdir /srv/sks/dump
# ls /srv/sks/dump
clean.log ema.pgp fastbuild.log gmon.out KDB rijeka.pgp Rijeka.pgp
```

Potrebno je pokrenuti skriptu sks_build.sh u root direktoriju SKS poslužitelja.

``` shell
# cd /srv/sks
```

Pokrećemo SKS bazu podataka:

``` shell
# service sks-db start
```

Ako dobijemo sljedeću poruku, znamo da smo imali puno sreće pri konfiguraciji i pokretanju baze poslužitelja:

```
Starting the SKS Database Server: [  OK  ]
```

### Lokalna komunikacija

U ovom poglavlju prikazana je komunikacija između običnog korisnika i administratora prijavljenih na istom računalu. Svaki korisnik je generirao svoj par ključeva, te je prikazana jednostavna međusobna razmjena. Kako bi se bolje vidjelo koji korisnik raspolaže s kojim ključevima, izlistan je popis. U komunikaciji korisnik ema poslao je administratoru ključ `Branko Bobic`, a primio je od administratora ključ `Ivan Ivic`.

``` shell
# locate gpg.conf
/home/ema/.gnupg/gpg.conf
/root/.gnupg/gpg.conf
```

U dvjema konfiguracijskim datotekama potrebno je staviti redak:

```
keyserver hkp://localhost
```

Kao korisnik `ema` imamo sljedeće:

``` shell
$ gpg --send-keys 7DF90C27
gpg: sending key 7DF90C27 to hkp server localhost

$ gpg --list-keys
/home/ema/.gnupg/pubring.gpg
----------------------------
pub 2048D/2A10AEC9 2013-01-15
uid Ema Matijevic <astaroth.sf@gmail.com>
sub 2048g/1FE79508 2013-01-15

pub 2048R/05837DBE 2013-01-24
uid Rijeka (Rijeka Kljuc) <rijeka.foo@gmail.com>
sub 2048R/A617C8F7 2013-01-24

pub 2048R/C80BE6DA 2013-01-24
uid Test Key (Test Proba) <proba@foo.com>
sub 2048R/B1242DF7 2013-01-24

pub 2048R/B2D00BF4 2013-01-29
uid Ana Maric <ana.maric@net.hr>
sub 2048R/B3108DA1 2013-01-29

pub 2048R/7DF90C27 2013-01-29
uid Branko Bobic (Branko Kljuc) <bbobic@ema.com>
sub 2048R/464878E5 2013-01-29

$ gpg --recv-keys 05F06AD8
gpg: requesting key 05F06AD8 from hkp server localhost
gpg: key 05F06AD8: public key "Ivan Ivic (Proba) <ivan@ivan.com>" imported
gpg: Total number processed: 1
gpg: imported: 1 (RSA: 1)
```

Kao korisnik `root` imamo sljedeće:

``` shell
# gpg --list-keys
/root/.gnupg/pubring.gpg
------------------------
pub 2048D/E2A80315 2013-01-15 [expired: 2013-01-16]
uid Ema Matijevic (Enki) <astaroth.sf@gmail.com>

pub 2048R/05837DBE 2013-01-24
uid Rijeka (Rijeka Kljuc) <rijeka.foo@gmail.com>
sub 2048R/A617C8F7 2013-01-24

pub 2048R/C80BE6DA 2013-01-24
uid Test Key (Test Proba) <proba@foo.com>
sub 2048R/B1242DF7 2013-01-24

pub 2048R/74C4949A 2013-01-29
uid Ema Matijevic (N/A) <astaroth.sf@gmail.com>
sub 2048R/2A464346 2013-01-29

pub 2048R/05F06AD8 2013-01-29
uid Ivan Ivic (Proba) <ivan@ivan.com>
sub 2048R/52C6FF36 2013-01-29

pub 2048R/72BAFE22 2013-01-29
uid Marina Miler <marina@gmail.com>
sub 2048R/53A7BCF4 2013-01-29

pub 2048R/7FABF75D 2013-01-29
uid Ema Matijevic <ema@gmail.com>
sub 2048R/A3787563 2013-01-29

# gpg --send-keys 05F06AD8
gpg: sending key 05F06AD8 to hkp server localhost

# gpg --recv-keys 7DF90C27
gpg: requesting key 7DF90C27 from hkp server localhost
gpg: key 7DF90C27: public key "Branko Bobic (Branko Kljuc) <bbobic@ema.com>" imported
gpg: Total number processed: 1
gpg: imported: 1 (RSA: 1)
```

### Komunikacija preko vanjskog poslužitelja ključeva

S obzirom da je komunikacija preko provjerenih, stabilnih i ažurnih poslužitelja najčešća, odabrala sam jedan takav poslužitelj: [MIT PGP Public Key Server](https://pgp.mit.edu/). U konfiguracijskim datotekama `gpg.conf` potrebno je uključiti redak:

```
keyserver hkp://pgp.mit.edu
```

Kao korisnik `ema` sad imamo:

``` shell
$ gpg --list-keys
/home/ema/.gnupg/pubring.gpg
----------------------------
pub 2048D/2A10AEC9 2013-01-15
uid Ema Matijevic <astaroth.sf@gmail.com>
sub 2048g/1FE79508 2013-01-15

pub 2048R/05837DBE 2013-01-24
uid Rijeka (Rijeka Kljuc) <rijeka.foo@gmail.com>
sub 2048R/A617C8F7 2013-01-24

pub 2048R/C80BE6DA 2013-01-24
uid Test Key (Test Proba) <proba@foo.com>
sub 2048R/B1242DF7 2013-01-24

pub 2048R/B2D00BF4 2013-01-29
uid Ana Maric <ana.maric@net.hr>
sub 2048R/B3108DA1 2013-01-29

pub 2048R/7DF90C27 2013-01-29
uid Branko Bobic (Branko Kljuc) <bbobic@ema.com>
sub 2048R/464878E5 2013-01-29

pub 2048R/05F06AD8 2013-01-29
uid Ivan Ivic (Proba) <ivan@ivan.com>
sub 2048R/52C6FF36 2013-01-29

$ gpg --recv-key 2BDF2221
gpg: requesting key 2BDF2221 from hkp server pgp.mit.edu
gpg: key 2BDF2221: public key "Viktor V <viktor@email.com>" imported
gpg: Total number processed: 1
gpg: imported: 1 (RSA: 1)
```

Kao korisnik `root` imamo:

``` shell
# gpg --list-keys
/root/.gnupg/pubring.gpg
------------------------
pub 2048D/E2A80315 2013-01-15 [expired: 2013-01-16]
uid Ema Matijevic (Enki) <astaroth.sf@gmail.com>

pub 2048R/05837DBE 2013-01-24
uid Rijeka (Rijeka Kljuc) <rijeka.foo@gmail.com>
sub 2048R/A617C8F7 2013-01-24

pub 2048R/C80BE6DA 2013-01-24
uid Test Key (Test Proba) <proba@foo.com>
sub 2048R/B1242DF7 2013-01-24

pub 2048R/74C4949A 2013-01-29
uid Ema Matijevic (N/A) <astaroth.sf@gmail.com>
sub 2048R/2A464346 2013-01-29

pub 2048R/05F06AD8 2013-01-29
uid Ivan Ivic (Proba) <ivan@ivan.com>
sub 2048R/52C6FF36 2013-01-29

pub 2048R/72BAFE22 2013-01-29
uid Marina Miler <marina@gmail.com>
sub 2048R/53A7BCF4 2013-01-29

pub 2048R/7FABF75D 2013-01-29
uid Ema Matijevic <ema@gmail.com>
sub 2048R/A3787563 2013-01-29

pub 2048R/7DF90C27 2013-01-29
uid Branko Bobic (Branko Kljuc) <bbobic@ema.com>
sub 2048R/464878E5 2013-01-29

pub 2048R/2BDF2221 2013-01-29
uid Viktor V <viktor@email.com>
sub 2048R/E02CC32F 2013-01-29

# gpg --send-key 2BDF2221
gpg: sending key 2BDF2221 to hkp server pgp.mit.edu
```

Iako je klijent javio da je ključ uspješno poslan i primljen, putem [web sučelja MIT-ovog poslužitelja](https://pgp.mit.edu/) može se provjeriti je li ključ zaista poslan na poslužitelj. Rezultat pretrage koji poslužitelj prikazuje strukturom je vrlo sličan onome koji daje GnuPG pa ne treba detaljnije objašnjavati.

## Zaključak

Kao što je već spomenuto kroz ovaj rad, enkripcija besplatnim programom GnuPG najpopularnija je i vjerovatno najjednostavnija enkripcija poruka ovog tipa. Bez obzira na jednostavnost korištenja, sigurnost i sve druge pozitivne stvari, ovaj postupak enkripcije danas se koristi pretežno u zatvorenim krugovima entuzijastičnih korisnika, no pitanje je da li je i u njihovim krugovima to još uvijek popularno i korisno.

Instalacija SKS Keyservera i njegova konfiguracija opisana je na više izvora jednostavnim koracima, no prosječni korisnik se može lako izgubiti u svim tim opcijama, s druge strane upitno je koliko prosječnih korisnika stvarno koristi kombinaciju kriptiranja poruka alatom GnuPG korištenjem poslužitelja ključeva.

## Literatura

!!! todo
    Reference treba pročistiti i povezati u tekstu.

1. [Key server (cryptographic)](https://en.wikipedia.org/wiki/Key_server_(cryptographic)), siječanj 2013.
1. [Ubuntu Forums](https://ubuntuforums.org/showthread.php?t=680292), siječanj 2013.
1. [Opis PGP i GnuPG alata](https://www.cis.hr/www.edicija/LinkedDocuments/CCERT-PUBDOC-2003-12-51.pdf), siječanj 2013.
1. [Zaštita datoteka na Linux operacijskim sustavima](https://www.cis.hr/www.edicija/LinkedDocuments/CCERT-PUBDOC-2007-05-192.pdf), siječanj 2013.
1. [GPG Quick Start](https://www.madboa.com/geek/gpg-quickstart/), siječanj 2013.
1. [Osobna kriptografija - GPG, TrueCrypt, SSL](https://security.foi.hr/wiki/index.php/Osobna_kriptografija_-_GPG,_TrueCrypt,_SSL.html), siječanj 2013.
1. [GnuPG](https://www.gnupg.org/), siječanj 2013.
1. [Kriptiranje - zaštita poruka u komunikaciji](https://informatika.buzdo.com/pojmovi/gpg-1.htm), siječanj 2013.

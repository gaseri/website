---
author: Vedran Miletić
---

# Virtualizacija korištenjem Virtual Machine Managera

Za jednostavno upravljanje virtualnim strojevima iz grafičkog sučelja koristit ćemo [Virtual Machine Manager](https://virt-manager.org/) (kraće virt-manager ili VMM). Virt-manager podržava brojne platforme uključujući [Linux Containers (LXC)](https://linuxcontainers.org/) i [Xen](https://xenproject.org/), a mi ćemo ga koristiti za [QEMU](https://www.qemu.org/) i [KVM](https://www.linux-kvm.org/).

## Stvaranje virtualnog stroja

Virtual Machine Manager pokrenut ćemo putem ikone u izborniku ili naredbom `virt-manager`. Nakon pokretanja VMM-a, pritiskom na gumb `Create a new virtual machine` pokrećemo proces stvaranja novog virtualnog stroja. U prvom koraku u dijelu `Choose how you would like to install the operating system` biramo:

* `Local install media (ISO image or CDROM)` ako želimo instalirati OS na klasičan način kao na fizičkom računalu
* `Import existing disk image` ako želimo koristiti unaprijed pripremljenu sliku za korištenje na virtualizacijskim sustavima i u oblaku.

### Preuzimanje, priprema i odabir slike

Primjerice, za [Arch Linux](https://archlinux.org/) možemo takve slike preuzeti u odjeljku VM images na [stranici Arch Linux Downloads](https://archlinux.org/download/). Preuzet ćemo sliku s nazivom `basic` u formatu QCOW2; ime datoteke je oblika `Arch-Linux-x86_64-basic-20230415.143140.qcow2`.

Napravimo kopiju preuzete slike koju ćemo koristiti kod stvaranja virtualnog stroja kako je ne bismo morali ponovno preuzimati za svako stvaranje novog virtualnog stroja.

Slika ima zadanu veličinu diska od 2 GiB, pri čemu se prazan prostor samo navodi, a zapravo ne zauzima gotovo ništa. Ta će nam veličina za praktične potrebe biti premalena pa je možemo povećati korištenjem [QEMU-ovih pomoćnih alata](https://www.qemu.org/docs/master/tools/index.html), specifično `qemu-img` ([dokumentacija](https://www.qemu.org/docs/master/tools/qemu-img.html)). Primjerice, za povećanje *na* veličinu od 25 GiB naredba je oblika:

``` shell
$ qemu-img resize moja-arch-linux-slika.img 25G
image resized
```

Alternativno, sliku možemo povećati *za* danu veličinu, npr. 30 GiB na način:

``` shell
$ qemu-img resize moja-arch-linux-slika.img +30G
image resized
```

Ako provedemo oba koraka, slika će imati 55 GiB.

Više informacija o korištenju alata `qemu-img` moguće je pronaći u [odjeljku Creating a hard disk image stranice QEMU na ArchWikiju](https://wiki.archlinux.org/title/QEMU#Creating_a_hard_disk_image).

U drugom koraku stvaranja virtualnog stroja pod `Provide the existing storage path:` kliknimo na `Browse...`. Zatim u dijalogu koji se otvori pod nazivom `Locale or create storage volume` kliknimo na gumb `Browse Local` pa pronađimo stvorenu kopiju.

U dijelu `Choose the operating system you are installing:` pronađimo `Arch Linux (archlinux)`.

### Postavke virtualnog stroja

U trećem koraku biramo postavke memorije i procesora za novostvoreni virtualni stroj. Varijanta Arch Linuxa koju koristimo nema grafičko sučelje i uredno će raditi s manje od predloženih 4096 MiB RAM-a i 2 CPU-a, odnosno 2048 MiB i 1 CPU će biti dosta. Ta štedljivost će nam biti korisna obzirom da ćemo vremenom stvarati veći broj virtualnih strojeva.

U posljednjem koraku imamo mogućnost prilagodbe konfiguracije uključivanjem kvačice pored `Customize configuration before install` i odabira mreže koja će se koristiti. Zasad ostavimo te postavke kakve jesu.

Nakon pokretanja virtualnog stroja uočit ćemo kako se možemo prijaviti korištenjem korisničkog imena `arch` i zaporke `arch`. Dostupan nam je i administratorski pristup korištenjem naredbe `sudo`.

Zasad nećemo koristiti drugu sliku koja sadrži `cloudimg` (kratica od **cloud** **im**a**g**e) u nazivu, npr. `Arch-Linux-x86_64-cloudimg-20230415.143140.qcow2`. Ako greškom preuzmemo tu sliku, lako se možemo uvjeriti da se na pokrenuti operacijski sustav nećemo moći prijaviti ma koje korisničko ime i zaporku isprobali. Tu sliku je potrebno konfigurirati korištenjem dodatnog izvora podataka i alata [cloud-init](https://cloud-init.io/).

## Povezivanje na virtualni stroj korištenjem OpenSSH-a

QEMU/KVM u zadanim postavkama mreže (prevođenje mrežnih adresa, engl. *network address translation*, kraće NAT) dodjeljuje svim virtualnim strojevima adrese u rasponu 192.168.122.0/24. Naredbom `ip addr` unutar virtualnog stroja možemo provjeriti koja je adresa dodijeljena tom stroju:

``` shell
$ ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
        valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host
        valid_lft forever preferred_lft forever
    2: enp1s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether e0:d5:5e:25:a1:29 brd ff:ff:ff:ff:ff:ff
    inet 192.168.122.147/24 brd 192.168.11.255 scope global dynamic noprefixroute enp5s0
        valid_lft 42841sec preferred_lft 42841sec
```

Na virtualni stroj možemo se povezati OpenSSH-om korištenjem korisničkog imena `arch` i zaporke koju smo postavili na način:

``` shell
$ ssh arch@192.168.122.147
arch@192.168.122.147's password:
```

Nakon toga možemo normalno pokretati naredbe, npr. možemo provjeriti koliko slobodnog prostora na disku u virtualnom stroju imamo naredbom `df -h`.

## Instalacija softvera u virtualnom stroju

Za instalaciju softvera u virtualnom stroju koristimo upravitelj paketa [pacman](https://archlinux.org/pacman/) (skraćeno od **pac**kage **man**ager).

Pokretanjem naredbe `pacman` s parametrom `--version` možemo saznati informacije o dostupnoj verziji pacmana:

``` shell
$ pacman --version 

 .--.                  Pacman v6.0.2 - libalpm v13.0.2
/ _.-' .-.  .-.  .-.   Copyright (C) 2006-2021 Pacman Development Team
\  '-. '-'  '-'  '-'   Copyright (C) 2002-2006 Judd Vinet
 '--'
Ovaj program može biti slobodno distribuiran 
sukladno pravilima GNU General Public License.
```

Možemo prepoznati [Pac-Man](https://en.wikipedia.org/wiki/Pac-Man)-a u obliku [ASCII arta](https://en.wikipedia.org/wiki/ASCII_art). Korištenjem parametra `--help`, odnosno `-h` možemo dobiti informacije o načinu korištenja:

``` shell
$ pacman --help
upotreba:  pacman <operacija> [...]
operacije:
    pacman {-h --help}
    pacman {-V --version}
    pacman {-D --database} <opcije> <paket(i)>
    pacman {-F --files}    [opcije] [datoteka(e)]
    pacman {-Q --query}    [opcije] [paket(i)]
    pacman {-R --remove}   [opcije] <paket(i)>
    pacman {-S --sync}     [opcije] [paket(i)]
    pacman {-T --deptest}  [opcije] [paket(i)]
    pacman {-U --upgrade}  [opcije] <datoteka(e)>

koristi 'pacman {-h --help}' sa operacijom za dostupne opcije
```

Detaljne informacije o načinu korištenja moguće je pronaći na [stranici pacman na ArchWikiju](https://wiki.archlinux.org/title/Pacman). [Službeni repozitoriji](https://wiki.archlinux.org/title/Official_repositories) Arch Linuxa sadrže [više od 14 000 paketa](https://archlinux.org/packages/).

!!! admonition "Zadatak"
    U virtualnom stroju instalirajte HTTP poslužitelj Nginx. Pokrenite pripadnu uslugu, a zatim iskoristite Firefox ili cURL kako biste se uvjerili da poslužitelj ispravno radi. Naposlijetku, (po)maknite zadanu datoteku `index.html` i na njenom mjestu stvorite novu datoteku sa sadržajem po vašoj želji.

### Instalacija paketa na temelju opisa u Arch User Repositoryju (AUR-u)

Osim službenih repozitorija, Arch Linux sadrži [gotovo 90 000 opisa paketa](https://aur.archlinux.org/packages) u [Arch User Repositoryju (AUR-u)](https://aur.archlinux.org/). Detalji oko oblika opisa paketa mogu se pronaći na [stranici PKGBUILD na ArchWikiju](https://wiki.archlinux.org/title/PKGBUILD), a detalji oko instalacije paketa na [stranici Arch User Repository](https://wiki.archlinux.org/title/Arch_User_Repository).

Za primjer, izgradit ćemo i instalirati paket [mkdocs](https://aur.archlinux.org/packages/mkdocs). Klikom na poveznicu `View PKGBUILD` na toj stranici i zatim na poveznicu `plain` otkrivamo URL `https://aur.archlinux.org/cgit/aur.git/plain/PKGBUILD?h=mkdocs` s kojeg možemo preuzeti PKGBUILD.

``` shell
$ mkdir mkdocs
$ cd mkdocs
$ curl -o PKGBUILD "https://aur.archlinux.org/cgit/aur.git/plain/PKGBUILD?h=mkdocs"
(...)
```

Pakete izgrađujemo korištenjem naredbe `makepkg` na način:

``` shell
$ makepkg
==> Making package: mkdocs 1.4.2-1 (nedjelja, 16. travnja 2023. 22:06:06 CEST)
==> Checking runtime dependencies...
==> Missing dependencies:
  -> python-babel
  -> python-ghp-import
  -> python-importlib-metadata
  -> python-jinja
  -> python-livereload
  -> python-markupsafe
  -> python-mergedeep
  -> python-mdx-gh-links
  -> python-pyyaml-env-tag
  -> python-watchdog
==> Checking buildtime dependencies...
==> Missing dependencies:
  -> python-hatchling
  -> python-pathspec
  -> python-build
  -> python-installer
  -> python-wheel
==> ERROR: Could not resolve all dependencies.
```

Više detalja o naredbi `makepkg` moguće je pronaći na [stranici makepkg na ArchWikiju](https://wiki.archlinux.org/title/Makepkg).

Potrebne pakete koji postoje u službenom repozitoriju ćemo od tamo i instalirati:

``` shell
$ sudo pacman -S python-babel python-ghp-import python-importlib-metadata python-jinja python-livereload python-markupsafe python-watchdog
$ sudo pacman -S python-hatchling python-pathspec python-build python-installer python-wheel
```

Za preostala tri paketa [python-mergedeep](https://aur.archlinux.org/packages/python-mergedeep), [python-mdx-gh-links](https://aur.archlinux.org/packages/python-mdx-gh-links) i [python-pyyaml-env-tag](https://aur.archlinux.org/packages/python-pyyaml-env-tag), preuzet ćemo PKGBUILD-ove iz AUR-a i na temelju njih:

``` shell
$ mkdir python-mergedeep
$ cd python-mergedeep
$ curl -o PKGBUILD "https://aur.archlinux.org/cgit/aur.git/plain/PKGBUILD?h=python-mergedeep"
(...)
$ makepkg
(...)
==> Finished making: python-mergedeep 1.3.4-2
$ cd ..
```

``` shell
$ mkdir python-mdx-gh-links
$ cd python-mdx-gh-links
$ curl -o PKGBUILD "https://aur.archlinux.org/cgit/aur.git/plain/PKGBUILD?h=python-mdx-gh-links"
(...)
$ makepkg
(...)
==> Finished making: python-mdx-gh-links 0.2-1
$ cd ..
```

``` shell
$ mkdir python-pyyaml-env-tag
$ cd python-pyyaml-env-tag
$ curl -o PKGBUILD "https://aur.archlinux.org/cgit/aur.git/plain/PKGBUILD?h=python-pyyaml-env-tag"
(...)
$ makepkg
(...)
==> Finished making: python-pyyaml-env-tag 0.1-1
$ cd ..
```

Instalaciju lokalno izgrađenih paketa vršimo narebom `pacman` s parametrom `--upgrade`, odnosno `-U` na način

``` shell
$ sudo pacman -U python-mergedeep/python-mergedeep-1.3.4-2-any.pkg.tar.zst python-mdx-gh-links/python-mdx-gh-links-0.2-1-any.pkg.tar.zst python-pyyaml-env-tag/python-pyyaml-env-tag-0.1-1-any.pkg.tar.zst
(...)
```

Nakon instalacije ta tri paketa moguće je uspješno izgraditi paket mkdocs korištenjem naredbe `makepkg` pa ga instalirati na sličan način naredbom `pacman` s parametrom `-U`. Uvjerimo se da je instalacija bila uspješna pokretanjem naredbe `mkdocs` s parametrom `--version`:

``` shell
$ mkdocs --version
mkdocs, version 1.4.2 from /usr/lib/python3.10/site-packages/mkdocs (Python 3.10)
```

## Stvaranje više virtualnih strojeva

Stvaranjem više kopija slike Arch Linuxa i ponavljanjem postupka možemo stvoriti više virtualnih strojeva. Svaka slika se može koristiti samo za jedan virtualni stroj.

!!! admonition "Zadatak"
    Stvorite još jedan virtualni stroj pa u njemu instalirajte HTTP poslužitelj Apache.

## Složenije topologije u oblaku

### Web poslužitelji i baza podataka

!!! admonition "Zadatak"
    Stvorite tri virtualna stroja. Na prvom virtualnom stroju instalirajte MariaDB poslužitelj (paket `mariadb-server`). Stvorite bazu i u njoj barem jednu tablicu sa stupcima po želji te je popunite podacima po želji. Omogućite pristup sa zaporkom *udaljenom* korisniku imena po želji (ime domaćina `'%'` omogućit će pristup s bilo koje adrese, razmislite je li vam to u interesu).

    Naposlijetku, kako biste učinili da se MariaDB veže i na adrese do kojih je moguće doći izvana, promijenite postavku `bind-address` u datoteci `/etc/mysql/mariadb.conf.d/50-server.cnf` iz vezivanja na lokalnu adresu:

    ``` ini
    bind-address            = 127.0.0.1
    ```

    u vezivanje na sve adrese:

    ``` ini
    bind-address            = 0.0.0.0
    ```

    Preostala dva virtualna stroja neka imaju Apache i modul za PHP te PHP-ov modul za MySQL (paket `php-mysql`). Na ta dva virtualna stroja stvorite `index.php` sadržaja ([izvor koda](https://www.php.net/manual/en/mysqli.quickstart.dual-interface.php)):

    ``` php
    <?php

    $mysqli = mysqli_connect("database.example.com", "user", "password", "database");
    $result = mysqli_query($mysqli, "SELECT mycolumn AS _column FROM mytable");
    $row = mysqli_fetch_assoc($result);
    echo $row["_column"];
    ```

    Pritom ćete postavke povezivanja u `mysqli_connect()` i upit u `mysqli_query()` prilagoditi svojim potrebama.

    Uvjerite se pristupanjem IP adresama tih dvaju virtualnih strojeva da sve uspješno radi. (**Uputa:** Pratite poruke o pogreškama u datoteci `/var/log/apache2/error.log`.)

### Balansiranje opterećenja web poslužitelja

!!! admonition "Zadatak"
    Dodajte još jedan virtualni stroj i na njemu instalirajte HAProxy (paket `haproxy`). Dopunite datoteku `/etc/haproxy/haproxy.cfg` sadržajem:

    ``` haproxy
    frontend myfrontend
        bind :80
        default_backend myservers

    backend myservers
        server apache1 srv1.example.com:80
        server apache2 srv2.example.com:80
    ```

    Prilagodite konfiguraciju svojim potrebama, a zatim se uvjerite da radi balansiranje opterećenja između dva poslužitelja.

### Odvojena distribucija statičkih resursa

!!! adomonition "Zadatak"
    Na dva virtualna stroja koji imaju HTTP poslužitelj Apache i modul za PHP promijenite format poslanih podataka da bude JSON.

    Dodajte još jedan virtualni stroj koji ćete koristiti za distribuciju statičkih sadržaja (HTML, CSS, fontovi, JavaScript, slike, audiovizualne datoteke itd.). Na njemu postavite HTTP poslužitelj Apache (ovaj put bez dodavanja interpretera skriptnih jezika) tako da poslužuje datoteku `index.html` oblika:

    ``` html
    <!DOCTYPE html>
    <html>
    <body>
        <p>Address received is: <span id="myIP"></span></p>

        <script>
        var myHeaders = new Headers({ Accept: "application/json" });
        const myInit = { headers: myHeaders };
        fetch("http://apps.group.miletic.net/ip/", myInit)
            .then(function (response) {
                return response.json().then(function (data) {
                    document.getElementById("myIP").innerHTML = data["address"];
                });
            })
            .catch(function (err) {
                console.log("Failed to fetch page: ", err);
            });
        </script>
    </body>
    </html>
    ```

    Prilagodite adresu u pozivu funkcije `fetch()` tako da hvata podatke s vašeg balansera opterećenja te ostatak JavaScripta i HTML-a po potrebi, ovisno o strukturi JSON-a koju vaš poslužitelj bude vraćao.

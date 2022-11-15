---
author: Vedran Miletić
---

# Virtualizacija korištenjem Virtual Machine Managera

Za jednostavno upravljanje virtualnim strojevima iz grafičkog sučelja koristit ćemo [Virtual Machine Manager](https://virt-manager.org/) (kraće virt-manager ili VMM). Virt-manager podržava brojne platforme uključujući [Linux Cointainers (LXC)](https://linuxcontainers.org/) i [Xen](https://xenproject.org/), a mi ćemo ga koristiti za [QEMU](https://www.qemu.org/) i [KVM](https://www.linux-kvm.org/).

## Stvaranje virtualnog stroja

Nakon pokretanja virt-managera pritiskom na gumb `Create a new virtual machine` pokrećemo proces stvaranja novog virtualnog stroja. U prvom koraku u dijelu `Choose how you would like to install the operating system` biramo:

* `Local install media (ISO image or CDROM)` ako želimo instalirati OS na klasičan način kao na fizičkom računalu
* `Import existing disk image` ako želimo koristiti unaprijed pripremljenu sliku za korištenje na virtualizacijskim sustavima i u oblaku.

### Preuzimanje, priprema i odabir slike

Primjerice, za [Ubuntu](https://ubuntu.com/) možemo takve slike preuzeti s [cloud-images.ubuntu.com](https://cloud-images.ubuntu.com/). Preuzet ćemo sliku iz direktorija `focal` (kodno ime za Ubuntu 20.04 LTS) pa `current`; kako radimo na računalu čiji se procesor temelji na arhitekturi x86-64, odnosno AMD64, datoteka koja nam odgovara je `focal-server-cloudimg-amd64.img`.

Napravimo kopiju preuzete slike koju ćemo koristiti kod stvaranja virtualnog stroja. Slika ima zadanu veličinu diska od 2 GiB, pri čemu se prazan prostor samo navodi, a zapravo ne zauzima gotovo ništa. Ta će nam veličina za praktične potrebe biti premalena pa je možemo povećati korištenjem [QEMU-ovih pomoćnih alata](https://www.qemu.org/docs/master/tools/index.html), specifično `qemu-img` ([dokumentacija](https://www.qemu.org/docs/master/tools/qemu-img.html)). Primjerice, za povećanje *na* veličinu od 25 GiB naredba je oblika:

``` shell
$ qemu-img resize moja-ubuntu-slika.img 25G
image resized
```

Alternativno, sliku možemo povećati *za* danu veličinu, npr. 30 GiB na način:

``` shell
$ qemu-img resize moja-ubuntu-slika.img +30G
image resized
```

Ako provedemo oba koraka, slika će imati 55 GiB.

U drugom koraku stvaranja virtualnog stroja pod `Provide the existing storage path:` kliknimo na `Browse...`. Zatim u dijalogu koji se otvori pod nazivom `Locale or create storage volume` kliknimo na gumb `Browse Local` pa pronađimo stvorenu kopiju.

U dijelu `Choose the operating system you are installing:` pronađimo `Ubuntu 20.04`.

### Postavke virtualnog stroja

U trećem koraku biramo postavke memorije i procesora za novostvoreni virtualni stroj. Varijanta Ubuntua koju koristimo nema grafičko sučelje i uredno će raditi s manje od predloženih 4096 MiB RAM-a i 2 CPU-a, odnosno 2048 MiB i 1 CPU će biti dosta. Ta štedljivost će nam biti korisna obzirom da ćemo vremenom stvarati veći broj virtualnih strojeva.

U posljednjem koraku imamo mogućnost prilagodbe konfiguracije uključivanjem kvačice pored `Customize configuration before install` i odabira mreže koja će se koristiti. Zasad ostavimo te postavke kakve jesu.

Nakon pokretanja virtualnog stroja uočit ćemo kako se ne možemo prijaviti ma koje korisničko ime i zaporku isprobali jer je potrebno iskorisiti [cloud-init](https://cloud-init.io/) za konfiguraciju cloud slike koju smo preuzeli.

!!! admonition "Zadatak"
    U virtualnom stroju instalirajte HTTP poslužitelj Apache (paket `apache2`). Iskoristite Firefox ili cURL kako biste se uvjerili da poslužitelj ispravno radi, a zatim (po)maknite zadanu datoteku `index.html` i na njenom mjestu stvorite novu datoteku sa sadržajem po vašoj želji.

!!! tip
    Ako ste zaboravili povećati veličinu preuzete slike prije pokretanja virtualnog stroja, moguće je da ćete instalacijom dodatnih paketa popuniti disk. U tom slučaju prvo zaustavite virtualni stroj, povećajte disk, a zatim nakon pokretanja iskoristite naredbu `apt` s parametrom `--fix-broken` i argumentom `install` kako biste instalacijom potrebnih paketa doveli stanje u red:

    ``` shell
    sudo apt --fix-broken install
    ```

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

Na virtualni stroj možemo se povezati OpenSSH-om korištenjem korisničkog imena `ubuntu` i zaporke koju smo postavili na način:

``` shell
$ ssh ubuntu@192.168.122.147
ubuntu@192.168.122.147's password:
```

Nakon toga možemo normalno pokretati naredbe, npr. možemo provjeriti koliko slobodnog prostora na disku u virtualnom stroju imamo naredbom `df -h`.

## Stvaranje više virtualnih strojeva

Stvaranjem više kopija Ubuntuove slike i više kopija ranije stvorene slike `user-data.img` možemo stvoriti više virtualnih strojeva. Svaki par slika se može koristiti samo za jedan virtualni stroj.

!!! admonition "Zadatak"
    Stvorite još jedan virtualni stroj pa u njemu instalirajte HTTP poslužitelj Apache i modul za PHP (paket `libapache2-mod-php`). Prilagodite PHP skriptu:

    ``` php
    <?php

    $stream = fopen('http://www.example.com/', 'r');
    echo stream_get_contents($stream);
    fclose($stream);
    ```

    tako da umjesto sadržaja s `www.example.com` povlači i ispisuje sadržaj s vašeg prvog poslužitelja te je spremite kao `index.php`, a zatim (po)maknite zadanu datoteku `index.html`.

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

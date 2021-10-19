---
author: Vedran Miletić
---

# Virtualizacija korištenjem Virtual Machine Managera

Za jednostavno upravljanje virtualnim strojevima iz grafičkog sučelja koristit ćemo [Virtual Machine Manager](https://virt-manager.org/) (kraće virt-manager ili VMM). Virt-manager podržava brojne platforme uključujući [Linux Cointainers (LXC)](https://linuxcontainers.org/) i [Xen](https://xenproject.org/), a mi ćemo ga koristiti za [QEMU](https://www.qemu.org/) i [KVM](https://www.linux-kvm.org/).

## Stvaranje virtualnog stroja

Nakon pokretanja virt-managera pritiskom na gumb `Create a new virtual machine` pokrećemo proces stvaranja novog virtualnog stroja. U prvom koraku u dijelu `Choose how you would like to install the operating system` biramo:

* `Local install media (ISO image or CDROM)` ako želimo instalirati OS na klasičan način kao na fizičkom računalu
* `Import existing disk image` ako želimo koristiti unaprijed pripremljenu sliku za korištenje na virtualizacijskim sustavima i u oblaku.

Primjerice, za [Ubuntu](https://ubuntu.com/) možemo takve slike preuzeti s [cloud-images.ubuntu.com](https://cloud-images.ubuntu.com/). Preuzet ćemo sliku iz direktorija `focal` (kodno ime za Ubuntu 20.04 LTS) pa `current`; kako radimo na računalu čiji se procesor temelji na arhitekturi x86-64, odnosno AMD64, datoteka koja nam odgovara je `focal-server-cloudimg-amd64.img`.

Napravimo kopiju preuzete slike koju ćemo koristiti kod stvaranja virtualnog stroja. U drugom koraku stvaranja virtualnog stroja pod `Provide the existing storage path:` kliknimo na `Browse...`. Zatim u dijalogu koji se otvori pod nazivom `Locale or create storage volume` kliknimo na gumb `Browse Local` pa pronađimo stvorenu kopiju.

U dijelu `Choose the operating system you are installing:` pronađimo `Ubuntu 20.04`.

U trećem koraku biramo postavke memorije i procesora za novostvoreni virtualni stroj. Varijanta Ubuntua koju koristimo nema grafičko sučelje i uredno će raditi s manje od predloženih 4096 MiB RAM-a i 2 CPU-a, odnosno 2048 MiB i 1 CPU će biti dosta. Ta štedljivost će nam biti korisna obzirom da ćemo vremenom stvarati veći broj virtualnih strojeva.

U posljednjem koraku imamo mogućnost prilagodbe konfiguracije uključivanjem kvačice pored `Customize configuration before install` i odabira mreže koja će se koristiti. Zasad ostavimo te postavke kakve jesu.

Nakon pokretanja virtualnog stroja uočit ćemo kako se ne možemo prijaviti ma koje korisničko ime i zaporku isprobali.

## Cloud-init

Ubuntove slike za korištenje u oblaku sadrže sustav [cloud-init](https://cloud-init.io/) koji omogućuje postavljanje:

* zadanih lokalnih i regionalnih postavki
* imena domaćina
* zaporki korisnika
* SSH ključeva koji se mogu koristiti za prijavu
* diskova koji će biti montirani

Kako bismo postavili zaporku korisnika, koristit ćemo alat `cloud-localds` iz paketa [cloud-utils](https://github.com/canonical/cloud-utils). Taj alat služi za pripremu slike diska u odgovarajućem formatu za `cloud-init` izvod podataka (engl. *data source*, odakle dolazi `ds` u `localds`) `NoCloud` ([dokumentacija](https://github.com/canonical/cloud-init/blob/main/doc/rtd/topics/datasources/nocloud.rst)). Taj će se disk uključiti prilikom stvaranja virtualnog stroja u posljednjem koraku kao drugi disk i `cloud-init` će ga očitati prilikom prvog pokretanja. Alat `cloud-localds` kod pokretanja bez argumenata pokazuje pomoć pri korištenju:

``` shell
$ cloud-localds 
Usage: cloud-localds [ options ] output user-data [meta-data]

   Create a disk for cloud-init to utilize nocloud

   options:
     -h | --help             show usage
     -d | --disk-format D    disk format to output. default: raw
                             can be anything supported by qemu-img or
                             tar, tar-seed-local, tar-seed-net
     -H | --hostname    H    set hostname in metadata to H
     -f | --filesystem  F    filesystem format (vfat or iso), default: iso9660

     -i | --interfaces  F    write network interfaces file into metadata
     -N | --network-config F write network config file to local datasource
     -m | --dsmode      M    add 'dsmode' ('local' or 'net') to the metadata
                             default in cloud-init is 'net', meaning network is
                             required.
     -V | --vendor-data F    vendor-data file
     -v | --verbose          increase verbosity

   Note, --dsmode, --hostname, and --interfaces are incompatible
   with metadata.

   Example:
    * cat my-user-data
      #cloud-config
      password: passw0rd
      chpasswd: { expire: False }
      ssh_pwauth: True
    * echo "instance-id: $(uuidgen || echo i-abcdefg)" > my-meta-data
    * cloud-localds my-seed.img my-user-data my-meta-data
    * kvm -net nic -net user,hostfwd=tcp::2222-:22 \
         -drive file=disk1.img,if=virtio -drive file=my-seed.img,if=virtio
    * ssh -p 2222 ubuntu@localhost
must provide output, userdata
```

Pripremimo datoteku `user-data` tako da je oblika:

``` yaml
#cloud-config
password: asdf1234
chpasswd: { expire: False }
ssh_pwauth: True
```

Ovom datotekom u konfiguraciji oblaka (`cloud-config`) koju će `cloud-init` čitati postavljamo zaporku (`password`) na `asdf1234`, označavamo da zaporka nije istekla (`chpasswd: { expire: False }`) i dozvoljavamo prijavu putem SSH korištenjem zaporke `ssh_pwauth: True`.

Stvorimo sliku diska naredbom:

``` shell
$ cloud-localds user-data.img user-data
```

Ponovimo instalaciju Ubuntua s novom kopijom slike. U posljednjem koraku ćemo iskoristiti mogućnost prilagodbe konfiguracije uključivanjem kvačice pored `Customize configuration before install`. U dijalogu koji dobijemo odabrati ćemo `Add Hardware` pa u odjeljku `Storage` na kartici `Details` odabrati `Select or create custom storage`. Klikom na gumb `Manage...` dobivamo `Locate or create storage volume` gdje možemo iskoristiti gumb `Browse Local` kako bismo pronašli upravo stvoreni `user-data.img`.

Nakon pokretanja Ubuntua moći ćemo se prijaviti sa s korisničkim imenom `ubuntu` i postavljenom zaporkom.

!!! admonition "Zadatak"
    U virtualnom stroju instalirajte HTTP poslužitelj Apache (paket `apache2`). Iskoristite Firefox ili cURL kako biste se uvjerili da poslužitelj ispravno radi, a zatim (po)maknite zadanu datoteku `index.html` i na njenom mjestu stvorite novu datoteku sa sadržajem po vašoj želji.

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

!!! admonition "Zadatak"
    Stvorite tri virtualna stroja. Na prvom virtualnom stroju instalirate MariaDB poslužitelj (paket `mariadb-server`). Stvorite bazu i u njoj barem jednu tablicu sa stupcima po želji te je popunite podacima po želji. Omogućite pristup sa zaporkom *udaljenom* korisniku imena po želji.

    Preostala dva virtualna stroja neka imaju Apache i modul za PHP. Na ta dva virtualna stroja stvorite `index.php` sadržaja ([izvor koda](https://www.php.net/manual/en/mysqli.quickstart.dual-interface.php)):

    ``` php
    <?php

    $mysqli = mysqli_connect("database.example.com", "user", "password", "database");
    $result = mysqli_query($mysqli, "SELECT 'Please do not use the deprecated mysql extension for new development. ' AS _msg FROM DUAL");
    $row = mysqli_fetch_assoc($result);
    echo $row['_msg'];
    ```

    Pritom ćete postavke povezivanja u `mysqli_connect()` i upit u `mysqli_query()` prilagoditi svojim potrebama.

    Uvjerite se pristupanjem IP adresama tih dvaju virtualnih strojeva da sve uspješno radi. (**Uputa:** Pratite poruke o pogreškama u datoteci `/var/log/apache2/error.log`.)

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

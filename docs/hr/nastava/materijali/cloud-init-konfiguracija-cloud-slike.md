---
author: Vedran Miletić
---

# Konfiguracija cloud slike korištenjem cloud-inita

Slike operacijskih sustava namijenje za korištenje u oblaku sadrže sustav [cloud-init](https://cloud-init.io/) koji omogućuje postavljanje:

* zadanih lokalnih i regionalnih postavki
* imena domaćina
* zaporki korisnika
* SSH ključeva koji se mogu koristiti za prijavu
* diskova koji će biti montirani

Takve slike nude [Arch Linux](https://gitlab.archlinux.org/archlinux/arch-boxes/-/jobs/artifacts/master/browse/output?job=build:secure), [Fedora](https://alt.fedoraproject.org/cloud/), [Debian](https://cloud.debian.org/images/cloud/), [Ubuntu](https://cloud-images.ubuntu.com/), [BSD-i](https://bsd-cloud-image.org/) i brojni drugi.

!!! hint
    Upute u nastavku namijenjene su za Ubuntu. Za Arch Linux ćemo koristiti upute sa [stranice cloud-init na ArchWikiju](https://wiki.archlinux.org/title/cloud-init) te po potrebi postavljati `lock_passwd: False` i `password: asdf1234`. Dodatno, `xorriso` se izravno koristi za generiranje ISO 9660 slike umjesto `cloud-localds`.

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

!!! tip
    Dodatne primjere konfiguracije moguće je pronaći u dijelu [Cloud config examples službene dokumentacije cloud-inita](https://cloudinit.readthedocs.io/en/latest/topics/examples.html).

Stvorimo sliku diska naredbom:

``` shell
$ cloud-localds user-data.img user-data
```

Ponovimo instalaciju Ubuntua s novom kopijom slike. U posljednjem koraku ćemo iskoristiti mogućnost prilagodbe konfiguracije uključivanjem kvačice pored `Customize configuration before install`. U dijalogu koji dobijemo odabrati ćemo `Add Hardware` pa u odjeljku `Storage` na kartici `Details` odabrati `Select or create custom storage`. Klikom na gumb `Manage...` dobivamo `Locate or create storage volume` gdje možemo iskoristiti gumb `Browse Local` kako bismo pronašli upravo stvoreni `user-data.img`.

Nakon pokretanja Ubuntua moći ćemo se prijaviti sa s korisničkim imenom `ubuntu` i postavljenom zaporkom.

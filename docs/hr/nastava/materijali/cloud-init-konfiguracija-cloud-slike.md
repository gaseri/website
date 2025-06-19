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

Sustav cloud-init u proteklom je desetljeću postao *de facto* standard za konfiguraciju slika koje se pokreću u oblaku; uz velike pružatelje usluga u oblaku, među koje spadaju [Amazon Web Services (AWS)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html), [Google Cloud Platform (GCP)](https://cloud.google.com/container-optimized-os/docs/how-to/create-configure-instance) i [Microsoft Azure](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/using-cloud-init), podržavaju ga i manji pružatelji kao što su [DigitalOcean](https://www.digitalocean.com/community/tutorials/how-to-use-cloud-config-for-your-initial-server-setup), [Vultr](https://www.vultr.com/docs/how-to-deploy-a-vultr-server-with-cloudinit-userdata/), [Scaleway](https://www.scaleway.com/en/docs/compute/instances/api-cli/using-cloud-init/) i [Hetzner](https://community.hetzner.com/tutorials/basic-cloud-config). Osim navedenih (i drugih) pružatelja usluga u oblaku, podržani su i [drugi izvori konfiguracijskih podataka](https://cloudinit.readthedocs.io/en/latest/reference/datasources.html).

Na strani razvijatelja distribucija Linuxa, slike koje uključuju cloud-init nude [Arch Linux](https://gitlab.archlinux.org/archlinux/arch-boxes/-/jobs/artifacts/master/browse/output?job=build:secure), [Fedora](https://alt.fedoraproject.org/cloud/), [CentOS](https://cloud.centos.org/)/[AlmaLinux](https://wiki.almalinux.org/cloud/Generic-cloud.html)/[Rocky Linux](https://rockylinux.org/cloud-images/), [Debian](https://cloud.debian.org/images/cloud/), [Ubuntu](https://cloud-images.ubuntu.com/), [SmartOS](https://docs.smartos.org/how-to-create-an-hvm-zone/) i [brojni drugi distributeri](https://cloudinit.readthedocs.io/en/latest/reference/availability.html).

Upute u nastavku namijenjene su za Arch Linux i služe kao nadopuna upute sa [stranice cloud-init na ArchWikiju](https://wiki.archlinux.org/title/cloud-init).

Među dostupnim slikama u odjeljku VM images na [stranici Arch Linux Downloads](https://archlinux.org/download/) koristit ćemo sliku koja sadrži `cloudimg` (kratica od **cloud** **im**a**g**e) u nazivu, npr. `Arch-Linux-x86_64-cloudimg-20230415.143140.qcow2`.

Kako se naš virtualni stroj ne pokreće u oblaku koji bi mogao poslužiti kao izvor konfiguracijskih podataka, i nastavku opisujemo kako stvoriti izvora podataka u obliku ISO 9660 slike CD-ROM-a koji ćemo umetnuti u virtualni CD-ROM čitač kod stvaranja virtualnog stroja.

## Datoteka `user-data`

Kako bismo postavili zaporku korisnika na `fidit1234`, treba nam zaporka u hashiranom obliku. Iskoristit ćemo OpenSSL-ovu naredbu `passwd` ([dokumentacija](https://docs.openssl.org/3.1/man1/openssl-passwd/)) na način:

``` shell
openssl passwd -6 -salt 0123456789abcdef fidit1234
```

``` shell-session
$6$0123456789abcdef$dShpkpJZaM1mate.CGJEnCIUr5OnqlKLzqgErqxukBdgGMteNCAMRA/3WWVZBOwsX444nHdSAmYpJcq09V5SP1
```

RSA ključ ćemo izgenerirati naredbom `ssh-keygen` ([dokumentacija](https://man.openbsd.org/ssh-keygen.1)) na način:

``` shell
ssh-keygen
```

``` shell-session
Generating public/private rsa key pair.
Enter file in which to save the key (/home/vedranm/.ssh/id_rsa): id_rsa_gaser
ksshaskpass: Unable to parse phrase "Enter passphrase (empty for no passphrase): "
ksshaskpass: Unable to parse phrase "Enter same passphrase again: "
Your identification has been saved in id_rsa_gaser
Your public key has been saved in id_rsa_gaser.pub
The key fingerprint is:
SHA256:0x2Zhd3h8WdYte5i4gU/wc+2EBQyrHTX5zwukaOlnME vedranm@spqr
The key's randomart image is:
+---[RSA 3072]----+
|          .o +o+=|
|         . o+=++*|
|        . + =.o==|
|         o Eo*.o+|
|        S o.B++..|
|         . =o.*. |
|           . B.= |
|          . + = .|
|           .   . |
+----[SHA256]-----+
```

Uzmimo da je dobiveni tajni ključ oblika:

``` pem
-----BEGIN OPENSSH PRIVATE KEY-----
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
-----END OPENSSH PRIVATE KEY-----
```

dok je javni ključ oblika:

``` text
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDdBUhfvkHUpPT41jk4TNTbb8bZ44mH8BsXbZskqb5VjcJD7S5NyaB3vI7r0vKz3Vkd1PeV3M
FZTnn2YH/RFmyQ5G2nL/XQRgzRXBKHNU4oSOsDGnW4Xpuis0+PE+ODRTB9qNIVCpM31J2v1AT5DUp393Lgda1yBFMlmk74sUIsWV2HCAy2UgD5
pOr5z5hYopIBodQc0rq6y3PgzyYuQwBZRAWiCbAVIYEmToUmdBO6bqY+yHE3KG+WF/x+CP58vX3dVgVV9x/XkIerc0RxjMeAHUd+fMYQJ9BqFW
Gc2OSvYIvqmQ71U1geaSTk1FmpKAe5oNu3C48ax2TkJr5k06j//uCnB3MissrkK3kpJvPRhoWdg261zvBcCzITT+X4nlNeneINE33p3Y5C10GR
9vDiYXvfYZlnQoU3XY6vKfNMp4BYRDyPRSRHAgoX4pQZKPRcSj0nzxtdlSapwMpGcPOxTEoL+OhNVGXcgjGkox7YQ+fQZPMUcg2qi+EcKnvae5
c= gaser@fidit
```

Pripremimo datoteku `user-data` tako da je oblika:

``` yaml
#cloud-config
users:
  - name: gaser
    gecos: Član grupe GASERI
    groups: wheel, adm
    sudo: ALL=(ALL) NOPASSWD:ALL
    lock_passwd: false
    passwd: $6$0123456789abcdef$dShpkpJZaM1mate.CGJEnCIUr5OnqlKLzqgErqxukBdgGMteNCAMRA/3WWVZBOwsX444nHdSAmYpJcq09V5SP1
    ssh_authorized_keys:
      - ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDdBUhfvkHUpPT41jk4TNTbb8bZ44mH8BsXbZskqb5VjcJD7S5NyaB3vI7r0vKz3Vkd1PeV3MFZTnn2YH/RFmyQ5G2nL/XQRgzRXBKHNU4oSOsDGnW4Xpuis0+PE+ODRTB9qNIVCpM31J2v1AT5DUp393Lgda1yBFMlmk74sUIsWV2HCAy2UgD5pOr5z5hYopIBodQc0rq6y3PgzyYuQwBZRAWiCbAVIYEmToUmdBO6bqY+yHE3KG+WF/x+CP58vX3dVgVV9x/XkIerc0RxjMeAHUd+fMYQJ9BqFWGc2OSvYIvqmQ71U1geaSTk1FmpKAe5oNu3C48ax2TkJr5k06j//uCnB3MissrkK3kpJvPRhoWdg261zvBcCzITT+X4nlNeneINE33p3Y5C10GR9vDiYXvfYZlnQoU3XY6vKfNMp4BYRDyPRSRHAgoX4pQZKPRcSj0nzxtdlSapwMpGcPOxTEoL+OhNVGXcgjGkox7YQ+fQZPMUcg2qi+EcKnvae5c= gaser@fidit
```

Ovdje smo postavili korisničko ime na `gaser`, komentar korisnika na `Član grupe GASERI` (za više detalja proučite [odjeljak Other examples of user management stranice Users and Groups na ArchWikiju](https://wiki.archlinux.org/title/Users_and_groups#Other_examples_of_user_management)), dodali korisnika u grupe administratorske grupe `wheel` i `adm` (za više detalja proučite [odjeljak User groups stranice Users and Groups na ArchWikiju](https://wiki.archlinux.org/title/Users_and_groups#User_groups)), isključili zaključavanje zaporke, postavili zaporku na `fidit1234` i SSH ključ na ranije navedeni.

Konfiguracijska datoteka `user-data` je u formatu YAML, ali `cloud-init` je prilično izbirljiv oko sintakse pa je potrebno paziti da su retci adekvatno uvučeni i da nema suvišnih razmaka na kraju retka.

Za više primjera konfiguracije proučite odjeljak [Cloud config examples](https://cloudinit.readthedocs.io/en/latest/reference/examples.html) u [službenoj dokumentaciji cloud-inita](https://cloudinit.readthedocs.io/en/latest/reference/index.html). Za značenje pojedinih ključeva proučite odjeljak [Module reference](https://cloudinit.readthedocs.io/en/latest/reference/modules.html), a za način korištenja parametara naredbenog retka odjeljak [CLI commands](https://cloudinit.readthedocs.io/en/latest/reference/cli.html).

!!! example "Zadatak"
    Istražite kako možete korištenjem cloud-inita stvoriti datoteku `fidit.txt` sa sadržajem `najjači fakultet` u korisničkom direktoriju te pokrenuti naredbu `ping -c 3 example.group.miletic.net`.

## Alat `cloud-localds`

Alat `cloud-localds` iz paketa [cloud-utils](https://github.com/canonical/cloud-utils) se može koristiti za generiranje ISO 9660 slike umjesto `xorriso` (kako je opisano u [odjeljku QEMU stranice Cloud-init na ArchWikiju](https://wiki.archlinux.org/title/Cloud-init#QEMU)).

Taj alat služi za pripremu slike diska u odgovarajućem formatu za `cloud-init` izvor konfiguracijskih podataka (engl. *data source*, odakle dolazi `ds` u `localds`) `NoCloud` ([dokumentacija](https://cloudinit.readthedocs.io/en/latest/reference/datasources/nocloud.html)). Taj će se disk uključiti prilikom stvaranja virtualnog stroja u posljednjem koraku kao drugi disk i `cloud-init` će ga očitati prilikom prvog pokretanja. Alat `cloud-localds` kod pokretanja bez argumenata pokazuje pomoć pri korištenju:

``` shell
cloud-localds
```

``` shell-session
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

Stvorimo sliku diska naredbom:

``` shell
cloud-localds user-data.img user-data
```

Preuzmimo `cloudimg` Arch Linuxa s [nekog od službenih izvora](https://geo.mirror.pkgbuild.com/images/latest/) i ponovimo stvaranje virtualnog stroja na temelju slike. U posljednjem koraku ćemo iskoristiti mogućnost prilagodbe konfiguracije uključivanjem kvačice pored `Customize configuration before install`. U dijalogu koji dobijemo odabrati ćemo `Add Hardware` pa u odjeljku `Storage` na kartici `Details` odabrati `Select or create custom storage`. Klikom na gumb `Manage...` dobivamo `Locate or create storage volume` gdje možemo iskoristiti gumb `Browse Local` kako bismo pronašli upravo stvoreni `user-data.img`. Dodatno, `Device type` postavit ćemo na `CDROM device`.

Nakon pokretanja Arch Linuxa moći ćemo se prijaviti s postavljenim korisničkim imenom i zaporkom.

!!! example "Zadatak"
    Nakon uspješne prijave proučite [informacije o mjestu gdje se nalaze zapisnici u često postavljanim pitanjima](https://cloudinit.readthedocs.io/en/latest/reference/faq.html#where-are-the-logs) i provjerite sadržaj logova za upozorenjima i greškama.

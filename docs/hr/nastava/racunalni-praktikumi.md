---
author: Vedran Miletić
---

# O računalnim praktikumima za nastavu

[Fakultet informatike i digitalnih tehnologija](https://www.inf.uniri.hr/) održava nastavu u četiri računalna praktikuma:

- O-350/O-351 i O-366/O-367 s po 36 računala (35 studentskih i jedno nastavničko),
- O-359 s 21 računalom (20 studentskih i jedno nastavničko) i
- O-365 s 26 računala (25 studentskih i jedno nastavničko).

## Upute za instalacija i konfiguracija računala u računalnim praktikumima

U nastavku je opisan način instalacije i konfiguracije računala u računalnim praktikumima. Pritom pretpostavljamo da je operacijski sustav Windows, bez obzira radi li se o verziji 10 ili 11, instaliran na računalima u konfiguraciji pokretanja putem UEFI-ja te da su instalirane sve dostupne nadogradnje. Nadalje pretpostavljamo da računala u sebi imaju SSD i HDD, da je sav hardver ispravan, da je UEFI firmware ("BIOS") osvježen na zadnju dostupnu verziju i da je u konfiguraciji firmwarea:

- isključen Secure Boot[^1] i
- uključena podrška za [hardverski potpomognutu virtualizaciju](https://en.wikipedia.org/wiki/Hardware-assisted_virtualization)[^2] ([Intel VT-x](https://www.intel.com/content/www/us/en/virtualization/virtualization-technology/intel-virtualization-technology.html)/[AMD-V](https://www.amd.com/en/solutions/hci-and-virtualization)).

[^1]: Distribucije temeljene na Arch Linuxu moguće je [konfigurirati tako da se pokreću kad je Secure Boot uključen](https://wiki.archlinux.org/title/Unified_Extensible_Firmware_Interface/Secure_Boot), ali to komplicira održavanje i nepotrebno je na računalima koja se koriste samo u nastavne svrhe.

[^2]: Sustav za virtualizaciju [QEMU](https://wiki.archlinux.org/title/QEMU)/[KVM](https://wiki.archlinux.org/title/KVM), kojeg koristi emulator operacijskog sustava [Android](https://wiki.archlinux.org/title/Android), zahtijeva hardverski potpomognutu virtualizaciju za svoj rad.

### Instalacija operacijskog sustava Manjaro uz operacijski sustav Windows

Operacijski sustav [Manjaro](https://manjaro.org/), popularnu distribuciju Linuxa [temeljenu na Arch Linuxu](https://wiki.archlinux.org/title/Arch-based_distributions), instalirat ćemo u konfiguraciji [dualnog pokretanja s operacijskim sustavom Windows](https://wiki.archlinux.org/title/Dual_boot_with_Windows).

Kod instalacije Manjara iskoristit ćemo njegov ugrađeni instalacijski alat [Calamares](https://calamares.io/). Prilikom instalacije ćemo:

- na stranici `Particije`:
    - iskoristiti postojeću EFI particiju stvorenu prilikom instalacije operacijskog sustava Windows
    - particiju koju ćemo montirati na `/` napraviti na SSD-u s datotečnim sustavom `btrfs`
    - particiju koju ćemo montirati na `/home` napraviti na HDD-u s datotečnim sustavom `ext4`
- na stranici `Korisnici`:
    - postaviti ime na `FIDIT Sensei`
    - postaviti korisničko ime na `sensei`
    - postaviti ime domaćina na `odj-oxxx-yyy`
    - postaviti zaporku po želji
    - isključiti automatsku prijavu bez traženja zaporke i uključiti korištenje iste zaporke za administratorski račun

Nakon instalacije ponovno ćemo pokrenuti računalo.

### Konfiguracija operacijskog sustava Manjaro nakon instalacije

Konfigurirat ćemo [GRUB](https://wiki.archlinux.org/title/GRUB). U datoteci `/etc/default/grub` ćemo:

- promijeniti liniju `GRUB_DEFAULT=0` u `GRUB_DEFAULT=2` tako da se kao zadani pokreće operacijski sustav Windows
- promijeniti liniju `GRUB_TIMEOUT=5` u `GRUB_TIMEOUT=90` tako da onaj tko pokreće sva računala u računalnom praktikumu za potrebe održavanja ima vremena odabrati pokretanje operacijskog sustava Manjaro prije nego pokrene zadani operacijski sustav
- odkomentirati liniju `#GRUB_DISABLE_LINUX_UUID=true`, odnosno maknuti `#` i ostaviti `GRUB_DISABLE_LINUX_UUID=true`, tako da se koriste nazivi uređaja umjesto UUID-a diskova što olakšava kloniranje particije operacijskog sustava među računalima

Dodatno ćemo instalirati [Avahi](https://wiki.archlinux.org/title/Avahi) i uključiti pokretanje njegovog daemona:

``` shell
$ sudo pamac install avahi
(...)
$ sudo systemctl enable --now avahi-daemon.service
```

Naposlijetku ćemo uključiti pokretanje [OpenSSH](https://wiki.archlinux.org/title/Secure_Shell) daemona:

``` shell
$ sudo systemctl enable --now sshd.service
```

i generirati novi SSH ključ sa zadanim postavkama i bez zaporke:

``` shell
$ ssh-keygen
Generating public/private rsa key pair.
Enter file in which to save the key (/home/sensei/.ssh/id_rsa):
Created directory '/home/sensei/.ssh'.
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/sensei/.ssh/id_rsa
Your public key has been saved in /home/sensei/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:x8BAgC63dlAV/QJclVtUHuB7w9XWdcQFVNyJ6z8vjUU sensei@odj-oxxx-yyy
The key's randomart image is:
+---[RSA 3072]----+
|   ..==+...oo+**X|
|  . . oo. ...o oO|
| . .   .o. o. o =|
|. +     .oo  + oE|
| o o    S.o o +. |
|  o .    .   o ..|
| . .          .+ |
|              oo.|
|               .+|
+----[SHA256]-----+
```

Dodat ćemo novostvoreni ključ u `.ssh/authorized_keys` ručno ili naredbom:

``` shell
$ ssh-copy-id localhost
/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: "/home/sensei/.ssh/id_rsa.pub"
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys

Number of key(s) added: 1

Now try logging into the machine, with:   "ssh 'localhost'"
and check to make sure that only the key(s) you wanted were added
```

### Kloniranje slike diska instaliranog operacijskog sustava na računala

Iskoristit ćemo [Clonezillu](https://clonezilla.org/) za stvaranje slike diska koja se može prebaciti na ostala računala i pohraniti je na neki vanjski medij. Zatim ćemo prebaciti tu sliku na ostala računala.

!!! note
    U nastavku ćemo pretpostaviti da su upute pisane za računalni praktikum O-359. Ostali računalni praktikumi se konfiguriraju analogno.

Nakon kloniranja i uspješnog pokretanja operacijskog sustava Manjaro postavit ćemo ime domaćina naredbom:

``` shell
$ sudo hostnamectl hostname odj-o359-101
```

te u datoteci `/etc/hosts` promijeniti liniju `127.0.1.1  odj-oxxx-yyy` u `127.0.1.1  odj-o359-101`. Varirat ćemo broj koji se odnosi na računalni praktikum i broj koji se odnosi na pojedino računalo po potrebi.

### Konfiguracija nastavničkog računala

Na nastavničkom računalu ćemo instalirati [Ansible](https://wiki.archlinux.org/title/Ansible) naredbom:

``` shell
$ sudo pamac install ansible
```

Stvorit ćemo datoteku `/etc/ansible/hosts` sadržaja:

``` ini
[o359]
odj-o359-101.local
odj-o359-102.local
odj-o359-103.local
odj-o359-104.local
odj-o359-105.local
odj-o359-106.local
odj-o359-107.local
odj-o359-108.local
odj-o359-109.local
odj-o359-110.local
odj-o359-111.local
odj-o359-112.local
odj-o359-113.local
odj-o359-114.local
odj-o359-115.local
odj-o359-116.local
odj-o359-117.local
odj-o359-118.local
odj-o359-119.local
odj-o359-120.local
odj-o359-121.local

[o359stud]
odj-o359-102.local
odj-o359-103.local
odj-o359-104.local
odj-o359-105.local
odj-o359-106.local
odj-o359-107.local
odj-o359-108.local
odj-o359-109.local
odj-o359-110.local
odj-o359-111.local
odj-o359-112.local
odj-o359-113.local
odj-o359-114.local
odj-o359-115.local
odj-o359-116.local
odj-o359-117.local
odj-o359-118.local
odj-o359-119.local
odj-o359-120.local
odj-o359-121.local
```

!!! todo
    Ovdje treba dodatno opisati proces instalacije i konfiguracije [Veyona](https://veyon.io/).

## Ograničavanje pristupa Moodle testu na temelju IP adrese

Moodleova aktivnost `Test` u pripadnim `Postavkama` sadrži odjeljak `Dodatna ograničenja tijekom rješavanja`. Među dodatnim ograničenjima postoji i mogućnost `Ograničavanja pristupa samo ovim IP adresama` ([dokumentacija](https://docs.moodle.org/401/en/Quiz_settings#Extra_restrictions_on_attempts)). Za IP adresu se navodi javna adresa odgovarajućeg računalnog praktikuma iz popisa u nastavku; u slučaju da postoji potreba za provjerom adrese na licu mjesta, može se iskoristiti [Detektor adrese IP](https://apps.group.miletic.net/ip/).

### O-350/351

``` ip
193.198.209.231
```

### O-359

``` ip
193.198.209.234
```

### O-365

``` ip
193.198.209.232
```

### O-366/367

``` ip
193.198.209.233
```

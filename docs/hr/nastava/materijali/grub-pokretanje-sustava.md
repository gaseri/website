---
author: Vedran Miletić
---

# Boot proces računala i učitavač GRUB

## Boot proces

Boot proces računala počinje pritiskom na gumb (kako god on bio izveden) koji putem matične ploče uzrokuje uključivanje napajanja, koje zatim počinje isporučivati struju komponentama unutar računala. Inicijalizaciju komponenata računala vrši BIOS ili, kod novijih računala, UEFI firmware pa specijalno nakon inicijalizacije grafičkog procesora i njegovog podsustava za ekrane vidimo tekst ili logotip proizvođača na ekranu.

### BIOS boot

Pokreće operacijski sustav s master boot recorda (MBR) ili bootabilnih particija odabranog diska na sustavu (optički mediji su također podržani kao mediji s kojih se može pokrenuti operacijski sustav, ali su [standardi drugačiji](https://en.wikipedia.org/wiki/El_Torito_(CD-ROM_standard))). Korištenje MBR-a ograničava veličinu diska na 2 TiB, ali moguće je [uvođenjem dodatne particije BIOS Boot zaobići to ograničenje i koristiti GUID Partition Table](https://www.anchor.com.au/blog/2012/10/the-difference-between-booting-mbr-and-gpt-with-grub/).

### UEFI boot

- [Unified Extensible Firmware Interface (UEFI)](https://uefi.org/) je sučelje između firmwarea i OS-a
- UEFI je modernija zamjena za BIOS

    - Može bootati s diskova većih od 2 TiB koji koriste GUID Partition Table (GPT)
    - Arhitektura neovisna o instrukcijskom skupu CPU-a
    - Mogućnost korištenja mreže bez pokretanja OS-a, e.g. "Check for firmware update on line"
    - Modularan dizajn

- Nudi pokretanje imenovanih operacijskih sustava koje pronađe na sustavskim EFI particijama na diskovima; primjerice, boot menu ima u popisu `Arch Linux`, `Windows Boot Manager`, `FreeBSD 12.2` ili `Fedora` umjesto `Seagate 1 TB HDD` ili `Samsung 256 GB M.2 SSD`.
- Omogućuje "Fast Boot" kod kojeg se inicijalizira samo dio uređaja unutar računala i specijalno samo onaj disk koji je nužan za pokretanje operacijskog sustava, što ubrzava boot proces.
- Većina današnjeg UEFI hardvera nudi kompatibilnost s legacy BIOS-om pa mogu pokretati operacijske sustave koji ne poznaju UEFI.

#### Secure Boot

- Ograničava pokretanje OS-a (ili bootloadera OS-a) samo na one u koje ima povjerenja
- Standard razvijen na inicijativu Microsofta; [Microsoft opravdava postojanje takve vrste ograničenja](https://docs.microsoft.com/en-us/windows-hardware/manufacture/desktop/windows-secure-boot-key-creation-and-management-guidance) s ciljem sprječavanja pre-bootloader [rootkita](https://en.wikipedia.org/wiki/Rootkit), od kojih su pojedini služili da sabotiraju mehanizam aktivacije Windowsa
- Kod pokretanja provjerava potpis bootloadera

    - Ako je potpis dobar, firmware predaje bootloaderu kontrolu nad računalom
    - Ako potpis nije dobar ili ne postoji, pokretanje bootloadera se ne događa

- Radi na temelju [četiri baze ključeva](https://docs.microsoft.com/en-us/windows-hardware/manufacture/desktop/windows-secure-boot-key-creation-and-management-guidance)

    - Baza potpisa (engl. signature database), db

        - Ključevi i hashevi kojima se vjeruje
        - Npr. Microsoft Windows Production PCA 2011, Microsoft Corporation UEFI CA 2011, Canonical Ltd. Master CA, ...

    - Baza opozvanih potpisa (engl. revoked signature database), dbx

        - Ključevi i hashevi kojima se eksplicitno ne vjeruje

    - Ključ razmjene ključeva (engl. Key Exchange Key), KEK

        - Ključ kojim su potpisani softver koji mogu mijenjati db i dbx
        - Npr. Microsoft Corporation KEK CA 2011, HP CA, Lenovo CA, Toshiba CA, ...

    - Ključ platforme (engl. Platform Key), PK

        - Ključ kojim je potpisan softver koji može mijenjati PK i KEK
        - Npr. HP CA, Lenovo CA, Toshiba CA, ...

- Secure Boot u praksi

    - Operacijski sustavi čiji bootloader je potpisan ključem Microsoft Corporation UEFI CA 2011:

        - [Fedora (18+)](https://fedoraproject.org/wiki/Secureboot)
        - [openSUSE (12.3+)](https://en.opensuse.org/openSUSE:UEFI)
        - [Ubuntu (12.04.2+)](https://help.ubuntu.com/community/UEFI)

    - Operacijski sustavi koji rade na podršci za Secure Boot

        - [Debian](https://wiki.debian.org/SecureBoot)
        - [FreeBSD](https://wiki.freebsd.org/SecureBoot)
        - Nejasan status: [Mageia](https://wiki.mageia.org/en/About_EFI_UEFI#Secure_Boot)

    - OEM-ovi NE MORAJU imati taj ključ u db-u
    - Microsoft može prestati potpisivati 3rd party binaryje kad god im to odgovara

- Kratka povijest Secure Boota

    - 2011\. Microsoft priprema teren za izlazak Windowsa 8; za UEFI mode:

        - Certificirana mašina MORA imati Secure Boot uključen po defaultu
        - Certificirana mašina MORA imati Microsoftove ključeve u odgovarajućoj bazi

    - Kreće propaganda protiv Microsofta, prozivajući ih zbog lock-ina
    - Različita pravila za ARM i x86

        - ARM

            - Secure Boot se NE SMIJE moći isključiti
            - Dodatni ključevi se NE SMIJU moći instalirati

        - x86 (nakon pritiska od strane zajednice)

            - Secure Boot se MORA moći isključiti
            - Dodatni ključevi se MORAJU moći instalirati

    - 2015\. Microsoft priprema teren za izlazak Windowsa 10; za x86 pravila su

        - Secure Boot se NE MORA moći isključiti
        - Dodatni ključevi se MORAJU moći instalirati ([primjer postupka](https://dannyvanheumen.nl/post/secure-boot-in-fedora/))

    - Ars Technica prva ima [vijest o novim pravilima](https://arstechnica.com/information-technology/2015/03/windows-10-to-make-the-secure-boot-alt-os-lock-out-a-reality/), i ponovno kreće propaganda protiv Microsofta

- Povodom [objave novih pravila o Secure Bootu u Windowsima 10](https://www.phoronix.com/scan.php?page=news_item&px=SecoreBoot-Windows-10) na Phoronixu, korisnik [chithanh](https://www.phoronix.com/forums/member/16465-chithanh) dao je zanimljiv pregled [jednog mogućeg slijeda događaja u budućnosti](https://www.phoronix.com/forums/forum/phoronix/general-discussion/49072-new-secureboot-concerns-arise-with-windows-10/page6#post623687):

    - Proizvođači hardvera moraju u svojim proizvodima podržavati značajku Secure Boot. Kritičari su umireni time što se može isključiti i mogućnošću korisnika da instalira vlastite ključeve. ✅
    - Secure Boot mora biti uključen u zadanim postavkama. ✅ (**do ovdje je došlo s Windowsima 8**)
    - Opcionalna tehnologija [Intel Boot Guard brani modifikaciju firmwarea](https://patrick.georgi.family/2015/02/17/intel-boot-guard/) (primjerice, flashanje [coreboota](https://coreboot.org/) umjesto proizvođačevog firmwarea). ✅
    - Secure Boot može biti uvijek uključen ako proizvođač hardvera tako odluči. ✅ (**do ovdje je došlo s Windowsima 10, trenutno stanje**)
    - Mogućnost instalacije ključeva može biti opcionalna ako proizvođač hardvera tako odluči. 🚧
    - Proizvođači hardvera moraju trajno uključiti Secure Boot. 🚧
    - Intel Boot Guard postaje obavezan. 🚧
    - Instalacija vlastitih ključeva u UEFI firmware postaje zabranjena. 🚧
    - *Rezultat:* moguće je pokretati samo bootloadere potpisane postojećim ključevima, što u praksi vjerojatno znači potpisane Microsoftovim ključevima, a Microsoft može u bilo kojem trenutku prestati potpisivati sve osim Windowsa.

- Općenitije o ograničavanju mogućnosti računala (tzv. rat protiv računarstva opće namjene) govorio je [Cory Doctorow](https://craphound.com/) iz [Electronic Frontier Foundationa](https://www.eff.org/about/staff/cory-doctorow) na [28C3](https://events.ccc.de/congress/2011/wiki/Welcome) pod naslovom [The coming war on general computation (The copyright war was just the beginning)](https://media.ccc.de/v/28c3-4848-en-the_coming_war_on_general_computation) ([transkript](https://joshuawise.com/28c3-transcript)) i na [DEF CON-u 23](https://www.defcon.org/html/defcon-23/dc-23-index.html) pod naslovom [Fighting Back in the War on General Purpose Computers](https://youtu.be/pT6itfUUsoQ).

## Boot učitavač GRUB

- [GNU GRUB](https://en.wikipedia.org/wiki/GNU_GRUB) je [boot učitavač](https://en.wikipedia.org/wiki/Booting#Modern_boot_loaders) (engl. *bootloader*)

    - pokreće se prije pokretanja samog operacijskog sustava i nudi korisniku mogućnost izbora koji operacijski sustav želi pokrenuti
    - primjerice, kod Debiana GRUB nudi normalan način rada, način rada za oporavak operacijskog sustava (engl. *recovery*), testiranje radne memorije računala i pokretanje preostalih operacijskih sustava na računalu

- [initrd](https://en.wikipedia.org/wiki/Initrd) i [initramfs](https://en.wikipedia.org/wiki/Initramfs)

    - sadrži module jezgre koji se koriste za dosezanje particije na kojoj se nalazi operacijski sustav
    - npr. `pata_atiixp`, `pata_amd`, `sata_nv`, `ahci`; `raid0`, `raid1`; `ext3`, `ext4`, `btrfs`

### Konfiguracija GRUB-a

Kako bismo mogli pokrenuti drugačiju verziju jezgre ili dodati određene parametre prilikom pokretanja, trebamo konfigurirati [boot učitavač](https://wiki.archlinux.org/title/Arch_boot_process#Boot_loader); u zadanoj instalaciji većine distribucija Linuxa i specijalno Arch Linuxa to je [GRUB](https://wiki.archlinux.org/title/GRUB).

[Konfiguracija GRUB-a](https://wiki.archlinux.org/title/GRUB#Configuration) nalazi se u datoteci `/boot/grub/grub.cfg` koju možemo proučiti i uočiti da je konfiguracija naše jezgre između `### BEGIN /etc/grub.d/10_linux ###` i `### END /etc/grub.d/10_linux ###`.

Konfiguraciju se ne preporučuje ručno uređivati, već se to vrši naredbom `grub-mkconfig`:

``` shell
$ sudo grub-mkconfig
Stvaranje grub datoteke podešavanja ...
#
# DO NOT EDIT THIS FILE
#
# It is automatically generated by grub-mkconfig using templates
# from /etc/grub.d and settings from /etc/default/grub
#

### BEGIN /etc/grub.d/00_header ###
(...)
Pronađena linux slika: /boot/vmlinuz-linux-lts
Pronađena initrd slika: /boot/initramfs-linux-lts.img
(...)
Pronađena linux slika: /boot/vmlinuz-linux
Pronađena initrd slika: /boot/initramfs-linux.img
(...)
### END /etc/grub.d/41_custom ###
Završeno
```

Želimo li generiranu konfiguraciju i zapisati u datoteku, naredbi ćemo dodati parametar `--output`, odnosno `-o` i argument u kojem ćemo navesti putanju do datoteke, u našem slučaju `/boot/grub/grub.cfg`.

Nakon ponovnog pokretanja operacijskog sustava i odabira jezgre `linux-lts` putem GRUB-a možemo se naredbom `uname -a` uvjeriti da se zaista koristi novoinstalirana varijanta jezgre.

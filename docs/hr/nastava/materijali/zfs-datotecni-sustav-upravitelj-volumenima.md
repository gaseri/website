---
author: Ana Tomasović, Vedran Miletić
---

# Datotečni sustav i upravitelj volumenima ZFS

Sun Microsystems s razvojem datotečnog sustava [ZFS](https://en.wikipedia.org/wiki/ZFS) za vlastiti operacijski sustav Solaris započinje 2001. godine. Ideja je bila razviti posljednji datotečni sustav koji će ikada biti potrebno razviti pa je i nazvan posljednjim slovom u abecedi. Ograničenja ZFS-a su toliko velika da bi za njihovo dostizanje trebalo izgraditi polje diskova koje je toliko energetski zahtjevno da bi [isparavalo sve oceane na Zemlji](https://web.archive.org/web/20151208192725/https://blogs.oracle.com/dcb/entry/zfs_boils_the_ocean_consumes). Voditelji projekta su Jeff Bonwick, Bill Moore i Matthew Ahrens. Rezultat je copy-on-write sustav koji kombinira dotada odvojene uloge datotečnog sustava, primjeri kojih su ext3, ext4 i XFS, i upravitelja volumenima, primjeri kojih su [Linux Software Raid (mdraid)](https://raid.wiki.kernel.org/) i [Logical Volume Manager (LVM)](https://opensource.com/business/16/9/linux-users-guide-lvm).

Otvoreni kod ZFS-a izdan je 2005. godine, kao dio operacijskog sustava OpenSolaris. Nedugo nakon, 2006. godine, započinje prenošenje koda na Linux kernel pod sučeljem FUSE (Filesystems in UserSpacE), a 2008. kreće razvoj porta ZFS-a na Linux koji bi se izvodio u jezgri, nazvanog [ZFS on Linux](https://zfsonlinux.org/), kojeg financira američki [Lawrence Livermore National Laboratory](https://www.llnl.gov/). Sljedećih godina kod se prenosi i na ostale operacijske sustave, prvenstveno macOS i FreeBSD. Neki od njih staju s daljnjim razvojem ili nastavljaju kao fork originalnog projekta.

OpenSolaris prestaje postojati 2010. godine, a s njime i razvoj ZFS-a kao softvera otvorenog koda. Daljnji razvoj Solarisa preuzima illumos iste godine, svojevrsno nastavljajući OpenSolaris. Naposlijetku, 2013. godine započeo je projekt [OpenZFS](https://openzfs.org/) kao zajednički trud programskih inženjera ZFS-a za illumos, Linux, FreeBSD i macOS. Među pokretačima je i jedan od vođa originalnog projekta unutar Suna, Matthew Ahrens. Osim cilja poboljšavanja koda, otklanjanja bugova i održavanja postojećeg sustava, također i promoviraju sam datotečni sustav, njegove prednosti te rade na prenosivosti između različitih operacijskih sustava tako da je ZFS moguće koristiti i na [Windowsima](https://openzfsonwindows.org/).

Sama jezgra OpenZFS-a je neovisna o platformi, ali unatoč tome postoji po jedan repozitorij za svaku od 4 platforme, zbog određenih specifičnih implementacija vezanih uz operacijski sustav, primjerice, upravljanje memorijom ili input/output diska. OpenZFS je izdan pod licencom [Common Development and Distribution License (CDDL)](https://opensource.org/licenses/CDDL-1.0), koju je naslijedio od prvotnog projekta započetog 2001. godine. Odogovor na pitanje je li ta licenca kompatibilna s licencom GNU General Public License, version 2 (GPLv2) pod kojom je izdana jezgra Linux varira ovisno o tome koji pravni stručnjaci ga daju; [Software Freedom Law Center tvrdi da se licence mogu kombinirati](https://softwarefreedom.org/resources/2016/linux-kernel-cddl.html), a [Software Freedom Conservancy tvrdi da ne mogu jer to krši odredbe GPL-a](https://sfconservancy.org/blog/2016/feb/25/zfs-and-linux/). Hipotetska promjena licence zahtijevala bi suglasnost brojnih tvrtki i programera koji su na ZFS-u radili od početka, stoga takav pothvat nije u planu projekta.

## Specifičnosti korištenja ZFS-a na Arch Linuxu

[ZFS na Arch Linuxu](https://wiki.archlinux.org/title/ZFS) je dio [Arch User Repositoryja (AUR-a)](https://wiki.archlinux.org/title/Arch_User_Repository). To znači da se paket izgrađuje ručno iz izvornog koda korištenjem:

- datoteke [PKGBUILD](https://wiki.archlinux.org/title/PKGBUILD), koja sadrži naredbe potrebne za izgradnju softvera i
- naredbe [makepkg](https://wiki.archlinux.org/title/makepkg), koja čita sadržaj datoteke `PKGBUILD` i na temelju istog izgrađuje paket.

Za dohvaćanje ključeva kojima su potpisane datoteke izvornog koda iskoristit ćemo [GnuPG](https://wiki.archlinux.org/title/GnuPG), specifično naredbu `gpg --recv-keys key-id`. Naredba `pacman-key`, koja je dio [sustava za potpisivanje Pacman paketa](https://wiki.archlinux.org/title/Pacman/Package_signing)), generalno je vrlo korisna, ali nam u ovom slučaju nije korisna jer radimo s potpisima datoteka izvornog koda, a ne s potpisima paketa.

Koristit ćemo [zfs-dkms](https://aur.archlinux.org/packages/zfs-dkms), koji nije vezan za specifičnu verziju jezgre Linuxa i koristi [Dynamic Kernel Module Support (DKMS)](https://wiki.archlinux.org/title/Dynamic_Kernel_Module_Support) za izgradnju modula jezgre za verziju jezgre Linuxa koju imamo instaliranu na sustavu. Kako koristimo [stabilnu verziju jezgre Linuxa](https://wiki.archlinux.org/title/Kernel) (paket `linux`), potrebna su nam pripadna zaglavlja dostupna u paketu `linux-headers`:

``` shell
$ sudo pacman -S linux-headers
```

Prvo ćemo preuzeti datoteku `PKGBUILD` za `zfs-dkms`, a zatim i sve datoteke izvornog koda navedene u dijelu `Sources`.

Također ćemo izgraditi i instalirati [zfs-utils](https://aur.archlinux.org/packages/zfs-utils) koji sadrži alate u korisničkom prostoru za rad sa ZFS-om, specifično naredbe:

- `zpool`, koja služi za baratanje ZFS volumenima i
- `zfs`, koja služi za baratanje ZFS datotečnim sustavima.

Na isti način, preuzet ćemo datoteku `PKGBUILD` za `zfs-utils`, a zatim i datoteke izvornog koda pod `Sources`. Kako ne bi došlo do konflikta, datoteku `PKGBUILD` možemo ili spremiti pod drugim imenom pa koristiti parametar `-p` naredbe `makepkg` ili postaviti u drugi direktorij u odnosu na datoteku `PKGBUILD` za `zfs-dkms`.

## Stvaranje volumena

Potrebno je [particionirati diskove](https://wiki.archlinux.org/title/Partitioning) tako da imaju particijsku tablicu tipa [GPT](https://en.wikipedia.org/wiki/GUID_Partition_Table) i na njoj samo jednu particiju tipa `Solaris /usr & Apple ZFS`, GUID `6A898CC3-1DD2-11B2-99A6-080020736631`. To možemo izvesti, primjerice, korištenjem [fdisk-a](https://wiki.archlinux.org/title/Fdisk).

Stvaranje ZFS bazena za pohranu podataka vršimo naredbom `zpool create`:

``` shell
$ sudo zpool create -m /mojbazen /dev/vdb1 /dev/vdc1
```

Ovime smo stvorili bazen za pohranu podataka bez redundancije gdje se podaci zapisuju na oba diska. Alternativno, moguće je iskoristiti argument `mirror` i dobiti bazen u kojem se isti podaci zapisuju na oba diska:

``` shell
$ sudo zpool create -m /mojezrcaljenje mirror /dev/vdc1 /dev/vdd1
```

!!! admonition "Zadatak"
    Osim zrcaljenja, dostupni su i RAID-Z nivoi `raidz`, `raidz2` i `raidz3`. Isprobajte kako rade i koliko diskova je potrebno u svakom od njih odvojiti za pohranu paritetnih podataka koja će osigurati nastavak rada u slučaju kvara.

Dodatne uređaje moguće je dodati u postojeći bazen naredbom `zpool add`, kao samo novi uređaj ili kao dio mirrora. [Proširenje RAID-Z-a je u razvoju](https://arstechnica.com/gadgets/2021/06/raidz-expansion-code-lands-in-openzfs-master/) i još uvijek nije dostupno za korištenje.

## Postavke datotečnog sustava

Naredbom `zfs get` moguće je dohvatiti postavke datotečnog sustava, a naredbom `zfs set` postaviti iste. Primjerice, za uključiti ZSTD kompresiju:

``` shell
$ sudo zfs set compression=zstd
```

Da je kompresija uspješno uključena možemo se uvjeriti naredbom:

``` shell
$ sudo zfs get compression
```

!!! admonition "Zadatak"
    Provjerite vrijednost omjera kompresije (varijabla `compressratio`) i razmislite je li kompresija učinkovita.

    Provjerite je li uključeno bilježenje vremena pristupa datotekama (varijabla `atime`).

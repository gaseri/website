---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov
---

# Diskovi, particije, datotečni sustavi i mjesta montiranja

## Diskovi i particije

- particioniranje je podjela diska u particije, omogućuje da na jedan fizičk disk odvojimo u više nezavisnih dijelova

    - informacije o particioniranju su zapisane u particijskoj tablici koja je dio [glavnog unosa pokretanja](https://en.wikipedia.org/wiki/Master_Boot_Record) (engl *Master Boot Record*, MBR)
    - stanadardna particijska tablica nudi dvije mogućnosti: 4 primarne particije ili 3 primarne i 1 sekundarna (koja sadrži do 64 logičke)
    - standardna particijska tablica podržava diskove do veličine 2 terabajta
    - spomenuto je prilično ograničavajuće za današnje pojmove, ali može se zaobići korištenjem

        - logičkih volumena (engl. *logical volumes*); implementacija pod Linuxom naziva se [LVM](https://en.wikipedia.org/wiki/Logical_Volume_Manager_(Linux)); više na , ili
        - [GUID particijske tablice](https://en.wikipedia.org/wiki/GUID_Partition_Table) (engl. *GUID partition table*, GPT).

- primjer particioniranja diska veličine 320 GB

    - `/` -- 25 GB
    - `/boot` -- 0.5 GB
    - `/home` -- 290 GB
    - `swap` -- 4.5 GB

- `fdisk` je alat za particioniranje

    - prilično je jednostavan za korištenje kada znate osnovne pojmove particioniranja (nećemo ga koristiti, ali ga spominjemo jer je vrlo važan)

- imenovanje diskova

    - `/dev/sda` -- prvi disk
    - `/dev/sdb` -- drugi disk
    - `/dev/sda1` -- prva particija prvog diska
    - `/dev/sda2` -- druga particija prvog diska
    - `/dev/sdb5` -- prva logička particija drugog diska
    - `/dev/sdd6` -- druga logička particija četvrtog diska

!!! admonition "Zadatak"
    Što znamo o sljedećim particijama diska:

    - `/dev/sdf7`,
    - `/dev/sde3`?

- imenovanje ostalih uređaja za pohranu podataka

    - `/dev/sr0` -- prvi CD/DVD uređaj
    - `/dev/st0` -- prvi uređaj pisanje i čitanje vrpce
    - **Napomena:** ranije su postojale dvije konvencije, jedna za IDE (PATA) (`/dev/hd*`), a druga za SCSI, SATA i SAS diskove (`/dev/sd*`); danas se za sve koristi ova druga

- `eject /dev/sr0` izbacuje ladicu CD/DVD uređaja `/dev/sr0`

- `sg_map` pokazuje koja imena uređaja za pohranu podataka korespondiraju kojima

    - `/dev/sd*`, `/dev/sr*` i `/dev/st*` postoje i pod nazivom oblika `/dev/sg*`
    - dio softverskog paketa `sg3-utils` (ne dolazi u zadanoj instalaciji većine distribucija)

!!! admonition "Zadatak"
    - Saznajte koje particije postoje na računalu na kojem radite. Jesu li one primarne, sekundarne ili logičke?
    - Saznajte ima li računalo na kojem radite CD/DVD uređaj. Možete li izbaciti ladicu? ;-)

## Datotečni sustavi

- datotečni sustav (engl. *file system*) je *sadržan* na particiji *sadrži* datoteke

    - FAT{16,32} -- datira iz doba [MS-DOS](https://en.wikipedia.org/wiki/MS-DOS)-a; [FAT 32](https://en.wikipedia.org/wiki/File_Allocation_Table#FAT32) omogućuje veću veličinu datotečnog sustava i pojedine datoteke, limit 4 gigabajta za veličinu datoteke
    - [NTFS](https://en.wikipedia.org/wiki/NTFS) kreće 1993. godine sa [Windows NT 3.1](https://en.wikipedia.org/wiki/Windows_NT_3.1)

        - jako dobar datotečni sustav
        - podržan i pod operacijskim sustavima sličnim Unixu putem [NTFS-3G](https://en.wikipedia.org/wiki/NTFS-3G); zbog nedostatka dokumentacije koja objašnjava kako implementirati podršku za NTFS stvoren je [reverse engineeringom](https://unix.stackexchange.com/q/117006)

    - **ext{2,3,4}** je Linux [extended file system](https://en.wikipedia.org/wiki/Extended_file_system)

        - [ext2](https://en.wikipedia.org/wiki/Ext2) ne podržava dnevničenje (pojam ćemo objasniti nešto kasnije)
        - [ext3](https://en.wikipedia.org/wiki/Ext3) i [ext4](https://en.wikipedia.org/wiki/Ext4) podržavaju dnevničenje

    - Reiser{FS,4} zasnovan na [B*-stablima](https://en.wikipedia.org/wiki/B-tree) kao podatkovnoj strukturi
    - [Btrfs](https://en.wikipedia.org/wiki/Btrfs) je vjerojatni nasljednik ext4, također je zasnovan na B-stablima, ali se još uvijek intenzivno razvija i ne smatra dovoljno stabilnim
    - [XFS](https://en.wikipedia.org/wiki/XFS)
    - [JFS](https://en.wikipedia.org/wiki/JFS), IBM-ov datotečni sustav
    - [NILFS2](https://en.wikipedia.org/wiki/NILFS)
    - [UFS](https://en.wikipedia.org/wiki/Unix_File_System) je Unix-ov datotečni sustav, koriste ga Solaris i brojni BSD-i

- [montiranje](https://en.wikipedia.org/wiki/Mount_(computing)) (engl. *mount*) je povezivanje datotečnog sustava particije na neki direktorij u postojećem datotečnom sustavu

    - Linux je jednokorijenski OS, pa se sve nalazi *negdje* pod direktorijem `/`
    - npr. `/dev/sda3` može biti montiran na `/home`

- `mount` je naredba za montiranje datotečnih sustava; **primjer:**

    ``` shell
    $ mount /dev/sda3 /home
    ```

- `umount` je naredba za odmontiranje datotečnih sustava; **primjer:**

    ``` shell
    $ umount /dev/sda3
    ```

    ili

    ``` shell
    $ umount /home
    ```

!!! admonition "Zadatak"
    - Saznajte koje su particije trenutno montirane i koji datotečni sustav imaju na sebi.
    - Provjerite poklapaju li se informacije koje ste dobili na bilo koji način sa sadržajem datoteka `/etc/mtab` i `/etc/fstab`. (**Uputa:** da bi razumijeli prvi stupac `/etc/fstab` proučite [stranicu o UUID na Wikipediji](https://en.wikipedia.org/wiki/UUID).)

## swap prostor

- virtualna memorija podrazumijeva istovremeno korištenje brže unutarnje (radne) i sporije vanjske memorije (čvrsti disk)

    - izmjenjuje sadržaj u radnoj i vanjskoj memoriji, s ciljem ostvarivanje većeg efektivnog kapaciteta
    - nekad veoma značajno, ali danas se dosta manje koristi (4-8-16 gigabajta radne memorije standard)

- `swap` prostor je particija ili datoteka; koristi se u dvije situacije

    - kada nema dovoljno mjesta u radnoj memoriji za smještenje nekog od programa,
    - za [hibernaciju](https://en.wikipedia.org/wiki/Hibernation_(computing)) (*suspend to disk*, STD, ACPI S4 stanje); kod isključivanja računala sadržaj radne memorije sprema se na disk i kod ponovnog pokretanja računala učitava u memoriju

- hibernacija != suspenzija

    - [suspenzija](https://en.wikipedia.org/wiki/Sleep_mode) (*suspend to RAM*, STR, ACPI S3 stanje); radna memorija ostaje pod naponom i čuva sadržaj, a kod buđenja se pokreću sve ostale komponente u računalu
    - ponekad ima problema zbog loše implementacije ACPI standarda u BIOS-u matične ploče

## Zauzeće i kapacitet diska

- `du` ispisuje koliko na disku zauzima određeni direktorij/datoteka
- `df` ispisuje slobodan prostor na datotečnom sustavu

!!! admonition "Zadatak"
    - Ispišite veličinu koju direktorij `/usr` zauzima na datotečnom sustavu na kojem se nalazi, i to na način da se vrijednost ispisuju u `MiB`, `GiB` za sve datoteke, a ne samo direktorije.
    - Pronađite način da ispišete ukupan zauzeti i slobodan prostor na svim particijma.

### Konzistentnost i dnevničenje

- kod iznenadnog prekida rada može doći do nekonzistentnog stanja datotečnog sustava (popis datoteka koje se nalaze na datotečnom sustavu nije točan)
- dnevnik (engl. *journal*) zapisuje koje će promjene napraviti na datotečnom sustavu prije nego ih napravi, omogućuje relativno brzu provjeru konzistentnosti datotečnog sustava

    - svi ranije navedeni datotečni sustavi osim FAT{16,32} i ext2 su dnevnički datotečni sustavi (engl. *journaling file system*)

- `fsck` je naredba za provjeru datotečnog sustava particije

    - [the second letter was different](https://en.wikipedia.org/wiki/Fsck)

- `mkfs` je naredba za stvaranje datotečnog sustava na particiji (spominjemo radi potpunosti)

!!! admonition "Zadatak"
    - Pokušajte napraviti provjeru datotečnog sustava particije `/dev/sda3`. Objasnite grešku koje vam sustav javlja.
    - Pokušajte napraviti provjeru datotečnog sustava `/boot`. Objasnite upozorenje i grešku koju vam sustav javlja.
    - Pokušajte napraviti provjeru datotečnog sustava `/usr`. Objasnite grešku koju vam sustav javlja.
    - Pokušajte napraviti provjeru datotečnog sustava `/`. Objasnite grešku koju vam sustav javlja.

!!! admonition "Ponovimo!"
    - Čemu služi MBR?
    - Kakve mogućnosti nudi standardna particijska tablica?
    - Kako se imenuju diskovi u UNIX-like operacijskim sustavima? Koristite primjer sustava na kojem radimo
    - Za što koristimo `fdisk`?
    - Koje smo datotečne sustave spominjali?
    - Kako provjeravamo zauzeće diska?
    - Pojasnite pojam dnevnika i njegovu ulogu u sustavu.

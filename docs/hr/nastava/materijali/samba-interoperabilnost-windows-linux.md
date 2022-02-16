---
author: Renato Kalanj, Vedran Miletić
---

# Interoperabilnost operacijskih sustava Windows i GNU/Linux korištenjem Sambe

Server Message Block (SMB) je mrežni protokol koji pruža usluge zajedničkog pristupa prema datotekama, serijskim portovima, printerima i drugim komunikacijskim mogućnostima između više čvorova u jednoj mreži. Common Internet File System (CIFS) je naprednija verzija protokola iste namjene.

Samba je besplatan software pod licencom GNU General Public License (GPL), reimplementacija SMB/CIFS mrežnih protokola, koju je razvio [Andrew Tridgell](https://www.samba.org/~tridge/). Sam naziv Samba potječe od naziva protokola SMB nakon umetanja dva vokala.

Samba je paket Unix aplikacija koji je koristan za komunikaciju u mreži u kojoj se nalaze i Windows i Unix sustavi, jer zapravo omogućuje komunikaciju između njih, te samo dijeljenje podataka i informacija. Microsoft Windows sustavi koriste SMB protokol za uspostavu klijent-server mreže, pri čemu Samba omogućava računalima koji se koriste Linux sustavom da zapravo komuniciraju sa Windows sustavima, tako da ih Windows sustavi doživljavaju kao još jedan takav sustav u mreži. Na taj način, omogućava se dijeljenje podataka između dvije vrste sustava, Linux i Windows.

Samba je paket Unix aplikacija koji je koristan za komunikaciju u mreži u kojoj se nalaze i Windows i Unix sustavi, jer zapravo omogućuje komunikaciju između njih, te samo dijeljenje podataka i informacija. Microsoft Windows sustavi koriste SMB protokol za uspostavu klijent-server mreže, pri čemu Samba omogućava računalima koji se koriste Linux sustavom da zapravo komuniciraju sa Windows sustavima, tako da ih Windows sustavi doživljavaju kao još jedan takav sustav u mreži. Na taj način, omogućava se dijeljenje podataka između dvije vrste sustava, Linux i Windows.

Samba sadrži tri daemona koji pružaju dijeljenje resursa svim SMB klijentima u mreži:

- smbd -- bavi se dijeljenjem datoteka i naredbi i pruža autentifikaciju i autorizaciju klijentima
- nmbd -- podrška za Network Basic Input/Output System (NetBIOS) Name Service i Windows Internet Name Service (WINS)
- winbindd -- od verzije 2.2, daje informacije o korisnicima i grupama sa Windows NT servera, te dopušta autorizaciju korisnika

Navest ću i neke od alata koje Samba nudi:

- findsmb -- traži uređaje koji odgovaraju na SMB protokol u lokalnoj mreži u ispisuje njihove informacije
- smbcontrol -- administrativni alat koji šalje poruke prema smbd i nmbd
- smbstatus -- javlja trenutne Samba veze
- testprns -- provjerava jesu li printeri na Samba hostu prepoznati sa strane smbd daemona
- testparm -- provjerava konfiguracijsku datoteku Sambe
- smbclient -- klijent koji se koristi za spajanje na SMB dijeljenja

## Konfiguracija poslužitelja

Instalacija paketa:

``` shell
# dnf install samba
```

Stvara se konfiguracijski smb.conf file koji može sadržavati više dijelova. Primjer jednog takvog bazičnog file-a je:

```
[global]
  workgroup = studenti
[test]
  comment = test
  path = /usr/local/samba/tmp
  read only = no
  guest ok = yes
```

Svi pripadnici grupe studenti mogu koristiti direktorij tmp koji se nalazi na Samba serveru. Pokazat ću kako sad to izgleda u stvarnosti. Na zadnjoj slici vidi se naredba za uređivanje smb.conf file-a, te ću tamo dodati dio koji će omogućivati spajanje. Sami file smb.conf je poprilično velik:

Tu vidimo zadnji dio, koji definira put do direktorija koji dijelimo, komentar te dozvolu za čitanje i pisanje, te za goste bez passworda. Također je potrebno napraviti taj direktorij u samom sustavu, kako ne bismo tu postavili put na nešto što ne postoji. To se može napraviti jednostavno naredbom mkdir. Zatim ću unutra napraviti tekstualnu datoteku `test.txt` koji ću zapravo pokušati podijeliti sa Windowsima:

``` shell
# touch /srv/samba/share/test.txt
```

Za pokretanje Sambe potrebno je aktivirati oba daemona:

``` shell
# systemctl restart smbd.service
# systemctl restart nmbd.service
```

Nakon toga spremni smo na korištenje Sambe, te se korisnici mogu spojiti sa drugih računala. U Windowsima, da bismo vidjeli dijeljenu mapu, potrebno je otići u Computer -> Network i tu vidimo da je prikazano naše računalo, klikom na njega vidimo mapu `share` i u njoj tekstualnu datoteku `test.txt`.

---
author: Renato Kalanj, Vedran Miletić
---

# Mrežni datotečni sustav NFS

Mrežni datotečni sustav je sustav koji služi za pristupanje udaljenim uređajima preko mreže. Ta komunikacija vrši se u obliku klijent-server komunikacije.

Network file system je protokol koji je razvila tvrtka Sun Microsystems 1984. godine. Omogućava klijentu pristup podacima preko mreže kao što se spaja na lokalnu memoriju. Temelji se na Open Network Computing Remote Procedure Call (ONC RPC) sustavu, te podržava mnoge platforme.

Verzije NFS-a:

- V1 je koristio Sun u eksperimentalne svrhe
- V2 je funkcionirao samo preko UDP-a
- V3 ima podršku za TCP, podržava veće datoteke
- V4 je prva verzija razvijena od strane Internet Engineering Task Force (IETF)

Za korištenje NFS-a potrebno je imati pakete (`nfs-kernel-server` na Debianu) instalirane na računalu.

Svaki korisnik može biti i klijent i server i oboje, stoga ću ovdje pokazati kako podesiti i server i klijent stranu.

## Konfiguracija poslužitelja

Potrebna su podešenja tri konfiguracijske datoteke za podešavanje NFS servera:

- `/etc/exports`
- `/etc/hosts.allow`
- `/etc/hosts.deny`

Datoteka `/etc/exports` je u obliku:

```
directory machine1(option11,option12)
```

Directory označava onaj direktorij koji želimo dijeliti. Machine 1 je klijent koji sadržava njegovu DNS ili IP adresu (preporučljivo je korištenje IP adrese). Options su zapravo ovlasti koje klijenti dobivaju, od kojih su najvažnije:

- `ro` -- direktorij je read-only
- `rw` -- klijenti mogu čitati i pisati

Primjer takve datoteke:

```
/export/shared *(rw)
```

/etc/hosts.allow i /etc/hosts.deny specificiraju koja računala na mreži mogu koristiti usloge servera. Svaka linija datoteke sadrži uslugu i popis klijenata koji se mogu njome služiti. Kada klijent pošalje serveru zahtjev, prvo se provjerava postoji li u /etc/hosts.allow pravilo koje tom klijentu dozvoljava pristup, ako da onda mu se dopušta. Ukoliko ne postoji takvo pravilo, provjerava se postoji li pravilo u /etc/hosts.deny kojim mu se odbija pristup, ako da pristup se odbija. Ukoliko ne postoji pravilo niti u jednoj datoteci, pristup je dozvoljen.

S obzirom da se prvo provjerava datoteka sa dozvolama, datoteku sa odbijenim pristupima možemo postaviti da odbija sve:

- portmap:ALL
- lockd:ALL
- mountd:ALL
- rquotad:ALL
- statd:ALL

Zatim oblikujemo datoteku sa dozvolama:

- portmap: ALL
- lockd: ALL
- mountd: ALL
- rquotad: ALL
- statd: ALL

Dodajemo onoliko IP adresa u svaki red koliko ima klijenata kojima želimo dopustiti pristup. Stvaram direktorij koji sam gore naveo da želim dijeliti i startam daemon:

``` shell
# systemctl restart nfs-kernel-server.service
# showmount -e
```

## Konfiguracija klijenta

Za uspostavu klijenta potrebni su portmapper, rpc.statd i rpc.lockd, i moraju biti uključeni i na serveru i na klijentu (paket `nfs-common` na Debianu). Nakon toga, mounta/unmounta se direktorij (koji se nalazi na serveru) naredbama (pri čemu je `/mnt/home` već napravljen).

U datoteci `/etc/fstab` imamo:

```
192.168.32.128:/export/shared /mnt/home
```

Napravit ću direktorij sa `mkdir -p /mnt/home`, te restartat sustav naredbom sudo reboot. Druga opcija je korištenje naredbe mount/unmount.

## Dodatno

NFS je popularan među korisnicima Linuxa, osim gore navedenih načina oblikovanja rada NFS-a, on se dodatno može optimizirati, može se mijenjati brzina transfera i druge opcije. Portmapper sadržava listu u kojoj se nalazi popis usluga koje se vrše na kojim portovima, tako da klijent koji se spaja može vidjeti kojem portu pristupiti s obzirom na uslugu koju traži.

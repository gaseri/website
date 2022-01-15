---
author: Vedran Miletić
---

# Sigurna ljuska i udaljeni rad alatom OpenSSH

[Secure Shell (SSH)](https://en.wikipedia.org/wiki/Secure_Shell) je kriptografski mrežni protokol za sigurno udaljeno pristupanje uslugama operacijskog sustava putem mreže koja se smatra nesigurnom. Može se reći da je takav oblik prijave na operacijski sustav na udaljenom računalu [sličan astralnoj projekciji](https://twitter.com/nixcraft/status/1232380850378948608). Protokol SSH je izvorno dokumentiran u pet glavnih RFC-a:

- [RFC 4250: SSH Assigned Numbers](https://datatracker.ietf.org/doc/html/rfc4250)
- [RFC 4251: SSH Protocol Architecture](https://datatracker.ietf.org/doc/html/rfc4251)
- [RFC 4252: SSH Authentication Protocol](https://datatracker.ietf.org/doc/html/rfc4252)
- [RFC 4253: SSH Transport Layer Protocol](https://datatracker.ietf.org/doc/html/rfc4253)
- [RFC 4254: SSH Connection Protocol (RFC 4254)](https://datatracker.ietf.org/doc/html/rfc4254)

Postoji i [nekolicina drugih RFC-a koji opisuju ekstenzije protokola](https://www.openssh.com/specs.html).

Najpoznatija implementacija protokola SSH je [OpenSSH](https://www.openssh.com/) i ona je otvorenog koda, koristi vrlo snažne kriptografske algoritme i ima [brojne druge dobre značajke](https://www.openssh.com/features.html). U praksi se OpenSSH najčešće koristi na operacijskim sustavima sličnim Unixu za pristupanje ljusci na udaljenom računalu iz ljuske na lokalnom računalu, a od 2018. godine [može se koristiti i na operacijskom sustavu Windows](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_overview). OpenSSH se sastoji od klijenta i poslužitelja, a vrijedi naglasiti da operacijski sustav na kojem se izvodi klijent i operacijski sustav na kojem se izvodi poslužitelj ne moraju biti isti. Primjerice, klijent koji se izvodi na macOS-u ili Linuxu može se povezati na poslužitelj koji se izvodi na FreeBSD-u ili Illumosu.

## Korištenje OpenSSH klijenta

OpenSSH klijent je aplikacija naredbenog retka (naredba `ssh`) dostupna pod većinom operacijskih sustava sličnim Unixu u zadanoj instalaciji. Prvo ćemo provjeriti verziju parametrom `-V`:

``` shell
$ ssh -V
OpenSSH_8.2p1, OpenSSL 1.1.1e  17 Mar 2020
```

!!! warning
    Za verziju koju imamo svakako treba provjeriti [popis poznatih ranjivosti](https://www.openssh.com/security.html). Poznate ranjivosti u OpenSSH-u su vrlo rijetke, no moramo biti svjesni da ih ima i koje su točno prisutne u verziji koju koristimo.

Korištenjem OpenSSH klijenta možemo se povezati na udaljeni poslužitelj na kojem je pokrenut SSH poslužitelj, primjerice `example.group.miletic.net`:

``` shell
$ ssh example.group.miletic.net
The authenticity of host 'example.group.miletic.net (135.181.105.39)' can't be established.
ECDSA key fingerprint is SHA256:0ru7bD+izhNW+qTNFkxqHtDoiyDRNLUHHvvuF0O0I84.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Ovdje ćemo odgovoriti [Yes](https://knowyourmeme.com/memes/yes-chad) (`yes`), odnosno da vjerujemo otisku ključa i da želimo nastaviti povezivanje.

```
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added 'example.group.miletic.net,135.181.105.39' (ECDSA) to the list of known hosts.
korisnik@example.group.miletic.net's password:
```

Naravno, kako bismo se mogli prijaviti na poslužitelj, na njemu moramo imati otvoren korisnički račun tako da se na `example.group.miletic.net` nećemo moći prijaviti. Sama prijava bi nam bila bitna kad bismo na poslužitelju zaista htjeli nešto i raditi, što jest razlog za korištenje OpenSSH-a u realnom svijetu, ali u ovom trenutku samo izučavamo kako proces povezivanja radi i prijava na poslužitelj nam nije toliko važna.

Prekid povezivanja izvodimo, kao i drugdje, kombinacijom tipki Control + C (`^C`). Za ilustraciju kako OpenSSH klijent radi, zamislimo da imamo otvoren korisnički račun imena `student` na `example.group.miletic.net`. Tada ćemo se povezati na način:

``` shell
$ ssh student@example.group.miletic.net
student@example.group.miletic.net's password:
```

Korisničko ime se može navesti i kao vrijednost parametra `-l` [uz obavezni samozadovoljni izraz na licu jer naredba u tom obliku postanje manje čitljiva laicima](https://dilbert.com/strip/1995-06-24):

``` shell
$ ssh -l student example.group.miletic.net
student@example.group.miletic.net's password:
```

SSH poslužitelj se obično izvodi na [TCP ili UDP vratima 22](https://www.iana.org/assignments/service-names-port-numbers). Ako je SSH poslužitelj pokrenut na nekim drugim vratima, parametrom `-p` možemo ih navesti:

``` shell
$ ssh -p 2223 student@example.group.miletic.net
student@example.group.miletic.net's password:
```

Navedemo li vrata na kojima nije pokrenut SSH poslužitelj, klijent će javiti grešku. Primjerice, pokušajmo se spojiti na vrata 443 na kojima je pokrenut HTTPS poslužitelj:

``` shell
$ ssh -p 443 student@example.group.miletic.net
kex_exchange_identification: Connection closed by remote host
```

Parametrom `-v` (kratica od verbose) možemo saznati više detalja o uspješnom ili neuspješnom spajanju:

``` shell
$ ssh -v -p 443 example.group.miletic.net
OpenSSH_8.2p1, OpenSSL 1.1.1e  17 Mar 2020
debug1: Reading configuration data /home/korisnik/.ssh/config
debug1: Reading configuration data /etc/ssh/ssh_config
debug1: Connecting to example.group.miletic.net [135.181.105.39] port 443.
debug1: Connection established.
debug1: identity file /home/korisnik/.ssh/id_rsa type -1
debug1: identity file /home/korisnik/.ssh/id_rsa-cert type -1
debug1: identity file /home/korisnik/.ssh/id_dsa type -1
debug1: identity file /home/korisnik/.ssh/id_dsa-cert type -1
debug1: identity file /home/korisnik/.ssh/id_ecdsa type -1
debug1: identity file /home/korisnik/.ssh/id_ecdsa-cert type -1
debug1: identity file /home/korisnik/.ssh/id_ecdsa_sk type -1
debug1: identity file /home/korisnik/.ssh/id_ecdsa_sk-cert type -1
debug1: identity file /home/korisnik/.ssh/id_ed25519 type -1
debug1: identity file /home/korisnik/.ssh/id_ed25519-cert type -1
debug1: identity file /home/korisnik/.ssh/id_ed25519_sk type -1
debug1: identity file /home/korisnik/.ssh/id_ed25519_sk-cert type -1
debug1: identity file /home/korisnik/.ssh/id_xmss type -1
debug1: identity file /home/korisnik/.ssh/id_xmss-cert type -1
debug1: Local version string SSH-2.0-OpenSSH_8.2
debug1: kex_exchange_identification: banner line 0: HTTP/1.1 400 Bad Request
debug1: kex_exchange_identification: banner line 1: Server: nginx/1.10.3
debug1: kex_exchange_identification: banner line 2: Date: Fri, 03 Apr 2020 18:28:42 GMT
debug1: kex_exchange_identification: banner line 3: Content-Type: text/html
debug1: kex_exchange_identification: banner line 4: Content-Length: 173
debug1: kex_exchange_identification: banner line 5: Connection: close
debug1: kex_exchange_identification: banner line 6:
debug1: kex_exchange_identification: banner line 7: <html>
debug1: kex_exchange_identification: banner line 8: <head><title>400 Bad Request</title></head>
debug1: kex_exchange_identification: banner line 9: <body bgcolor="white">
debug1: kex_exchange_identification: banner line 10: <center><h1>400 Bad Request</h1></center>
debug1: kex_exchange_identification: banner line 11: <hr><center>nginx/1.10.3</center>
debug1: kex_exchange_identification: banner line 12: </body>
debug1: kex_exchange_identification: banner line 13: </html>
kex_exchange_identification: Connection closed by remote host
```

Ovdje vidimo da je HTTP poslužitelj [nginx](https://nginx.org/) odgovorio sa stranicom [HTTP greške](https://http.cat/) 400 (neispravan zahtjev), iz čega možemo zaključiti da se SSH klijent ne može povezati na HTTPS poslužitelj.

Za usporedbu, kod uspješnog spajanja dobivamo nešto duži ispis, navode se algoritmi koji se koriste za različite postupke šifriranja i brojne druge informacije, a naposlijetku i upit za zaporkom kakav smo već vidjeli ranije:

``` shell
$ ssh -v example.group.miletic.net
OpenSSH_8.2p1, OpenSSL 1.1.1e  17 Mar 2020
debug1: Reading configuration data /home/vedranm/.ssh/config
debug1: Reading configuration data /etc/ssh/ssh_config
debug1: Connecting to example.group.miletic.net [135.181.105.39] port 22.
debug1: Connection established.
(...)
debug1: Local version string SSH-2.0-OpenSSH_8.2
debug1: Remote protocol version 2.0, remote software version OpenSSH_7.4p1 Debian-10+deb9u7
debug1: match: OpenSSH_7.4p1 Debian-10+deb9u7 pat OpenSSH_7.0*,OpenSSH_7.1*,OpenSSH_7.2*,OpenSSH_7.3*,OpenSSH_7.4*,OpenSSH_7.5*,OpenSSH_7.6*,OpenSSH_7.7* compat 0x04000002
debug1: Authenticating to example.group.miletic.net:22 as 'korisnik'
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: curve25519-sha256
debug1: kex: host key algorithm: ecdsa-sha2-nistp256
debug1: kex: server->client cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: client->server cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
debug1: Server host key: ecdsa-sha2-nistp256 SHA256:0ru7bD+izhNW+qTNFkxqHtDoiyDRNLUHHvvuF0O0I84
debug1: Host 'example.group.miletic.net' is known and matches the ECDSA host key.
debug1: Found key in /home/korisnik/.ssh/known_hosts:1
(...)
debug1: Next authentication method: password
korisnik@example.group.miletic.net's password:
```

OpenSSH klijent ima još nekoliko parametara koji se mogu koristiti po potrebi, a njihov se opis može pronaći u `man` stranici `ssh(1)`.

### Kopiranje i prijenos datoteka

OpenSSH naredba za sigurno kopiranje datoteka `scp` omogućuje kopiranje datoteka s udaljenog i na udaljeno računalo slično kako to lokalno izvodi naredba `cp`. Detaljne upute za korištenje te naredbe dane su u `man` stranici `scp(1)`.

OpenSSH naredba za sigurni prijenos datoteka `sftp` omogućuje prijenos datoteka s udaljenog i na udaljeno računalo slično kako to bez šifriranja izvodi FTP klijent (naredba `ftp`). Detaljne upute za korištenje te naredbe dane su u `man` stranici `sftp(1)`.

## Pokretanje i konfiguracija OpenSSH poslužitelja

OpenSSH poslužitelj možemo pokrenuti i najčešće pokrećemo kao uslugu operacijskog sustava. Provjeru je li pokrenut na modernijim distribucijama GNU/Linuxa koje koriste [systemd](https://www.freedesktop.org/wiki/Software/systemd/) izvest ćemo naredbom:

``` shell
$ systemctl status sshd.service
ssh.service - OpenBSD Secure Shell server
Loaded: loaded (/lib/systemd/system/ssh.service; enabled; vendor preset: enabled)
Active: active (running) since Fri 2020-04-03 22:43:45 CEST; 15h ago
Docs: man:sshd(8)
man:sshd_config(5)
Process: 1942 ExecStartPre=/usr/sbin/sshd -t (code=exited, status=0/SUCCESS)
Main PID: 2119 (sshd)
Tasks: 1 (limit: 19034)
Memory: 4.8M
CGroup: /system.slice/ssh.service
2119 sshd: /usr/sbin/sshd -D [listener] 0 of 10-100 startups

tra 01 22:43:39 hephaestus systemd[1]: Starting OpenBSD Secure Shell server...
tra 01 22:43:45 hephaestus sshd[2119]: Server listening on 0.0.0.0 port 22.
tra 01 22:43:45 hephaestus systemd[1]: Started OpenBSD Secure Shell server.
tra 01 22:43:45 hephaestus sshd[2119]: Server listening on :: port 22.
tra 01 22:44:03 hephaestus sshd[2525]: Accepted password for vedranm from 192.168.1.14 port 39428 ssh2
tra 01 22:44:03 hephaestus sshd[2525]: pam_unix(sshd:session): session opened for user vedranm by (uid=0)
```

Za eksperimentiranje i učenje kakvo nas ovdje zanima možemo OpenSSH poslužitelj pokrenuti i neovisno o već pokrenutoj usluzi operacijskog sustava naredbom `sshd`:

``` shell
$ sshd
sshd re-exec requires execution with an absolute path
```

Uočavamo da OpenSSH ima dodatnu sigurnosnu mjeru gdje očekuje da navedete čitavu putanju da točno znate što pokrećete. Naime, napadač može na nekom od mjesta gdje operacijski sustav traži naredbe (vrijednost varijable ljuske `PATH`) staviti vlastiti program i nazvati ga `sshd` ili pak instalirati njegovu stariju, nesigurnu verziju s poznatim napadima. Navedimo apsolutnu putanju:

``` shell
$ /usr/sbin/sshd
sshd: no hostkeys available -- exiting.
```

OpenSSH poslužitelj ne možemo pokrenuti bez prethodnog stvaranja potrebnih ključeva.

### Upravljanje domaćinskim ključevima

OpenSSH koristi [četiri algoritma javnog ključa](https://security.stackexchange.com/a/178986) i to su redom:

- [RSA](https://en.wikipedia.org/wiki/RSA_(cryptosystem)),
- [DSA](https://en.wikipedia.org/wiki/Digital_Signature_Algorithm) ([od verzije 7.0 ih se smatra preslabima](https://www.openssh.com/legacy.html)),
- [ECDSA](https://en.wikipedia.org/wiki/Elliptic_Curve_Digital_Signature_Algorithm),
- [Ed25519](https://ed25519.cr.yp.to/) ([Edwards-curve Digital Signature Algorithm (EdDSA)](https://en.wikipedia.org/wiki/EdDSA)).

OpenSSH alat `ssh-keygen` služi za generiranje domaćinskih (poslužiteljskih) i korisničkih (klijentskih) ključeva; bez parametara `ssh-keygen` će generirati korisničke ključeve. Pozabavimo se prvo generiranje domaćinskih ključeva, što ćemo napraviti dodavanjem parametra `-A`:

``` shell
$ ssh-keygen -A
ssh-keygen: generating new host keys: RSA Could not save your public key in /etc/ssh/ssh_host_rsa_key.mYJvrepuWR: Permission denied
ssh-keygen: generating new host keys: DSA Could not save your public key in /etc/ssh/ssh_host_dsa_key.zG3QFGy1zd: Permission denied
ssh-keygen: generating new host keys: ECDSA Could not save your public key in /etc/ssh/ssh_host_ecdsa_key.JG7rzkXpNm: Permission denied
ssh-keygen: generating new host keys: ED25519 Could not save your public key in /etc/ssh/ssh_host_ed25519_key.G4BsHFctX2: Permission denie
```

Uočimo da bez administrativnih privilegija ne možemo spremiti generirane ključeve. Iskoristimo zato parametar `-f` da postavio ključeve u direktorij po želji koji ćemo prethodno stvoriti:

``` shell
$ mkdir -p moj-ssh-server/etc/ssh
$ ssh-keygen -A -f moj-ssh-server
ssh-keygen: generating new host keys: RSA DSA ECDSA ED25519
```

Ova naredba ima brojne druge parametara koji se mogu koristiti po potrebi, a njihov se opis može pronaći u `man` stranici `ssh-keygen(1)`. Dio njih ćemo iskoristiti kasnije kod generiranja korisničkih ključeva.

Generirani ključevi nalaze se u direktoriju `moj-ssh-server/etc/ssh` (parametar `-f` domaćinske ključeve automatski smješta u poddirektorij `etc/ssh` unutar danog direktorija). Stvorit ćemo konfiguracijsku datoteku OpenSSH poslužitelja i za nju nam treba apsolutna putanja do ključeva, koju možemo dohvatiti naredbom `realpath`:

``` shell
$ realpath moj-ssh-server/etc/ssh
/home/korisnik/moj-ssh-server/etc/ssh
$ ls -1 moj-ssh-server/etc/ssh
ssh_host_dsa_key
ssh_host_dsa_key.pub
ssh_host_ecdsa_key
ssh_host_ecdsa_key.pub
ssh_host_ed25519_key
ssh_host_ed25519_key.pub
ssh_host_rsa_key
ssh_host_rsa_key.pub
```

### Konfiguracija SSH poslužitelja

Stvorimo konfiguracijsku datoteku OpenSSH poslužitelja imena `sshd_config`:

``` shell
$ touch moj-ssh-server/etc/ssh/sshd_config
```

U proizvoljnom uređivaču teksta uredimo datoteku `moj-ssh-server/etc/ssh/sshd_config` tako da popišemo ključeve kao parametre konfiguracijske naredbe `HostKey` (DSA ključeve ne navodimo jer se smatraju nesigurnima):

```
HostKey /home/korisnik/moj-ssh-server/etc/ssh/ssh_host_rsa_key
HostKey /home/korisnik/moj-ssh-server/etc/ssh/ssh_host_ecdsa_key
HostKey /home/korisnik/moj-ssh-server/etc/ssh/ssh_host_ed25519_key
```

Pokrenemo li sad `sshd` s parametrom `-f` u kojem navodimo ovu konfiguracijsku datoteku i s parametrom `-d` (debug mode) koji uključuje ispis poruka o radu OpenSSH poslužitelja, dobit ćemo izlaz oblika:

``` shell
$ /usr/sbin/sshd -f moj-ssh-server/etc/ssh/sshd_config -d
debug1: sshd version OpenSSH_8.2p1, OpenSSL 1.1.1e  17 Mar 2020
debug1: private host key #0: ssh-rsa SHA256:+TP+gmo09NPfA5gVKyktIC30nDoCcrejnbo8G8Cp5Nk
debug1: private host key #1: ecdsa-sha2-nistp256 SHA256:fVro3huxlPLu9BgLgeDvQo2C7rVnvqc69E2dLsuAnRg
debug1: private host key #2: ssh-ed25519 SHA256:Bcz5mbVmDY0wl9kD2NARr2MbuGFxa4pRwCCOC2FQlYc
debug1: setgroups() failed: Operation not permitted
debug1: rexec_argv[0]='/usr/sbin/sshd'
debug1: rexec_argv[1]='-f'
debug1: rexec_argv[2]='moj-ssh-server/etc/ssh/sshd_config'
debug1: rexec_argv[3]='-d'
debug1: Set /proc/self/oom_score_adj from 0 to -1000
debug1: Bind to port 22 on 0.0.0.0.
Bind to port 22 on 0.0.0.0 failed: Permission denied.
debug1: Bind to port 22 on ::.
Bind to port 22 on :: failed: Permission denied.
Cannot bind any address.
```

!!! warning
    Ukoliko privatni SSH ključevi imaju dozvolu čitanja od strane grupe i ostalih korisnika, `sshd` će kod pokretanja upozoriti na to i odbiti pokrenuti poslužitelj:

    ```
    debug1: sshd version OpenSSH_8.2, OpenSSL 1.1.1f  31 Mar 2020
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @         WARNING: UNPROTECTED PRIVATE KEY FILE!          @
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Permissions 0777 for '/mnt/c/Users/Korisnik/moj-ssh-server/etc/ssh/ssh_host_rsa_key' are too open.
    It is required that your private key files are NOT accessible by others.
    This private key will be ignored.
    Unable to load host key "/mnt/c/Users/Korisnik/moj-ssh-server/etc/ssh/ssh_host_rsa_key": bad permissions
    Unable to load host key: /mnt/c/Users/Korisnik/moj-ssh-server/etc/ssh/ssh_host_rsa_key
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @         WARNING: UNPROTECTED PRIVATE KEY FILE!          @
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Permissions 0777 for '/mnt/c/Users/Korisnik/moj-ssh-server/etc/ssh/ssh_host_ecdsa_key' are too open.
    It is required that your private key files are NOT accessible by others.
    This private key will be ignored.
    Unable to load host key "/mnt/c/Users/Korisnik/moj-ssh-server/etc/ssh/ssh_host_ecdsa_key": bad permissions
    Unable to load host key: /mnt/c/Users/Korisnik/moj-ssh-server/etc/ssh/ssh_host_ecdsa_key
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @         WARNING: UNPROTECTED PRIVATE KEY FILE!          @
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Permissions 0777 for '/mnt/c/Users/Korisnik/moj-ssh-server/etc/ssh/ssh_host_ed25519_key' are too open.
    It is required that your private key files are NOT accessible by others.
    This private key will be ignored.
    Unable to load host key "/mnt/c/Users/Korisnik/moj-ssh-server/etc/ssh/ssh_host_ed25519_key": bad permissions
    Unable to load host key: /mnt/c/Users/Korisnik/moj-ssh-server/etc/ssh/ssh_host_ed25519_key
    sshd: no hostkeys available -- exiting.
    ```

    Rješenje je maknuti dozvole grupi i ostalima:

    ```
    chmod go-rwx moj-ssh-server/etc/ssh/*_key
    ```

    U konkretnom slučaju se koristi WSL i navedeno neće biti dovoljno jer datotečni sustav direktorija `/mnt/c/Users/Korisnik` koji je NTFS ne podržava upravljanje dozvolama. Rješenje je ključeve stvoriti u direktoriju `/home/korisnik` čiji datotečni sustav podržava upravljanje dozvolama.

Vidimo da ne možemo koristiti vrata 22, koja su zadana vrata SSH poslužitelja. Iskoristimo neka od vrata čiji je broj veći od 1024 i koja možemo koristiti po želji. Uredimo konfiguracijsku datoteku `moj-ssh-server/etc/ssh/sshd_config` tako da joj pored već navedenih ključeva dodamo i konfiguracijsku naredbu `Port`:

```
Port 4022
```

Želimo konfigurirati takav poslužitelj da se na njega možemo prijaviti kako bismo ga mogli isprobati od početka do kraja. Pritom ćemo se prijavljivati s vlastitim korisničkim imenom i zaporkom na lokalnom računalu na kojem radimo (prazne zaporke nisu dozvoljene u zadanim postavkama; ako nemate postavljenu zaporku, postavite je naredbom `passwd`). Samo korijenski korisnik može čitati datoteku sa zaporkama korisnika na sustavu. Kako bismo se mogli prijaviti s vlastitim korisničkim imenom i zaporkom na računalu na kojem radimo kad je SSH poslužitelj pokrenut od strane običnog korisnika, moramo uključiti podršku za podsustav [pluggable authentication modules (PAM)](https://en.wikipedia.org/wiki/Pluggable_authentication_module), dostupan i običnim korisnicima konfiguracijskom naredbom:

```
UsePAM yes
```

Sad ćemo uspjeti pokrenuti `sshd`:

``` shell
$ /usr/sbin/sshd -f moj-ssh-server/etc/ssh/sshd_config -d
debug1: sshd version OpenSSH_8.2p1, OpenSSL 1.1.1e  17 Mar 2020
debug1: private host key #0: ssh-rsa SHA256:+TP+gmo09NPfA5gVKyktIC30nDoCcrejnbo8G8Cp5Nk
debug1: private host key #1: ecdsa-sha2-nistp256 SHA256:fVro3huxlPLu9BgLgeDvQo2C7rVnvqc69E2dLsuAnRg
debug1: private host key #2: ssh-ed25519 SHA256:Bcz5mbVmDY0wl9kD2NARr2MbuGFxa4pRwCCOC2FQlYc
debug1: setgroups() failed: Operation not permitted
debug1: rexec_argv[0]='/usr/sbin/sshd'
debug1: rexec_argv[1]='-f'
debug1: rexec_argv[2]='moj-ssh-server/etc/ssh/sshd_config'
debug1: rexec_argv[3]='-d'
debug1: Set /proc/self/oom_score_adj from 0 to -1000
debug1: Bind to port 4022 on 0.0.0.0.
Server listening on 0.0.0.0 port 4022.
debug1: Bind to port 4022 on ::.
Server listening on :: port 4022.
```

Kako je `sshd` [daemon](https://en.wikipedia.org/wiki/Daemon_(computing)) (zbog toga ime naredbe i završava slovom `d`), on će ostati pokrenut i pružati uslugu SSH poslužitelja sve dok ga ne prekinemo kombinacijom tipki Control + C. Ostavimo ga pokrenutog do daljnjega.

Prijavimo se iz drugog terminala korištenjem OpenSSH klijenta naredbom:

``` shell
$ ssh -p 4022 localhost
Password:
Environment:
  USER=korisnik
  LOGNAME=korisnik
  HOME=/home/korisnik
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
  SHELL=/bin/bash
  TERM=xterm-256color
  MOTD_SHOWN=pam
  LANG=hr_HR.UTF-8
  SSH_CLIENT=127.0.0.1 57172 4022
  SSH_CONNECTION=127.0.0.1 57172 127.0.0.1 4022
  SSH_TTY=/dev/pts/3
```

Sada možemo koristiti naredbe po želji, a odjavu vršimo naredbom `logout`:

``` shell
$ logout
Connection to localhost closed.
```

!!! warning
    Nakon odjave klijenta OpenSSH poslužitelj će završiti izvođenje s greškom oblika:

    ```
    debug1: do_cleanup
    debug1: temporarily_use_uid: 1000/1000 (e=1000/1000)
    debug1: restore_uid: (unprivileged)
    debug1: do_cleanup
    debug1: PAM: cleanup
    debug1: PAM: closing session
    debug1: PAM: deleting credentials
    debug1: temporarily_use_uid: 1000/1000 (e=1000/1000)
    debug1: restore_uid: (unprivileged)
    debug1: audit_event: unhandled event 12
    ```

    To je nešto što se neće događati u primijeni u realnom svijetu (gdje će OpenSSH poslužitelj raditi pod korijenskim korisnikom), a za potrebe ove vježbe trebat će ponovno pokrenuti poslužitelj nakon svake odjave kako bi buduće prijave bile moguće. Ranije verzije OpenSSH poslužitelja podržavale su ovaj način rada korištenjem konfiguracijske naredbe `UsePrivilegeSeparation no`, ali [od verzije 7.5 ta se konfiguracijska naredba smatra zastarjelom](https://www.openssh.com/releasenotes.html#7.5).

U `sshd_config` možemo dodati još brojne druge konfiguracijske naredbe. Neke od naredbi su:

- `AcceptEnv`, koja navodi koje varijable ljuske će biti prenesene iz lokalne u udaljenu sesiju
- `Banner`, koja sadržaj datoteke navedene u svom argumentu šalje korisniku prije autentifikacije
- `Ciphers`, koja postavlja dostupne algoritme za šifriranje (popis svih podržanih može se dobiti naredbom `ssh -Q cipher`)
- `HostKeyAlgorithms`, koja postavlja algoritme javnog ključa korištene za ključeve domaćina (popis svih podržanih može se dobiti narebom `ssh -Q HostKeyAlgorithms`)
- `MACs`, koja postavlja dostupne algoritme za provjeru autentičnosti poruka (popis svih podržanih može se dobiti naredbom `ssh -Q mac`)
- `PasswordAuthentication`, koja definira mogu li se za prijavu koristiti zaporke
- `PermitEmptyPasswords`, koja omogućava da zaporke korisnika budu prazan niz znakova
- `PermitRootLogin`, koja regulira smije li se korijenski korisnik prijaviti na OpenSSH poslužitelj
- `PubkeyAuthentication`, koja definira mogu li se za prijavu koristiti (javni) ključevi

Ovo su zaista tek neke od naredbi. Čitav popis podržanih konfiguracijskih naredbi može se pronaći u `man` stranici `sshd_config(5)`.

## Načini prijave

Osim prijave pomoću zaporki, moguće je koristiti i korisničke ključeve. Već spomenutom naredbom `ssh-keygen` to ćemo napraviti na način:

``` shell
$ ssh-keygen
Generating public/private rsa key pair.
Enter file in which to save the key (/home/korisnik/.ssh/id_rsa):
Created directory '/home/korisnik/.ssh'.
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/korisnik/.ssh/id_rsa
Your public key has been saved in /home/korisnik/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:3x+D9GXDyoZ2mJ9J4hzGwTCDHkl5oZX7VxVXmt1+Adw korisnik@hephaestus
The key's randomart image is:
+---[RSA 3072]----+
|       ..oo ....=|
|      ..=o   ..E+|
|       =.+.   o.+|
|      . ..=   .o.|
|       .S .o. .o=|
|         ..+*+.oo|
|          .X+*+  |
|          = B.oo |
|           o +.  |
+----[SHA256]-----+
```

Zadana vrsta ključa je RSA, a druge vrstu ključa možemo odabrati parametrom `-t`. Specijalno, ECDSA i Ed25519 ključeve stvoriti naredbama:

``` shell
$ ssh-keygen -t ecdsa
Generating public/private ecdsa key pair.
Enter file in which to save the key (/home/korisnik/.ssh/id_ecdsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/korisnik/.ssh/id_ecdsa
Your public key has been saved in /home/korisnik/.ssh/id_ecdsa.pub
The key fingerprint is:
SHA256:RxwLKEToEpbyclxu7kQQQ29RhfKCTOQkbJ13VR/G0uU korisnik@hephaestus
The key's randomart image is:
+---[ECDSA 256]---+
|ooO==..++.o.oo.. |
|oO+=+oo. o +ooo  |
|+B.=++.   + .. E |
|o B.= .  .       |
| + + .  S .      |
|    o    .       |
|   o             |
|    .            |
|                 |
+----[SHA256]-----+

$ ssh-keygen -t ed25519
Generating public/private ed25519 key pair.
Enter file in which to save the key (/home/korisnik/.ssh/id_ed25519):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/korisnik/.ssh/id_ed25519
Your public key has been saved in /home/korisnik/.ssh/id_ed25519.pub
The key fingerprint is:
SHA256:V6l5XYIJoojLwDT8pHPiFjQnSNXRovnFtI16wSZr15I korisnik@hephaestus
The key's randomart image is:
+--[ED25519 256]--+
|++....o . .      |
|+=.+ + + . . +   |
|o.O + * +   = . .|
| B * . O . + . o |
|. B . * S + . .  |
| o   = E o .     |
|.   . o .        |
|                 |
|                 |
+----[SHA256]-----+
```

Provjerimo generirane ključeve:

``` shell
$ ls -1 .ssh/
id_ecdsa
id_ecdsa.pub
id_ed25519
id_ed25519.pub
id_rsa
id_rsa.pub
known_hosts
```

Tajni ključevi su slično zapisani kao što smo navikli kod OpenSSL-a:

``` shell
$ cat .ssh/id_ecdsa
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAaAAAABNlY2RzYS
1zaGEyLW5pc3RwMjU2AAAACG5pc3RwMjU2AAAAQQTXtLTu++vPBysDp1ysheqBrBSUWKzl
CDZJIYehFBMCMYUFgPxiGg+x/UqdfM/R6OdKItNm6DqAxAyTjJ0Kah+qAAAAqNdovdbXaL
3WAAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBNe0tO77688HKwOn
XKyF6oGsFJRYrOUINkkhh6EUEwIxhQWA/GIaD7H9Sp18z9Ho50oi02boOoDEDJOMnQpqH6
oAAAAgXhPxbYVqp48SncI95Pq3bPoJw4o8vy/zNBiChOVe8xMAAAAPdmVkcmFubUB3aWxs
ZXJzAQ==
-----END OPENSSH PRIVATE KEY-----
```

dok su javni ključevi nešto drugačijeg zapisa gdje se redom odvojeni razmakom navode tip, sam ključ kodiran u zapisu Base64 i njegov naziv:

``` shell
$ cat .ssh/id_ecdsa.pub
ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBNe0tO77688HKwOnXKyF6oGsFJRYrOUINkkhh6EUEwIxhQWA/GIaD7H9Sp18z9Ho50oi02boOoDEDJOMnQpqH6o= vedranmiletic@example.group.miletic.net
```

Kako bismo omogućili prijavu korištenjem nekog para ključeva umjesto zaporke, dodat ćemo njegov javni ključ u `.ssh/authorized_keys`, primjerice za gornji ECDSA ključ to možemo učiniti ručnim kopiranjem i ljepljenjem u nekom uređivaču teksta ili na način:

``` shell
$ touch .ssh/authorized_keys
$ chmod go-rwx .ssh/authorized_keys
$ cat .ssh/id_ecdsa.pub >> .ssh/authorized_keys
```

Uočite da smo ograničili čitanje datoteke s autoriziranim ključevima samo na korisnika koji je vlasnik datoteke. Nakon ovog koraka prijava će proći bez unošenja zaporke:

``` shell
$ ssh -p 4022 localhost
Environment:
(...)
```

Provjerimo li popis poruka na poslužiteljskoj strani, vidjet ćemo među njima poruke oblika:

```
debug1: trying public key file /home/korisnik/.ssh/authorized_keys
debug1: fd 4 clearing O_NONBLOCK
debug1: /home/korisnik/.ssh/authorized_keys:1: matching key found: ECDSA SHA256:RxwLKEToEpbyclxu7kQQQ29RhfKCTOQkbJ13VR/G0uU
debug1: /home/korisnik/.ssh/authorized_keys:1: key options: agent-forwarding port-forwarding pty user-rc x11-forwarding
Accepted key ECDSA SHA256:RxwLKEToEpbyclxu7kQQQ29RhfKCTOQkbJ13VR/G0uU found at /home/korisnik/.ssh/authorized_keys:1
```

Vrijedi spomenuti da se za istu svrhu može koristiti i naredba `ssh-copy-id` koja je detaljnije opisana u `man` stranici `ssh-copy-id(1)`.

### Korištenje vanjske autentifikacije

!!! todo
    Ovdje treba objasniti kako se koriste druge vrste autentifikacije osim zaporki i ključeva (npr. LDAP).

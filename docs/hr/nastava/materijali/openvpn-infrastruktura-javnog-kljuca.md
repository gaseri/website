---
author: Manuel Maraš, Vedran Miletić
---

# Korištenje infrastrukture javnog ključa u postavljanju virtualne privatne mreže alatom OpenVPN

U prethodnom smo se dijelu bavili postavljanjem OpenVPN-a u načinu rada točka-do-točke s jednim poslužiteljem i jednim klijentom koji šifriraju podatke u međusobnoj komunikaciji korištenjem jednog ili dva statička ključa. Postoji i drugi način rada koji je složeniji za postavljanje jer zahtijeva korištenje infrastrukture javnog ključa, ali podržava povezivanje više klijenata na poslužitelj i opcionalno omogućuje međusobnu komunikaciju klijenata. Kako je povezivanje više klijenata potrebno kod gotovo svake primjene virtualne privatne mreže, taj nam je način rada u praksi puno korisniji pa se njime bavimo u nastavku. Detaljno ćemo opisati proces stvaranja autoriteta certifikata za virtualnu privatnu mrežu i potpisivanje pojedinih certifikata od strane autoriteta certifikata. Poznavanje tog procesa će nam dobro doći i van virtualnih privatnih mreža jer se s certifikatima na sličan način radi i kod web poslužitelja i klijenata kad se koristi HTTPS.

!!! note
    Ovaj dio je složen prema članku [Setting up your own Certificate Authority (CA)](https://openvpn.net/community-resources/setting-up-your-own-certificate-authority-ca/) koji je dio [službene dokumentacije OpenVPN-a namijenjene za zajednicu](https://openvpn.net/community-resources/). Cjelovite upute su dane u članku [2x HOW TO](https://openvpn.net/community-resources/how-to/).

[Easy-RSA](https://openvpn.net/community-resources/rsa-key-management/) ([službena dokumentacija](https://easy-rsa.readthedocs.io/en/latest/)) je alat za upravljanje infrastrukturom javnog ključa koja slijedi [ITU-T standard X.509](https://en.wikipedia.org/wiki/X.509). [Infrastruktura javnog ključa koju ćemo koristiti u nastavku](https://easy-rsa.readthedocs.io/en/latest/intro-to-PKI/) sastoji se od:

- autoriteta certifikata (engl. *certificate authority*, kraće CA), odnosno korijenskog certifikata i pripadnog privatnog ključa koji su temelj infrastrukture javnih ključeva
- certifikata koje je autoritet certifikata potpisao, koji sadrže javni ključ, metapodatke koji opisuju certifikat i digitalni potpis od strane privatnog ključa autoriteta certifikata
- zahtjeva za potpisom, koji sadrže javni ključ, metapodatke koji opisuju certifikat i digitalni potpis od strane vlastitog privatnog ključa
- para ključeva, koji se sastoji od javnog i tajnog ključa

!!! warning
    U nastavku radimo s Easy-RSA 3.0 i novijim. Easy-RSA 2.0 ima isti način rada, ali umjesto jedne naredbe `easyrsa` i nekolicine podnaredbi (npr. `easyrsa build-ca` i `easyrsa gen-dh`), ima skup odvojenih naredbi koje često imaju i druga imena (npr. `build-ca` ima isto ime, ali `build-dh` nema).

Uvjerimo se da nam Easy-RSA radi i provjerimo koje naredbe ima:

``` shell
$ easyrsa help

Easy-RSA 3 usage and overview

USAGE: easyrsa [options] COMMAND [command-options]

A list of commands is shown below. To get detailed usage and help for a
command, run:
  easyrsa help COMMAND

For a listing of options that can be supplied before the command, use:
  easyrsa help options

Here is the list of commands available with a short syntax reminder. Use the
'help' command above to get full usage details.

  init-pki
  build-ca [ cmd-opts ]
  gen-dh
  gen-req <filename_base> [ cmd-opts ]
  sign-req <type> <filename_base>
  build-client-full <filename_base> [ cmd-opts ]
  build-server-full <filename_base> [ cmd-opts ]
  revoke <filename_base> [cmd-opts]
  renew <filename_base> [cmd-opts]
  build-serverClient-full <filename_base> [ cmd-opts ]
  gen-crl
  update-db
  show-req <filename_base> [ cmd-opts ]
  show-cert <filename_base> [ cmd-opts ]
  show-ca [ cmd-opts ]
  import-req <request_file_path> <short_basename>
  export-p7 <filename_base> [ cmd-opts ]
  export-p12 <filename_base> [ cmd-opts ]
  set-rsa-pass <filename_base> [ cmd-opts ]
  set-ec-pass <filename_base> [ cmd-opts ]
  upgrade <type>

DIRECTORY STATUS (commands would take effect on these locations)
  EASYRSA: /home/vedranm
      PKI: /home/vedranm/pki
```

Na nekim distribucijama Linuxa naredba `easyrsa` nije ni u jednom od direktorija koji su navedeni u varijabli okoline `PATH` te je potrebno varijablu dopuniti odgovarajućim direktorijem:

- na Ubuntuu naredbom `export PATH=/usr/share/easy-rsa:$PATH`
- na Fedori naredbom `export PATH=/usr/share/easy-rsa/3:$PATH`

Na Arch Linuxu i derivatima (npr. Manjaro i Garuda Linux) se naredba `easyrsa` nakon instalacije [easy-rsa](https://archlinux.org/packages/extra/any/easy-rsa/) nalazi u `/usr/bin` te je odmah dostupna za korištenje.

Na FreeBSD-u se naredba `easyrsa` nakon instalacije [security/easy-rsa](https://www.freshports.org/security/easy-rsa/) nalazi u `/usr/local/bin` te je odmah dostupna za korištenje.

## Osnovna struktura infrastrukture javnog ključa i autoritet certifikata

Izradu certifikata i ključeva počinjemo naredbom `init-pki` koja će stvoriti potrebne datoteke direktorije:

``` shell
$ easyrsa init-pki

init-pki complete; you may now create a CA or requests.
Your newly created PKI dir is: /home/vedranm/pki
```

Unutar upravo stvorenog direktorija `pki` nalaze se prazni direktoriji `private` i `reqs` te datoteka `openssl-easyrsa.cnf` s konfiguracijom [OpenSSL](https://www.openssl.org/)-a, koji Easy-RSA koristi za stvaranje ključeva, stvaranje zahtjeva za potpisivanjem certifikata i potpisivanje certifikata. Teoretski bismo mogli sve što radimo u nastavku napraviti i ručno pokrećući `openssl` više puta s različitim parametrima, ali to je vrlo nepraktično pa baš zbog toga postoji Easy-RSA koji to radi za nas.

Kako se ne bi morali baviti metapodacima certifikata koje OpenSSL stvara, možemo koristiti datoteku `vars`. Primjer te datoteke na nekim distribucijama Linuxa postoji kao `vars.example` u `/usr/share/easy-rsa`, a mi ćemo je sami stvoriti neovisno o tom primjeru. Iz `vars.example` za nas su relevantne samo linije:

```
#set_var EASYRSA_REQ_COUNTRY    "US"
#set_var EASYRSA_REQ_PROVINCE   "California"
#set_var EASYRSA_REQ_CITY       "San Francisco"
#set_var EASYRSA_REQ_ORG        "Copyleft Certificate Co"
#set_var EASYRSA_REQ_EMAIL      "me@example.net"
#set_var EASYRSA_REQ_OU         "My Organizational Unit"
#set_var EASYRSA_REQ_CN         "ChangeMe"
#set_var EASYRSA_KEY_SIZE       2048
```

Prepoznajemo metapodatke X.509 certifikata i postavku veličine ključa. Odkomentirati ćemo linije i postavit ćemo vrijednosti. U direktoriju `pki` stvorimo datoteku `vars` sadržaja:

```
set_var EASYRSA_REQ_COUNTRY    "HR"
set_var EASYRSA_REQ_PROVINCE   "Primorje-Gorski Kotar County"
set_var EASYRSA_REQ_CITY       "Rijeka"
set_var EASYRSA_REQ_ORG        "Odjel za informatiku Sveucilista u Rijeci"
set_var EASYRSA_REQ_EMAIL      "info@primjer-vpn.rm.miletic.net"
set_var EASYRSA_REQ_OU         "Laboratorij za racunalne mreze"
set_var EASYRSA_REQ_CN         "primjer-vpn.rm.miletic.net"
set_var EASYRSA_KEY_SIZE       2048
```

Uočimo da smo postavili CN kao da se radi o domeni, što je dakako opcionalno. Veličinu ključa smo ostavili na zadanih 2048 bita.

Izgradimo sad autoritet certifikata naredbom `build-ca` i pritom zaporku postavimo po želji:

``` shell
$ easyrsa build-ca

Note: using Easy-RSA configuration from: /home/vedranm/pki/vars

Using SSL: openssl OpenSSL 1.1.1f  31 Mar 2020

Enter New CA Key Passphrase:
Re-Enter New CA Key Passphrase:
Generating RSA private key, 2048 bit long modulus (2 primes)
.................................................................................................................................................................++++
................................................................................................................................................................................................................................................................................++++
e is 65537 (0x010001)
Can't load /home/vedranm/pki/.rnd into RNG
139911430903104:error:2406F079:random number generator:RAND_load_file:Cannot open file:../crypto/rand/randfile.c:98:Filename=/home/vedranm/pki/.rnd
You are about to be asked to enter information that will be incorporated
into your certificate request.
What you are about to enter is what is called a Distinguished Name or a DN.
There are quite a few fields but you can leave some blank
For some fields there will be a default value,
If you enter '.', the field will be left blank.
-----
Common Name (eg: your user, host, or server name) [Easy-RSA CA]:

CA creation complete and you may now import and sign cert requests.
Your new CA certificate file for publishing is at:
/home/vedranm/pki/ca.crt
```

Pogledajmo što smo upravo stvorili:

``` shell
$ ls -1 pki pki/private
pki:
ca.crt
certs_by_serial
index.txt
issued
openssl-easyrsa.cnf
private
renewed
reqs
revoked
safessl-easyrsa.cnf
serial
vars

pki/private:
ca.key
```

U direktoriju `pki` vidimo certifikat `ca.crt`, a u direktoriju `pki/private` njegov pripadni privatni ključ. Naredbom `show-ca` možemo pregledati stvoreni certifikat:

``` shell
$ easyrsa show-ca

Note: using Easy-RSA configuration from: /home/vedranm/pki/vars

Using SSL: openssl OpenSSL 1.1.1f  31 Mar 2020

Showing  details for 'ca'.
This file is stored at:
/home/vedranm/pki/ca.crt

Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number:
            15:4b:69:f0:48:dc:fa:60:30:12:19:6d:c1:88:de:de:c4:72:1f:65
        Signature Algorithm: sha256WithRSAEncryption
        Issuer:
            commonName                = Easy-RSA CA
        Validity
            Not Before: May 28 14:18:27 2020 GMT
            Not After : May 26 14:18:27 2030 GMT
        Subject:
            commonName                = Easy-RSA CA
        X509v3 extensions:
            X509v3 Subject Key Identifier:
                B0:9B:24:08:5B:A6:08:6A:B8:57:2A:54:EE:40:08:F2:7A:66:82:84
            X509v3 Authority Key Identifier:
                keyid:B0:9B:24:08:5B:A6:08:6A:B8:57:2A:54:EE:40:08:F2:7A:66:82:84
                DirName:/CN=Easy-RSA CA
                serial:15:4B:69:F0:48:DC:FA:60:30:12:19:6D:C1:88:DE:DE:C4:72:1F:65

            X509v3 Basic Constraints:
                CA:TRUE
            X509v3 Key Usage:
                Certificate Sign, CRL Sign
```

Svako iduće pokretanje naredbe `build-ca` prvo će provjeriti imamo li već izgrađen autoritet certifikata i neće prepisati ključeve i ceritifikat novim:

``` shell
$ easyrsa build-ca

Note: using Easy-RSA configuration from: /home/vedranm/pki/vars

Using SSL: openssl OpenSSL 1.1.1f  31 Mar 2020

Easy-RSA error:

Unable to create a CA as you already seem to have one set up.
If you intended to start a new CA, run init-pki first.
```

Sad smo napravili autoritet certifikata. U stvarnosti ovaj autoritet certifikata koji će potpisivati serverski i klijentske certifikate može (ali ne mora) biti na odvojenom računalu od onoga na kojem će raditi OpenVPN poslužitelj.

## Poslužiteljski certifikat

Poslužiteljski certifikat generiramo naredbom `build-server-full`. Proučimo kako se ona koristi naredbom `help`:

``` shell
$ easyrsa help build-server-full

Note: using Easy-RSA configuration from: /home/vedranm/pki/vars

  build-client-full <filename_base> [ cmd-opts ]
  build-server-full <filename_base> [ cmd-opts ]
  build-serverClient-full <filename_base> [ cmd-opts ]
      Generate a keypair and sign locally for a client and/or server

  This mode uses the <filename_base> as the X509 CN.

  cmd-opts is an optional set of command options from this list:

        nopass  - do not encrypt the private key (default is encrypted)
```

Uočimo opciju `nopass` koja će nam pomoći da kasnije ne moramo unositi zaporku ključa kod njegovog korištenja. Nazovimo naš poslužitelj jednostavno `mojposluzitelj`; za potpisivanje njegovog certifikata od strane autoriteta certifikata bit će potrebno unijeti zaporku autoriteta certifikata:

``` shell
$ easyrsa build-server-full mojposluzitelj nopass

Note: using Easy-RSA configuration from: /home/vedranm/pki/vars

Using SSL: openssl OpenSSL 1.1.1f  31 Mar 2020
Generating a RSA private key
........................................................................................++++
.....................................++++
writing new private key to '/home/vedranm/pki/private/mojposluzitelj.key.Q9pCTpHqNe'
-----
Using configuration from /home/vedranm/pki/safessl-easyrsa.cnf
Enter pass phrase for /home/vedranm/pki/private/ca.key:
Check that the request matches the signature
Signature ok
The Subject's Distinguished Name is as follows
commonName            :ASN.1 12:'mojposluzitelj'
Certificate is to be certified until May 13 14:53:12 2023 GMT (1080 days)

Write out database with 1 new entries
Data Base Updated
```

Dobili smo datoteke `pki/issued/mojposluzitelj.crt` i `pki/private/mojposluzitelj.key`, od kojih je prva dakako certifikat, a druga njegov privatni ključ.

Kao i kod `build-ca`, ponovno pokretanje naredbe `build-server-full` javit će da ključ već postoji. Naredbom `show-cert` možemo pregledati stvoreni certifikat:

``` shell
$ easyrsa show-cert mojposluzitelj

Note: using Easy-RSA configuration from: /home/vedranm/pki/vars

Using SSL: openssl OpenSSL 1.1.1f  31 Mar 2020

Showing cert details for 'mojposluzitelj'.
This file is stored at:
/home/vedranm/pki/issued/mojposluzitelj.crt

Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number:
            91:87:c6:0c:44:7b:18:c1:d2:29:8a:9e:f3:13:11:49
        Signature Algorithm: sha256WithRSAEncryption
        Issuer:
            commonName                = Easy-RSA CA
        Validity
            Not Before: May 28 14:53:12 2020 GMT
            Not After : May 13 14:53:12 2023 GMT
        Subject:
            commonName                = mojposluzitelj
        X509v3 extensions:
            X509v3 Basic Constraints:
                CA:FALSE
            X509v3 Subject Key Identifier:
                B4:13:51:21:C6:86:B0:A6:28:5F:4B:7E:38:97:0A:D9:7B:94:2C:78
            X509v3 Authority Key Identifier:
                keyid:B0:9B:24:08:5B:A6:08:6A:B8:57:2A:54:EE:40:08:F2:7A:66:82:84
                DirName:/CN=Easy-RSA CA
                serial:15:4B:69:F0:48:DC:FA:60:30:12:19:6D:C1:88:DE:DE:C4:72:1F:65

            X509v3 Extended Key Usage:
                TLS Web Server Authentication
            X509v3 Key Usage:
                Digital Signature, Key Encipherment
            X509v3 Subject Alternative Name:
                DNS:mojposluzitelj
```

## Diffie-Hellmanovi parametri

Poslužitelju trebaju i parametri za Diffie-Hellmanovu razmjenu ključeva. Naime, [OpenVPN koristi TLS](https://openvpn.net/faq/why-openvpn-uses-tls/) pa nakon povezivanja klijenta i poslužitelja dogovara ključeve koji će se koristiti unutar te komunikacije i pritom koristi Diffie-Hellmanovu razmjenu ključeva. Datoteku koja sadrži te parametre ćemo izraditi naredbom `gen-dh`:

``` shell
$ easyrsa gen-dh

Note: using Easy-RSA configuration from: /home/vedranm/pki/vars

Using SSL: openssl OpenSSL 1.1.1f  31 Mar 2020
Generating DH parameters, 2048 bit long safe prime, generator 2
This is going to take a long time
..............................................................................................+.............................................................................................................+...............................................................++*++*++*++*

DH parameters of size 2048 created at /home/vedranm/pki/dh.pem
```

Dobili smo datoteku `dh.pem` u direktoriju `pki`.

## Klijentski certifikat

Klijentski certifikat generiramo analogno poslužiteljskom, naredbom `build-client-full`; ponovno ćemo kod potpisivanja trebati zaporku privatnog ključa autoriteta certifikata:

``` shell
$ easyrsa build-client-full mojklijent nopass

Note: using Easy-RSA configuration from: /home/vedranm/pki/vars

Using SSL: openssl OpenSSL 1.1.1f  31 Mar 2020
Generating a RSA private key
...................+++++
................+++++
writing new private key to '/home/vedranm/pki/private/mojklijent.key.qfKXZZMOJA'
-----
Using configuration from /home/vedranm/pki/safessl-easyrsa.cnf
Enter pass phrase for /home/vedranm/pki/private/ca.key:
Check that the request matches the signature
Signature ok
The Subject's Distinguished Name is as follows
commonName            :ASN.1 12:'mojklijent'
Certificate is to be certified until May 13 15:12:31 2023 GMT (1080 days)

Write out database with 1 new entries
Data Base Updated
```

Dobili smo datoteke `pki/issued/mojklijent.crt` i `pki/private/mojklijent.key`. Naredbu `show-cert` koristili bismo na isti način kao iznad.

Naredbom `build-serverClient-full` možemo u istom koraku napraviti certifikat koji je istovremeno i poslužiteljski i klijentski te njegov pripadni privatni ključ.

## Zahtjev za potpis certifikata i potpisani certifikat

U praksi je često nepraktično generirati privatne ključeve na strani autoriteta certifikata i onda ih slati klijentima. Tada se koristi kombinacija naredbi `gen-req` i `sign-req`, od kojih prvu pokreće klijent na svom računalu i zatim autoritetu certifikata šalje samo generirani zahtjev za potpis certifikata, dok privatni ključ zadržava za sebe. Autoritet certifikata klijentu šalje potpisani certifikat. Mi ovdje sve radimo na istom računalu pa nema potrebe da razdvajamo te korake.

## Pokretanje poslužitelja i klijenta

Konfigurirajmo poslužitelj; stvorimo datoteku `server.ovpn` sadržaja:

``` squid hl_lines="1-5"
server 172.21.0.0 255.255.255.0
ca pki/ca.crt
cert pki/issued/mojposluzitelj.crt
key pki/private/mojposluzitelj.key
dh pki/dh.pem
dev tun
```

Konfiguracijska naredba `server` označava da se radi o poslužitelju koji prima više klijenata i koristi autoritet certifikata za njihovu provjeru. Argumenti te naredbe su adresa mreže i maska podmreže, ovdje postavljamo mrežu 172.21.0.0/24. Konfiguracijska naredba `ca` zatim navodi datoteku gdje se nalazi certifikat autoriteta certifikata. Uočite da nigdje nema privatnog ključa autoriteta certifikata; to je u skladu s ranije spomenutim načinom rada gdje se autoritet certifikata (pa specijalno i njegov privatni ključ) nalazi na odvojenom računalu od onoga na kojem se izvodi OpenVPN poslužitelj. Zatim naredbe `cert` i `key` navode putanje do poslužiteljskog certifikata i privatnog ključa (respektivno), a naredba `dh` do datoteke s Diffie-Hellmanovim parametrima. Naredba `dev` nam je već poznata od ranije.

Pokrenimo poslužitelj na isti način kao i dosad:

``` shell
$ sudo openvpn --config server.ovpn
Fri May 29 00:03:10 2020 OpenVPN 2.4.7 x86_64-pc-linux-gnu [SSL (OpenSSL)] [LZO] [LZ4] [EPOLL] [PKCS11] [MH/PKTINFO] [AEAD] built on Sep  5 2019
Fri May 29 00:03:10 2020 library versions: OpenSSL 1.1.1f  31 Mar 2020, LZO 2.10
Fri May 29 00:03:10 2020 WARNING: --keepalive option is missing from server config
Fri May 29 00:03:10 2020 TUN/TAP device tun0 opened
Fri May 29 00:03:10 2020 /sbin/ip link set dev tun0 up mtu 1500
Fri May 29 00:03:10 2020 /sbin/ip addr add dev tun0 local 172.21.0.1 peer 172.21.0.2
Fri May 29 00:03:10 2020 Could not determine IPv4/IPv6 protocol. Using AF_INET
Fri May 29 00:03:10 2020 UDPv4 link local (bound): [AF_INET][undef]:1194
Fri May 29 00:03:10 2020 UDPv4 link remote: [AF_UNSPEC]
Fri May 29 00:03:10 2020 Initialization Sequence Completed
```

Poslužitelj je za sebe uzeo adrese 172.21.0.1 i 172.21.0.2, znači prve dvije u postavljenom rasponu, slično kao kod varijante točka-do-točke.

Konfigurirajmo klijent; stvorimo datoteku `server.ovpn` sadržaja:

``` squid hl_lines="1-4"
client
ca pki/ca.crt
cert pki/issued/mojklijent.crt
key pki/private/mojklijent.key
remote localhost
nobind
dev tun
```

Konfiguracijska naredba `client` označava da se radi o klijentu koji koristi autoritet certifikata za provjeru certifikata poslužitelja na koji se povezuje. Naredba `ca` ima istu ulogu kao i kod poslužitelja, a naredbe `cert` i `key` analognu. Ostale naredbe (`remote`, `nobind` i `dev`) su nam poznate od ranije.

Bez zaustavljanja poslužitelja pokrenimo klijent:

``` shell
$ sudo openvpn --config client.ovpn
Fri May 29 00:03:15 2020 OpenVPN 2.4.7 x86_64-pc-linux-gnu [SSL (OpenSSL)] [LZO] [LZ4] [EPOLL] [PKCS11] [MH/PKTINFO] [AEAD] built on Sep  5 2019
Fri May 29 00:03:15 2020 library versions: OpenSSL 1.1.1f  31 Mar 2020, LZO 2.10
Fri May 29 00:03:15 2020 WARNING: No server certificate verification method has been enabled.  See http://openvpn.net/howto.html#mitm for more info.
Fri May 29 00:03:15 2020 TCP/UDP: Preserving recently used remote address: [AF_INET]127.0.0.1:1194
Fri May 29 00:03:15 2020 UDP link local: (not bound)
Fri May 29 00:03:15 2020 UDP link remote: [AF_INET]127.0.0.1:1194
Fri May 29 00:03:15 2020 [mojposluzitelj] Peer Connection Initiated with [AF_INET]127.0.0.1:1194
Fri May 29 00:03:16 2020 TUN/TAP device tun1 opened
Fri May 29 00:03:16 2020 /sbin/ip link set dev tun1 up mtu 1500
Fri May 29 00:03:16 2020 /sbin/ip addr add dev tun1 local 172.21.0.6 peer 172.21.0.5
Fri May 29 00:03:16 2020 WARNING: this configuration may cache passwords in memory -- use the auth-nocache option to prevent this
Fri May 29 00:03:16 2020 Initialization Sequence Completed
```

Klijent se povezao na poslužitelj i zauzeo je adrese 172.21.0.5 i 172.21.0.6. Općenito se u ovoj varijanti adrese pojedinim klijenatim koji se povezuju na poslužitelj dodjeljuju ovisno o poretku kojim se klijenti povežu. Uočimo da je i poslužitelj napisao poruku o novom klijentu:

```
Fri May 29 00:03:15 2020 127.0.0.1:36623 peer info: IV_VER=2.4.7
Fri May 29 00:03:15 2020 127.0.0.1:36623 peer info: IV_PLAT=linux
Fri May 29 00:03:15 2020 127.0.0.1:36623 peer info: IV_PROTO=2
Fri May 29 00:03:15 2020 127.0.0.1:36623 peer info: IV_NCP=2
Fri May 29 00:03:15 2020 127.0.0.1:36623 peer info: IV_LZ4=1
Fri May 29 00:03:15 2020 127.0.0.1:36623 peer info: IV_LZ4v2=1
Fri May 29 00:03:15 2020 127.0.0.1:36623 peer info: IV_LZO=1
Fri May 29 00:03:15 2020 127.0.0.1:36623 peer info: IV_COMP_STUB=1
Fri May 29 00:03:15 2020 127.0.0.1:36623 peer info: IV_COMP_STUBv2=1
Fri May 29 00:03:15 2020 127.0.0.1:36623 peer info: IV_TCPNL=1
Fri May 29 00:03:15 2020 127.0.0.1:36623 [mojklijent] Peer Connection Initiated with [AF_INET]127.0.0.1:36623
Fri May 29 00:03:15 2020 mojklijent/127.0.0.1:36623 MULTI_sva: pool returned IPv4=172.21.0.6, IPv6=(Not enabled)
```

## Dodatne konfiguracijske naredbe i parametri

Korištenje poslužitelja s više klijenata donosi nam nekoliko dodatnih konfiguracijskih naredbi:

- `ifconfig-pool start-IP end-IP` navodi početnu IP adresu (`start-IP`) i završnu IP adresu (`end-IP`) iz bazena IP adresa koje će biti dodijeljene klijentima, slično kako to radi DHCP poslužitelj, opcionalno se može navesti i maska podmreže `netmask`
- `ifconfig-pool-persist file` u datoteku `file` sprema dodijeljene IP adrese kod završetka izvođenja, odnosno učitava dodijeljene IP adrese iz datoteke `file` na početku izvođenja
- `client-to-client` omogućuje komunikaciju između različitih klijenata
- `tls-version-min version` postavlja minimalnu verziju TLS-a koja se smije koristiti, zadana je `1.0`, a podržane su i vrijednosti `1.1` i `1.2`
- `tls-cipher l` navodi popis `l` dozvoljenih TLS šifrarnika; zadana vrijednost je `DEFAULT:!EXP:!LOW:!MEDIUM:!kDH:!kECDH:!DSS:!PSK:!SRP:!kRSA` kad se koristi OpenSSL; preporuča se korištenje postavki koje nudi [Mozillin generator](https://ssl-config.mozilla.org/) za web poslužitelje (konfiguracijske naredbe `ssl_ciphers` u nginx-u i `SSLCipherSuite` u Apacheju)
- `tls-auth file` dodaje sloj autentifikacije upravljačkih poruka komunikacijskog kanala korištenjem statičkog ključa koji se nalazi u datoteci `file`; kao i kod navođenja statičkog ključa naredbom `secret`, moguće je dodatno navesti smjer `0` ili `1`
- `tls-crypt keyfile` šifrira upravljačke poruke ključem navedenim u datoteci `keyfile`
- `remote-cert-tls client|server` provjerava da potpis certifikata klijenta koji se povezuje (opcija `client`) ili poslužitelja na koji se povezuje (opcija `server`) navodi način korištenja ključa i prošireni način korištenja ključa u skladu s pravilima iz RFC-a 3280

Detaljniji opis ovih i brojnih drugih parametara dan je u man stranici `openvpn(8)` (naredba `man 8 openvpn`).

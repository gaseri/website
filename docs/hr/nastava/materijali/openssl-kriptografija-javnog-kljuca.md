---
author: Domagoj Margan, Vedran Miletić
---

# Kriptografija javnog ključa

## Stvaranje i transformacije para ključeva

### Stvaranje (tajnog) RSA ključa

Za generiranje RSA ključa koristimo opciju `genrsa`:

``` shell
$ openssl genrsa
Generating RSA private key, 2048 bit long modulus (2 primes)
................+++++
................................+++++
e is 65537 (0x010001)
-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEA41Eyt2xgFhxOvDRefKJaQ0ReYoDRMqDiHpToYroUIhyulTqM
CkBhGFZSrwP8EQRLHe/0XCUXTXFKtfPkw9xs2HhMkqbufjcDEQF2bJR3xNwUctPS
i6MTwVX1cWvURd2kJRzRR5pBBnWkxsb0Xp5bgu05nlkCsjSTN/U0vnncfaBElkbD
Ao54taTqkaCBBvNdrqBHY6cK9mfbL2Zo9A3rwfMU/0Vc2ZcM2MBGqu8g9QZvKNN5
cKYJ8LRn/4NoaKzaD5w11ss6qpGjbkBYph+2q5yeUiDarD4Br7e8boOJnNv13qwI
cCjp4t6Dk0eyHGpAEuFfyO0vgn8sLq68ZJiVZQIDAQABAoIBAQC+Im+6boLcW6cb
0u3pb97x/uC3oanZoCoijEjoM16dvcHlfkgeVwUf1yxnyXxwO1gdXVTWMgwQsZ9d
G/iQPvIoCk96Jvi7R4ZgFgoY/gJD/hV4imbEd44Rm7Wlvhyap64hgL4oFwUmwwYl
YcqKzggbNhOIuEkCB+nP12DbpZBg3jaJT4km1Iz58kacbCItm+rilycq5mNWfKOq
M8eipf+a2QO+1R4pP6vATMgCSz4euAjr3AISYiUa2WtQCrhy3tlpTF6FAazhZXO5
7MqB7/liM5k/hp5P2334vbh3bKVJcJHW9Uf4ViYSeHGIDXonUTED2tX4kuztt+V7
C8dPyrghAoGBAPR3tzSsIDROmfH+wkedk4hsppQT/n9jCezmiCatVd0+8UUHL3R/
DDGdyVIHEsb3sZBdUh8Oin9AofdaYpZyJgJXkBRph4Wf0973nvAaR5F6nLTBjVS+
eX9mFz80KUJB7XzXQn1gz83iKzX32rYbE2eWmFc+8NL2b7gmVonIp94ZAoGBAO4K
XfK+ItF2+dtxcNfqXJNf3R7i5OoOhGNjQu2QR3gX/m7iJ87fOsh5tGtkADdA8Io1
+8jIPcaedp81UlId3ux9o8FKl8YewEPufFz+ZjP8Q+DwQyISITrGBU0ifh97S78+
8ZVFQkx9RI+6gU2PU62C21asGeqHVvhRUp2CZkMtAoGBAPKmqUg02TpSEmeq8PfY
pomxHp64QdH7Yeys3dNWFXYndZ+IhEfjmxzceulab/7h+HNMzahZ7Ipmgt1b76NY
5fVJKI+6N3QgslIMAsxbqVHzG/wmabwF297iXIy1n4ZOngVePHbqUxkONsm4nHRI
57fYOJnQtYUQas+j7h2Q915xAoGBAMJ1UZ5V2VfInAC6wXaBjDMQozDyJhNW3Kvc
kPZFYT0oTMAEnISRNBJF6i+4t7xrnpUp2JCDlIPHPBx/kMpogI4tbTMgXrCIuoRE
NPA7Gv7o3ALMA+u3Z9H9pqMGxIWvUYfgQbaxp6GYzAOmVq8noTIjrk81tM401cVx
mc32ktfZAoGALL8/ZE8rOyE2wzoIjlOnnQWRWPYdD3hu0ruxEUGtCYxPokTYGL4y
i/DyrJ5xLe7ydPJ5QdPb3tPraLVYuhGpKUx6K2df1TJ+Q6AjokZH/Utr52m/dmaI
+doKmc6nJ28/T2+OdEFgStua8NDGOqNvq/wEf9TmcT+uU8ExlEEipUA=
-----END RSA PRIVATE KEY-----
```

Ovdje generiramo 2048-bitni ključ koji ispisujemo na standardni izlaz. Veća veličina ključa povećava vrijeme šifriranja i dešifriranja, [ali isto tako smanjuje mogućnost provaljivanja šifriranog sadržaja od treće strane](https://xkcd.com/538/).

Ukoliko želimo spremiti ključ u posebnu izlaznu datoteku, koristimo parametar `-out`. Također možemo navesti veličinu ključa koju želimo:

``` shell
$ openssl genrsa -out mojkljuc.pem 1024
openssl genrsa -out mojkljuc.pem 1024
Generating RSA private key, 1024 bit long modulus (2 primes)
.................................+++++
.................+++++
e is 65537 (0x010001)
```

Ovime smo generirali 1024-bitni ključ i spremili ga u datoteku `mojkljuc.pem`. Želimo li osigurati naš ključ, možemo ga šifrirati i zaštititi lozinkom. Dodajemo parametar željenog algoritma za šifriranje:

``` shell
$ openssl genrsa -camellia256 -out mojkljuc.pem 4096
Generating RSA private key, 4096 bit long modulus (2 primes)
........+++++
.....................+++++
e is 65537 (0x010001)
Enter pass phrase for mojkljuc.pem:
Verifying - Enter pass phrase for mojkljuc.pem:
```

Ovime smo generirali 4096-bitni ključ i spremili ga u datoteku `mojkljuc.pem`, uz šifriranje algoritmom [Camellia-256](https://en.wikipedia.org/wiki/Camellia_(cipher)).

### Stvaranje javnog RSA ključa

Kako bi generirali pripadni javni ključ našeg privatnog RSA ključa, koristimo opciju `rsa` s parametrom `-pubout`:

``` shell
$ openssl rsa -in mojkljuc.pem -pubout
Enter pass phrase for mojkljuc.pem:
writing RSA key
-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDAJn/omkMIes+2XQC8WumxJuk0
n4uSrfoVJg/TmFecsfnxfGwX+AIKg7vYZE21IqqrgB1lUmBp+hqGxTmw55b/9iPo
pBk+I1aaRZdHNvkbztU+t8/PPwFAMN7969jpMXYGhGcZACn0h+ok/i7MglSOQWzB
im00OOWKyVh0EWrT9wIDAQAB
-----END PUBLIC KEY-----
```

Kao i dosad, parametrom `-out` možemo ključ pohraniti u datoteku umjesto ispisivanja na standardni izlaz:

``` shell
$ openssl rsa -in mojkljuc.pem -out javnikljuc.pem -pubout
Enter pass phrase for mojkljuc.pem:
writing RSA key
```

Opcija `rsa` ima i druge parametre koji omogućuju, primjerice, konverziju formata (`-inform` i `-outform`), ispis modulusa (`-modulus`) i šifriranje ključa (`-` i naredba bilo kojeg od podržanih algoritama za šifriranje).

### Mjerenje performansi RSA ključeva

OpenSSL omogućuje mjerenje brzine stvaranja tajnih i javnih RSA ključeva opcijom `speed rsa`:

``` shell
$ openssl speed rsa
Doing 512 bits private rsa's for 10s: 276416 512 bits private RSA's in 9.98s
Doing 512 bits public rsa's for 10s: 4679174 512 bits public RSA's in 9.98s
Doing 1024 bits private rsa's for 10s: 130765 1024 bits private RSA's in 9.98s
Doing 1024 bits public rsa's for 10s: 1944320 1024 bits public RSA's in 9.98s
Doing 2048 bits private rsa's for 10s: 18717 2048 bits private RSA's in 9.98s
Doing 2048 bits public rsa's for 10s: 625187 2048 bits public RSA's in 9.98s
Doing 3072 bits private rsa's for 10s: 6198 3072 bits private RSA's in 9.99s
Doing 3072 bits public rsa's for 10s: 302978 3072 bits public RSA's in 9.98s
Doing 4096 bits private rsa's for 10s: 2782 4096 bits private RSA's in 9.99s
Doing 4096 bits public rsa's for 10s: 175534 4096 bits public RSA's in 9.97s
Doing 7680 bits private rsa's for 10s: 289 7680 bits private RSA's in 9.98s
Doing 7680 bits public rsa's for 10s: 49636 7680 bits public RSA's in 9.89s
Doing 15360 bits private rsa's for 10s: 54 15360 bits private RSA's in 10.00s
Doing 15360 bits public rsa's for 10s: 13626 15360 bits public RSA's in 9.97s
OpenSSL 1.1.1b  26 Feb 2019
built on: Wed Apr  3 10:50:23 2019 UTC
options:bn(64,64) rc4(16x,int) des(int) aes(partial) blowfish(ptr)
compiler: gcc -fPIC -pthread -m64 -Wa,--noexecstack -Wall -Wa,--noexecstack -g -O2 -fdebug-prefix-map=/build/openssl-uEA50R/openssl-1.1.1b=. -fstack-protector-strong -Wformat -Werror=format-security -DOPENSSL_USE_NODELETE -DL_ENDIAN -DOPENSSL_PIC -DOPENSSL_CPUID_OBJ -DOPENSSL_IA32_SSE2 -DOPENSSL_BN_ASM_MONT -DOPENSSL_BN_ASM_MONT5 -DOPENSSL_BN_ASM_GF2m -DSHA1_ASM -DSHA256_ASM -DSHA512_ASM -DKECCAK1600_ASM -DRC4_ASM -DMD5_ASM -DAES_ASM -DVPAES_ASM -DBSAES_ASM -DGHASH_ASM -DECP_NISTZ256_ASM -DX25519_ASM -DPADLOCK_ASM -DPOLY1305_ASM -DNDEBUG -Wdate-time -D_FORTIFY_SOURCE=2
            sign    verify    sign/s verify/s
rsa  512 bits 0.000036s 0.000002s  27697.0 468855.1
rsa 1024 bits 0.000076s 0.000005s  13102.7 194821.6
rsa 2048 bits 0.000533s 0.000016s   1875.5  62644.0
rsa 3072 bits 0.001612s 0.000033s    620.4  30358.5
rsa 4096 bits 0.003591s 0.000057s    278.5  17606.2
rsa 7680 bits 0.034533s 0.000199s     29.0   5018.8
rsa 15360 bits 0.185185s 0.000732s      5.4   1366.7
```

Specijalno se može mjeriti brzina stvaranja tajnih i javnih RSA ključeva duljine redom 512, 1024, 2048 i 4096 bitova opcijama `speed rsa512`, `speed rsa1024`, `speed rsa2048` i `speed rsa4096`. Opcija `speed` nudi mogućnost mjerenja brzine izvođenja i drugih kriptoalgoritama o čemu se više može pročitati u `speed(1ssl)`.

### Stvaranje para DSA ključeva

Generiranje DSA ključeva izvodi se putem opcije `dsaparam` i opcije `gendsa` u dva koraka. Pokažimo stvaranje 1024-bitnog ključa u datoteci `dsakljuc.pem`. Prvo stvaramo datoteku s parametrima algoritma DSA opcijom `dsaparam`:

``` shell
$ openssl dsaparam -out dsaparam.pem 1024
Generating DSA parameters, 1024 bit long prime
This could take some time
..+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*
..+......+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*
```

Zatim stvaramo tajni ključ naredbom `gendsa`:

``` shell
$ openssl gendsa dsaparam.pem
Generating DSA key, 1024 bits
-----BEGIN DSA PRIVATE KEY-----
MIIBuwIBAAKBgQDOhyPo/H/ULvcuw3KiY1L19ccVUbn98x6/nlP9sW6npmVTfdcz
QnBVikmDhzSVlNF5kXznGotporfSnISBeaT/znTLV8/u394FgpDFw4vFuuq+njyp
xQ8HahHPrNe+495OXwXpAhrw+DkcpIrXZCRZJR4AfNixvxKQuiuJV6QbRQIVAOWb
rPV+O1LV8eDKLs9q5WI6W5enAoGAVkRz5CeaIg6XNN8y98ItGrf6FhKEr033rOiv
UWgR24h/vHuj7HMANA27pDxvntZKUVwnObdTXRowRBCVXQQnLCZZ/ltlSPBt9hBT
gapcUh00UfpXo0ybjbD8Y0v9SXu6fXlqPGo6JUI88O5nHA1WpN5NNENGOCxRkzm2
qAUpdlQCgYBafObwm2vyGzjDsrmY+jqKy3TnNTbhRLAANpNKeSwwu3+4XN6j/0at
hlU68uWIUkLxXnkK+IZdhiNaUI5G3QxkKvnj11oGnMT9IAaSrWiQq7Mgf15s/VO7
U1wn6E0N+80LqIN2LkpaSFA+hI3/1xzx877zHfeHerEqYYNsbIhKAwIVAMniYkKx
UNABQps6Jm6Kf1gz8Lhi
-----END DSA PRIVATE KEY-----
```

Dodamo li parametar `-out`, ključ će umjesto ispisivanja na standardni izlaz biti pohranjen u datoteku:

``` shell
$ openssl gendsa -out dsakljuc.pem dsaparam.pem
Generating DSA key, 1024 bits
```

Ovih dva koraka moguće je svesti na jedan korištenjem parametra `-genkey` opcije `dsaparam`.

``` shell
$ openssl dsaparam -out dsakljuc.pem -genkey 1024
Generating DSA parameters, 1024 bit long prime
This could take some time
...+.......+...+....+..+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*
..+.+.....+..+.+.+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*
```

Kako bi generirali pripadni javni ključ našeg privatnog DSA ključa, koristimo opciju `dsa` s parametrom `-pubout` analogno primjeru za RSA:

``` shell
$ openssl dsa -in dsakljuc.pem -pubout
read DSA key
writing DSA key
-----BEGIN PUBLIC KEY-----
MIIBtjCCASsGByqGSM44BAEwggEeAoGBAM6HI+j8f9Qu9y7DcqJjUvX1xxVRuf3z
Hr+eU/2xbqemZVN91zNCcFWKSYOHNJWU0XmRfOcai2mit9KchIF5pP/OdMtXz+7f
3gWCkMXDi8W66r6ePKnFDwdqEc+s177j3k5fBekCGvD4ORykitdkJFklHgB82LG/
EpC6K4lXpBtFAhUA5Zus9X47UtXx4Mouz2rlYjpbl6cCgYBWRHPkJ5oiDpc03zL3
wi0at/oWEoSvTfes6K9RaBHbiH+8e6PscwA0DbukPG+e1kpRXCc5t1NdGjBEEJVd
BCcsJln+W2VI8G32EFOBqlxSHTRR+lejTJuNsPxjS/1Je7p9eWo8ajolQjzw7mcc
DVak3k00Q0Y4LFGTObaoBSl2VAOBhAACgYB/APco+HtrGsyeGGFpZWgu9N/e2y+e
p4/RQH9iatV3AkbjuK5vECi9cgCUH3R2eUIRYRy7MxABmp0ARwCCreF95LsUlHdx
ytf5/FhB0OMSmc2Zxf55I2QEnWq9kNgoGbl75slUIpeF7Vuqe7K3Q31kRAr/L75N
gdwQ8Xo2lO727g==
-----END PUBLIC KEY-----
```

Također analogno primjeru za RSA, parametrom `-out` možemo ključ pohraniti u datoteku umjesto ispisivanja na standardni izlaz:

``` shell
$ openssl dsa -in dsakljuc.pem -out javnidsakljuc.pem -pubout
read DSA key
writing DSA key
```

Opcija `dsa` ima i druge parametre vrlo slične već opisanoj opciji `rsa`.

### Mjerenje performansi DSA ključeva

OpenSSL omogućuje mjerenje brzine potpisivanja DSA ključevima i provjere potpisa opcijom `speed dsa`:

``` shell
$ openssl speed dsa
Doing 512 bits sign dsa's for 10s: 192335 512 bits DSA signs in 9.99s
Doing 512 bits verify dsa's for 10s: 325914 512 bits DSA verify in 10.00s
Doing 1024 bits sign dsa's for 10s: 110027 1024 bits DSA signs in 9.99s
Doing 1024 bits verify dsa's for 10s: 148698 1024 bits DSA verify in 10.00s
Doing 2048 bits sign dsa's for 10s: 43267 2048 bits DSA signs in 10.00s
Doing 2048 bits verify dsa's for 10s: 49308 2048 bits DSA verify in 9.99s
OpenSSL 1.1.1b  26 Feb 2019
built on: Wed Apr  3 10:50:23 2019 UTC
options:bn(64,64) rc4(16x,int) des(int) aes(partial) blowfish(ptr)
compiler: gcc -fPIC -pthread -m64 -Wa,--noexecstack -Wall -Wa,--noexecstack -g -O2 -fdebug-prefix-map=/build/openssl-uEA50R/openssl-1.1.1b=. -fstack-protector-strong -Wformat -Werror=format-security -DOPENSSL_USE_NODELETE -DL_ENDIAN -DOPENSSL_PIC -DOPENSSL_CPUID_OBJ -DOPENSSL_IA32_SSE2 -DOPENSSL_BN_ASM_MONT -DOPENSSL_BN_ASM_MONT5 -DOPENSSL_BN_ASM_GF2m -DSHA1_ASM -DSHA256_ASM -DSHA512_ASM -DKECCAK1600_ASM -DRC4_ASM -DMD5_ASM -DAES_ASM -DVPAES_ASM -DBSAES_ASM -DGHASH_ASM -DECP_NISTZ256_ASM -DX25519_ASM -DPADLOCK_ASM -DPOLY1305_ASM -DNDEBUG -Wdate-time -D_FORTIFY_SOURCE=2
            sign    verify    sign/s verify/s
dsa  512 bits 0.000052s 0.000031s  19252.8  32591.4
dsa 1024 bits 0.000091s 0.000067s  11013.7  14869.8
dsa 2048 bits 0.000231s 0.000203s   4326.7   4935.7
```

Specijalno se može mjeriti brzina stvaranja tajnih i javnih DSA ključeva duljine redom 512, 1024 i 2048 bitova opcijama `speed dsa512`, `speed dsa1024` i `speed dsa2048` (respektivno).

### Stvaranje para ključeva algoritmom eliptične krivulje

Za generiranje ključeva algoritmom eliptične krivulje, koristi se opcija `ecparam` s parametrima `-name` i `-genkey`. Pokažimo stvaranje ključa u datototeci `krivulja.pem` korištenjem eliptične krivulje `sect571r1`.

``` shell
$ openssl ecparam -out parametri.pem -name sect571r1
```

Time smo generirali datoteku s parametrima eliptičke krivulje. Ako želimo uz parametre generirati i ključ, koristimo i parametar `-genkey`.

``` shell
$ openssl ecparam -out krivuljakljuc.pem -name sect571r1 -genkey
```

Uočimo kako smo generirali parametre i ključ u istom koraku te da prethodno generirane parametre nismo iskoristili. Opcijom `-name` određujemo krivulju, te je nužno navesti željeni algoritam. Listu možemo vidjeti navođenjem `ecparam -list_curves`:

``` shell
$ openssl ecparam -list_curves
  secp112r1 : SECG/WTLS curve over a 112 bit prime field
  secp112r2 : SECG curve over a 112 bit prime field
  secp128r1 : SECG curve over a 128 bit prime field
  secp128r2 : SECG curve over a 128 bit prime field
  secp160k1 : SECG curve over a 160 bit prime field
  secp160r1 : SECG curve over a 160 bit prime field
  secp160r2 : SECG/WTLS curve over a 160 bit prime field
  secp192k1 : SECG curve over a 192 bit prime field
  secp224k1 : SECG curve over a 224 bit prime field
  secp224r1 : NIST/SECG curve over a 224 bit prime field
  secp256k1 : SECG curve over a 256 bit prime field
  secp384r1 : NIST/SECG curve over a 384 bit prime field
  secp521r1 : NIST/SECG curve over a 521 bit prime field
  prime192v1: NIST/X9.62/SECG curve over a 192 bit prime field
  prime192v2: X9.62 curve over a 192 bit prime field
  prime192v3: X9.62 curve over a 192 bit prime field
  prime239v1: X9.62 curve over a 239 bit prime field
  prime239v2: X9.62 curve over a 239 bit prime field
  prime239v3: X9.62 curve over a 239 bit prime field
  prime256v1: X9.62/SECG curve over a 256 bit prime field
  sect113r1 : SECG curve over a 113 bit binary field
  sect113r2 : SECG curve over a 113 bit binary field
  sect131r1 : SECG/WTLS curve over a 131 bit binary field
  sect131r2 : SECG curve over a 131 bit binary field
  sect163k1 : NIST/SECG/WTLS curve over a 163 bit binary field
  sect163r1 : SECG curve over a 163 bit binary field
  sect163r2 : NIST/SECG curve over a 163 bit binary field
  sect193r1 : SECG curve over a 193 bit binary field
  sect193r2 : SECG curve over a 193 bit binary field
  sect233k1 : NIST/SECG/WTLS curve over a 233 bit binary field
  sect233r1 : NIST/SECG/WTLS curve over a 233 bit binary field
  sect239k1 : SECG curve over a 239 bit binary field
  ...
```

Kako bi generirali pripadni javni ključ našeg privatnog ključa eliptičke krivoljke, koristimo opciju `ec` s parametrom `-pubout` analogno primjerima za RSA i DSA:

``` shell
$ openssl ec -in krivuljakljuc.pem -pubout
read EC key
writing EC key
-----BEGIN PUBLIC KEY-----
MIGnMBAGByqGSM49AgEGBSuBBAAnA4GSAAQFZytZxxNf738OUhCOKikBeMIN+NE1
KZ+OhEhywMxw45IBLQVHduPs0Lx2ty6u7yguRsy2zo8WzFIQu6LO4/76vPScTbqo
KyYHB7Nm5WM0b308kMCNK2Wy2fIdHc/TuYQKUU4jeQyjmTdQ0JMHHRi8bMULmZuT
AQ+8P8umVzatfd6RVW4tDuu2l6ych7ONsHs=
-----END PUBLIC KEY-----
```

Opcija `ec` ima i druge parametre vrlo slične već opisanoj opcijama `rsa` i `dsa`, uključujući parametar `-out` kojim se javni ključ sprema u datoteku umjesto ispisa na standardni izlaz.

### Mjerenje performansi ECDSA ključeva

OpenSSL omogućuje mjerenje brzine potpisivanja ECDSA ključevima i provjere potpisa opcijom `speed ecdsa`.

### Stvaranje parametara Diffie-Hellmanove razmjene ključeva

Parametri Diffie-Hellmanove razmjene ključeva generiraju se opcijom `dhparam` i navođenjem broja bitova:

``` shell
$ openssl dhparam 2048
Generating DH parameters, 2048 bit long safe prime, generator 2
This is going to take a long time
....................................................................................................+.......................................................+............................................................+...............++*++*++*++*
-----BEGIN DH PARAMETERS-----
MIIBCAKCAQEAlaJAQw56AgC8JQ8reVyMuR6NG+rwvfMJvW0T9BhDKgy24OaoiLeZ
hQ4Xs039AKYIa9MMmS6U+yVSNa42gj8PfW80sV10LbOQ5/+vAcN43Xhk9+b4YsB5
lCmXG69HZXKj3PA9MOYGOzI1t6CM/88lNHcLiwBTDqdVudOXClCNScbicmqaJ7Ke
VZbsut55yV1XODuP2ggidpeFipjNHEvSWy8RQUrGeUL/ywmXIkEdI8jx4HmdvQMT
71u3jwjyuJ32xGJNAqVNwluJJUv/H8P5jEHk6PbeH59eMCDL+yP9AMLZE79TJ93s
TaHER4aLNUNk2SzAOpWQx8TNem9nHQq54wIBAg==
-----END DH PARAMETERS-----
```

Parametrom `-out` možemo umjesto na standardni izlaz parametre zapisati u datoteku koju želimo:

``` shell
$ openssl dhparam -out parametri.pem 2048
Generating DH parameters, 2048 bit long safe prime, generator 2
This is going to take a long time
.................................................................................................................................+.....+..........................................................................................................................+...........................................................++*++*++*++*
```

### Mjerenje performansi Diffie-Hellmanove razmjene ključeva

OpenSSL omogućuje mjerenje brzine Diffie-Hellmanove razmjene ključeva temeljene na eliptičkim krivuljama (engl. [Elliptic-curve Diffie-Hellman](https://en.wikipedia.org/wiki/Elliptic-curve_Diffie%E2%80%93Hellman)) opcijom `speed ecdsa`.

### Stvaranje i obrada ostalih ključeva

Grupa opcija `genpkey`, `pkeyparam` i `pkey` nudi alternativu gornjim opcijama i omogućuje generiranje ključeva raznih vrsta putem unificiranog sučelja naredbenog retka. Više o tim naredbama može se naći u istoimenim man stranicama u odjeljku `1ssl`.

## Korištenje ključeva

### Potpisivanje hashirane datoteke i provjera potpisa

Ako se želimo osigurati da se stvorena hashirana datoteka neće mijenjati bez naše dozvole, možemo je potpisati našim privatnim ključem. Koristimo opcije `-sign` za potpisivanje i `-out` za izlaznu datoteku. Potpisana datoteka bit će `datoteka.txt.sha1`.

``` shell
$ openssl dgst -sha1 -sign mojkljuc.pem -out datoteka.txt.sha1 datoteka.txt
```

Za provjeru potpisane hash datoteke, potrebna je originalna datoteka iz koje je stvoren hash, potpisana hash datoteka i javni ključ potpisatelja. Koristimo parametar `-verify` za javni ključ, te `-signature` za potpisanu hash datoteku.

``` shell
$ openssl dgst -sha1 -verify javnikljuc.pem -signature datoteka.txt.sha1 datoteka.txt
Verified OK
```

### Šifriranje i dešifriranje parom RSA ključeva

Šifriranje i dešifriranje korištenjem RSA ključeva vršimo naredbom `rsautl`. Za šifriranje dodajemo parametar `-encrypt` i koristimo javni ključ (opcija `-pubin`) u datoteci koju navodimo pod `-inkey`:

``` shell
$ openssl rsautl -encrypt -in datoteka.txt -inkey javnikljuc.pem -pubin -out datoteka.txt.enc
```

Dešifriranje vršimo parametrom `-decrypt` i korištenjem tajnog ključa pod `-inkey`:

``` shell
$ openssl rsautl -decrypt -in datoteka.txt.enc -inkey mojkljuc.pem -out desifrirana-datoteka.txt
```

Uočimo kako sad nema ni opcije `-pubin` jer je korišteni ključ tajni.

!!! warning
    Kriptografija javnog ključa nije namijenjena za šifriranje proizvoljno dugih datoteka; za tu svrhu koristi se simetrično šifriranje, npr. algoritmom AES. U tom slučaju se javni ključ koristi za šifriranje slučajne šifre za simetrično šifriranje, a tajni ključ za njeno dešifriranje. U praksi taj način rada koristi TLS i zato je za njegov rad potrebna [grupa šifrarnika](https://en.wikipedia.org/wiki/Cipher_suite) (engl. *cipher suite*): primjerice, RSA se može koristiti za autentifikaciju u procesu rukovanja i razmjenu ključeva, a zatim se za šifriranje poruka može koristiti AES.

### Šifriranje i dešifriranje ostalim ključevima

Za šifriranje ostalim vrstama ključeva koristi se opcija `pkeyutl`, donekle slična `rsautl`. Više informacija može se naći u `pkeyutl(1ssl)`.

---
author: Vedran Miletić
---

# Rad s Python modulom pyca/cryptography

Python u okviru svoje standardne biblioteke nudi kriptografske funkcije u modulu `crypto`, ali puno elegantniji za korištenje je modul `cryptography` ([službeno web sjedište](https://cryptography.io/)) razvijen od strane [Python Cryptographic Authorityja](https://github.com/pyca).

Modul `cryptopgraphy` sastoji se od dva sloja:

- sloj recepata (engl. *recipes layer*) kod kojeg nije potrebno izvoditi detaljnu konfiguraciju, odnosno odabir parametara korištenih algoritama
- sloj opasnih materijala (engl. *hazardous materials layer*) koji nudi implementaciju kriptografskih algoritama za koje je nužno dobro poznavati teoriju koja se tih koncepata tiče

## Kodiranje znakova

Znakovni nizovi u Pythonu dizajnirani su da slijede svakdonevno poimanje pojma teksta. Primjerice, znakovni niz `"riječ"` je u Pythonu duljine 5 znakova, iako se posljednji znak u računalu kodira s 2 bajta (primjerice kod spremanja u datoteku ili slanja putem mreže) pa je duljina znakovnog niza 6 znakova.

``` python
s = "riječ"
print("Znakovni niz", s, "ima duljinu", len(s))

e = g.encode()
print("Znakovni niz", e, "ima duljinu", len(e))
```

Kodiranje u obliku Base64 dostupno je korištenjem modula `base64` ([dokumentacija](https://docs.python.org/3/library/base64.html)) koji je dio standardne biblioteke:

``` python
from base64 import b64encode, b64decode

b = b64encode(e)
print(b)
d = b64decode(b)
print(d)
```

!!! admonition "Zadatak"
    Usporedite duljinu:

    1. znakovnog niza u Pythonu
    1. niza bajtova dobivenih kodiranjem znakovnog niza
    1. znakovnog niza kodiranog metodom Base64

    za znakovni niz "Energični, za sve vrste egzistencije sposobni pojedinac najveći je kapital i jedini temelj našeg narodnog kapitala, koji pada samo zato, jer rapidno pada kult moralni, fizički i intelektualni naših energija u pravcu što veće osobne i privatne inicijative." ([Matošev citat](https://hr.wikiquote.org/wiki/Antun_Gustav_Mato%C5%A1)).

## Simetrična kriptografija

Fernet ([dokumentacija](https://cryptography.io/en/latest/fernet/)) je skup recepata za simetričnu enkripciju u modulu `crytopgraphy`. Ime Fernet dolazi od [talijanskog likera popuarnog u Argentini](https://en.wikipedia.org/wiki/Fernet).

Inicijalizacija se izvodi na način:

``` python
#!/usr/bin/env python

from cryptography.fernet import Fernet

key = Fernet.generate_key()
f = Fernet(key)
```

S tako inicijaliziranim modulom šifriranje se izvodi na način:

``` python
encrypted_message = f.encrypt(b"""\
Jezik, čist materinji jezik poznavati je prva i najglavnija dužnost svakog\
pisca. Tko ga ne poznaje, može biti uman, odličan, zanimljiv čovjek, ali\
dobar, uspješan pisac ― nikada. Pokažite mi na jednog jedinog većeg pisca u\
stranom svijetu što griješi proti pravilima svog jezika.\
""")

print("Šifrirana poruka je", encrypted_message)
```

Dešifriranje se vrši kodom:

``` python
decrypted_message = f.decrypt(encrypted_message)

print("Dešifrirana poruka je", decrypted_message)
```

!!! admonition "Zadatak"
    Šifrirajte poruku po želji ključem koji ćete generirati, a zatim pokušajte dešifrirati drugim ključem koji ćete generirati i objasnite zašto to možete ili ne možete učiniti.

Fernetom je moguće šifrirati i dešifrirati korištenjem grupe ključeva. U tom slučaju će dešifriranje biti isprobano svakim od dostupnih ključeva. Primjerice, za dva ključa kod za inicijalizaciju je oblika:

``` python
#!/usr/bin/env python

from cryptography.fernet import Fernet, MultiFernet
k1 = Fernet.generate_key()
f1 = Fernet(k1)
k2 = Fernet.generate_key()
f2 = Fernet(k2)
f = MultiFernet([f1, f2])
```

Šifriranje dvije poruke različitim ključevima vršimo na način:

``` python
encrypted_message1 = f1.encrypt(b"Tajna poruka!")
encrypted_message2 = f2.encrypt(b"Isto tako tajna poruka!")
```

Dešifriranje:

``` python
decrypted_message1 = f.decrypt(encrypted_message1)
print(decrypted_message1)

decrypted_message2 = f.decrypt(encrypted_message2)
print(decrypted_message2)
```

Simetrične ključeve moguće je dodatno zaštiti zaporkom, o čemu se više može naći u [službenoj dokumentaciji](https://cryptography.io/en/latest/fernet/#using-passwords-with-fernet).

## Asimetrična kriptografija

Modul `cryptography` podržava niz asimetričnih kriptografskih algoritama ([dokumentacija](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/)), ali mi ćemo se ograničiti na korištenje RSA ([dokumentacija](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/rsa/)).

Stvaranje privatnog ključa veličine 2048 bita i zapis istog u formatu PKCS#8 vrši se na način:

``` python
#!/usr/bin/env python

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

private_key_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.BestAvailableEncryption(b'1234')
)

print("Ključ je", private_key_pem)
```

Stvaranje pripadnog javnog ključa i pretvorba u format PKCS#1 vrši se na način:

``` python
public_key = private_key.public_key()

public_key_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.PKCS1
)

print("Ključ je", public_key_pem)
```

Ključem možemo potpisivati i šifrirati. Potpisivanje vršimo tajnim ključem na način:

``` python
message = """Energični, za sve vrste egzistencije sposobni pojedinac \
najveći je kapital i jedini temelj našeg narodnog kapitala,""".encode()
signature = private_key.sign(
    message,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)

print("Potpis je", signature)
```

Provjeru vršimo javnim ključem na način:

``` python
verification = public_key.verify(
    signature,
    message,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)
```

Šifriranje vršimo javnim ključem na način:

``` python
short_message = b"0123556789"
ciphertext = public_key.encrypt(
    short_message,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)

print("Šifrirana poruka je", ciphertext)
```

Dešifriranje vršimo tajnim ključem na način:

``` python
plaintext = private_key.decrypt(
    ciphertext,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)

print("Dešifrirana poruka je", plaintext)
print("Poruka je jednaka početnoj?" plaintext == short_message)
```

!!! admonition "Zadatak"
    Isprobajte potpisati nekoliko poruka i usporedite dobivene potpise pa zatim provjerite koliko je dugačka najduža poruka koju možete potpisati, ako ograničenje postoji.

!!! admonition "Zadatak"
    Isprobajte šifrirati nekoliko poruka i uvjerite se da su rezultati različiti, a zatim provjerite koliko je dugačka najduža poruka koju možete šifrirati, ako ograničenje postoji.

!!! note
    U nastavku ćemo, kao i dosad, šifrirati i dešifrirati unutar jednog procesa bez korištenja međuprocesne komunikacije (primjerice, putem socketa), obzirom da razdvajanje koda na dvije strane koje međusobno komuniciraju komplicira kod i odvlači fokus sa usvajanja načina korištenja kriptografskih algoritama. Međuprocesnu komunikaciju ćemo uvesti na kraju.

## Razmjena simetričnih ključeva korištenjem asimetričnog šifriranja

Jedna od tipičnih primjena asimetričnog šifriranja je razmjena ključeva za simetrično šifriranje. Ključ za simetrično šifriranje se bira na jednoj strani:

``` python
symmetric_key = "MojPasFido25".encode()
```

Kako simetrični ključ korišten unutar jedne sesije korisnik ne mora pamtiti, bolje ga je slučajno generirati:

``` python
import os

symmetric_key_client = os.urandom(12)
```

Neka su sad spremljeni tajni ključ u varijabli `private_key` i javni ključ u varijabli `public_key`. Na strani koja je generirala simetričn ključ on se šifrira javnim ključem:

``` python
# public_key = ...

ciphertext = public_key.encrypt(
    symmetric_key_client,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)

print("Simetrični ključ šifriran javnim ključem je", ciphertext)
```

Nakon primanja šifriranog ključa na drugoj strani on se dešifrira tajnim ključem na način:

``` python
# private_key = ...

symmetric_key_decrypted = private_key.decrypt(
    ciphertext,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)

print("Dešifrirani simetrični ključ je", symmetric_key_decrypted)

```

## Autentificirano šifriranje

U situaciji kad je provedena razmjena ključeva mehanizmom opisanim iznad moguće je iskoristiti bilo koji od algoritama za šifriranje. Specijalno, ako za simetrično šifriranje želimo koristiti AES-GCM, onda ćemo to izvesti na način:

``` python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

aes_key = AESGCM.generate_key(bit_length=128)

print("AES ključ je", aes_key)
```

Nakon razmjene ključeva i druga strana bi imala isti ključ. Dodatno nam treba broj koji se koristi samo jednom (engl. number used once, kraće nonce):

``` python
nonce = os.urandom(12)
```

Taj broj možemo razmijeniti na isti način kao ključeve. Šifriranje zatim vršimo na način:

``` python
msg = "Ako ljubav nije ludost, onda nije ni ljubav.".encode()

encrypted_msg = aesgcm.encrypt(nonce, msg, None)
print("Šifrirana poruka je", encrypted_msg)
```

Posljednji parametar su podaci koji se šalju bez šifriranja za kojima nemamo potrebu pa ih postavljano na `None`. Dešifriranje se vrši na način:

``` python
decrypted_msg = aesgcm.decrypt(nonce, encrypted_msg, None)
print("Dešifrirana poruka je", decrypted_msg)
```

!!! admonition "Zadatak"
    - Zamijenite AES-GCM za AES-CCM i provjerite koristi li se na isti način.
    - Zamijenite AES-GCM za ChaCha20Poly1305 i proučite u dokumentaciji koje su razlike kod generiranja simetričnog ključa.

## Autentifikacija poruka

Autentifikaciju poruka je moguće vršiti temeljeno na hashiranju ili temeljeno na šifriranju:

- [HMAC: Keyed-Hashing for Message Authentication (RFC 2104)](https://datatracker.ietf.org/doc/html/rfc2104)
- [The AES-CMAC Algorithm (RFC 4493)](https://datatracker.ietf.org/doc/html/rfc4493)

### Autentifikacija poruka temeljena na hashiranju

HMAC ([dokumentacija](https://cryptography.io/en/latest/hazmat/primitives/mac/hmac/)) inicijaliziramo za svaku poruku na način:

``` python
import os
from cryptography.hazmat.primitives import hashes, hmac

key = os.urandom(20)

poruka = "I kada sniježi, a spušta se tama, u pahuljama tišina je sama.".encode()

h = hmac.HMAC(key, hashes.MD5())
h.update(poruka)

hash_poruke = h.finalize()

print("Poruka", poruka, "ima hash", hash_poruke)
```

Poruka i hash šalju se drugoj strani koja ima ključ dobiven razmjenom ključeva opisanom iznad. Provjera poruke se vrši na način:

``` python
h_provjera = hmac.HMAC(key, hashes.MD5())
h_provjera.update(poruka)

h_provjera.verify(hash_poruke)
```

!!! admonition "Zadatak"
    - Iskoristite SHA1 umjesto SHA256 i usporedite duljinu sažetka.
    - Iskoristite MD5 umjesto SHA256 i usporedite duljinu, a zatim razmislite koliko bi vam pokušaja trebalo da pogodite točan MD5 hash neke poruke u situaciju u kojoj ne znate tajni ključ pa ga ne možete izračunati.

### Autentifikacija poruka temeljena na šifriranju

CMAC ([dokumentacija](https://cryptography.io/en/latest/hazmat/primitives/mac/cmac/)) se također inicijalizira za svaku pojedinu poruku na način:

``` python
import os
from cryptography.hazmat.primitives import cmac
from cryptography.hazmat.primitives.ciphers.algorithms import AES

key = os.urandom(16)

aes = AES(key)

poruka = "Kad si sretan, i sunce za tobom žuri.".encode()

c = cmac.CMAC(aes)
c.update(poruka)

cmac_poruke = c.finalize()
print("CMAC poruke", poruka, "je", cmac_poruke)
```

Poruka i njen CMAC šalju se drugoj strani koja ima ključ dobiven razmjenom ključeva opisanom iznad. Provjera poruke se vrši na način:

``` python
c_provjera = cmac.CMAC(aes)
c_provjera.update(poruka)
c_provjera.verify(cmac_poruke)
```

!!! admonition "Zadatak"
    Iskoristite neki drugi algoritam umjesto AES-a, npr. `TripleDES`, `Blowfish` ili `ChaCha20`.

## Dvofaktorska autentifikacija i jednokratne zaporke

Dvofaktorska autentifikacija uz standardnu zaporku koristi i jednokratnu zaporku koja se generira prilikom prijave i šalje korisniku drugim kanalom, primjerice e-mailom ili SMS-om. Dva su pristupa stvaranju jednokratnih zaporki:

- [HOTP: An HMAC-Based One-Time Password Algorithm (RFC 4226)](https://datatracker.ietf.org/doc/html/rfc4226.html)
- [TOTP: Time-Based One-Time Password Algorithm (RFC 6238)](https://datatracker.ietf.org/doc/html/rfc6238)

### Jednokratne zaporke temeljene na HMAC-u

Inicijalizacija HOTP-a ([dokumentacija](https://cryptography.io/en/latest/hazmat/primitives/twofactor/)) vrši se na način:

``` python
import os
from cryptography.hazmat.primitives.twofactor.hotp import HOTP
from cryptography.hazmat.primitives.hashes import SHA1

key = os.urandom(20)

hotp = HOTP(key, 6, SHA1())
```

Drugi parametar kod HOTP-a je broj znakova koje će generirane zaporke imati. Nakon inicijalizacije moguće je generirati niz jednokratnih zaporki:

``` python
hotp_value0 = hotp.generate(0)
hotp_value1 = hotp.generate(1)
hotp_value2 = hotp.generate(75)

print("Jednokratne zaporke su", hotp_value0, hotp_value1, hotp_value2)
```

Generirane vrijednosti se tada šalju nekim kanalom kojem jedino korisnik koji se prijavljuje ima pristup. Prima se korisnikov unos i provjerava na način:

``` python
hotp.verify(hotp_value0, 0)
hotp.verify(hotp_value1, 1)
hotp.verify(hotp_value2, 75)
```

!!! admonition "Zadatak"
    - Provjerite prolaze li HOTP vrijednosti provjeru kad su navedeni pogrešni redni brojevi.
    - Provjerite možete li generirati i provjeravati HOTP vrijednosti u proizvoljnom poretku.

### Jednokratne zaporke temeljene na vremenu

Inicijalizacija TOTP-a ([dokumentacija](https://cryptography.io/en/latest/hazmat/primitives/twofactor/)) vrši se na način:

``` python
import os
import time
from cryptography.hazmat.primitives.twofactor.totp import TOTP
from cryptography.hazmat.primitives.hashes import SHA1

key = os.urandom(20)

totp = TOTP(key, 6, SHA1(), 30)
```

Drugi parametar kod TOTP-a je, kao i kod HOTP-a, broj znakova koje će generirane zaporke imati. Posljednji navedeni parametar je vremenski korak u sekundama, odnosno interval nakon kojeg će doći iduća jednokratna zaporka.

Nakon inicijalizacije moguće je generirati niz jednokratnih zaporki navođenjem vremena:

``` python
time_moment1 = time.time()
totp_value1 = totp.generate(time_moment1)
print("Zaporka stvorena u vremenskom trenutku", time_moment1, "je", totp_value1)

time.sleep(30)

time_moment2 = time.time()
totp_value2 = totp.generate(time_moment2)
print("Zaporka stvorena u vremenskom trenutku", time_moment2, "je", totp_value2)
```

Generirana zaporka šalje se korisniku i od njega prima, a zatim se provjera vrši na način:

``` python
totp.verify(totp_value1, time_moment1)
totp.verify(totp_value2, time_moment2)
```

!!! admonition "Zadatak"
    - Provjerite prolaze li TOTP vrijednosti provjeru kad su navedeni pogrešni vremenski intervali.
    - Provjerite možete li generirati i provjeravati TOTP vrijednosti za proizvoljni vremenski trenutak koji nije sadašnji. (Uputa: iskoristite funkcije `time.struct_time()` i `time.mktime()` za stvaranje vremenskog trenutka u adekvatnom obliku.)

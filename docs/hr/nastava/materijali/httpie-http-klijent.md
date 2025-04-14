---
author: Matea Turalija, Vedran Miletić
---

# Korištenje REST API-ja pomoću HTTP klijenta HTTPie

[HTTPie](https://httpie.io/) je moderan HTTP klijent koji koristi sučelje naredbenog retka i primarno se koristi za testiranje REST API-ja. Podržava isticanje sintakse JSON-a, HTML-a, CSS-a i JavaScripta te se njegova funkcionalnost može dodatno proširiti korištenjem priključaka.

U primjerima u nastavku za to koristimo HTTPie pomoću naredbe `httpie`. Osim toga, alat je dostupan pomoću naredbi `http` i `https`:

``` shell
httpie
```

``` shell-session
usage: httpie [-h] [--debug] [--traceback] [--version] {plugins} ...
httpie: error: Please specify one of these: 'plugins'

This command is only for managing HTTPie plugins.
To send a request, please use the http/https commands:

  $ http POST pie.dev/post hello=world

  $ https POST pie.dev/post hello=world
```

## Aplikacijsko programsko sučelje

Aplikacijsko programsko sučelje (engl. *Application Programming Interface*, skraćenica **API**) predstavlja sučelje programskog dijela aplikacije. API je softverski posrednik koji omogućuje dvijema aplikacijama da razmjenjuju informacije.

API možete zamisliti kao konobara u restoranu. Vi kao gost, sjedite za stolom s izborom jelovnika po narudžbi, a kuhinja je dobavljač koji će ispuniti vašu narudžbu. Potrebna vam je poveznica da odnesete svoju narudžbu kuharu, a onda i da hrana stigne do vas. Trebate nešto da povežete gosta koji ima zahtjev i stranu koja odgovara na zahtjev.

Ovdje API preuzima zadatak. Poslužitelj (server) preuzima zahtjev od klijenta i dostavlja ga gdje je potrebno. On vam tada dostavlja odgovor. Ako je API ispravno dizajniran, vaša se narudžba neće izgubiti u procesu.

Podaci za API pohranjeni su u bazi podataka koja se nalazi na poslužitelju. Da biste imali pristup tim podacima, morate na neki način pitati što želite. Primjerice, kada u web-preglednik upišete URL da biste posjetili stranicu, odmah šaljete zahtjev poslužitelju za informacijama koje su vam potrebne za pregled te web stranice. Kada želite pogledati videoisječak na Internetu, kada kliknete na gumb za početak snimanja, poslužitelju se automatski šalje zahtjev za pristup datoteci u kojoj se nalazi snimka. API ima zadatak prenijeti zahtjev s jedne strane na drugu, od pošiljatelja do primatelja. To je ono što API radi.

Postoji nekoliko različitih tipova API-ja. Web API je najčešće korišten, ali postoje i drugi, primjerice *REpresentational State Transfer*, kraće REST. REST označava način komunikacije između klijenta i poslužitelja prilikom korištenja mrežnih resursa pomoću HTTP protokola. HTTP pruža operacije poznatijih kao metode kao što su GET, POST, PUT i DELETE.

Web API koji poštuje REST ograničenja kažemo da je *RESTful*.

Resursima (podacima) se pristupa pomoću jedinstvenih identifikatora resursa (engl. *Uniform Resource Identifiera*, kraće URI). Recimo, možemo imati RESTful web servis koji putem URI-ja `https://jsonplaceholder.typicode.com/users` korištenjem HTTP metode GET

``` shell
httpie GET https://jsonplaceholder.typicode.com/users
```

omogućuje dohvaćanje popisa svih korisnika koji je oblika:

``` json
[
  {
    "id": 1,
    "name": "Leanne Graham",
    "username": "Bret",
    "email": "Sincere@april.biz",
    "address": {
      "street": "Kulas Light",
      "suite": "Apt. 556",
      "city": "Gwenborough",
      "zipcode": "92998-3874",
      "geo": {
        "lat": "-37.3159",
        "lng": "81.1496"
      }
    },
    "phone": "1-770-736-8031 x56442",
    "website": "hildegard.org",
    "company": {
      "name": "Romaguera-Crona",
      "catchPhrase": "Multi-layered client-server neural-net",
      "bs": "harness real-time e-markets"
    }
  },
  {
    "id": 2,
    "name": "Ervin Howell",
    "username": "Antonette",
    "email": "Shanna@melissa.tv",
    "address": {
      "street": "Victor Plains",
      "suite": "Suite 879",
      "city": "Wisokyburgh",
      "zipcode": "90566-7771",
      "geo": {
        "lat": "-43.9509",
        "lng": "-34.4618"
      }
    },
    "phone": "010-692-6593 x09125",
    "website": "anastasia.net",
    "company": {
      "name": "Deckow-Crist",
      "catchPhrase": "Proactive didactic contingency",
      "bs": "synergize scalable supply-chains"
    }
  },
...
]
```

Za usporedbu, putem URI-ja `/users/5` može se dohvatiti korisnik s identifikatorom 5

``` shell
httpie GET https://jsonplaceholder.typicode.com/users/5
```

i primljeni odgovor će biti oblika:

``` json
{
  "id": 5,
  "name": "Chelsey Dietrich",
  "username": "Kamren",
  "email": "Lucio_Hettinger@annie.ca",
  "address": {
    "street": "Skiles Walks",
    "suite": "Suite 351",
    "city": "Roscoeview",
    "zipcode": "33263",
    "geo": {
      "lat": "-31.8129",
      "lng": "62.5342"
    }
  },
  "phone": "(254)954-1289",
  "website": "demarco.info",
  "company": {
    "name": "Keebler LLC",
    "catchPhrase": "User-centric fault-tolerant solution",
    "bs": "revolutionize end-to-end systems"
  }
}
```

U nastavku ćemo implementirati web servis koji na upite na te URI-je vraća odgovore s navedenim sadržajima. Naravno, pored čitanja podataka, naš će web servis omogućavati i druge radnje nad podacima kao što su stvaranje, osvježavanje i brisanje (engl. *create, read, update, and delete*, kraće **CRUD**).

## CRUD i HTTP metode

Korištenjem HTTP metoda i URI-ja možemo izvesti sve četiri navedene radnje nad podacima.

- Stvaranje (engl. *create*) vrši se:
    - HTTP metodom **POST** na URI `/users` stvara se korisnik s prvim slobodnim identifikatorom ili
    - HTTP metodom **PUT** na URI `/users/{id}`, pri čemu je `{id}` dotad neiskorišten identifikator korisnika, stvara se novi korisnik s navedenim identifikatorom.

- Čitanje (engl. *read*) vrši se:
    - HTTP metodom **GET** na URI `/users` dohvaćaju se svi korisnici ili
    - HTTP metodom **GET** na URI `/users/{id}`, pri čemu je `{id}` identifikator korisnika, dohvaća se korisnik s navedenim identifikatorom.

- Osvježavanje (engl. *update*) vrši se:
    - HTTP metodom **PUT** na URI `/users/{id}`, pri čemu je `{id}` identifikator korisnika, osvježava se čitav korisnik.

- Brisanje (engl. *delete*) vrši se:
    - HTTP metodom **DELETE** na URI `/users/{id}`, pri čemu je `{id}` identifikator korisnika.

## Praktična vježba

U nastavku ćemo koristiti lažne podatke s [JSONPlaceholder-a](https://jsonplaceholder.typicode.com/). JSONPlaceholder dolazi s 6 uobičajenih resursa:

- [/posts](https://jsonplaceholder.typicode.com/posts): 100 posts
- [/comments](https:/[/](https://jsonplaceholder.typicode.com/comments)): 500 comments
- [/albums](https://jsonplaceholder.typicode.com/albums): 100 albums
- [/photos](https://jsonplaceholder.typicode.com/photos): 5000 photos
- [/todos](https://jsonplaceholder.typicode.com/todos): 200 todos
- [/users](https://jsonplaceholder.typicode.com/users): 10 users

Spomenute HTTP metode možete vježbati s navedenim podacima na online besplatnom alatu [REST test test...](https://resttesttest.com/)

Odredite HTTP metodu i URI te kliknite na Ajax zahtjev. Primjerice,

- za metodu **GET**:

    ``` text
    https://jsonplaceholder.typicode.com/users/2
    ```

    - U navedenim resursima postoji 10 korisnika, što će se dogoditi ako stavimo npr. `{id}` korisnika 85?
    - Dohvatite ostale podatke za `posts`, `albums`, `photos` i `todos`.

- za metodu **POST**:

    ``` text
    https://jsonplaceholder.typicode.com/users
    ```

    U ovom primjeru stvara se korisnik s `"id": 11`.

- za metodu **PUT**:

    ``` text
    https://jsonplaceholder.typicode.com/users/1
    ```

    Ako se u PUT navede `{id}` već postojećeg korisnika, tada metoda PUT služi za ažuriranje korisnika. U slučaju kada `{id}` korisnika prvi put navodimo tada se metoda PUT ponaša kao POST, dakle stvara se novi korisnik.

- za metodu **DELETE**:

    ``` text
    https://jsonplaceholder.typicode.com/users/5
    ```

    Prilikom vježbe možda ćete se susresti s nekim HTTP statusnim kodovima. Značenje statusnih kodova možete pogledati [ovdje](https://www.restapitutorial.com/httpstatuscodes.html).

## Zadaci

Radi jedostavnosti, možete se ograničiti na korištenje metoda danih u tablici:

| HTTP metode | Uloga (u okviru CRUD) |
| ----------- | ----------- |
| POST | Stvaranje (engl. _**C**reate_) |
| GET | Čitanje (engl. _**R**ead_) |
| PUT | Osvježavanje (engl. _**U**pdate_) |
| DELETE | Brisanje (engl. _**D**elete_) |

### Upute za rješavanje

- Sve zadatke rješavat ćete u online besplatnom alatu **REST test test...**: `https://resttesttest.com/`
- Kod odabira `URI`-ja uvijek koristite `https://`
- `URI` i odgovore pišete unutar triju kvačica: ` ``` # moj primjer ``` `
- Za pojedine zadatke potrebno je napisati koja je metoda korištena: `POST`, `GET`, `PUT` ili `DELETE`

### 1. zadatak

Koristeći se podacima s JSONPlaceholder-a dostupnog na adresi `https://jsonplaceholder.typicode.com` koji ima

`/posts` (100 posts)
`/comments` (500 comments)
`/albums` (100 albums)
`/photos` (5000 photos)
`/todos` (200 todos)
`/users` (10 users)

- Dohvatite točno jedan ToDo po želji.

**Korištena metoda**:

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor**:

``` json
# ovdje pišete odgovor koji ste zaprimili
```

- Obrišite sliku rednog broja 10.

**Korištena metoda**:

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor**:

``` json
# ovdje pišete odgovor koji ste zaprimili
```

- Ažurirajte komentar rednog broja 15.

**Korištena metoda**:

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor**:

``` json
# ovdje pišete odgovor koji ste zaprimili
```

- Kreirajte novu objavu.

**Korištena metoda**:

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor**:

``` json
# ovdje pišete odgovor koji ste zaprimili
```

## 2. Zadatak

Koristeći se Public API-jem izlistajte sve kategorije koje se nude.

**Napomena:** primjer `URI-ja` pronađite na web sjedištu: `https://github.com/davemachado/public-api`

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor**:

``` json
# ovdje pišete odgovor koji ste zaprimili
```

## 3. Zadatak

**Wayback machine** jednostavan je API za test da se vidi je li dati `URI` arhiviran i trenutno dostupan u Wayback Machineu.

Ovaj API je koristan za pružanje 404 ili drugog rukovatelja pogreškama koji provjerava ima li arhiviranu kopiju spremnu za prikaz. Kada je dostupan, `URI` je veza na arhiviranu snimku u Wayback Machineu.

Provjerite je li spremljena snimka web sjedišta `gaseri.org`.

**Napomena:** primjer `URI-a` pronađite na web sjedištu: `https://archive.org/help/wayback_api.php`

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor**:

``` json
# ovdje pišete odgovor koji ste zaprimili
```

## 4. Zadatak

Koristeći se Cat Facts API-jem saznajte nekoliko činjenica o mačkama.

**Napomena:** primjer `URI-a` pronađite na web sjedištu: `https://catfact.ninja/#/Breeds`

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor**:

> ovdje pišete 1. činjenicu
>
> ovdje pišete 2. činjenicu
>
> ovdje pišete 3. činjenicu

## 5. Zadatak

Koristeći se podacima HTTPbin-a na adresi: `https://httpbin.org/`

### HTTP metode

- Iskoristite parametar DELETE zahtjeva.

**Korištena metoda**:

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor**:

``` json
# ovdje pišete odgovor koji ste zaprimili
```

- Iskoristite parametar PUT zahtjeva da biste postavili naziv parametra `ime`, a njegovu vrijednost po želji.

**Korištena metoda**:

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor**:

``` json
# ovdje pišete odgovor koji ste zaprimili
```

### Statusni kodovi

- Postavite statusni kod 200.

**Korištena metoda**:

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor statusnog koda**:

``` text
# ovdje pišete odgovor koji ste zaprimili
```

- Dohvatite statusni kod 500. Objasnite što znači taj statusni kod.

**Korištena metoda**:

**URI**:

``` text
# ovdje pišete URI koji ste koristili
```

**Odgovor statusnog koda**:

``` text
# ovdje pišete odgovor koji ste zaprimili
```

**Objašnjenje značenja statusnog koda:**:

> ovdje pišete objašnjenje odgovora koji ste zaprimili

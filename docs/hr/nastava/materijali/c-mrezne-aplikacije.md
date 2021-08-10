---
author: Vedran Miletić
---

# Programiranje mrežnih aplikacija u programskim jezicima C i C++

[BSD sockets](https://en.wikipedia.org/wiki/Berkeley_sockets), poznati i kao "Berkeley sockets" i kao "Unix sockets", su *de facto* standardno sučelje za međuprocesnu komunikaciju lokalno i preko računalne mreže. Oni koriste klijent-poslužitelj model komunikacije.

Nastaju 1983. kada Bill Joy i ekipa s UCB-a izbacuje 4.2BSD koji prvi puta uvodi mrežne utičnice, a 2008. godine preimenovani su u **POSIX sockets** i standardizirani; osnovne razlike između tog standarda i ranijih su u imenovanju funkcija.

Prema adresiranju mogu se podijeliti na:

- Unix domain sockets -- `AF_UNIX`

    - formalno koriste datotečni sustav kao prostor adresa (sama komunikacija ne ide preko datoteka)
    - procesi međusobno mogu slati podatke i opisnike datoteka

- Unix network sockets -- `AF_INET` (IP verzija 4) i `AF_INET6` (IP verzija 6)

    - adresa je uređeni par oblika `(ip, port)`
    - komunikacija se može vršiti lokalno ili putem Interneta

Prema pouzdanosti mogu se podijeliti na:

- datagramski -- `SOCK_DGRAM`

    - omogućuju jednosmjernu komunikaciju kod koje klijent šalje podatke, a poslužitelj prima
    - nema mogućnosti baratanja s više klijenata, svi podaci stižu na jednu utičnicu bez obzira koji od klijenata ih šalje
    - nema osiguranja da će poslani podaci stići

- tokovni -- `SOCK_STREAM`

    - imaju stvarnu konekciju dvije strane, garantiraju dostavu poruka
    - omogućuju dvosmjernu komunikaciju
    - češće korištene

Zaglavlje `sys/socket.h` ([dokumentacija](https://www.gnu.org/software/libc/manual/html_node/Sockets.html)) nudi pristup POSIX socket sučelju pod operacijskim sustavima sličnim Unixu.

## Poslužiteljska strana

U ovom primjeru napisat ćemo poslužiteljsku stranu mrežne aplikacije. Pored zaglavlja uobičajenih za programski jezik C koja nude često korištene pomoćne funkcije, uključujemo i `sys/socket.h` za baratanje socketima i `netinet/in.h` za baratanje internetskim adresama.

``` c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
```

Definiramo pomoćnu funkciju `error()` koja prima dani niz znakova i ispisuje ga na ekran kao grešku korištenjem funkcije `perror()`, a zatim završava izvođenje programa korištenjem funkcije `exit()` i argumentom različitim od 0, koji označava da program nije završio izvođenje na uobičajen način.

``` c
void
error (const char *msg)
{
  perror (msg);
  exit (1);
}
```

Unutar funkcije `main()` provjerava se je li korisnik unio dovoljan broj argumenata; u slučaju da nije, program završava izvođenje.

``` c
int
main (int argc, char *argv[])
{
  if (argc < 2)
    {
      fprintf (stderr, "ERROR, no port provided\n");
      exit (1);
    }
```

Zatim deklariramo varijable koje ćemo kasnije koristiti:

- `sockfd` je socket na koji se klijent spaja, `newsockfd` je socket koji će biti stvoren za klijenta koji se spoji i biti korišten za komunikaciju s njim,
- `portno` je broj TCP vrata na koji se klijent spaja, a `n` je pomoćni broj u koji se hvata povratna vrijednost funkcija `read()` i `write()` koje će se kasnije korsititi,
- `clilen` je duljina adrese klijenta (kod IP adresa verzije 4 to je 32 bita, ali općenito ne mora biti),
- `serv_addr` i `cli_addr` su adrese poslužitelja i klijenta.

``` c
int sockfd, newsockfd, portno, n;
char buffer[256];
socklen_t clilen;
struct sockaddr_in serv_addr, cli_addr;
```

Vršimo stvaranje socketa, i javljamo grešku u slučaju da nije uspješno.

``` c
sockfd = socket (AF_INET, SOCK_STREAM, 0);
if (sockfd < 0)
  error ("ERROR opening socket");
```

Postavljamo bajtove u memoriji koja pripada strukturi `serv_addr` na vrijednost 0, kako bi u njih mogli zapisati familiju adresa (ovdje `AF_INET`) i samu adresu (ovdje `INADDR_ANY`, što znači da će poslužitelj primati sa svih adresa koje poznaje kao svoje). Zatim postavljamo vrijednost broja vrata na vrijednost koju je korisnik dao kao argument, uz prethodnu pretvorbu znakovnog niza u cijeli broj korištenjem funkcije `atoi()` ([C dokumentacija](https://en.cppreference.com/w/c/string/byte/atoi), [C++ dokumentacija](https://en.cppreference.com/w/cpp/string/byte/atoi)). Funkcija `htons()` služi za pretvorbu zapisa broja iz domaćinskog u mrežni, [u slučaju da nisu isti po pitanju poretka bajtova](https://en.wikipedia.org/wiki/Endianness).

``` c
memset ((char *) &serv_addr, 0, sizeof (serv_addr));
serv_addr.sin_family = AF_INET;
serv_addr.sin_addr.s_addr = INADDR_ANY;
portno = atoi (argv[1]);
serv_addr.sin_port = htons (portno);
```

Nakon stvaranja socketa i postavljanja adrese slijedi `bind()`, odnosno pridruživanje socketa adresi, te `listen()`, odnosno postavljanje socketa u stanje u kojem čeka povezivanje klijenata. U slučaju da dođe do greške, ona se ispisuje na ekran.

``` c
if (bind (sockfd, (struct sockaddr *) &serv_addr,
          sizeof (serv_addr)) < 0)
  error ("ERROR on binding");
listen (sockfd, 5);
```

Nakon povezivanja klijenta prihvaća ga se funkcijom `accept()`, a njegova se adresa sprema u varijablu `cli_addr`. U slučaju da dođe do greške, ona se ispisuje na ekran.

``` c
clilen = sizeof (cli_addr);
newsockfd = accept (sockfd,
                    (struct sockaddr *) &cli_addr,
                    &clilen);
if (newsockfd < 0)
  error ("ERROR on accept");
```

Sve vrijednosti u međuspremniku se postavljaju na 0, a zatim se vrši čitanje (funkcija `read()`). Vrijednost varijable `n` bit će broj pročitanih znakova; u slučaju da dođe do greške, bit će manji od 0 i greška će se ispisati na ekran. Zatim se na ekrani ispisuje `Here is the message:` i sadržaj poruke koju je klijent poslao.

``` c
memset (buffer, 0, 256);
n = read (newsockfd, buffer, 255);
if (n < 0) error ("ERROR reading from socket");
printf ("Here is the message: %s\n", buffer);
```

Klijentu se šalje `I got your message` poruka zapisivanjem u socket ((funkcija `write()`). Vrijednost varijable `n` bit će broj zapisanih znakova; u slučaju da dođe do greške, bit će manji od 0 i greška će se ispisati na ekran.

``` c
n = write (newsockfd, "I got your message", 18);
if (n < 0) error ("ERROR writing to socket");
```

Zatvara se prvo socket koji je korišten za komunikaciju s tim klijentom, a zatim i socket koji služi za povezivanje klijenata. Naredba `return 0;` završava funkciju `main()`.

``` c
  close (newsockfd);
  close (sockfd);
  return 0;
}
```

Cjelokupan programski primjera koji opisujemo u dijelovima dan je postavljen ispod kako bi olakšali kopiranje koda za isprobavanje i vježbu.

``` c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

void
error (const char *msg)
{
  perror (msg);
  exit (1);
}

int
main (int argc, char *argv[])
{
  if (argc < 2)
    {
      fprintf (stderr, "ERROR, no port provided\n");
      exit (1);
    }
  int sockfd, newsockfd, portno;
  socklen_t clilen;
  char buffer[256];
  struct sockaddr_in serv_addr, cli_addr;
  int n;
  sockfd = socket (AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0)
    error ("ERROR opening socket");
  memset ((char *) &serv_addr, 0, sizeof (serv_addr));
  portno = atoi (argv[1]);
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons (portno);
  if (bind (sockfd, (struct sockaddr *) &serv_addr,
            sizeof (serv_addr)) < 0)
    error ("ERROR on binding");
  listen (sockfd, 5);
  clilen = sizeof (cli_addr);
  newsockfd = accept (sockfd,
                      (struct sockaddr *) &cli_addr,
                      &clilen);
  if (newsockfd < 0)
    error ("ERROR on accept");
  memset (buffer, 0, 256);
  n = read (newsockfd, buffer, 255);
  if (n < 0) error ("ERROR reading from socket");
  printf ("Here is the message: %s\n", buffer);
  n = write (newsockfd, "I got your message", 18);
  if (n < 0) error ("ERROR writing to socket");
  close (newsockfd);
  close (sockfd);
  return 0;
}
```

## Klijentska strana

Sad ćemo napisati i klijentsku stranu za ranije napisani poslužitelj. Zaglavlja su ista, osim što klijentska strana uključuje i `netdb.h` koje nam nudi funkciju za pretvorbu DNS adresa u IP adrese.

``` c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
```

Funkcija za ispis greške ista je kao kod poslužitelja.

``` c
void
error (const char *msg)
{
  perror (msg);
  exit (1);
}
```

Kao i kod poslužitelja, unutar funkcije `main()` provjerava se je li korisnik unio dovoljan broj argumenata; u slučaju da nije, program završava izvođenje.

``` c
int
main (int argc, char *argv[])
{
  if (argc < 3)
    {
      fprintf (stderr, "usage %s hostname port\n", argv[0]);
      exit (1);
    }
```

Zatim deklariramo varijable koje ćemo kasnije koristiti:

- `sockfd` je socket koji klijent koristi za komunikaciju s poslužiteljem,
- `portno` je broj TCP vrata na poslužitelju na koji se klijent spaja, a `n` je pomoćni broj u koji se hvata povratna vrijednost funkcija `read()` i `write()` koje će se kasnije korsititi,
- `server` je struktura koja sadrži informacije o poslužitelju,
- `serv_addr` je adresa poslužitelja.

``` c
int sockfd, portno, n;
struct sockaddr_in serv_addr;
struct hostent *server;
char buffer[256];
```

Vršimo stvaranje socketa, i javljamo grešku u slučaju da nije uspješno.

``` c
sockfd = socket (AF_INET, SOCK_STREAM, 0);
if (sockfd < 0)
  error ("ERROR opening socket");
```

Funkcija `gethostbyname()` prima kao argument niz znakova koje je korisnik unio kao argument kod pokretanja programa. Taj niz znakova može biti IP ili DNS adresa zapisana kao niz znakova (obzirom da ćemo koristiti klijent i poslužitelj na istom računalu, to će biti "127.0.0.1" ili "localhost").

``` c
server = gethostbyname (argv[1]);
if (server == NULL)
  {
    fprintf (stderr, "ERROR, no such host\n");
    exit (1);
  }
```

Postavljamo bajtove u memoriji koja pripada strukturi `serv_addr` na vrijednost 0, kako bi u njih mogli zapisati familiju adresa (ovdje `AF_INET`) i samu adresu poslužitelja. Adresu poslužitelja dobivamo iz rezultata pretrage funkcijom `gethostbyname()`, koji je spremljen u varijablu `server`. Odatle funkcijom `memcpy()` vršimo kopiranje adrese u `serv_addr`.

Zatim postavljamo vrijednost broja vrata na vrijednost koju je korisnik dao kao argument. Funkcije `atoi()` i `htons()` imaju isti cilj kao i kod poslužitelja.

``` c
memset ((char *) &serv_addr, 0, sizeof (serv_addr));
serv_addr.sin_family = AF_INET;
memcpy ((char *) &serv_addr.sin_addr.s_addr,
       (char *) server->h_addr,
       server->h_length);
portno = atoi (argv[2]);
serv_addr.sin_port = htons (portno);
```

Povezivanje na server vrši se funkcijom `connect()`; ako je rezultat koji ona vrati manji od 0, došlo je do pogreške i informacija o tome ispisuje se na ekran.

``` c
if (connect (sockfd, (struct sockaddr *) &serv_addr, sizeof (serv_addr)) < 0)
  error ("ERROR connecting");
```

Od korisnika se traži unos poruke koja će se poslati, međuspremnik za poruku se postavlja na vrijednost 0, a zatim se poruka koju korisnik unese zapisuje u njega.

``` c
printf ("Please enter the message: ");
memset (buffer, 0, 256);
fgets (buffer, 255, stdin);
```

Poruka se zapisuje u socket korištenjem funkcije `write()`. Vrijednost varijable `n` bit će broj zapisanih znakova; u slučaju da dođe do greške, bit će manji od 0 i greška će se ispisati na ekran.

``` c
n = write (sockfd, buffer, strlen (buffer));
if (n < 0)
  error ("ERROR writing to socket");
```

Vrijednost međuspremnika se ponovno postavlja na 0, i u njega se učitava korištenjem funkcije `read()` poruka primljena od poslužitelja. Vrijednost varijable `n` bit će broj pročitanih znakova; u slučaju da dođe do greške, bit će manji od 0 i greška će se ispisati na ekran. Poruka se zatim ispisuje na ekran.

``` c
memset (buffer, 0, 256);
n = read (sockfd, buffer, 255);
if (n < 0)
  error ("ERROR reading from socket");
printf ("%s\n", buffer);
```

Vrši se zatvaranje socketa. Naredba `return 0;` završava funkciju `main()`.

``` c
  close (sockfd);
  return 0;
}
```

Cjelokupan programski primjera koji opisujemo u dijelovima dan je postavljen ispod kako bi olakšali kopiranje koda za isprobavanje i vježbu.

``` c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

void
error (const char *msg)
{
  perror (msg);
  exit (1);
}

int
main (int argc, char *argv[])
{
  if (argc < 3)
    {
      fprintf (stderr, "usage %s hostname port\n", argv[0]);
      exit (1);
    }
  int sockfd, portno, n;
  struct sockaddr_in serv_addr;
  struct hostent *server;
  char buffer[256];
  portno = atoi (argv[2]);
  sockfd = socket (AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0)
    error ("ERROR opening socket");
  server = gethostbyname (argv[1]);
  if (server == NULL)
    {
      fprintf (stderr, "ERROR, no such host\n");
      exit (1);
    }
  memset ((char *) &serv_addr, 0, sizeof (serv_addr));
  serv_addr.sin_family = AF_INET;
  memcpy ((char *) &serv_addr.sin_addr.s_addr,
         (char *) server->h_addr,
         server->h_length);
  serv_addr.sin_port = htons (portno);
  if (connect (sockfd, (struct sockaddr *) &serv_addr, sizeof (serv_addr)) < 0)
    error ("ERROR connecting");
  printf ("Please enter the message: ");
  memset (buffer, 0, 256);
  fgets (buffer, 255, stdin);
  n = write (sockfd, buffer, strlen (buffer));
  if (n < 0)
    error ("ERROR writing to socket");
  memset (buffer, 0, 256);
  n = read (sockfd, buffer, 255);
  if (n < 0)
    error ("ERROR reading from socket");
  printf ("%s\n", buffer);
  close (sockfd);
  return 0;
}
```

## Pregled korištenih funkcija

Srž programa je, kronološki poredano:

- poslužitelj: `bind()`
- poslužitelj: `listen()`
- klijent: `connect()`
- poslužitelj: `accept()`
- klijent: `write()`
- poslužitelj: `read()`
- poslužitelj: `write()`
- klijent: `read()`
- klijent: `close()`
- poslužitelj: `close()` socketa koji je pripadao klijentu, `close()` socketa za povezivanje klijenata

## Prevođenje i pokretanje programa

Spremite li kod programa u datoteke `server.c` i `client.c` unutar kućnog direktorija, tada u terminalu možete naredbom:

``` shell
$ gcc -o server server.c
$ gcc -o client client.c
```

izvesti prevođenje oba programa. Zatim ćemo pokrenuti poslužitelj i klijent u dva odvojena terminala i pritom odabrati TCP vrata preko kojih će komunicirati (u našem primjeru 5000). U prvom terminalu pokrećemo poslužitelj naredbom:

``` shell
$ ./server 5000
```

U drugom terminalu pokrećemo klijent naredbom:

``` shell
$ ./client localhost 5000
```

Umjesto vrata 5000 možete koristiti bilo koja u rasponu 1024--65535 dostupna običnim korisnicima; vrata 0--1023 može otvoriti samo korisnik `root`.

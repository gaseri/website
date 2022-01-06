---
author: Luka Vretenar
---

# Regularni izrazi i konačni automati

## Regularni izrazi

- Nizovi znakova sa posebnim značenjem koji nam služe za pretragu i opisivanje nekog ulaznog niza znakova.
- Način specificiranja konačnog automata koji procesira zadani ulazni niz znakova.
- Imaju podršku u većini programskih jezika i alata koji se danas koriste.
- Najčešće nam služe za:

    - opisivanje uzoraka u tekstu i pretragu teksta
    - izmjenu i manipulaciju opisanih uzoraka u tekstu

- Neke od primjena regularnih izraza:

    - sustavi za bojanje sintakse (`syntax highlighting`)
    - provjera validnosti podataka kod korisničkog unosa (`form validation`, `text validation`)
    - eskstrakcija pojedinih elemenata iz formatiranog niza

- Primjer regularnog izraza koji opisuje sve prirodne brojeve u nekom tekstu: `[1-9][0-9]*`.
- Kako bi nam olakšali izradu i testiranje regularnih izraza postoje razni pomoćni web alati:

    - [regex101](https://regex101.com/)
    - [RegExr](https://regexr.com/)
    - [regexpal](https://www.regexpal.com/)
    - [Regex Crossword](https://regexcrossword.com/)

- U ovim vježbama se koriste prošireni regularni izrazi ili `ERE`.

### Uzorci teksta i ostali znakovi

- Uzorak koji odgovara određenom nizu znakova možemo specificirati kao točno taj niz znakova.
- Ako pretražujemo tekst za niz `Dobar dan`, uzorak za traženje tog niza bi bio `Dobar dan`.

!!! hint
    Važno je paziti na mala i velika slova u uzorcima jer su regularni izrazi osjetljivi na veličinu slova.

- Ako želimo zadati uzorak koji na određenom mjestu ima bilo koji znak, onda možemo koristiti specijalni znak `.`.
- Primjer takvog uzorka je `d..r`, nizovi koji tom uzorku odgovaraju mogu biti:

    - `daar`
    - `dabr`
    - `dacr`
    - i tako dalje za sve kombinacije dva znaka na mjestima gdje je `.`

!!! admonition "Zadatak"
    - Zadan je tekst na dnu poglavlja.
    - Regularnim izrazom označiti sva pojavljivanja riječi `Perl`.
    - Regularnim izrazom označiti pojavljivanje sve riječi koje počinju sa `izraz` a završavaju bilo kojim slovom.

- Određeni znakovi u regularnim izrazima imaju posebno značenje, da bi ih koristili u uzorku kao znakove koje zapravo predstavljaju, potrebno ih je označiti znakom `\` ispred samog znaka.
- Ti znakovi su:

    - `\` i `/`
    - `^` i `$`
    - `.`
    - `|`
    - `?`
    - `*` i `+`
    - `(` i `)`
    - `[` i `]`
    - `{`

- Primjer je traženje izraza `$20`, kako sadrži jedan od posebnih znakova potrebno je napraviti regularni izraz `\$20` da bi ga mogli tražiti.

- Specijalni znakovi koji predstavljanju neki element niza su:

    - `.` -- bilo koji znak
    - `^` -- označava početak linije
    - `$` -- označava kraj linije
    - `\b` -- označava početak ili kraj riječi (gdje je riječ odvojena razmacima)

!!! admonition "Zadatak"
    - Regularnim izrazom označiti izraz `(automate)` uključujući i zagrade.
    - Regularnim izrazom označiti izraz `Perl` koji se nalazi na početku linije.

### Višestruki izbor i grupiranje

- Moguće je pretraživati izraze koji mogu biti sastavljeni od jednog ili više različitih podnizova.
- Da bi specificirali pretragu po jednom i drugom nizu koristimo simbol `|` kojem ih odvajamo.
- Regularni izraz `jedan|dva` će nam pronaći nizove `jedan` i `dva`.

!!! admonition "Zadatak"
    - Regularnim izrazom označiti sve nizove `reg` i `izraz` u zadanom tekstu.

- Ako želimo specificiratid da su pojedini podnizovi djelovi većeg niza koji tražimo, možemo grupirati pojedine višestruke izbore u odvojene grupe koristeći `(` i `)`.
- Regularni izraz `gr(a|e)y` će nam pronaći nizove `gray` i `grey`.

!!! admonition "Zadatak"
    - Regularnim izrazom označiti sve nizove `ama` i `aka` grupirajući po slovu `m` ili `k` u zadanom tekstu.

### Klase znakova

- Već smo vidjeli da sa `.` možemo specificirati da želimo bilo koji znak na određenom mjestu, no ako želimo da bude samo jedan od definiranih znakova možežmo koristiti klase znakova.
- Klasa znakova se definira unutar `[` i `]`, primjer `[abcd]` i `[0-9]`.
- Klasom znakova definiramo da određeni znak u nizu možež biti jedan od znakova u toj klasi.
- Regularni izraz `[Dd]obar dan` će nam pronaći nizove `Dobar dan` i `dobar dan`.

!!! admonition "Zadatak"
    - Pronaći sva velika slova u zadanom tekstu.
    - Pronaći sva velika slova i znamenke `0` i `1`.
    - Pronaći sve brojeve duljine `4` u tekstu, pronađeni niz ne smije biti sastavljen ododvojenih znakova.

- U klasama možemo definirati i raspone znakova koristeći `-` da odvojimo početni i krajnji znak u rasponu.
- Klasu `[abcdefghijklmnoprsquvwxz]` koja nam predstavlja sva mala slova, možemo zapisatiti i kao `[a-z]`.
- Ako želimo obrnuti znakove koje predstavlja klasa, koristimo znak `^` na početku klase.
- Regularni izraz `[^0-9]` će nam pronaći sve znakove koji nisu znamenke `0123456789`.

!!! admonition "Zadatak"
    - Pronaći sve znamenke koji nisu `0`.
    - Pronaći sva slova koja nisu u rasponu od `a` do `j` i koja nisu znakovi `(` i `)`.

### Ponavljanja izraza

- Ako želimo da se određeni znak, klasa znakova ili grupa znakova ponavlja nula, jednom ili više puta, možemo koristiti modifikatore ponavljanja.
- Modifikatori ponavljanja su `?`, `+` i `*` i nalaze se nakon znaka, klase ili grupe čije ponavljanje određuju.
- Značenja pojedinih modifikatora su:

    - `?` -- znak koji prethodi se može pojaviti jednom ili nijednom
    - `+` -- znak koji prethodi se može pojaviti jednom ili više puta
    - `*` -- znak koji prethodi se može pojaviti nijednom ili više puta

- Regularni izraz `[a-z]*` će nam pronaći sve nizove koji sadrže samo mala slova: `a`, `aa`, `ab`, `asdf`...
- Regularni izraz `abc?` će pronaći nizove `ab` i `abc`.

!!! admonition "Zadatak"
    - Pronaći sve brojeve duljine jedan ili više koristeći klasu i ponavljanja.
    - Pronaći sve nizove velikih slova koristeći klasu i ponavljanja.
    - Pronaći sve riječi koji započinju velikim slovom i završavaju malim slovima.

- Ponavljanja također možemo specificirati i brojčanim vrijednostima `m` i `n` u modifikatoru oblika `{m,n}` gdje su:

    - `{0,1}` -- ekvivalentno `?`
    - `{1,}` -- ekvivalentno `+`
    - `{0,}` -- ekvivalentno `*`
    - `{m,n}` -- znak, klasa ili grupa se ponavlja najmanje `m` a najviše `n` puta

!!! admonition "Zadatak"
    - Pronađi sve riječi koji su duljine veće od `8` znakova.

### Primjena u alatu `grep`

- Na svim Linux sustavima postoji alat koji nam omogućuje pretragu teksta koristeći regularne izraze.
- Alat `grep` podržava proširene regularne izraze i možemo ga pozvati slijedećom naredbom:

    ``` shell
    $ egrep "regex" datoteka
    ```

- Gdje je `regex` regularni izraz koji pretražujemo a `datoteka` naziv daoteke u kojoj pretražujemo.
- Izlaz iz programa su sve linije koje sadrže tražene izraze.

!!! admonition "Zadatak"
    - Koristeći alat `egrep` pretraživati datoteku `/etc/passwd` iz komandne linije.
    - Pronaći sve linije koje imaju izraze `/var` ili `/bin`.
    - Pronaći sve linije koje počinju sa nizom slova duljim od `4`.
    - Pronaći sve linije koje završavaju na `bash`.
    - Pronaći sve linije dulje od `50` znakova.
    - Pronaći sve brojeve koji započinju sa znamenkom `1`.

### Tekst za vježbe

```
Porijeklo regularnih izraza leži u teoriji automata i teoriji formalnih jezika, pri čemu su oboje discipline teoretskog računarstva.
Ove discipline proučavaju modele računanja (automate) te načine opisa i klasifikacije formalnih jezika.
Matematičar Stephen Kleene je 1950-ih opisao ove modele koristeći matematičku notaciju zvanu regularni skupovi.
Ken Thompson je ugradio ovu notaciju u uređivač QED, a potom i u Unix editor ed, što je s vremenom dovelo do uporabe regularnih izraza u grep-u.
Otad se regularni izrazi naširoko koriste u Unixu i Unixoidnim pomoćnim programima kao što su expr, awk, Emacs, vi, lex i Perl.

Perl i Tcl regularni izrazi su izvedeni iz regex biblioteke koju je napisao Henry Spencer, iako je Perl kasnije proširio Spencerovu regex biblioteku i dodao mnogo novih svojstava.
Philip Hazel je razvio PCRE (Perl Compatible Regular Expressions) koji jako dobro oponaša funkcionalnost regularnih izraza u Perlu, i kojeg koriste mnogi moderni programski alati kao što su PHP, ColdFusion, i Apache.
Dio napora uloženog u dizajn Perla 6 je baš u smjeru poboljšanja integracije Perlovih regularnih izraza, te povećanju njihovog područja djelovanja u svrhu dozvole definicije tzv. 'parsing expression grammar'.
Rezultat je mini-jezik zvan Perl 6 pravila, koja se koriste kako za definiciju gramatike Perla 6 tako i kao alat Perl programerima.
Ova pravila čuvaju sva svojstva regularnih izraza, ali i dozvoljavaju definicije u BNF stilu parsera tehnikom rekurzivnog spusta preko potpravila.

Korištenje regularnih izraza u strukturiranim informacijskim standardima (za modeliranje dokumenata i baza podataka) se pokazalo vrlo važnim, počevši od 1960-ih te se proširujući 1980-ih konsolidacijom industrijskih standarda kao što je ISO SGML.
Jezgra standarda jezika specifikacije strukture su regularni izrazi.
Jednostavnija i evidentnija uporaba jest u groupnoj sintaksi DTD-a.
```

Izvor: [Wikipedia](https://hr.wikipedia.org/wiki/Regularni_izraz)

## Konačni automati

- Regularni izraz zadan nizom specijalnih znakova se prije korištenja u računalu pretvori u konačni automat.
- Optimalan alat za izvršavanje regularnog izraza.
- Mogu opisivati i mnoge druge pojave vezane za svijet oko nas.

``` dot
digraph finite_state_machine {
  rankdir=LR;

  node [ shape = point ]; s
  node [ shape = doublecircle, label = "S1" ] S1;
  node [ shape = circle, label = "S2" ] S2;

  s -> S1;
  S1 -> S1 [ label = "1" ];
  S1 -> S2 [ label = "0" ];
  S2 -> S1 [ label = "0" ];
  S2 -> S2 [ label = "1" ];
}
```

- Čitanjem određenog ulaznog znaka, konačni automat prelazi u novo stanje i generira neki izlazni znak.
- Određeni konačni automat definiran je:

    - stupom stanja u kojima se taj automat može nalaziti
    - skupom ulaznih znakova koje taj automat može pročitati sa ulaza
    - skupom prijelaza koji definiraju kako automat prelazi u nova stanja

### Deterministički konačni automati

- Najjednostavniji oblik konačnog automata.
- Determinizam konačnog automata nam govori da za pojedini ulaz i određeno stanje postoji samo jedno novo stanje u koji taj automat može preći.
- Konačni automat možemo specificirati:

    - matematičkom definicijom
    - tablicom prijelaza
    - grafičkim prikazom

- Matematički, konačni automat možemo specificirati parom skupova: $A=(Q, \Sigma, \Delta, q_0, F)$

    - $Q$ -- skup svih stanja u kojima može biti automat
    - $\Sigma$ -- skup svih ulaznih znakova koje automat može pročitati
    - $\Delta$ -- skup svih funkcija prijelaza, funkcije su oblika $\delta(q, \sigma) = q$
    - $q_0$ -- početno stanje u kojem se nalazi automat prije čitanja znakova
    - $F$ -- skup svih prihvatljivih stanja, stanja u kojima prihvaćamo pročitani niz kao ispravan

!!! admonition "Zadatak"
    - Za konačni automat zadan matematičkom definicijom napraviti tablicu prijelaza i grafički prikaz:

    $$
    Q = \{S, A\};
    \Sigma = \{a, b\};
    q_0 = S;
    F = \{A\};
    \delta_0(S, a) = A;
    \delta_1(S, b) = A;
    \delta_2(A, a) = S;
    \delta_3(A, b) = A
    $$

    - Za isti automat provjeriti da li prihvaća ulazni niz znakova `baab`.

- Iz zadane matematičke definicije konstruiramo tablicu prijelaza:

    |   |   | a | b |   |
    | - | - | - | - | - |
    | > | S | A | A | 0 |
    |   | A | S | A | 1 |

- Grafički prikaz istog automata:

    ``` dot
    digraph finite_state_machine {
      rankdir=LR;

      node [ shape = point ]; s
      node [ shape = circle, label = "S" ] S;
      node [ shape = doublecircle, label = "A" ] A;

      s -> S;
      S -> A [ label = "a, b" ];
      A -> S [ label = "a" ];
      A -> A [ label = "b" ];
    }
    ```

- Provjera da li konačni automat prihvaća zadani niz `baab` se vrši postupkom:

    $$
    S \xrightarrow{b} A \xrightarrow{a} S \xrightarrow{a} A \xrightarrow{b} A
    $$

- Niz je ispravan i prihvaća se jer je završno stanje $A$ u skupu prihvatljivih stanja $F$.

!!! admonition "Zadatak"
    - Za konačni automat zadan matematičkom definicijom napraviti tablicu prijelaza i grafički prikaz:

    $$
    Q = \{pp, np, pn, nn\};
    \Sigma = \{0, 1\};
    q_0 = pp;
    F = \{pp, nn\}

    \delta_0(pp, 1) = pn;
    \delta_1(pp, 0) = np;
    \delta_2(pn, 1) = pp;
    \delta_3(pn, 0) = nn;
    \delta_4(np, 1) = nn;
    \delta_5(np, 0) = pp;
    \delta_6(nn, 1) = np;
    \delta_7(nn, 0) = pn
    $$

    - Za isti automat provjeriti da li prihvaća ulazni niz znakova `10110`.

---
author: Vedran Miletić
---

# Distribuirani sustav za upravljanje verzijama Git

Upravljanje verzijama podrazumijeva baratanje promjenama nad nekim skupom datoteka pamćenjem povijesti promjena. Spremište povijesti promjena nazivamo [repozitorijem](https://en.wikipedia.org/wiki/Repository_(revision_control)) (engl. *repository*). Obzirom na strukturu repozitorija, razlikujemo dva pristupa:

- [centralizirani pristup](https://en.wikipedia.org/wiki/Revision_control#Source-management_models), kod kojeg samo osnovni poslužitelj ima repozitorij, a klijenti imaju ono što od poslužitelja zatraže; ovaj pristup koriste CVS, Subversion i drugi
- [distribuirani pristup](https://en.wikipedia.org/wiki/Distributed_revision_control#Distributed_vs._centralized), kod kojeg svaki klijent ima repozitorij i time nestaje razlika između poslužitelja i klijenta; ovaj pristup koriste Git, Mercurial i drugi

[Git](https://git-scm.com/) je vrlo moćan i vrlo popularan alat za upravljanje verzijama koji koristi decentralizirani (distribuirani) pristup. Ime Git ima [vrlo duboko i višeslojno značenje](https://git.wiki.kernel.org/index.php/GitFaq#Why_the_.27Git.27_name.3F):

> Quoting Linus: "I'm an egotistical bastard, and I name all my projects after myself. First 'Linux', now 'Git'".
>
> ('git' is British slang for "pig headed, think they are always correct, argumentative").
>
> Alternatively, in Linus' own words as the inventor of Git: "git" can mean anything, depending on your mood:
>
> - Random three-letter combination that is pronounceable, and not actually used by any common UNIX command. The fact that it is a mispronunciation of "get" may or may not be relevant.
> - Stupid. Contemptible and despicable. Simple. Take your pick from the dictionary of slang.
> - "Global information tracker": you're in a good mood, and it actually works for you. Angels sing and light suddenly fills the room.
> - "Goddamn idiotic truckload of sh*t": when it breaks

Git svoju popularnost može uvelike zahvaliti servisima kao što su [GitHub](https://github.com/) i [Atlassian Bitbucket](https://bitbucket.org/) koji korisnicima omogućuju postavljanje repozitorija sa izvornim kodom na web. Putem tih servisa olakšana je suradnja, a uključuju i brojne druge značajke poznate sa društvenih mreža.

Više o Gitu može se saznati na [službenim stranicama](https://git-scm.com/). Specijalno, [službena dokumentacija](https://git-scm.com/doc) uključuje man stranice alata, knjigu Pro Git, video materijale i poveznice na tutoriale, knjige i video materijale. Brojni tutoriali su dostupni, a neki od najboljih su upravo GitHubov [Getting started](https://docs.github.com/en/get-started) i Atlassianov [Git Tutorials and Training](https://www.atlassian.com/git/tutorials).

## Osnovna konfiguracija

### Globalna konfiguracija

Globalna konfiguracija odnosi se na sve repozitorije i operira na datoteci `.gitconfig` unutar kućnog direktorija korisnika.

- ime, prezime i mail adresa

    - `git config --global user.name "Ime Prezime"`
    - `git config --global user.email user@mail.com`

- ostalo, ali ne i manje važno

    - `git config --global color.ui true` -- uključivanje korištenja boja u prikazu
    - `git config --global core.editor "emacs"` -- uređivač teksta koji se koristi postaje Emacs
    - `git config --global merge.tool "kdiff3"` -- alat za rad sa spajanjima grana koji se koristi postaje [KDiff3](https://apps.kde.org/kdiff3/)

### Lokalna konfiguracija

Lokalna konfiguracija odnosi se samo na trenutni repozitorij i operira na datoteci `.git/config` unutar repozitorija.

!!! todo
    Ovdje treba ubaciti uputu kako promijeniti origin.

## Sučelje naredbenog retka i grafička sučelja

### Pregled naredbi alata Git

- `git version` -- informacije o verziji gita
- `git help` -- ispis liste najčešće korištenih naredbi s pripadajućim opisom
- `git add` -- dodavanje sadržaja datoteke u index
- `git branch` -- ispis, stvarnaje ili brisanje grane
- `git checkout` -- provjera brancha do radnog stabla
- `git clean` -- brisanje nepraćenih datoteka iz radnog stabla
- `git clone` -- kloniranje repozitorija u novi direktorij
- `git commit` -- snimanje promjena u repozitoriju
- `git describe` -- prikaz zadnje oznake dohvatljive iz commita
- `git diff` -- prikaz promjena između commitova
- `git fetch` -- dohvaćanje objekata iz drugog repozitorija
- `git init` -- stvaranje praznog git repozitorija ili reinicijalizacija postojećeg
- `git merge` -- spajanje dvije ili više povijesti razvoja
- `git pull` -- dohvaćanje i spajanje sa drugim repozitorijom ili lokalnom granom
- `git push` -- osvježavanje udaljenog repozitorija
- `git revert` -- povratak starog commita
- `git status` -- prikaz statusa radnog stabla
- `git tag` -- stvaranje, ispis ili brisanje oznake objekta potpisanog sa GPG

### Grafička sučelja alata Git

- [gitk](https://git-scm.com/docs/gitk) (službeni GUI, koristi [Tk](https://en.wikipedia.org/wiki/Tk_(framework)))
- [Tig](https://jonas.github.io/tig/) (koristi [ncurses](https://en.wikipedia.org/wiki/Ncurses), priladan za remote rad)
- [gitg](https://wiki.gnome.org/Apps/Gitg) (koristi [GTK+](https://en.wikipedia.org/wiki/GTK), za samostalno proučavanje)
- [giggle](https://wiki.gnome.org/Apps/giggle) (koristi [GTK+](https://en.wikipedia.org/wiki/GTK), za samostalno proučavanje)
- [QGit](https://github.com/tibirna/qgit) (koristi [Qt](https://en.wikipedia.org/wiki/Qt_(software)), za samostalno proučavanje)
- [git-cola](https://git-cola.github.io/) (koristi [Qt](https://en.wikipedia.org/wiki/Qt_(software)), za samostalno proučavanje)

## Vraćanje promjena

!!! todo
    Ovaj dio treba napisati u cijelosti.

## Rad s oznakama

!!! todo
    Ovaj dio treba napisati u cijelosti.

## Grananje i spajanje

!!! todo
    Ovaj dio treba napisati u cijelosti.

## Rješavanje konflikata kod spajanja

!!! todo
    Ovaj dio treba napisati u cijelosti.

## Napredna konfiguracija

!!! todo
    Ovaj dio treba napisati u cijelosti prema [dijelu u knjizi Pro Git](https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration)

## Atributi

!!! todo
    Ovaj dio treba napisati u cijelosti prema [dijelu u knjizi Pro Git](https://git-scm.com/book/en/v2/Customizing-Git-Git-Attributes).

## Zakačke

!!! todo
    Ovaj dio treba napisati u cijelosti prema [dijelu u knjizi Pro Git](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks).

## Ostali sustavi za upravljanje verzijama

### CVS

[Concurrent Versions System](https://en.wikipedia.org/wiki/Concurrent_Versions_System) (kraće CVS, naredba `cvs`) je jedan od najstarijih danas korištenih sustava za upravljanje verzijama.

### Apache Subversion

[Apache Subversion](https://en.wikipedia.org/wiki/Apache_Subversion) (kraće SVN, naredba `svn`) je popularan centralizirani sustav za upravljanje verzijama.

- `checkout` -- preuzima radnu kopiju repozitorija
- `update` -- postavlja postojeću radnu kopiju na određenu reviziju

### Mercurial

[Mercurial](https://en.wikipedia.org/wiki/Mercurial) je, kao i Git, popularan decentralizirani sustav za upravljanje verzijama.

- `clone` -- preuzima repozitorij
- `pull` -- preuzima skupove promjena iz udaljenog repozitorija u lokalni repzitorij
- `update` -- postavlja postojeću radnu kopiju na određenu reviziju iz lokalnog repozitorija

!!! admonition "Zadatak"
    - Mercurial može klonirati i lokalne i udaljene repozitorije na isti način.
    - S poveznice `https://hg.python.org/cpython/` ili iz `/home/vedran/repositories/` izvršite kloniranje repozitorija `cpython`.
    - Postavite repozitorij na reviziju `f3d96d28a86e`. Objasnite razliku efekta naredbi `pull` i `update` u ovoj situaciji.

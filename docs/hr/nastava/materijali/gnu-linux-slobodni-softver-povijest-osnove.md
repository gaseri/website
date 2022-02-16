---
author: Vedran Miletić, Vanja Slavuj
---

# Razvoj slobodnog operacijskog sustava sličnog Unixu

## Slobodni softver

- softver koji možete prilagođavati svojim potrebama i dijeliti prilagođene verzije, npr.

    - [BSD](https://docs.freebsd.org/en/books/design-44bsd/) (Bell Labs, Kalifornijsko sveučilište u Berkeleyu)
    - [X Window System](https://www.x.org/) ([Digital Equipment Corporation](https://en.wikipedia.org/wiki/Digital_Equipment_Corporation), MIT, IBM)
    - [TeX](https://www.tug.org/) ([Donald E. Knuth](https://www-cs-faculty.stanford.edu/~knuth/))

## Projekt GNU

- [Richard M. Stallman](https://stallman.org/)

    - radi u MIT AI Labu; [priča o printeru i napuštanju MIT-a](https://en.wikipedia.org/wiki/Richard_Stallman#Events_leading_to_GNU)
    - 1983\. osniva projekt [GNU](https://www.gnu.org/), kratica za *GNU's Not Unix* (GNU zaista nije Unix jer je neovisno razvijen kao slobodni softver, a Unix je neslobodni softver)
    - želi napraviti GNU OS, slobodni operacijski sustav [sličan Unixu](https://en.wikipedia.org/wiki/Unix-like), razvija [Emacs](https://www.gnu.org/software/emacs/) i [GCC](https://gcc.gnu.org/)
    - 1985\. osniva Free Software Foundation (FSF) kako bi financirao razvoj slobodnog softvera

- Stallman definira slobodni softver kroz [četiri slobode koje korisnici imaju](https://www.gnu.org/philosophy/free-sw.hr.html):

    - Sloboda pokretanja programa kako želite, u bilo koje svrhe (sloboda 0).
    - Sloboda proučavanja rada i prilagodba programa kako bi vršio računalne aktivnosti koje želite (sloboda 1). Dostupnost izvornog kôda je za to preduvjet.
    - Sloboda distribucije kopijâ kako biste pomogli bližnjemu (sloboda 2).
    - Sloboda distribucije izmijenjenih inačica programa (sloboda 3) čime vaše izmjene koriste cijeloj zajednici. Dostupnost izvornog kôda je za to preduvjet.

## Licenca GPL

- Licenca osigurava da će softver uvijek ostati slobodan (tzv. [copyleft](https://www.gnu.org/licenses/copyleft.hr.html))

    - 1989\. GNU General Public License (GPL),
    - 1991\. GPLv2,
    - 2007\. GPLv3 (inicijalno dosta kontroverzna; MS--Novell).

- GPL je različita od licence [Creative Commons](https://creativecommons.org/) (cilja primarno na umjetnička djela) i [BSD licenci](https://en.wikipedia.org/wiki/BSD_licenses) (drugačije poimanje slobode softvera).
- Softver u javnoj domeni, bez vlasništva, nije slobodan, jer ga bilo tko može učiniti svojim vlasništvom.

## Jezgra operacijskog sustava Linux

- razvoj jezgre GNU OS-a [HURD](https://www.gnu.org/software/hurd/) odgađan do 1990. godine
- 1991\. većina GNU OS-a je spremna, ali nedostaje jezgra (engl. *kernel*)
- [Linux](https://en.wikipedia.org/wiki/Linux), započinje 1991. godine neovisno o projektu GNU, autor je [Linus Torvalds](https://en.wikipedia.org/wiki/Linus_Torvalds), student Sveučilišta u Helsinkiju

    > Hello everybody out there using minix -
    >
    > I'm doing a (free) operating system (just a hobby, won't be big and professional like gnu) for 386(486) AT clones. (...)
    >
    > PS. Yes -- it's free of any minix code, and it has a multi-threaded fs. It is NOT portable (uses 386 task switching etc), and it probably never will support anything other than AT-harddisks, as that's all I have :-(.

    - Torvalds ipak nije razvio čitav operacijski sustav, već "samo" njegovu jezgru
    - [Debata Torvaldsa i Tanenbauma](https://en.wikipedia.org/wiki/Tanenbaum%E2%80%93Torvalds_debate) na temu arhitekture jezgre Linuxa i operacijskih sustava uopće, 1992.
    - [jezgra Linuxa](https://www.kernel.org/) je danas velik projekt, [razvija ga 3700 ljudi (podatak iz 2013. godine)](https://youtu.be/abTSM8hvkb8?t=11m19s), a razvoj i dalje vodi Torvalds kao dobronamjerni diktator

- GNU/Linux -- kombinacije jezgre Linux i korisničke okoline GNU čini cjeloviti operacijski sustav sličan Unixu

    - nazivlje Linux ili GNU/Linux -- [postoje zagovornici oba](https://en.wikipedia.org/wiki/GNU/Linux_naming_controversy), [nema](https://devrant.com/rants/1051771/id-just-like-to-interject-for-a-moment-what-you-re-referring-to-as-linux-is-in-f) [konsenzusa](https://wiki.installgentoo.com/index.php/Interjection)
    - operacijski sustavi slični Unixu prilično su različiti od Windowsa, [pa je to i Linux](https://linux.oneandoneis2.org/LNW.htm)

        - zbog tih razlika i brojnih mogućnosti koje nude ponekad su neprivlačni novim korisnicima
        - zbog tih razlika i brojnih mogućnosti koje nude odlična su podloga za izučavanje značajki operacijskih sustava

- nastaju GNU/Linux distribucije (Linux jezgra + GNU korisnička okolina + ostali slobodan softver)

    - 1992\. SuSE, nasljednik je [openSUSE](https://www.opensuse.org/)
    - 1993\. [Debian GNU/Linux](https://www.debian.org/)
    - 1994\. Red Hat Linux, nasljednik je [Fedora](https://fedoraproject.org/)
    - 1998\. Mandrake Linux, nasljedik je [Mageia](https://www.mageia.org/)

## Linux distribucije danas

- [Linux distribucija](https://en.wikipedia.org/wiki/Linux_distribution)

    - skupina programa koje su spojile grupe ljudi (najčešće iz idealističkih motiva) ili tvrtke (najčešće zbog profita)
    - uključuje sve komponente koje su potrebne da bi korisnik mogao koristiti operacijski sustav
    - najviše korištene distribucije se mijenjaju vremenom, nepreciznu statistku vodi [DistroWatch Page Hit Ranking](https://distrowatch.com/dwres.php?section=popularity)
    - deset najpopularnijih distribucija s opisom: [DistroWatch: Top Ten Distributions](https://distrowatch.com/dwres.php?resource=major)

- Neke od najvažnijih Linux distribucija:

    - [Ubuntu](https://ubuntu.com/desktop/features), započeo 2004.

        - osnova joj je distribucija [Debian GNU/Linux](https://www.debian.org/), čiji je razvoj započeo 1993.
        - primarno namijenjena krajnim korisnicima, velik naglasak na uključivanje popularnih desktop aplikacija
        - na njoj se zasniva [Linux Mint](https://www.linuxmint.com/)

    - [Fedora](https://getfedora.org/), započela 2003.

        - osnova joj je distribucija [Red Hat Linux](https://en.wikipedia.org/wiki/Red_Hat_Linux), čiji je rauzvoj započeo 1994. i čiji je nasljednik upravo Fedora
        - primarno namijenjena Linux entuzijastima, velik naglasak na slobodu softvera i nove značajke
        - na njoj se zasnivaju [Red Hat Enterprise Linux](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux) i [CentOS](https://centos.org/), namijenjene poslovnim korisnicima

    - [Arch Linux](https://archlinux.org/), započeo 2002.

        - neovisna distribucija
        - primarno namijenjena naprednim korisnicima; za ilustraciju: [btw I use Arch](https://knowyourmeme.com/memes/btw-i-use-arch), [i use arch btw](https://iusearchbtw.lol/)
        - na njoj se zasnivaju [Manjaro](https://manjaro.org/) i [Garuda Linux](https://garudalinux.org/), namijenjene krajnjim korisnicima
        - sve popularnija među određenim skupinama korisnika, npr. [igračima računalnih igara na Linuxu](https://www.gamingonlinux.com/users/statistics/)

## X Window System

- značajan dio većine distribucija Linuxa i ostalih danas
- počeo 1984. na MIT-u,
- kraće nazvan X, X11,
- mrežna transparentnost,
- najpoznatija implementacija [X.Org](https://www.x.org/),
- X11 je 11. verzija standarda, danas se koristi, datira iz 1987. godine,
- od 2004. X.Org intenzivno moderniziran,
- djelomična zamjena od 2013. [Wayland](https://wayland.freedesktop.org/)

## Paketi i upravitelj paketa

- u svijetu Windowsa: korisnik sam nabavlja softver negdje na internetu i instalira ga

    - [Microsoft Store](https://www.microsoft.com/en-us/store/apps) je vrlo malo promijenio ovu naviku

- u svijetu operacijskih sustava sličnim Unixu: svaki distributer pakira softver za svoju distribuciju u takozvane pakete (slični zip arhivama), a korisnik zatraži instalaciju softvera koji želi, sve ostalo rješava upravitelj paketima

    - omogućene nadogradnje svog softvera
    - na istim načelima zasnivaju se [trgovina Google Play za Android aplikacije](https://play.google.com/store/apps) i [Apple App Store](https://www.apple.com/app-store/)
    - primjer grafičkog sučelja: [GNOME Software](https://wiki.gnome.org/Apps/Software) i [KDE Apper](https://apps.kde.org/apper/)

## GNOME i KDE

- korisnička sučelja: imaju web preglednik, mail klijent, klijent za trenutačno poručivanje, tekst editor, terminal, kalkulator, audio svirač, preglednik slika, preglednik videa, uredske alate, igre, ...

- 1996\. godine Matthias Ettrich želi napraviti grafičko sučelje dovoljno atraktivno da ga njegova cura želi koristititi (💓) i razvija [KDE](https://kde.org/)

    - KDE je zasnovan na skupu biblioteka za razvoj grafičkih sučelja [Qt](https://www.qt.io/) koji je tada neslobodan softver

- 1997\. započinje razvoj alternativa KDE-u i Qt-u, [GNOME](https://www.gnome.org/) i [GTK+](https://www.gtk.org/), projekt vode Miguel de Icaza i Federico Mena

- beskrajne rasprave GNOME vs. KDE, danas sve manje

- druga korisnička sučelja: [Xfce](https://xfce.org/), [LXDE](https://www.lxde.org/), [Enlightenment](https://www.enlightenment.org/about-enlightenment)

## Red Hat

> What we realized very early on is the one unique thing we were delivering was not technology at all. This is not a better operating system. Solaris is a better operating system than Linux for many things. Windows NT is a better operating system for many other things. The one unique thing that we could do that no one else could do was, for the first time, we were giving control of the operating system to the user.

-- [Bob Young](https://www.zdnet.com/article/bob-young-talks-about-the-origins-of-red-hat/), CEO, [Red Hat Inc.](https://www.redhat.com/en/about/company) (citat preuzet iz Pattinson, H. M. (2005). [Mapping implemented strategies of bringing Internet-based software applications to market using storytelling research methods.](https://opus.lib.uts.edu.au/bitstream/10453/37359/2/02whole.pdf) PhD Thesis. University of Technology Sydney., str. 109)

!!! admonition "Ponovimo!"
    - Što je slobodni softver?
    - Kada je započeo GNU projekt i koji je njegov cilj?
    - Što je Linux, a što GNU/Linux?
    - Navedite barem dvije poznate Linux distribucije.
    - Što su X i X11? Koja je poznata implementacija X-a koja se danas koristi?
    - Čemu služe GNOME i KDE?

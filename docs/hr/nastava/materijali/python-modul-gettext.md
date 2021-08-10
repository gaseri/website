---
author: Vedran Miletić
---

# Python: usluge za višejezičnost

- `module gettext` ([službena dokumentacija](https://docs.python.org/3/library/gettext.html)) nudi internacionalizacijske i lokalizacijske usluge po uzoru na [GNU gettext](https://www.gnu.org/software/gettext/), [često korišteni alat za izradu višejezičnih programa](https://en.wikipedia.org/wiki/Gettext) na operacijskim sustavima sličnim Unixu

- `gettext.bindtextdomain(domain, localedir)` veže domenu `domain` na direktorij s lokalizacijama `localedir`; prijevodi aplikacije će se tražiti u `localedir/language/LC_MESSAGES/domain.mo`, npr. sustavski direktorij s lokalizacijama je `/usr/share/locale` pa za aplikaciju `grep` (domena je isto `grep`) i hrvatski jezik (`hr`) ta putanja je `/usr/share/locale/hr/LC_MESSAGES/grep.mo`
- `gettext.textdomain(domain)` postavlja globalnu domenu u programu na `domain`
- `gettext.gettext(message)` dohvaća prijevod poruke `message` obzirom na trenutnu globalnu domenu, jezik i direktorij u kojem se prijevodi nalaze

    ``` python
    import gettext

    gettext.bindtextdomain('kalendar', '/home/korisnik/locale')
    # prijevod se nalazi u /home/korisnik/locale/hr/LC_MESSAGES/kalendar.mo
    gettext.textdomain('kalendar')

    print(gettext.gettext('Print a calendar'))
    # ako su postavljene hrvatske lokalne i regionalne postavke, bit će ispisano "Ispis kalendara"

    # zbog lakše čitljivosti koda preporuča se skratiti ime gettext.gettext na _
    _ = gettext.gettext
    print(_('Print a calendar'))
    # isti ispis kao iznad
    ```

- `gettext.dgettext(domain, message)` dohvaća prijevod poruke `message` u navedenoj domeni `domain` umjesto u trenutnoj globalnoj domeni
- `gettext.ngettext(singular, plural, n)` dohvaća prijevod poruka `singular` i `plural` u množini u odgovarajućem obliku za `n`
- `gettext.dngettext(domain, singular, plural, n)`  dohvaća prijevod poruka `singular` i `plural` u množini u odgovarajućem obliku za `n` u navedenoj domeni `domain` umjesto u trenutnoj globalnoj domeni

!!! admonition "Zadatak"
    (Zasad još uvijek ne znamo stvoriti potrebnu `.mo` datoteku; to ćemo napraviti u idućem zadatku.)

    Napišite Python program koji koristi tekstualnu domenu `mljekara` i ispisuje četiri poruke koje imaju mogućnost prevođenja:

    - `Acquire milk`
    - `Acquire yogurt`
    - `Deliver milk`
    - `Deliver yogurt`

- naredba ljuske `xgettext` parsira programski kod u brojnim programskim jezicima (uključujući Python) i generira [PO datoteku](https://www.gnu.org/software/gettext/manual/html_node/PO-Files.html) `messages.po`, primjerice:

    ``` shell
    $ xgettext kalendar.py
    $ cat messages.po
    # SOME DESCRIPTIVE TITLE.
    # Copyright (C) YEAR THE PACKAGE'S COPYRIGHT HOLDER
    # This file is distributed under the same license as the PACKAGE package.
    # FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.
    #
    #, fuzzy
    msgid ""
    msgstr ""
    "Project-Id-Version: PACKAGE VERSION\n"
    "Report-Msgid-Bugs-To: \n"
    "POT-Creation-Date: 2020-05-07 01:27+0200\n"
    "PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\n"
    "Last-Translator: FULL NAME <EMAIL@ADDRESS>\n"
    "Language-Team: LANGUAGE <LL@li.org>\n"
    "Language: \n"
    "MIME-Version: 1.0\n"
    "Content-Type: text/plain; charset=CHARSET\n"
    "Content-Transfer-Encoding: 8bit\n"

    #: kalendar.py:7 kalendar.py:12
    msgid "Print a calendar"
    msgstr ""
    ```

    - u ovoj datoteci je nužno postaviti jezik na hrvatski promjenom `"Language: \n"` u `"Language: hr\n"` i kodiranje na UTF-8 promjenom `"Content-Type: text/plain; charset=CHARSET\n"` u `"Content-Type: text/plain; charset=UTF-8\n"`, a ostala polja bi bilo dobro popuniti
    - prijevodi se zatim upisuju pod `msgstr ""` na način

        ```
        #: kalendar.py:7 kalendar.py:12
        msgid "Print a calendar"
        msgstr "Ispis kalendara"
        ```

- naredba ljuske `msgfmt` omogućuje prevođenje PO datoteka u MO datoteke koje se mogu koristiti u programima i nisu namijenjene da ih ljudi čitaju i uređuju

    ``` shell
    $ msgfmt messages.po
    $ cat messages.mo
    ,<PQb|Print a calendarProject-Id-Version: PACKAGE VERSION
    Report-Msgid-Bugs-To:
    PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
    Last-Translator: FULL NAME <EMAIL@ADDRESS>
    Language-Team: LANGUAGE <LL@li.org>
    Language: hr
    MIME-Version: 1.0
    Content-Type: text/plain; charset=UTF-8
    Content-Transfer-Encoding: 8bit
    Ispis kalendara
    ```

    - datoteka `messages.mo` se zatim naziva `kalendar.mo` i stavlja u `hr/LC_MESSAGES` unutar direktorija s lokalizacijama (npr. `/home/korisnik/locale/hr/LC_MESSAGES/kalendar.mo`)

!!! admonition "Zadatak"
    Za program iz prethodnog zadatka napravite MO datoteku s hrvatskim prijevodom, nazovite ju na odgovarajući način i postavite na odgovarajuće mjesto. Vežite direktorij s lokalizacijama na domenu. Uvjerite se da se kod pokretanja programa zaista koriste prijevodi iz MO datoteke.

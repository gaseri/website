---
author: Milan Petrović, Vedran Miletić
---

# Instalacija i konfiguracija softvera za vježbe iz kolegija Programiranje za web

## Visual Studio Code

Razvojno okruženje u kojem se radi na vježbama je Visual Studio (VS) Code koji možete preuzeti na [službenim stranicama](https://code.visualstudio.com/).

Nakon što preuzimete i instalirate VS Code, pokrenite ga, zatim kraticom ++control+shift+x++ otvorite dio za instalaciju ekstenzija, pronađite Python i instalirajte ga. Dodatna upustva za instalaciju Python ekstenzije unutar VS Codea možete pronaći u [službenom tutorialu](https://code.visualstudio.com/docs/python/python-tutorial).

Specijalno, na [Manjaru](https://manjaro.org/), popularnom derivatu [Arch Linuxa](https://archlinux.org/), možete instalirati [službeni VS Code binary](https://aur.archlinux.org/packages/visual-studio-code-bin) iz AUR-a:

```
$ pamac build visual-studio-code-bin
(...)
```

Ime paketa na [Garuda Linuxu](https://garudalinux.org/) je isto, a naredba za instalaciju je `paru -S`.

## Pythonovi pomoćni alati

Preduvjet je instalacija [pip](https://pypi.org/project/pip/)-a i [Pylint](https://pylint.org/)-a.

Naredba za instalaciju na Manjaru je:

```
$ pamac install python-pip python-pylint
(...)
```

Imena paketa na Garuda Linuxu su ista, a naredba za instalaciju je `sudo pacman -S`.

Naredba za instalaciju na [Ubuntu WSL](https://ubuntu.com/wsl)-u i drugima je:

``` shell
$ sudo apt install python-is-python3 python3-pip pylint
(...)
```

## Django

Naredba za instalaciju [Djanga](https://www.djangoproject.com/) na Manjaru je:

``` shell
$ pamac install python-django
(...)
```

Na Ubuntu WSL-u i drugima instalacija Djanga vrši se korištenjem pip-a:

``` shell
$ pip3 install Django
(...)
```

ili

``` shell
$ python3 -m pip install Django
(...)
```

Alternativno, na Ubuntu WSL-u i drugima moguća je i instalacija pakirane verzije Djanga na Ubuntu WSL-u, ali se to ne preporuča jer je verzija znatno starija od trenutne aktualne:

``` shell
$ sudo apt install python3-django
(...)
```

## HTTPie

Naredba za instalaciju [HTTPie]-a na Manjaru je:

``` shell
$ pamac install httpie
(...)
```

Na Ubuntu WSL-u i drugima instalacija HTTPie-a, kao i Djanga, vrši se korištenjem pip-a:

``` shell
$ pip3 install httpie
(...)
```

ili

``` shell
$ python3 -m pip install httpie
(...)
```

Alternativno, na Ubuntu WSL-u i drugima moguća je i instalacija pakirane verzije HTTPie-a na Ubuntu WSL-u, ali se, kao i kod Djanga, to ne preporuča jer je verzija znatno starija od trenutne aktualne:

``` shell
$ sudo apt install httpie
(...)
```

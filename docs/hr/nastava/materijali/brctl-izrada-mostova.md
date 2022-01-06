---
author: Vedran Miletić
---

# Premošćenje mrežnih adaptera i izrada mostova

Kako bismo omogućili međusobnu komunikaciju više od dva računala putem mreže, pojedini čvorovi moraju služiti kao prijenosni čvorovi. Iako to mogu biti i najčešće jesu specijalizirani uređaji, za istu svrhu mogu poslužiti i računala s više mrežnih adaptera. (Preklopnik (engl. *switch*) je naprosto vrsta most koja ima mnogo vrata, a u praksi se najčešće ne realizira pomoću običnih računala koja koriste složenu konfiguraciju mosta već pomoću specijaliziranog hardvera i softvera.)

Skup alata [bridge-utils](https://wiki.linuxfoundation.org/networking/bridge) ([izvorni kod](https://git.kernel.org/pub/scm/network/bridge/bridge-utils.git/)) sadrži alat `brctl` koji nam omogućuje premošćenje mrežnih adaptera. Detaljne upute što sve `brctl` nudi mogu se naći na [Debianovom](https://wiki.debian.org/BridgeNetworkConnections), odnosno [Archevom wikiju](https://wiki.archlinux.org/title/Network_bridge). Maleni dio funkcionalnosti koji je nama značajan da bi filtriranje paketa na veznom sloju radilo izložit ćemo u dijelu gdje je opisano filtriranje paketa na veznom sloju.

!!! todo
    Napiši detaljnije o premošćenju.

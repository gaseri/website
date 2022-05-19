---
author: Domagoj Margan, Vedran Miletić
---

# Ograničavanje pristupa uslugama vatrozidom nftables

[Nftables](https://netfilter.org/projects/nftables/) je suvremena zamjena za iptables, ip6tables, arptables i ebtables. Osnovna strutktura je tablica koja sadrži lance pravila. Predefinirane tablice i lanci ne postoje, a kod stvaranja novih imena su im proizvoljna.

Koristit ćemo tablice tipa `ip` za IPv4, `ip6` za IPv6 i `net` za IPv4 i IPv6. Koristit ćemo lance pravila tipa `filter` za filtriranje paketa i `nat` za prevođenje adresa. Kod lanaca pravila navodi se tip kuke (engl. *hook type*) koji definira kada se pravila u lancu primijenjuju, a može biti `prerouting` prije usmjeravanja, `input` kod primanja paketa s mreže, `forward` kod prosljeđivanja paketa, `output` kod slanja paketa s domaćina, ili `postrouting` nakon usmjeravanja. Ovi tipovi imaju suštinski ulogu kao kod iptablesa i ip6tablesa, što je i očekivano jer nftables i ti alati koriste isti okvir netfilter unutar jezgre Linuxa. Razlike između nftablesa i starijih alata za konfiguraciju filtriranja paketa i prevođenja adresa je u dijelu implementacije koji se nalazi u korisničkom prostoru te korisničkom sučelju, odnosno sintaksi naredbi.

Nftables svu funkcionalnost nudi putem naredbe `nft`. Iako je danas ta kratica popularna, [nftables je već bio dio jezgre Linuxa početkom 2014. godine](https://kernelnewbies.org/Linux_3.13#nftables.2C_the_successor_of_iptables) i time prethodi [nepromjenjivim tokenima čija povijest počinje sredinom 2014. godine](https://en.wikipedia.org/wiki/Non-fungible_token#History). Osnovni primjeri korištenja dostupni su na [stranici nftables na ArchWikiju](https://wiki.archlinux.org/title/Nftables), a dodatni primjeri naredbi mogu se naći u [dijelu Examples na službenom wikiju](https://wiki.nftables.org/wiki-nftables/index.php/Main_Page#Examples).

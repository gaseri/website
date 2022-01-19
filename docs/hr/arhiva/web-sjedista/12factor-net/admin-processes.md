---
author: Adam Wiggins
---

!!! note
    Sadržaj u nastavku je prijevod stranice [XII. Admin processes](https://12factor.net/admin-processes) na web sjedištu [The Twelve-Factor App](https://12factor.net/).

## XII. Administrativni procesi

### Pokrenite administrativne/upravljačke zadatke kao jednokratne procese

[Formiranje procesa](concurrency.md) je niz procesa koji se koriste za obavljanje redovnog poslovanja aplikacije (kao što je rukovanje web zahtjevima) dok se ona izvodi. Zasebno, razvojni programeri će često htjeti obaviti jednokratne administrativne ili zadatke održavanja za aplikaciju, kao što su:

* Pokretanje migracija baze podataka (npr. `manage.py migrate` u Djangu, `rake db:migrate` u Railsu).
* Pokretanje konzole (također poznata kao ljuska [REPL](https://en.wikipedia.org/wiki/Read-eval-print_loop) za pokretanje proizvoljnog kôda ili provjeru modela aplikacije u odnosu na živu bazu podataka. Većina jezika pruža REPL pokretanjem interpretera bez ikakvih argumenata (npr. `python` ili `perl`) ili u nekim slučajevima imaju zasebnu naredbu (npr. `irb` za Ruby, `rails console` za Rails).
* Pokretanje jednokratnih skripti predanih u repozitorij aplikacije (npr. `php scripts/fix_bad_records.php`).

Jednokratni administrativni procesi trebali bi se izvoditi u identičnom okruženju kao i redoviti [dugotrajni procesi](processes.md) aplikacije. Pokreću se u odnosu na [izdanje](build-release-run.md), koristeći istu [bazu izvornog kôda](codebase.md) i [konfiguraciju](config.md) kao i bilo koji proces koji se pokreće u odnosu na to izdanje. Administrativni kôd mora biti isporučen s kôdom aplikacije kako bi se izbjegli problemi sa sinkronizacijom.

Iste tehnike [izolacije zavisnosti](dependencies.md) trebale bi se koristiti na svim tipovima procesa. Na primjer, ako Ruby web proces koristi naredbu `bundle exec thin start`, tada bi migracija baze podataka trebala koristiti `bundle exec rake db:migrate`. Isto tako, Python program koji koristi Virtualenv trebao bi koristiti isporučeni `bin/python` za pokretanje i Tornado web poslužitelja i svih `manage.py` administrativnih procesa.

Metodologija dvanaest faktora snažno daje prednost jezicima koji pružaju REPL ljusku od početka korištenja i koji olakšavaju pokretanje jednokratnih skripti. U lokalnoj implementaciji, razvojni programeri pozivaju jednokratne administrativne procese izravnom naredbom ljuske unutar direktorija za naplatu aplikacije. U produkcijskoj implementaciji razvojni programeri mogu koristiti ssh ili drugi mehanizam za daljinsko izvršavanje naredbi kojeg nudi izvršna okolina te implementacije za pokretanje takvog procesa.

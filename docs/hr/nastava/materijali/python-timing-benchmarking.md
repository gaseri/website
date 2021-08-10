---
author: Vedran Miletić
---

# Mjerenje brzine izvođenja Python aplikacija

Pamćenjem vrijednosti vremena zidnog sata prije početka i nakon završetka izvođenja (dijela) koda možemo izmjeriti brzinu izvođenja.

``` python
import time

start = time.time ()

# kod čije vrijeme izvođenja mjerimo

end = time.time()

print("Trajanje izvođenja dijela koda", end - start)
```

Dodatno, jedno korisno proširenje je korištenje argumenata Python skripte, na način

``` python
import time
import sys

broj = int(sys.argv[1])

start = time.time ()

# kod čije vrijeme izvođenja mjerimo i koristi vrijednost varijable broj

end = time.time()

print("Trajanje izvođenja dijela koda", end - start)
```

pri čemu varijabla može biti `broj` iteracija, željena preciznost (onda je češće tipa `float`) ili bilo što drugo što utječe na veličinu problema i trajanje izvođenja.

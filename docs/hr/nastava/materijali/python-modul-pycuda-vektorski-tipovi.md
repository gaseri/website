---
author: Vedran Miletić, Kristijan Lenković
---

# Python modul PyCUDA: rad s vektorskim tipovima podataka

PyCUDA, kao i CUDA, ima ugrađenu podršku za vektorske tipove: višedimenzionalni podaci s jednom do četiri komponente, adresirane kao `.x`, `.y`, `.z`, `.w`. Neki od postojećih predefiniranih vektorskih tipova podataka su:

``` c
struct uchar1
{
  unsigned char x;
};

struct __align__(4) ushort2
{
  unsigned short x, y;
};

struct uint3
{
  unsigned int x, y, z;
};

struct __align__(16) float4
{
  float x, y, z, w;
};
```

Uočimo funkciju `__align__()`, koja se koristi za poravnanje podataka. Naime, GPU može iz memorije u jednoj instrukciji dohvatiti:

- 32 bita, odnosno 4 bajta, za što se koristi `__align(4)`,
- 64 bita, odnosno 8 bajta, za što se koristi `__align(8)`,
- 128 bita, odnosno 16 bajta, za što se koristi `__align(16)`.

Primjeri slučajeva korištenja vektorskih tipova podataka su:

- slike s 8-bitnom dubinom boje (`uchar3`) (RGB => `x`, `y`, `z`),
- slike s 24-bitnom dubinom boje i transparencijom (`uint4`) (RGBA => `x`, `y`, `z`, `w`),
- sustavi fizikalnih objekata u prostoru (`float3`) (`x`, `y`, `z`: prostorne koordinate),
- sustavi fizikalnih objekata u prostorno-vremenskom kontinuumu (`float4`) (`x`, `y`, `z`, `w`: 3 prostorne koordinate i jedna vremenska).

Sa predefiniranim vektorskim tipovima podataka radimo na intuitivan način; zrno je oblika

``` c
#include <stdio.h>

__global__ void print_coordinates (float3 point)
{
  printf ("x: %3.2f, y: %3.2f, z: %3.2f\\n", point.x, point.y, point.z);
}
```

Pridruženi Python kod je oblika

``` python
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as ga
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule(open("vector_types.cu").read())

print_coordinates = mod.get_function("print_coordinates")

point_in_space = ga.vec.make_float3(3.6, 5.4, 2.25)

print_coordinates(point_in_space, block=(1,1,1), grid=(1,1))
```

!!! admonition "Zadatak"
    Na domaćinu definirajte tri točke u *ravnini* (točke će biti tipa `float2` i imati dvije koordinate). Izračunajte površinu trokuta koji te tri točke zatvaraju korištenjem formule:

    $$
    P_{ABC} = \frac{1}{2} \big| (x_A - x_C) (y_B - y_A) - (x_A - x_B) (y_C - y_A) \big|
    $$

    Zrno se izvodi **bez korištenja paralelizacije**, dakle kut se računa u jednom bloku s jednom niti po bloku. Za provjeru, isti izračun izvedite na domaćinu.

!!! admonition "Zadatak"
    Kosinus kuta dvaju vektora $a = (a_x, a_y)$ i $b = (b_x, b_y)$ računa se po formuli

    $$
    \cos \theta = \frac{a \cdot b}{|a| |b|},
    $$

    odnosno jednak je skalarnom produktu vektora podijeljenom s produktom normi vektora.

    Dane su četiri točke u ravnini, tipa `float2`. Definirajte:

    - funkciju uređaja koja računa skalarni produkt, koju ćete uz predefiniranu funkciju `sqrt()` iskoristiti za izračun norme vektora, te
    - zrno koje računa kosinus kuta, pozivajući funkciju uređaja koja računa skalarni produkt,

    Zrno se izvodi **bez korištenja paralelizacije**, dakle kut se računa u jednom bloku s jednom niti po bloku. Za provjeru, isti izračun izvedite na domaćinu.

!!! note
    Prethodni zadaci ne koristi nikakvu vrstu paralelizacije, odnosno izvode se na jednoj jezgri uređaja. U narednim primjerima uključiti ćemo vektorske tipove podataka u širi kontekst i iskoristiti ih u paralelnim algoritmima.

---
author: Vedran Miletić
---

# Python modul PyCUDA: paralelni algoritmi na matricama

## Rad s višedimenzionalnim poljima. Zbrajanje matrica

Intuitivni pristup zbrajanju matrica bio bi oblika

``` c
__global__ void matrix_sum (float **dest, float **a, float **b)
{
  const int i = threadIdx.x;
  const int j = threadIdx.y;
  dest[i][j] = a[i][j] + b[i][j];
}
```

međutim, on ovdje **ne** daje dobar rezultat. Kod, ako uopće radi, nije optimalan zbog specifičnosti arhitekture GPU-a (očekuje vektor, ne višedimenzionalnu strukturu).

Potrebno je napraviti tzv. *serijalizaciju*. C/C++ koristi [Row-major order](https://en.wikipedia.org/wiki/Row-major_order) (Fortran koristi [Column-major order](https://en.wikipedia.org/wiki/Column-major_order)).

- koristimo dvodimenzionalne koordinate
- uočimo da je `blockDim.x` broj procesnih niti po `x` koordinati, a `blockDim.y` broj procesnih niti po `y` koordinati

    - ukupan broj procesnih niti koje rade je `blockDim.y * blockDim.x`, i ta veličina mora biti u skladu s ograničenjima hardvera

- **što je x, a što y, ostavljeno je programeru na volju**, nije predefinirano

Zrno je oblika

``` c
__global__ void matrix_sum (float *dest, float *a, float *b)
{
  const int i = threadIdx.y * blockDim.x + threadIdx.x;
  dest[i] = a[i] + b[i];
}
```

``` python
mod = SourceModule(open("matrix_operations.cu").read())

matrix_sum = mod.get_function("matrix_sum")
```

Inicijalizacija matrica kao `numpy` polja je oblika

``` python
a = np.ones((20, 20), dtype=np.float32)
b = np.ones((20, 20), dtype=np.float32)

result_gpu = np.empty_like(a)
```

Izračun rezultata na GPU-u i CPU-u i usporedba rezultata izvodi se kodom

``` python
zbroji_matrice(drv.Out(result_gpu), drv.In(a), drv.In(b), block=(20,20,1), grid=(1,1))

result_cpu = a + b

print("CPU rezultat\n", result_cpu)
print("GPU rezultat\n", result_gpu)
print("CPU i GPU daju isti rezultat?\n", result_cpu == result_gpu.get())
```

!!! admonition "Zadatak"
    Prilagodite gornji kod tako da se zrnu prosljeđuje i veličina matrice koja se koristi u izračunu; za to trebate napraviti tri stvari:

    - U zrnu promijeniti kod da potpis bude `__global__ void zbroji_matrice (float *dest, float *a, float *b, int n)`,
    - Inicijalizirati u Python kodu `n` na željenu vrijednost odgovarajućeg `numpy` tipa (proučite [ranije navedeni popis](https://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions/#how-do-i-specify-the-correct-types-when-calling-and-preparing-pycuda-functions)),
    - Izvesti poziv funkcije s `matrix_sum(result_gpu, a_gpu, b_gpu, n, ...)`, obzirom da se kod prosljeđivanja argumenata po vrijednosti može raditi s nekim `numpy` tipovima (za detalje proučite [dokumentaciju objekta pycuda.driver.Function](https://documen.tician.de/pycuda/driver.html#pycuda.driver.Function)).

    **Napomena:** veličinu matrice potrebno je znati kada broj niti koje rade na matrici nije identičan kao ta veličina. Primjerice, za zbroj matrica veličine `(16, 16)` koji izvodi 20 niti kod bi mogao biti oblika

    ``` c
    const int idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (threadIdx.x < n && threadIdx.y < n)
      {
        dest[idx] = a[idx] + b[idx];
      }
    ```

    Međutim, radi jednostavnosti i boljih performansi, preferira se usklađivanje veličine polja i broja niti koje rade na tom polju ukoliko je moguće napraviti na danom problemu.

!!! admonition "Zadatak"
    Napišite program koji na dvije matrice matrica formata 10 * 10 element po element radi operaciju: $2 \cdot \sin(a) + 3 \cdot \cos(b) + 4$

    Ovu operaciju učinite device funkcijom i pozovite ju unutar zrna.

## Množenje matrice i vektora

Python kod je oblika

``` python
mod = SourceModule(open("matrix_operations.cu").read())

mat_vec_mult = mod.get_function("mat_vec_mult")

a = np.ones((20, 20), dtype=np.float32)
b = np.ones(20, dtype=np.float32)

result_gpu = np.empty_like(20, dtype=np.float32)

mat_vec_mult(drv.Out(result_gpu), drv.In(a), drv.In(b), block=(20,1,1), grid=(1,1))

result_cpu = np.dot(a, b)
```

Zrno je oblika

``` c
__global__ void mat_vec_mult (float *dest, float *mat, float *vec)
{
  const int idx = threadIdx.x;

  float sum_product = 0;
  for (int i = 0; i < 20; i++)
    {
      sum_product += mat[idx * 20 + i] * vec[i];
    }

  dest[idx] = sum_product;
}
```

!!! admonition "Zadatak"
    - Promijenite veličinu matrice na `(16, 16)`, a veličinu vektora na 16.
    - Promijenite kod tako da se izvodi u 16 niti po x-u i 16 niti po y-u, i da se pritom redukcija (zbrajanje produkata) izvodi paralelno na način koji smo već ranije opisali.

## Naivno množenje matrica

Serijski kod za množenje matrica je oblika

``` c
const int N = 4; // matrix size
float mat1[N][N], mat2[N][N], result[N][N];
// initialize values ...
for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
      {
        float sum = 0;
        for (k = 0; k < N; k++)
          {
            sum += mat1[i][k] * mat2[k][j];
          }
          result[i][j] = sum;
      }
  }
```

Kao i kod zbrajanja, na GPU-u ćemo množenje matrica raditi s jednodimenzionalnim poljem.

- Vanjski indeks je `threadIdx.y`, unutarnji indeks je `threadIdx.x`.
- Iz serijskog koda ostaje samo unutarnja petlja; vanjske dvije obilaze elemente rješenja, a u našem slučaju svaki element rješenja računa jedna procesna nit nit.

``` c
__global__ void matrix_mult (float *result, float *mat1, float *mat2)
{
  const int idx = threadIdx.y * blockDim.x + threadIdx.x;

  float sum_product = 0;
  for (int k = 0; k < 20; k++)
    {
      sum_product += mat1[threadIdx.y * blockDim.x + k] * mat2[k * blockDim.x + threadIdx.x];
    }

  result[idx] = sum_product;
}
```

Ostatak program vrlo je sličan kao kod zbrajanja.

``` python
mod = SourceModule(open("matrix_operations.cu").read())

matrix_mult = mod.get_function("matrix_mult")

a = np.ones((20, 20), dtype=np.float32)
b = np.ones((20, 20), dtype=np.float32)

result_gpu = np.empty_like(a)

matrix_mult(drv.Out(result_gpu), drv.In(a), drv.In(b), block=(20,20,1), grid=(1,1))

# result_cpu = np.dot(a, b)
result_cpu = np.matrix(a) * np.matrix(b)
```

!!! note
    Kada se izvodi zrno koje radi sa višedimenzionalnim poljem i više blokova u svakoj od dimenzija:

    - svaki blok po `y`-u sadrži `gridDim.x` blokova po `x`-u,
    - svaki blok po `x`-u sadrži `blockDim.y * blockDim.x` niti,
    - svaka nit po `y`-u sadrži `blockDim.x` niti po `x`-u,

    tada je indeks elementa suma:

    - `blockIdx.y * gridDim.x * blockDim.y * blockDim.x`,
    - `blockIdx.x * blockDim.y * blockDim.x`,
    - `threadIdx.y * blockDim.x`,
    - `threadIdx.x`.

!!! admonition "Zadatak"
    Prilagodite kod tako da množi matrice formata `(200, 200)` u 10 blokova po x koordinati i 10 blokova po y koordinati.

!!! admonition "Zadatak"
    Modificiriajte program koji vrši jednoblokovno modificirano množenje matrica tako da umjesto produkta gdje je je element $(i, j)$ oblika $\sum_k a_{ik} \cdot b_{kj}$, on bude oblika $\sum_k 2 \cdot a_{ik}^{b_{kj}}$.

## Druge primjene matrica

!!! admonition "Zadatak"
    Računate prosječne vrijednosti temperature za određeno mjesto.

    Definirajte zrno koje prima tri argumenta, matricu tipa `float`, vektor tipa `int` i vektor tipa `float`.

    - U matrici su u retcima zapisane vrijednosti u rasponu od -5.0 do 35.0 koje su očitanja temperatura za određeno mjesto. Matrica neka ima 4 retka i 16 stupaca, i u retcima neka budu vrijednosti po vašoj želji.
    - Vektor tipa `int` ima 4 elementa i kaže koliko ima izmjerenih vrijednosti temperature, odnosno koliko stupaca ima različitih od 0 (krenuvši od prvog stupca).
    - Vektor tipa `float`, koji ima 4 elementa, služi za spremanje izračunatih vrijednosti.

    (**Uputa:** napravite zrno koje se izvodi na 4 bloka i 16 niti po bloku; iskoristite dijeljenu memoriju unutar svakog bloka da bi redukcijom sumirali elemente; iz vektora tipa `int` očitajte koliko je elementa različitih od 0, i na temelju toga izračunajte prosječnu vrijednost.)

    (**Primjer::** u slučaju kad bi imali tri mjesta i matricu od 5 stupaca, matrica može biti oblika:

    ```
    2.3    6.2    24.8   5.6   2.2
    22.7   8.3    16.5   0     0
    31.7   23.2   1.4    2.5   0
    ```

    a pripadni vektor je tada oblika

    ```
    5
    3
    4
    ```

    Uočimo također da vrijednosti koje se ne uzimaju u obzir mogu biti proizvoljne i ne moraju nužno biti 0.)

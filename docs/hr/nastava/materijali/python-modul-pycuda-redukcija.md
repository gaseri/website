---
author: Vedran Miletić
---

# Python modul PyCUDA: paralelna redukcija, norma i skalarno množenje vektora

## Paralelna redukcija korištenjem jednog bloka

Redukcija podrazumijeva dobivanje jedne vrijednosti iz velikog broja njih; tipični primjeri su:

- pronalaženje minimuma ili maksimuma u polju,
- izračun skalarnog produkta dva vektora,
- svođenje matrice sustava linearnih jednadžbi na gornju trokutastu matricu Gaussovom eliminacijom.

Dva intuitivna pristupa nameću se sami po sebi:

- `sum += a[threadIdx.x]` -- ne radi, zašto?
- `atomicAdd (&sum, a[threadIdx.x])` -- radi, ali sporo, zašto?

Paralelna redukcija zahtijeva korištenje dijeljene memorije za razmjenu informacija između procesnih niti unutar jednog bloka. Grupa procesnih niti koje koriste dijeljenu memoriju se naziva **Cooperative Thread Array** (kraće CTA). Dijeljena memorija alocira se unutar zrna korištenjem ključne riječi `__shared__`. Svakom bloku pridružuje se dana količina dijeljene memorije, koja je dostupna za čitanje i pisanje od strane svih procesnih niti unutar bloka.

Kada procesna nit treba pročitati podatke zapisane od strane neke druge procesne niti unutar istog bloka, mora osigurati da je zapisivanje završilo. To se rješava čekanjem na ostale procesne niti korištenjem `__syncthreads()` naredbe. (Dakle, ne koriste se zaključavanja varijabli.)

Polja u dijeljenoj memoriji mogu se deklarirati *statički* u zrnu (veličina se definira u trenutku prevođenja programa) ili *dinamički* (veličina se definira kod poziva).

Primjer statičke alokacije (dinamičku alokaciju demonstriramo kasnije):

``` c
__global__ void some_kernel(...)
{
  __shared__ float shared_memory[32];
  // ...
}
```

``` python
some_kernel(..., block=(1,1,1), grid=(1,1))
```

Redukcija je sama po sebi paralelan proces. Da bi ga paralelizirali za izvođenje na heterogenim sustavima, vršimo dekompoziciju na veći broj nezavisnih, paralelnih mini-redukcija. U dijeljenoj memoriji, kada primjerice tražimo minimalnu vrijednost u polju, to se čini korištenjem petlje oblika:

``` c
// ...
int active = blockDim.x / 2;
do
  {
    __syncthreads();
    if (threadIdx.x < active)
      {
        shared_memory[threadIdx.x] = fmin (shared_memory[threadIdx.x], shared_memory[threadIdx.x + active]);
      }
    active /= 2;
  }
while (active > 0);
// ...
```

U svakom koraku prepolavljamo broj aktivnih procesnih niti, i to tako da svaka aktivna procesna nit uspoređuje svoju vrijednost s pripadnom vrijednosti u drugoj polovici dijeljene memorije

- konkretno, 16 procesnih niti reducira 32 vrijednosti na 16 vrijednosti,
- zatim 8 procesnih niti reducira 16 vrijednosti na 8 vrijednosti,
- i tako redom, sve do trenutka dok ne ostane samo jedna vrijednost.

Primjer; zrno koje radi redukciju je oblika

``` c
__global__ void find_min (float *dest, float *src)
{
  __shared__ float cache[16];

  const int idx = threadIdx.x;
  cache[idx] = src[idx];
  int active = blockDim.x / 2;

  do
    {
      __syncthreads();
      if (idx < active)
        {
          cache[idx] = fmin(cache[idx], cache[idx + active]);
        }
      active /= 2;
    }
  while (active > 0);

  if (idx == 0)
    {
       dest[blockIdx.x] = cache[0];
    }
}
```

Ostatak koda je oblika

``` python
mod = SourceModule(open("redukcija.cu").read())

find_min = mod.get_function("find_min")

a = np.random.rand(16).astype(np.float32)
result_gpu = np.empty(1).astype(np.float32)

find_min(drv.Out(result_gpu), drv.In(a), block=(16,1,1), grid=(1,1))

result_cpu = min(a)
```

!!! admonition "Zadatak"
    - Promijenite kod primjera tako da umjesto traženja minimuma traži maksimum i dodajte `printf` na odgovarajuće mjesto kako bi vidjeli vrijednosti varijabli `idx`, `active` i elemenata u polju `cache` s kojima ta nit radi.
    - Promijenite kod primjera tako da umjesto traženja minimuma, odnosno maksimuma, vrši zbrajanje.

Informacije o korištenoj memoriji možemo dobiti pozivom ovako definirane funkcije `kenrnel_meminfo()`

``` python
def kernel_meminfo(kernel):
    shared = kernel.shared_size_bytes
    regs = kernel.num_regs
    local = kernel.local_size_bytes
    const = kernel.const_size_bytes
    mbpt = kernel.max_threads_per_block
    print("""=== Memory usage informartion ===\nLocal: %d,\nShared: %d,\nRegisters: %d,\nConst: %d,\nMax Threads/B: %d""" % (local, shared, regs, const, mbpt))
```

Primjer dinamičke alokacije

- Uočite da argument `shared` u pozivu funkcije `some_kernel()` mjeri veličinu u **broju elemenata polja**, ne u bitovima ili bajtovima

``` c
__global__ void some_kernel(...)
{
  extern __shared__ float shared_memory[];
  // ...
}
```

``` python
some_kernel(..., block=(1,1,1), grid=(1,1), shared=32)
```

!!! admonition "Zadatak"
    Promijenite kod primjera koji traži minimalnu vrijednost u vektoru da koristi dinamičku umjesto statičke alokacije.

!!! note
    Zbog određenih problema na koje smo naišli prilikom korištenja dinamičke alokacije dijeljene memorije odavde nadalje koristit ćemo statičku alokaciju.

## Paralelna redukcija korištenjem više od jednog bloka

Svaki blok može zbog toga reducirati onoliko elemenata u dijeljenoj memoriji koliko ima procesnih niti u tom bloku na jedan element.

- Pokretanje jednog zrna zbog toga reducira polje na onoliko vrijednosti koliko je blokova bilo pokrenuto.
- Ako je pokrenuto više od jednog bloka, potrebni su dodatni koraci za redukciju do slučaja u kojem je jedan blok dovoljan.

Kod učitavanja iz globalne u dijeljenu memoriju, moguće je napraviti inicijalnu redukciju time što svaka procesna nit može učitati više od jedne vrijednosti iz globalne memorije, i sprema u dijeljenu memoriju samo krajnji rezultat. Primjerice, kod

``` c
uint gix = threadIdx.x + blockDim.x*blockIdx.x;
float acc = CUDART_NAN_F;
while (gix < dim) {
    acc = fmin(acc, dSrc[gix]);
    gix += blockDim.x*gridDim.x;
}
shMem[tid] = acc;
```

omogućuje proizvoljan broj blokova, i oni će pročitati koliko god elemenata iz globalne memorije je potrebno da se popuni čitavo polje. To znači da pokretanje jednog zrna može izvesti čitavu redukciju polja; međutim, to nije efikacsno jer se većina procesnih resursa uređaja ne koristi.

Zrno koje pokrećemo slično je kao i ranije, jedino što sada brine i o indeksu niti unutar bloka.

- Uočite da je veličina dijeljene memorije jednaka 16, jer je ona pridružena *jednom* bloku i dijeljena isključivo među nitima *istog* bloka, pa je ukupna veličina memorije koja se koristi 64.

``` c
__global__ void find_min (float *dest, float *src)
{
  __shared__ float cache[16];

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int tidx = threadIdx.x;
  cache[tidx] = src[idx];
  int active = blockDim.x / 2;

  do
    {
      __syncthreads();
      if (tidx < active)
        {
          cache[tidx] = fmin(cache[tidx], cache[tidx + active]);
        }
      active /= 2;
    }
  while (active > 0);

  if (tidx == 0)
    {
       dest[blockIdx.x] = cache[0];
    }
}
```

Najefikasniji od svih pristupa pokreće dva zrna, jednu s brojem blokova dovoljnim da zasiti hardver, a drugu s jednim blokom koji "završava" redukciju.

``` python
a = np.random.rand(64).astype(np.float32)
mid_result_gpu = np.empty(4).astype(np.float32)
result_gpu = np.empty(1).astype(np.float32)

find_min(drv.Out(mid_result_gpu), drv.In(a), block=(16,1,1), grid=(4,1))

find_min(drv.Out(result_gpu), drv.In(mid_result_gpu), block=(4,1,1), grid=(1,1))

result_cpu = min(a)
```

!!! admonition "Zadatak"
    - Modificirajte gornji kod da izračuna srednju vrijednost umjesto da traži minimum.
    - Učinite da se kod izvodi na vektoru veličine 512 u 16 blokova.

## Izračun norme i skalarnog produkta vektora

Neka su $a = (a_1, a_2, \ldots, a_n)$ i $b = (b_1, b_2, \ldots, b_n)$ dva vektora. Skalarni produkt vektora $a$ i $b$ je suma oblika

$$
a \cdot b = \sum_{i=1}^n a_i b_i.
$$

Norma vektora $a$ je korijen iz skalarnog produkta vektora sa samim sobom, odnosno

$$
|a| = \sqrt{a \cdot a} = \sqrt{\sum_{i=1}^n a_i a_i}.
$$

Python kod za normu vektora je oblika

``` python
compute_norm = mod.get_function("compute_norm")

a = 2 * np.ones(16, dtype=np.float32)

compute_norm(drv.Out(result_gpu), drv.In(a), block=(16,1,1), grid=(1,1))
```

Zrno je oblika

``` c
__global__ void compute_norm (float *dest, float *src)
{
  __shared__ float cache[16];

  const int tidx = threadIdx.x;
  // kvadriramo brojeve kod spremanja u međuspremnik
  cache[tidx] = src[idx] * src[idx];
  int active = blockDim.x / 2;

  do
    {
      __syncthreads();
      if (tidx < active)
        {
          // redukcija korištenjem zbrajanja
          cache[tidx] += cache[tidx + active];
        }
      active /= 2;
    }
  while (active > 0);

  if (tidx == 0)
    {
      // korjenujemo kod spremanja u završni rezultat
      dest[blockIdx.x] = sqrt(cache[0]);
    }
}
```

!!! todo
    Ovdje nedostaje zadatak.

Python kod za skalarni produkt je oblika

``` python
compute_scalar_product = mod.get_function("compute_scalar_product")

a = 2 * np.ones(16, dtype=np.float32)
b = 3 * np.ones(16, dtype=np.float32)

compute_scalar_product(drv.Out(result_gpu), drv.In(a), drv.In(b), block=(16,1,1), grid=(1,1))
```

Zrno je oblika

``` c
__global__ void compute_scalar_product (float *dest, float *src1, float *src2)
{
  __shared__ float cache[16];

  const int tidx = threadIdx.x;
  // brojeve množimo kod spremanja u međuspremnik
  cache[tidx] = src1[idx] * src2[idx];
  int active = blockDim.x / 2;

  do
    {
      __syncthreads();
      if (tidx < active)
        {
          // redukcija korištenjem zbrajanja
          cache[tidx] += cache[tidx + active];
        }
      active /= 2;
    }
  while (active > 0);

  if (tidx == 0)
    {
      dest[blockIdx.x] = cache[0];
    }
}
```

!!! todo
    Ovdje nedostaje zadatak.

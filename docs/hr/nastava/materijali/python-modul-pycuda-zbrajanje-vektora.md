---
author: Vedran Miletić, Kristijan Lenković
---

# Python modul PyCUDA: zbrajanje vektora

- CUDA C/C++ kod zrna spremamo u `zbroj.cu`

    - uočimo da se indeks dohvaća iz varijable `threadIdx.x`

    ``` c
    __global__ void zbroj_vektora (float *dest, float *a, float *b)
    {
      const int i = threadIdx.x;
      dest[i] = a[i] + b[i];
    }
    ```

- dohvaćanje zrna u Python

    ``` python
    mod = SourceModule(open("zbroj.cu").read())

    zbroj_vektora = mod.get_function("zbroj_vektora")
    ```

- inicijalizacija dva vektora čije su sve vrijednosti jednake 1 i vektora čje su sve vrijednosti jednake 0 i jednake je duljine kao ova dva

    ``` python
    a = np.ones(400, dtype=np.float32)
    b = np.ones(400, dtype=np.float32)

    result_gpu = np.zeros_like(a)
    ```

!!! note
    Pripazite na usklađenost Pythonovih `numpy` tipova podataka sa C-ovim tipovima podataka koji se koriste u kernelu (ovdje je to `numpy.float32` s `float`). [Potpun popis](https://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions/#how-do-i-specify-the-correct-types-when-calling-and-preparing-pycuda-functions) može se naći među često postavljanim pitanjima u vezi PyCUDA-e.

- izvođenje koda na GPU-u ([službena dokumentacija](https://documen.tician.de/pycuda/driver.html#pycuda.driver.Function))

    - prvo se navode argumenti kernela **istim redom** kao u definiciji kernela u C-u
    - `block` i `grid` su stvari kojima se bavimo kasnije, ali uočite da stavljamo istu vrijednost kao što je veličina vektora

    ``` python
    zbroj_vektora(drv.Out(result_gpu), drv.In(a), drv.In(b), block=(400,1,1), grid=(1,1))
    ```

    - `drv.In()`, `drv.Out()` i `drv.InOut()` pretvaraju `numpy` polja u polja s kojima GPU može manipulirati ([službena dokumentacija](https://documen.tician.de/pycuda/driver.html#pycuda.driver.In) objašnjava razliku između te tri funkcije i kada je potrebno koju od njih koristiti)

- izvođenje zbrajanja na CPU-u (jednostavnost sintakse osigurava preopterećeni operator zbrajanja od strane `numpy` modula)

    ``` python
    result_cpu = a + b
    ```

- ispis rezultata i provjera jednakosti po pojedinim poljima

    ``` python
    print("CPU rezultat\n", result_cpu)
    print("GPU rezultat\n", result_gpu)
    print("CPU i GPU daju isti rezultat?\n", result_cpu == result_gpu)
    ```

!!! admonition "Zadatak"
    - Prilagodite kod primjera tako da računa zbroj oblika $2a + b$.
    - Prilagodite kod primjera tako da računa zbroj oblika $2a + b + 3c$.

## Nit, blok i rešetka

!!! admonition "Zadatak"
    Prilagodite kod prethodnog primjera tako da vektori imaju 500 elemenata (umjesto 400).

Zrna se pokreću na rešetci. Rešetka se sastoji od **blokova**, koji se sastoje od **niti**.

- Postoji definirana varijabla `gridDim` i ima komponente `x`, `y`; opisuje veličinu rešetke.
- Postoji definirana varijabla `blockIdx` i ima komponente `x`, `y`; daje indeks bloka u rešetci.
- Postoji definirana varijabla `blockDim` i ima komponente `x`, `y`, `z`; opisuje veličinu bloka.
- postoji definirana varijabla `threadIdx` i ima komponente `x`, `y`, `z`; daje indeks niti u bloku.

Indeks `i` ovisi o indeksu bloka (`blockIdx`), dimenziji bloka (`blockDim`) i indeksu niti (`threadIdx`).

``` c
__global__ void vector_sum (float *dest, float *a, float *b)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  dest[i] = a[i] + b[i];
}
```

Svaki od dva bloka sada ima 250 niti i one pokrivaju čitav vektor od 500 komponenata.

``` python
zbroj_vektora(drv.Out(result_gpu), drv.In(a), drv.In(b), block=(250,1,1), grid=(2,1))
```

!!! admonition "Zadatak"
    - Provjerite možete li se s dva bloka izvesti zbrajanje vektora od 3000 elemenata. (**Napomena:** čim je zadano ovako, vjerojatno ne možete.)
    - Izvedite zbrajanje vektora od 3000 elemenata u 3 bloka.

!!! note
    GPU-i zasnovani na Tesla arhitekturi podržavaju maksimalno 512 niti po bloku, dok GPU-i zasnovani na arhitekturama Fermi, Keppler i Maxwell podržavaju maksimalno 1024 niti po bloku.

Sposobnosti GPU uređaja možemo saznati pomoću idućeg koda

``` python
def device_meminfo():
    (free, total) = drv.mem_get_info()
    print("=== Global memory occupancy for device 0 ===")
    print("Free: %d" % free)
    print("Total: %d" % total)
    print("Percentage free: %f%%" % (float(free) * 100 / float(total)))

    for devicenum in range(drv.Device.count()):
        device = drv.Device(devicenum)
        attrs = device.get_attributes()

        print("\n=== Attributes for device %d ===" % devicenum)
        for (key, value) in attrs.iteritems():
            print("%s: %s" % (str(key), str(value)))
```

## Atomične operacije

Atomične aritmetičke operacije su:

- `atomicAdd()`
- `atomicSub()`
- `atomicExch()`
- `atomicMin()`
- `atomicMax()`
- `atomicInc()`
- `atomicDec()`
- `atomicCAS()`

Atomične bitovne operacije su:

- `atomicAnd()`
- `atomicOr()`
- `atomicXor()`

!!! todo
    Ovdje nedostaje objašnjenje i zadatak.

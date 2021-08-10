---
author: Vedran Miletić, Kristijan Lenković
---

# Python modul PyCUDA: funkcije uređaja i domaćina

- Ključna riječ `__global__` -- poziva se na domaćinu, izvršava na uređaju
- Ključna riječ `__device__` -- poziva se na uređaju, izvršava na uređaju
- Ključna riječ `__host__` -- poziva se na domaćinu, izvršava na domaćinu ("obična funkcija", ne koriste se u PyCUDA-i)

Funkcije koje će se izvoditi na uređaju definiraju se na gotovo isti način kao u C/C++-u, s ograničenjem da se ne može definirati rekurzija.

``` c
__device__ float my_second_device_function(float y)
{
  return my_device_function(y) / 2;
}

__device__ int my_illegal_recursive_device_function(int x)
{
  if(x == 0) return 1;
  return x * my_illegal_recursive_device_function(x-1);
}
```

Primjer zbrajanja matrica korištenjem dvaju pomoćnih funkcija.

``` c
__device__ int get_thread_index(void)
{
  return threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int get_constant(void)
{
  return 7;
}

__global__ void matrix_sum (float *dest, float *a, float *b)
{
  const int i = get_thread_index();
  dest[i] = get_constant() * a[i] + b[i];
}
```

Ostatak koda vrlo je sličan kao ranije.

``` python
mod = SourceModule(open("zbroj_matrica2.cu").read())

matrix_sum = mod.get_function("matrix_sum")

a = np.ones((20, 20), dtype=np.float32)
b = np.ones((20, 20), dtype=np.float32)

result_gpu = np.empty_like(a)

matrix_sum(drv.Out(result_gpu), drv.In(a), drv.In(b), block=(20,20,1), grid=(1,1))

result_cpu = 7 * a + b
```

!!! admonition "Zadatak"
    - Definirajte funkciju `kvadriraj(float x)` koja se izvodi na uređaju i kvadrira dani `x` tako da ga množi samim sobom.
    - Definirajte funkciju `potenciraj(float x, int n)` koja se izvodi na uređaju i potencira dani `x` tako da ga množi opetovano samim sobom unutar `for` petlje `n` puta.
    - Unutar zrna za množenje matrica na uređaju izračunajte postavite rezultat na 42. potenciju zbroja `a[i] + b[i]`. Isto napravite na domaćinu.

!!! admonition "Zadatak"
    Definirajte funkciju uređaja koja vraća broj djelitelja danog prirodnog broja.

    Definirajte zrno koje prima 2 polja tipa `int` iste veličine; u prvo polje pohranite brojeve za koje će se određivati broj djelitelja; u drugo polje postavite sve vrijednosti na -1. Zrno u element u drugom polju upisuje broj djelitelja prirodnog broja koji se nalazi na istoj poziciji u prvom polju.

    Isto se izvodi na CPU-u, i rezultati se uspoređuju.

---
author: Vedran Miletić
---

# Python modul PyCUDA: funkcije i tipovi podatka dostupni u CUDA bibliotekama

Kompletna dokumentacija svih funkcija koje opisujemo u nastavu dostupna je u PyCUDA dokumentaciji, u [dijelu koji opisuje funkcije za rad s višedimenzionalnim poljima na GPU-u](https://documen.tician.de/pycuda/array.html).

## Modul pycuda.gpuarray

Pored dosad korištenih modula, koristimo i `pycuda.gpuarray`; on radi konverziju iz `numpy` polja u polja s kojima GPU može manipulirati. Konverzija se inače izvodi automatski korištenjem pomoćnih funkcija `pycuda.driver.In()`, `pycuda.driver.Out()` i `pycuda.driver.InOut()`.

Inicijalizacija, izračun rezultata na GPU-u za zbroj matrica je oblika

``` python
a = np.ones((20, 20), dtype=np.float32)
b = np.ones((20, 20), dtype=np.float32)

a_gpu = ga.to_gpu(a_cpu)
b_gpu = ga.to_gpu(b_cpu)

result_gpu = ga.empty_like(a_gpu)

zbroji_matrice(result_gpu, a_gpu, b_gpu, block=(20,20,1), grid=(1,1))
```

Uočite da su `result_gpu`, `a_gpu` i `b_gpu` već `gpuarray` polja i nema potrebe za `pycuda.driver.In()`, `pycuda.driver.Out()` i `pycuda.driver.InOut()`.

Korištenjem činjenice da `gpuarray` preopterećuje operator zbrajanja, gornji kod može se i dodatno pojednostaviti.

``` python
a = np.ones((20, 20), dtype=np.float32)
b = np.ones((20, 20), dtype=np.float32)

a_gpu = ga.to_gpu(a_cpu)
b_gpu = ga.to_gpu(b_cpu)

result_gpu = a_gpu + b_gpu
```

## Modul pycuda.cumath: funkcije koje rade element-po-element na poljima tipa GPUArray

CUDA Math API ima predefinirane funkcije uređaja s [jednostrukom preciznošću](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html) i [dvostrukom preciznošću](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html). Kada se koristi PyCUDA, navedene funkcije su dostupne putem modula `pycuda.cumath` koji opisujemo u nastavku.

Zaokruživanje i apsolutna vrijednost:

- `pycuda.cumath.fabs(array)`
- `pycuda.cumath.ceil(array)`
- `pycuda.cumath.floor(array)`

Exponencijalne i logaritamske funkcije, korjenovanje:

- `pycuda.cumath.exp(array)`
- `pycuda.cumath.log(array)`
- `pycuda.cumath.log10(array)`
- `pycuda.cumath.sqrt(array)`

Trigonometrijske funkcije:

- `pycuda.cumath.sin(array)`
- `pycuda.cumath.cos(array)`
- `pycuda.cumath.tan(array)`
- `pycuda.cumath.asin(array)`
- `pycuda.cumath.acos(array)`
- `pycuda.cumath.atan(array)`

Računamo li zbroj oblika $sin(a) + cos(b)$ element po element, kod je oblika

``` python
a_cpu = np.ones((20, 20), dtype=np.float32)
b_cpu = np.ones((20, 20), dtype=np.float32)

result_cpu = a_cpu + b_cpu

print("CPU rezultat\n", result_cpu)

a_gpu = ga.to_gpu(a_cpu)
b_gpu = ga.to_gpu(b_cpu)

result_gpu = pycuda.cumath.sin(a_gpu) + pycuda.cumath.cos(b_gpu)

print("GPU rezultat\n", result_gpu.get())
```

!!! admonition "Zadatak"
    Promijenite kod da računa element po element rezultat oblika $\sqrt{e^a + \tan(b)}$.

!!! admonition "Zadatak"
    Incijalizirate GPUArray sa 10 redaka i 5 stupaca u kojem ćemo retke tretirati kao vektore. Napišite kod koji vrši zbrajanje svakog vektora sa svakim vektorom a rezultat vraća kao GPUArray veličine $\binom{10}{2}$ redaka i 5 stupaca.

### Aproksimativno računanje broja $\pi$ kao sume niza

Broj $\pi$ može se aproksimirati formulom

$$
\pi = \int_0^1 \frac{4}{1 + x^2} dx \approx \frac{1}{n} \sum_{i = 0}^{n - 1} \frac{4}{1 + (\frac{i + 0.5}{n})^2}.
$$

- sekvencijalni kod je oblika

    ``` python
    import numpy as np

    n_iterations = 10

    def compute_pi(n):
        h = 1.0 / n
        s = 0.0
        for i in range(n):
            x = h * (i + 0.5)
            s += 4.0 / (1.0 + x**2)
        return s * h

    pi = compute_pi(n_iterations)
    error = abs(pi - np.math.pi)
    print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
    ```

- vektorizirani kod koji koristi `numpy` je oblika

    ``` python
    import numpy as np

    n_iterations = 1000

    def compute_pi(n):
        h = 1.0 / n
        s = 0.0 # nepotrebno
        x = h * (np.arange(1, n) + 0.5)
        s = np.sum(4.0 / (1.0 + x**2))
        return s * h

    pi = compute_pi(n_iterations)
    error = abs(pi - np.math.pi)
    print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
    ```

- paralelni kod je oblika

    ``` python
    import pycuda.autoinit
    import pycuda.gpuarray as ga
    import numpy as np

    n_iterations = 1000

    def compute_pi(n):
        h = 1.0 / n
        s = 0.0 # nepotrebno
        x = h * (ga.arange(1, n, dtype=np.float32) + 0.5)
        s = ga.sum(4.0 / (1.0 + x**2), dtype=np.float32)
        return s.get() * h

    pi = compute_pi(n_iterations)
    error = abs(pi - np.math.pi)
    print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
    ```

!!! admonition "Zadatak"
    - Modificirajte gornji primjer tako da dodate kod koji mjeri vrijeme izvođenja.
    - Usporedite vrijeme izvođenja algoritma na CPU-u i na GPU-u za 10^4^, 10^5^, 10^6^ iteracija; ukoliko neku od ovih veličina ne budete mogli izvesti zbog nedostatka memorije na GPU-u, zanemarite tu i sve veće. Opišite svoje zaključke.

## Modul pycuda.curandom: generiranje polja sa slučajnim brojevima

- `pycuda.curandom.rand(shape, dtype=numpy.float32)` -- generira `GPUArray` oblika `shape` sa "nekvalitetnim" slučajnim brojevima (njihova slučajnost se može dovesti u pitanje)

``` python
nekvalitetni_slucajni = pycuda.curandom.rand(10000)
print(nekvalitetni_slucajni)
```

### XORWOW i MRG32k3a generatori

- kvalitetniji generatori slučajnih brojeva; oba generatora imaju period barem 2^190^
- `pycuda.curandom.XORWOWRandomNumberGenerator()` -- konstruira instancu XORWOW generatora
- `pycuda.curandom.MRG32k3aRandomNumberGenerator()` -- konstruira instancu MRG32k3a generatora
- identično sučelje za pristup funkcionalnosti, primjer sa XORWOW generatorom

    ``` python
    kvalitetni_slucajni = pycuda.gpuarray.GPUArray(10000, dtype=np.float64)

    rng = pycuda.curandom.XORWOWRandomNumberGenerator()

    rng.fill_uniform(kvalitetni_slucajni)
    print("Uniformna distribucija", kvalitetni_slucajni)

    rng.fill_normal(kvalitetni_slucajni)
    print("Normalna distribucija", kvalitetni_slucajni)
    ```

!!! admonition "Zadatak"
    - Promijenite gornji primjer da koristi MRG32k3a generator umjesto XORWOW generatora i da se generiraju 32-bitni brojevi s pomičnim zarezom umjesto 64-bitnih.
    - Usporedite u terminu vremena izvođenja numpy generator koji se izvodi na CPU-u (funkcija `np.random.random()`) i PyCUDA MRG32k3a generator koji se izvodi na GPU-u kod generiranja:

        - 10000 slučajnih brojeva,
        - 100000 slučajnih brojeva,
        - 1000000 slučajnih brojeva,
        - 10000000 slučajnih brojeva,
        - 100000000 slučajnih brojeva.

!!! tip
    Poziv funkcije `set_printoptions()` iz modula numpy oblika `np.set_printoptions(threshold=np.nan)` čini da se ispisuju svi brojevi numpy polja, bez obzira na njegovu veličinu.

### Generatori zasnovani sa Soboljevim nizovima

!!! note
    Generatori `Sobol32`, `ScrambledSobol32`, `Sobol64` i `ScrambledSobol64` zasnovani su na [Soboljevim nizovima](https://en.wikipedia.org/wiki/Sobol_sequence). Više informacija o njima možete naći u [službenoj PyCUDA dokumentaciji](https://documen.tician.de/pycuda/array.html#pycuda.curandom.Sobol32RandomNumberGenerator).

### Aproksimativno računanje broja $\pi$ korištenjem Monte Carlo metode

[Monte Carlo metode](https://en.wikipedia.org/wiki/Monte_Carlo_method) su skupina metoda koja na temelju slučajnih brojeva i velikog broja pokusa daju aproksimaciju određene vrijednosti.

Broj $\pi$ može se aproksimativno računati i korištenjem Monte Carlo metode. Da bi to vidjeli, uzmimo da je u koordinatnom sustavu ucrtan jedinični krug unutar jediničnog kvadrata. Promotrimo njegovu četvrtinu u prvom kvadrantu:

- četvrtina kvadrata je površine 1 (obzirom da su mu obje stranice duljine 1),
- čevrtina kruga je površine $\frac{\pi}{4}$ (obzirom da mu je radijus 1).

Slučajno odabrana točka unutar četvrtine kvadrata ima vjerojatnost $\frac{\pi}{4}$ da upadne unutar četvrtine kruga. Dakle, ako s $n$ označimo broj slučajno odabranih točaka, a s $h$ broj točaka koje se od $n$ slučajno odabranih nalaze unutar četvrtine kruga, aproksimativno možemo odrediti broj $\pi$ kao

$$
\pi = 4 \times \frac{h}{n}.
$$

Povećavajući $n$ dobivamo točniju aproksimaciju, a navedena metoda naziva se Monte Carlo metoda. Programski kod koji aproksimira broj $\pi$ korištenjem Monte Carlo metode je oblika:

- sekvencijalni kod

    ``` python
    import numpy as np

    n_random_choices = 10000
    hits = 0
    throws = 0

    for i in range (0, n_random_choices):
        throws += 1
        x = np.random.random()
        y = np.random.random()
        dist = np.math.sqrt(x * x + y * y)
        if dist <= 1.0:
            hits += 1

    pi = 4 * (hits / throws)

    error = abs(pi - np.math.pi)
    print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
    ```

- paralelni kod

    - lošiji pristup; `for` petlja vrši 10000 * 4 dohvaćanja elemenata iz memorije GPU-a

        ``` python
        import pycuda.autoinit
        import pycuda.driver as drv
        import pycuda.gpuarray as ga
        import pycuda.curandom
        import pycuda.cumath
        from pycuda.compiler import SourceModule
        import numpy as np

        n_random_choices = 10000
        hits = 0
        throws = n_random_choices

        x_polje = pycuda.curandom.rand(n_random_choices)
        y_polje = pycuda.curandom.rand(n_random_choices)
        for i in range (0, n_random_choices):
            dist = np.math.sqrt(x[i] * x[i] + y[i] * y[i])
            if dist <= 1.0:
                hits += 1

        pi = 4 * (hits / throws)

        error = abs(pi - np.math.pi)
        print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
        ```

    - bolji pristup, eliminacija `for` petlje, radimo na polju

        ``` python
        import pycuda.autoinit
        import pycuda.driver as drv
        import pycuda.gpuarray as ga
        import pycuda.curandom
        import pycuda.cumath
        from pycuda.compiler import SourceModule
        import numpy as np

        n_random_choices = 10000
        hits = 0
        throws = n_random_choices

        x_polje = pycuda.curandom.rand(n_random_choices)
        y_polje = pycuda.curandom.rand(n_random_choices)
        dist = pycuda.cumath.floor(pycuda.cumath.sqrt(x_polje * x_polje + y_polje * y_polje))
        hits = n_random_choices - dist.get().sum()

        pi = 4 * (hits / throws)

        error = abs(pi - np.math.pi)
        print("pi is approximately %.16f, error is approximately %.16f" % (pi, error))
        ```

!!! admonition "Zadatak"
    - Modificirajte primjer dan za Monte Carlo simulaciju da dodate kod koji mjeri vrijeme izvođenja.
    - Usporedite vrijeme izvođenja simulacije na CPU-u i na GPU-u za 10^4^, 10^5^, 10^6^, 10^7^, 10^8^, 10^9^ iteracija; ukoliko neku od ovih veličina ne budete mogli izvesti zbog nedostatka memorije na GPU-u, zanemarite tu i sve veće. Opišite svoje zaključke.

!!! admonition "Zadatak"
    Usporedite vrijeme izvođenja programa i točnost aproksimacije pi korištenjem:

    - sumiranja elemenata niza,
    - Monte Carlo simulacije,

    na CPU-u i na GPU-u za 10^4^, 10^5^, 10^6^ iteracija. Opišite svoje zaključke.

!!! admonition "Zadatak"
    Simulirajmo ekonomiju. Inicijalizirajte GPUArray veličine 200 elemenata koji predstavlja financijsko stanje 200 jediniki u ekonomiji koju simuliramo. Incijalizirajte vrijednosti elemenata na 10. Definirajte dvije operacije

    - porast_pad_ekonomije(), koji svačije financijsko stanje množi sa koeficijentom k koji se određuje kao slučajan broj u intervalu $[0.5, 1.5]$.
    - preraspodjela_trgovina(), koja vrši transakciju određenog iznosa (slučajan broj u rasponu $[0, 0.5]$) između dvije slučajno odabrane jedinke.

    Izvedite Monte Carlo simulaciju veličine 100000 iteracija od kojih svaka iteracija vrši

    - slučajno između 5 i 10 operacija preraspodjela_trgovina
    - jednu operaciju porast_pad_ekonomije.

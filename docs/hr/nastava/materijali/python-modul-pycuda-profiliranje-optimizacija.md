---
author: Vedran Miletić
---

# Python modul PyCUDA: mjerenje performansi, profiliranje i optimizacija

## Mjerenje performansi zrna

Da bi mjerili brzinu izvođenja zrna i operacija kopiranja memorije s domaćina na uređaj i obrnuto, koristimo **profiler** (engl. *profiler*). [PyCUDA podržava profiliranje](https://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions/#is-it-possible-to-profile-cuda-code-with-pycuda), i ono se uključuje postavljanjem varijable okoline `CUDA_PROFILE` na vrijednost `1`, što možemo učiniti na način:

``` bash
$ CUDA_PROFILE=1 python program.py
```

Uzmemo li da je `program.py` sada kod koji zbraja dva vektora, datoteka `cuda_profile_0.log` koju dobivamo kao izlaz je sadržaja:

``` shell
# CUDA_PROFILE_LOG_VERSION 2.0
# CUDA_DEVICE 0 GeForce GT 520
# CUDA_CONTEXT 1
# TIMESTAMPFACTOR fffff6a8cb41f9a0
method,gputime,cputime,occupancy
method=[ memcpyHtoD ] gputime=[ 1.120 ] cputime=[ 26.000 ]
method=[ memcpyHtoD ] gputime=[ 0.864 ] cputime=[ 9.000 ]
method=[ vector_sum ] gputime=[ 3.232 ] cputime=[ 26.000 ] occupancy=[ 0.271 ]
method=[ memcpyDtoH ] gputime=[ 2.912 ] cputime=[ 27.000 ]
```

Za usporedbu, ista datoteka na drugom domaćinu i drugom uređaju je sadržaja:

``` shell
# CUDA_PROFILE_LOG_VERSION 2.0
# CUDA_DEVICE 0 GeForce GTX 560 Ti
# CUDA_CONTEXT 1
# TIMESTAMPFACTOR fffff6a83d18f898
method,gputime,cputime,occupancy
method=[ memcpyHtoD ] gputime=[ 0.864 ] cputime=[ 35.000 ]
method=[ memcpyHtoD ] gputime=[ 0.736 ] cputime=[ 12.000 ]
method=[ vector_sum ] gputime=[ 2.464 ] cputime=[ 27.000 ] occupancy=[ 0.271 ]
method=[ memcpyDtoH ] gputime=[ 1.472 ] cputime=[ 21.000 ]
```

Vidimo da je zbog jačeg grafičkog procesora vrijeme izvođenja pojedinih funkcije manje, tako da bi najjednostavnija metoda optimizacije bila "pokrenite kod na jačem grafičkom procesoru".

!!! admonition "Zadatak"
    - Promijenite kod da zbraja dva vektora veličine 800 elemenata i usporedite vrijeme izvođenja. Specijalno, kakvo je zauzeće?
    - Promijenite kod da zbraja dva vektora veličine 1600 elemenata i usporedite vrijeme izvođenja kad:

        - imate dva bloka veličine 800 elemenata, i
        - imate četiri bloka veličine 400 elemenata. Koja od dvije varijante daje bolji rezultat?

!!! admonition "Zadatak"
    Napišite kod koji množi matricu veličine (2048, 2084) s vektorom veličine 2048:

    - u 128 blokova s po 32 niti,
    - u 256 blokova s po 16 niti,
    - u 512 blokova s po 8 niti.

    Koja varijanta daje najbolje performanse?

## Vizualni profiler

Naredbom `nvvp` pokrećemo NVIDIA Visual Profiler. On je namijenjen za profiliranje CUDA C/C++ aplikacija, ali može profilirati i PyCUDA aplikacije na idući način: pod `File/New Session` podaci su

- `File: /usr/bin/python`
- `Working directory: /home/<username>`
- `Arguments: /home/<username>/<ime_programa>.py`
- `Environment:` -> `Add`

    - `Name: CUDA_PROFILE`
    - `Value: 1`

Zatim, `Next >` i na idućoj stranici `Finish`.

!!! note
    Ukoliko radite na udaljenom računalu, `nvvp` vam neće moći na ispravan način prikazati pomoć; iskoristite [kopiju iste dokumentacije dostupnu putem weba](https://docs.nvidia.com/cuda/profiler-users-guide/index.html).

## Optimizacija

GPU nudi najbolje performanse kada je potpuno zasićen, odnosno kada sve njegove jezgre računaju. Optimizacija se uglavnom vrši kombinacijom tri načina:

- maksimizacijom paralelizacije,
- minimizacijom kašnjenja pristupa memoriji,
- maksimizacijom propusnosti instrukcija.

### Maksimizacija paralelizacije

Maksimizacija paralelizacije uključuje tri faktora:

- korištenje svih jezgara GPU-a,
- zadržavanje serializacije lokalnom,
- prikrivanje kašnjenja sabirnice.

Korištenje svih jezgara zahtijeva veliki broj blokova (barem onoliko koliko ima multiprocessora na GPU uređaju). Na uređajima koji imaju računsku sposobnost 2.0+ (mi ih koristimo), moguće je istovremeno pokrenuti dva zrna. Ako imate potreba za serializacijom, odnosno procesne niti ovise međusobno o rezultatima izračuna, pokušajte zadržati međuovisne procesne niti na istom bloku: za to možete koristiti `__syncthreads()` umjesto odvojenih pokretanja zrna.

Skrivanje kašnjenja eksploatira paralelnost između komponenata GPU-a: ako postoje računske osnove koje mogu biti otpremljene dok druge računske osnove čekaju, kašnjenje je time pokriveno. Skrivanje kašnjenja zahtijeva poznavanje otpreme instrukcija, što ovisi od računskoj sposobnosti uređaja. Uređaji s računskom sposobnošću 2.0 skrivaju kašnjenje na slijedeći način:

- 1 instrukcija po računskoj osnovi u 2 instrukcijska ciklusa, 2 računske osnove u svakom trenutku => `N` instrukcija skriva `N` instrukcijskih ciklusa

Tipična instrukcija zahtijeva 22 instrukcijska ciklusa da se izvrši. To implicira da je potrebno barem 22 računske osnove (za sposobnost 2.x) da bi sakrili standardno sekvencijalno izvođenje (instrukcija ovisi o rezultatu izvođenja prethodne instrukcije).

Pristup globalnoj i lokalnoj memoriji traje od 400 do 800 instrukcijskih ciklusa. Računske osnove potrebne da bi sakrile to kašnjenje ovise o gustoći između instrukcije koje ne ovise o pristupu memoriji i instrukcija koje ovise o tome.

Kada se koristi sinkronizacija, sve računske osnove u bloku zastaju i njihovo kašnjenje je jednako kašnjenju procesne niti koja se najduže izvodi. To se pokriva pokretanjem više od jednog bloka po multiprocesoru.

Zauzeće, koje se dobiva kao omjer rezidentnih računskih osnova i maksimalnog broja rezidentnih računskih osnova, može se izračunati korištenjem proračunske tablice *CUDA Occupancy Calculator*. Veće zauzeće pomaže u skrivanju kašnjenja zbog sinkronizacije.

### Minimizacija kašnjenja memorije

Pristup memoriji vrši se u transkacijama. Jedna transkacija pristupa 32, 64 ili 128 bajta, poravnato njihovoj veličini (primjerice, 128-bajtne transakcije mogu biti poravnate na 128-bajtnu granicu).

Instrukcije pristupa memoriji automatski se slažu u transakcije. Broj transakcija potreban za pristup memoriji ovisi o:

- tipovima podataka,
- poravnanju podataka,
- sposobnostima hardvera.

Jedna memorijska instrukcija se šalje za pristup broju bajtova koji je potencija broja 2 ($2^1, 2^2, \dots, 2^{16}$), koji su prirodno poravnati. Svi ostali pristupi dijele se u više od jedne instrukcije.

Funkcija `cudaMalloc()` vraća memoriju koja je poravnata na 256 bajta.

Poravnanje redaka za linearizirana 2D polja može se poboljšati korištenjem funkcije `cudaMallocPitch()`:

``` c
data_t *devData; /* e.g. float *devData */
size_t pitch;
cudaMallocPitch (&devData, &pitch, numCols*sizeof (data_t), numRows);
```

Varijabla `pitch` sadržava veličinu danu u bajtovima. Elementu `(redak, stupac)` se tada može pristupiti s `((data_t*)((char*)devData + row*pitch))[col]`.

Kopiranje tako alocirane memorije vrši se pomoću funkcije `cudaMemcpy2D()`:

``` c
cudaMemcpy2D (dest, destpitch, src, srcpitch, row_width, numrows, direction);
```

gdje su `destpitch`, `srcpitch` i `row_width` dane u bajtovima i redom su:

- korak cilja (broj bajtova koji razdvajaju početak svakog retka)
- korak izvora (kao iznad)
- stvarna veličina u bajtovima podataka u retku (koja je manja ili jednaka koraku izvora)

### Pravila udruživanja

Pravila udruživanja za Fermi (računska sposobnost 2.x) su:

- pristup globalnoj memoriji spremljen je u pričuvnu memoriju sa 128-bajtnim linijama
- L1 pričuvna memorija koristi iste memorijske banke kao i dijeljena memorija
- omjer korištenja memorije iste memorijske banke između L1 pričuvne i dijeljene memorije može se odrediti pomoću `cudaFuncSetCacheConfig(kernelIme, odabir)` gdje je drugi argument `odabir` jedan od slijedećih:
- `cudaFuncCachePreferShared`: 48K dijeljene memorije, 16K L1 pričuvne memorije
- `cudaFuncCachePreferL1`: 16K dijeljene memorije, 48K of L1 pričuvne memorije
- `cudaFuncCachePreferNone`: (predodređen) koristi iste postavke koje uređaj koristi

### Dijeljena memorija i konflikti banke

Dijenjena memorija sastoji se od memorijskih banaka. 32-bitne (4-bajtne) riječi su slijedno dodijeljene bankama. Širina pojasa je 32 bita kroz dva ciklusa sata.

Dijeljena memorija je tipično jednako brza kao registri. Konflikti banke (kada cuda_runtime_api.h dvije ili više niti istovremeno pristupa istoj banci) uvode kašnjenje.

Svaka polovica računske osnove vrši zahtijeve za memorijom neovisno. Procesne niti u prvom dijelu računske osnove ne mogu imati konflikte banke s procesnim nitima u drugom dijelu. Ako sve procesne niti pristupaju istoj riječi, podaci se predaju svima na broadcast način i nema konflikata banke.

Zajednički pristup vrši se u koracima veličine 32 bita (`float`, `float2`, `float3`, `float4` itd.). Konflikti banke događaju se ako je korak paran (primjerice, `float2` ili `float4`), a u slučaju da je neparan nema konflikata banke  (primjerice, `float`, `float3`).

Kod 8-bitnog and 16-bitnog pristupa memoriji, svi pristupi uzrokuju konflikte banke osim u slučaju kada su podaci isprepleteni (engl. *interleaved*). Kod 64 ili više-bitnog pristupa konflikte banke nije moguće izbjeći.

Računska sposobnost 2.x uvodi nekoliko promjena.

- Postoje 32 banke umjesto 16 banaka kod sposobnosti 1.x.
- Pristupi memoriji prema istoj 32-bitnoj riječi su broadcast.
- Nema konflikata banke za 8-bitni i 16-bitni pristup.
- 64-bitni pristupi razdvojeni su u nekonfliktne 32-bitne pristupe.
- 128-bitni pristupi uzrokuju dvosmjerne konflikte banke.

### Maksimizacija propusnosti instrukcija

Maksimizacija propusnosti instrukcija ima tri faktora:

- preferiranje brzih instrukcija pred sporima (primjerice, korištenje jednostruke preciznosti pred dvostrukom obzirom da hardver s njom radi višestruko brže, korištenje intrinzičnih funkcija umjesto vlastitih obzirom da program prevoditelj brine da ih zamijeni najbržom implementacijom),
- smanjite broj `if-else` i `switch-case` naredbi, i minimizirajte divergenciju u slučaju da ih ima,
- smanjite broj točaka sinkronizacije i međuovisnost operacija.

Uređaji koji imaju računsku sposobnost 1.x imaju mogućnost brzog množenja 24-bitnih cijelih brojeva. `__mul24()` može se koristiti umjesto standardnog produkta kada množenje ne izaziva aritmetički preljev. Uređaji koji imaju računsku sposobnost 2.x imaju mogućnost brzog množenja 32-bitnih cijelih brojeva, i `__mul24()` je sporiji od od standardnog produkta.

Određene zastavice program prevoditelja upravljaju korištenjem funkcija kao što su logaritamska, eksponencijalna i trigonometrijske funkcije. (`--use_fast_math` i bool parametri `--ftz=`, `--prec-div=`, `--prec-sqrt=` mijenjaju način baratanja denormalnim brojevima te preciznost dijeljenja i kvadratnog korjenovanja).

### Preporuke za optimizaciju koda

Sumarno, preporuke za optimizaciju koda su:

- Pronađite način da paralelizirate sekvencijalni kod. Proučite koje paralelne algoritme i strukture podataka možete iskoristiti za vaš problem.
- Smanjite količinu prenesenih podataka i broj prijenosa između uređaja i domaćina, u oba smjera.
- Prilagodite konfiguraciju pokretanja zrna kako bi se maksimalno iskoristile jezgre uređaja.
- Osigurajte da su pristupi globalnoj memoriji sjedinjeni.
- Zamijenite pristup globalnoj memoriji pristupom dijeljenoj memoriji kad god je to moguće.
- Izbjegnite sukobe memorijskih banaka u dijeljenoj memoriji.
- Izbjegnite različite puteve kod izvršavanja unutar jedne računske osnove.

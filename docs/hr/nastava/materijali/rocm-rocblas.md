---
author: Mia Doričić, Vedran Miletić
---

# rocBLAS: ROCm Basic Linear Algebra Subprograms

U nastavku koristimo kod iz repozitorija [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) ([službena dokumentacija](https://rocblas.readthedocs.io/)).

rocBLAS nudi za korištenje osnovne podprograme koji u domeni linearne algebre.

## Tipovi podataka

Postoji velik broj tipova podataka koje rocBLAS koristi, a najvažniji od njih su:

- `rocblas_int` služi za određivanje koristi li se `int32` (podatkovni model LP64) ili `int64` (podatkovni model ILP64) ([više informacija o 64-bitnim podatkovnim modelima](https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models))
- `rocblas_stride` prolazi između matrica ili vektora u funkcijama tipa `strided_batched`
- `rocblas_half` služi za predstavljanje 16-bitnog broja s pomičnim zarezom (tzv. polovična preciznost, engl. *half-precision*)
- `rocblas_bfloat16` služi za predstavljanje 16-bitnog Brainovog broja s pomičnim zarezom ([više informacija o formatu s pomičnim zarezom bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format))
- `rocblas_float_complex` služi za predstavljanje realnih i imaginarnih dijelova kompleksnog broja u zapisu s jednostrukom preciznošću
- `rocblas_double_complex` služi za predstavljanje realnih i imaginarnih dijelova kompleksnog broja u zapisu s dvostrukom preciznošću
- `rocblas_handle` je struktura koja sadrži kontekst biblioteke rocBLAS; mora se inicijalizirati putem funkcije `rocblas_create_handle()`

Preostale tipove podataka moguće je pronaći u [službenoj dokumentaciji u dijelu rocBLAS Types](https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/samples/example_sscal.cpp).

## Enumeratori

- enumerator `rocblas_operation` služi za određivanje treba li matrica biti transponirana ili ne

    - `rocblas_operation_none` operira nad nepromijenjenom matricom
    - `rocblas_operation_transpose` operira sa transponiranom matricom
    - `rocblas_operation_conjugate_transpose` operira sa konjugatom transponirane matrice

- enumerator `rocblas_status`

    - `rocblas_status_success` je uspješan statusni kod, ostali označuju pojedine greške

## Funkcije

Podržana su sva tri nivoa BLAS-a i pojedina proširenja. Čitav popis funkcija i njihov opis moguće je naći u službenoj dokumentaciji u dijelu [rocBLAS Functions](https://rocblas.readthedocs.io/en/master/functions.html#rocblas-functions).

## Primjeri

### Primjer množenja vektora skalarom

Službeni primjer `clients/samples/example_sscal.cpp` ([poveznica na kod](https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/samples/example_sscal.cpp)) implementira primjer množenja vektora `x` skalarom `alpha`, što radi [BLAS-ova funkcija SSCAL](https://www.netlib.org/lapack/explore-html/df/d28/group__single__blas__level1_ga3252f1f70b29d59941e9bc65a6aefc0a.html#ga3252f1f70b29d59941e9bc65a6aefc0a) (Single-precision floating-point SCALar vector multiply; [pregled svih BLAS rutina](https://www.netlib.org/blas/#_blas_routines)).

Program izgleda ovako:

``` c++
int main()
{
  rocblas_int    N      = 10240;
  rocblas_status status = rocblas_status_success;
  float          alpha  = 10.0;

  std::vector<float> hx(N);
  std::vector<float> hz(N);
  float*             dx;

  double gpu_time_used;

  rocblas_handle handle;
  rocblas_create_handle(&handle);

  hipMalloc(&dx, N * sizeof(float));

  srand(1);
  rocblas_init<float>(hx, 1, N, 1);

  hz = hx;

  hipMemcpy(dx, hx.data(), sizeof(float) * N, hipMemcpyHostToDevice);

  printf("N        rocblas(us)     \n");

  gpu_time_used = get_time_us_sync_device(); // in microseconds


  status = rocblas_sscal(handle, N, &alpha, dx, 1);
  if(status != rocblas_status_success)
  {
      return status;
  }

  gpu_time_used = get_time_us_sync_device() - gpu_time_used;

  hipMemcpy(hx.data(), dx, sizeof(float) * N, hipMemcpyDeviceToHost);

  bool error_in_element = false;
  for(rocblas_int i = 0; i < N; i++)
  {
      if(hz[i] * alpha != hx[i])
      {
          printf("error in element %d: CPU=%f, GPU=%f ", i, hz[i] * alpha, hx[i]);
          error_in_element = true;
          break;
      }
  }

  printf("%d    %8.2f\n", (int)N, gpu_time_used);

  if(error_in_element)
  {
      printf("SSCAL TEST FAILS\n");
  }
  else
  {
      printf("SSCAL TEST PASSES\n");
  }

  hipFree(dx);
  rocblas_destroy_handle(handle);
  return rocblas_status_success;
}
```

Krenimo promatrati funkciju `main()`. Na početku se definiraju varijable veličine vektora, statusa izvođenja i vrijednost skalara te im se dodjeljuju vrijednosti. Koristi se ranije navedeni tip podataka `rocblas_int` za varijablu `N` koja ima vrijednost 10240, što znači da će vektor imati 10240 elemenata. Inicijaliziramo varijablu `status` tipa `rocblas_status` na vrijednost `rocblas_status_success` koji označava da dosad nije došlo do greške; skalar `alpha` ima vrijednost postavljenu na 10.0.

``` c++
rocblas_int    N      = 10240;
rocblas_status status = rocblas_status_success;
float          alpha  = 10.0;
```

Slijedi inicijalizacija dva vektora `hx` i `hz` (`h` označava memoriju domaćina (engl. *host*), odnosno osnovnog procesora) te pokazivača tipa `float` naziva `dx` (memorija uređaja (engl. *device*), odnosno grafičkog procesora). Također, stvara se varijabla `gpu_time_used` koja se koristi u kasnijim koracima za dohvaćanje vremena izvođenja operacija.

``` c++
std::vector<float> hx(N);
std::vector<float> hz(N);
float*             dx;

double gpu_time_used;
```

Stvara se rocBLAS drška `handle` preko koje ćemo provjeriti je li se putem pojavila greška. Nakon što smo stvorili `handle`, alocira se memorija na uređaju za vektor `x`.

``` c++
rocblas_handle handle;
rocblas_create_handle(&handle);

hipMalloc(&dx, N * sizeof(float));
```

Koristeći funkciju `srand()`, program postavlja sjeme generatora slučajnih brojeva na vrijednost 1. Zatim funkcijom `rocblas_init()` generira slučajne vrijednosti elemenata vektora na domaćinu i vektor `z` postavlja na iste vrijednosti kao vektor `x` tako da se kopira vrijednost vektora `hx` u `hz`.

``` c++
srand(1);
rocblas_init<float>(hx, 1, N, 1);

hz = hx;
```

Kopiramo podatke s domaćina na uređaj:

``` c++
hipMemcpy(dx, hx.data(), sizeof(float) * N, hipMemcpyHostToDevice);
```

Dohvaćamo vrijeme koje je bilo potrebno za operaciju putem funkcije `get_time_us_sync_device()` u mikrosekundama i počinjemo ispis iskorištenog vremena za izračun.

``` c++
printf("N        rocblas(us)     \n");

gpu_time_used = get_time_us_sync_device();
```

Naposlijetku pokrećemo funkciju SSCAL na grafičkom procesoru. U varijabla `status` spremit ćemo status nakon izvođenja funkcije `rocblas_sscal()` u kojoj se koriste prije-definirani `handle`, `N`, adresa od `alpha` i `dx`. Ako `status` nije uspjeh (`rocblas_status_success`), vraćamo taj status.

``` c++
status = rocblas_sscal(handle, N, &alpha, dx, 1);
if(status != rocblas_status_success)
{
    return status;
}
```

Računa se vrijeme potrebno za izvođenje operacije do ovog koraka, na način da se oduzima vrijednost koju smo računali prethodno sa istom varijablom.

``` c++
gpu_time_used = get_time_us_sync_device() - gpu_time_used;
```

Zatim, kopira se ispis podataka sa uređaja na domaćin.

``` c++
hipMemcpy(hx.data(), dx, sizeof(float) * N, hipMemcpyDeviceToHost);
```

Nakon svega, provjerava se je li uspješno provedena izračun vrijednosti svakog pojedinog elementa korištenjem funkcije `rocblas_scal`. Na početku je vrijednost varijable `error_in_element` koja označava moguću grešku u operaciji (tipa `bool`) na `false`. Ako u nekom elementu postji greška, ta vrijednost se mijenja u `true`.

``` c++
bool error_in_element = false;
for(rocblas_int i = 0; i < N; i++)
{
    if(hz[i] * alpha != hx[i])
    {
        printf("error in element %d: CPU=%f, GPU=%f ", i, hz[i] * alpha, hx[i]);
        error_in_element = true;
        break;
    }
}
```

Slijedi ispis vremena potrebnog za izvršavanje operacije:

``` c++
printf("%d    %8.2f\n", (int)N, gpu_time_used);
```

Za kraj primjera, ispisuje se da, u slučaju da se kroz prijašnju `for` petlju pojavila greška, je izračun bio neuspješan. U suprotnom, ispisuje se da je test bio uspješan.

``` c++
if(error_in_element)
{
    printf("SSCAL TEST FAILS\n");
}
else
{
    printf("SSCAL TEST PASSES\n");
}
```

Oslobađa se prethodno alocirana memorija, te se `handle` uništava.

``` c++
hipFree(dx);
rocblas_destroy_handle(handle);
```

Ukoliko je program uspješno došao do kraja, vraća se status uspješnog izvođenja:

``` c++
return rocblas_status_success;
```

!!! admonition "Zadatak"
    - Promijenite kod da izvodi zbroj vektora umjesto množenja vektora skalarom.
    - Promijenite kod da izvodi množenje matrice i vektora umjesto množenja vektora skalarom.

### Primjer množenja matrica

Službeni primjer `clients/samples/example_sgemm.cpp` ([poveznica na kod](https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/samples/example_sgemm.cpp)) prikazuje kako provesti množenje matrica čiji su elementi realni brojevi jednostruke preciznosti [BLAS-ovom funkcijom SGEMM](https://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html#gafe51bacb54592ff5de056acabd83c260) (Single-precision floating-point GEneral Matrix Multiply; [pregled svih BLAS rutina](https://www.netlib.org/blas/#_blas_routines)).

Promotrimo funkciju `main()`:

``` c++
int main()
{
  rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
  float             alpha = 1.1, beta = 0.9;

  rocblas_int m = DIM1, n = DIM2, k = DIM3;
  rocblas_int lda, ldb, ldc, size_a, size_b, size_c;
  int         a_stride_1, a_stride_2, b_stride_1, b_stride_2;
  rocblas_cout << "sgemm example" << std::endl;
  if (transa == rocblas_operation_none)
  {
      lda        = m;
      size_a     = k * lda;
      a_stride_1 = 1;
      a_stride_2 = lda;
      rocblas_cout << "N";
  }
  else
  {
      lda        = k;
      size_a     = m * lda;
      a_stride_1 = lda;
      a_stride_2 = 1;
      rocblas_cout << "T";
  }
  if (transb == rocblas_operation_none)
  {
      ldb        = k;
      size_b     = n * ldb;
      b_stride_1 = 1;
      b_stride_2 = ldb;
      rocblas_cout << "N: ";
  }
  else
  {
      ldb        = n;
      size_b     = k * ldb;
      b_stride_1 = ldb;
      b_stride_2 = 1;
      rocblas_cout << "T: ";
  }
  ldc    = m;
  size_c = n * ldc;

  // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
  std::vector<float> ha(size_a);
  std::vector<float> hb(size_b);
  std::vector<float> hc(size_c);
  std::vector<float> hc_gold(size_c);

  // initial data on host
  srand(1);
  for(int i = 0; i < size_a; ++i)
  {
      ha[i] = rand() % 17;
  }
  for(int i = 0; i < size_b; ++i)
  {
      hb[i] = rand() % 17;
  }
  for(int i = 0; i < size_c; ++i)
  {
      hc[i] = rand() % 17;
  }
  hc_gold = hc;

  // allocate memory on device
  float *da, *db, *dc;
  CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

  // copy matrices from host to device
  CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));

  rocblas_handle handle;
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

  CHECK_ROCBLAS_ERROR(
      rocblas_sgemm(handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc));

  // copy output from device to CPU
  CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

  rocblas_cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda
               << ", " << ldb << ", " << ldc << std::endl;

  float max_relative_error = std::numeric_limits<float>::min();

  // calculate golden or correct result
  mat_mat_mult<float>(alpha,
                      beta,
                      m,
                      n,
                      k,
                      ha.data(),
                      a_stride_1,
                      a_stride_2,
                      hb.data(),
                      b_stride_1,
                      b_stride_2,
                      hc_gold.data(),
                      1,
                      ldc);

  for(int i = 0; i < size_c; i++)
  {
      float relative_error = (hc_gold[i] - hc[i]) / hc_gold[i];
      relative_error       = relative_error > 0 ? relative_error : -relative_error;
      max_relative_error
          = relative_error < max_relative_error ? max_relative_error : relative_error;
  }
  float eps       = std::numeric_limits<float>::epsilon();
  float tolerance = 10;
  if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
  {
      rocblas_cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
  }
  else
  {
      rocblas_cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
  }

  CHECK_HIP_ERROR(hipFree(da));
  CHECK_HIP_ERROR(hipFree(db));
  CHECK_HIP_ERROR(hipFree(dc));
  CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
  return EXIT_SUCCESS;
}
```

Puno je pripreme prije i čišćenja nakon poziva funkcije `rocblas_sgemm()` ([dokumentacija](https://rocblas.readthedocs.io/en/master/functions.html#_CPPv413rocblas_sgemm14rocblas_handle17rocblas_operation17rocblas_operation11rocblas_int11rocblas_int11rocblas_intPKfPKf11rocblas_intPKf11rocblas_intPKfPf11rocblas_int)) koji čini srž programa pa analirajmo kod dio po dio.

Program počinje deklaracijom dvije varijable tipa `rocblas_operation` imena `transa` i `transb` čije su vrijednosti postavljene na `rocblas_operation_none` (nepromijenjena matrica) i `rocblas_operation_transpose` (transponirana matrica).

Zatim se postavljaju varijable `alpha` i `beta` tipa `float` koje će biti iskorištene kao skalari na određene vrijednosti:

``` c++
rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
float             alpha = 1.1, beta = 0.9;
```

Također se definiraju tri dimenzije koje će se koristiti, u obliku varijabli `m`, `n` i `k` ranije spomenutog tipa `rocblas_int`. Isto tako, definiraju se i vodeće dimenzije za matrice `a`, `b` i `c` te njihove veličine, također tipa `rocblas_int`.

``` c++
rocblas_int m = DIM1, n = DIM2, k = DIM3;
rocblas_int lda, ldb, ldc, size_a, size_b, size_c;
```

Nakon toga slijedi definicija int varijabli `stride` za `a` i `b`, koje označavaju broj lokacija u memoriji između elemenata nekog polja (u našem slučaju elementa retka i elementa idućeg retka unutar jednog stupca, odnosno elementa stupca i elementa idućeg stupca unutar jednog retka). U ovom slučaju imamo:

``` c++
int         a_stride_1, a_stride_2, b_stride_1, b_stride_2;
```

Ispisuje se putem `rocblas_cout`-a (koji proširuje `std::cout` da je ovo primjer SGEMM raučunanja, te krećemo na provjere o kojoj se varijanti izračuna radi ovisno o tome koje su matrice transponirane, a koje nepromijenjene.

Vršimo provjeru je li matrica `a` bez operacije transponiranja, a ako je matrica `a` ostaje matrica formata `m * k` pa postavljamo:

- vodeća dimenzija za A postaje `m` (dakle `DIM1`)
- veličina za `a` postaje jednaka umnošku `k` i `lda`
- `stride_1` od A se postavlja na 1
- `stride_2` od A se postavlja na vrijednost lda
- putem `rocblas_cout` program vraća slovo `N`, potvrđujući da matrica `a` ne prolazi transponiranje

``` c++
rocblas_cout << "sgemm example" << std::endl;
if (transa == rocblas_operation_none)
{
    lda        = m;
    size_a     = k * lda;
    a_stride_1 = 1;
    a_stride_2 = lda;
    rocblas_cout << "N";
}
```

U protivnom, ako je matrica `a` transponirana (formata `k * m`) postavljamo:

- vodeća dimenzija za A postaje `k` (dakle `DIM3`)
- veličina za A postaje jednaka umnošku `m` i `lda`
- `stride_1` od A se postavlja na vrijednost lda
- `stride_2` od A se postavlja na 1
- putem `rocblas_cout`-a program vraća slovo `T`, potvrđujući da matrica `a` prolazi transponiranje

``` c++
else
{
    lda        = k;
    size_a     = m * lda;
    a_stride_1 = lda;
    a_stride_2 = 1;
    rocblas_cout << "T";
}
```

Ako pogledamo sljedeći provjeru naredbom `if`, primjetit ćemo da je analogna prethodnoj i postavlja varijable vezane za matricu `b`:

``` c++
if (transb == rocblas_operation_none)
{
    ldb        = k;
    size_b     = n * ldb;
    b_stride_1 = 1;
    b_stride_2 = ldb;
    rocblas_cout << "N: ";
}
else
{
    ldb        = n;
    size_b     = k * ldb;
    b_stride_1 = ldb;
    b_stride_2 = 1;
    rocblas_cout << "T: ";
}
```

Slijedi postavljanje vodeće dimezije za matricu `c` koja će biti formata `m * n`. Postavljamo je na `m` (`DIM1`) i njenu veličinu računamo kao umnožak `n` (`DIM2`) i `ldc`.

``` c++
ldc    = m;
size_c = n * ldc;
```

Inicijaliziramo tri matrice na domaćinu pa ih nazivamo početnim slovom `h`:

``` c++
std::vector<float> ha(size_a);
std::vector<float> hb(size_b);
std::vector<float> hc(size_c);
std::vector<float> hc_gold(size_c);
```

Uočimo da koristimo `std::vector` kao strukturu za pohranu podataka. Mogli smo koristiti i jednodimenzionalna polja, ali ne i ništa drugo jer BLAS očekuje matrice u tom obliku uz dodatne parametre koji navode njihov broj redaka i stupaca.

Koristeći funkciju `rand()`, kroz tri `for` petlje postavljaju se početne vrijednosti na svaki element sva tri vektora. Nakon svih petlji varijabla `hc_gold` u koju ćemo spremiti rezultat računanja na domaćinu izjednačena je sa dodjeljenim vrijednostima od `hc` u koju ćemo spremiti rezultat računanja na uređaju.

``` c++
srand(1);
for(int i = 0; i < size_a; ++i)
{
    ha[i] = rand() % 17;
}
for(int i = 0; i < size_b; ++i)
{
    hb[i] = rand() % 17;
}
for(int i = 0; i < size_c; ++i)
{
    hc[i] = rand() % 17;
}
hc_gold = hc;
```

Alociramo memoriju za iste matrice na uređaju putem funkcije `hipMalloc` i provjeravamo uspješnost korištenjem pomoćne makro funkcije `CHECK_HIP_ERROR()`:

``` c++
float *da, *db, *dc;
CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));
```

Zatim se matrice sa domaćina kopiraju na uređaj, također pritom provjeravajući hoće li se pojaviti greška:

``` c++
CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));
```

Stvara se rocBLAS-ova drška `handle` koja služi kao za provjeru je li se putem pojavila greška. Nakon što smo stvorili drška, napokon pozivamo funkciju `rocblas_sgemm` sa svim parametrima koje smo dosad definirali:

``` c++
rocblas_handle handle;
CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

CHECK_ROCBLAS_ERROR(
    rocblas_sgemm(handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc));
```

Rezultat ove funkcije kopiramo sa uređaja domaćin uz provjeru makro funkcijom te ispisujemo dobivene vrijednosti:

``` c++
CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

rocblas_cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda
             << ", " << ldb << ", " << ldc << std::endl;
```

Slijedi inicijalizacija varijable maksimalne relativne greške `max_relative_error` tipa `float` na najnižu moguću pozitivnu vrijednost i onemogućuje da vrijednost ode u negativne brojeve (za više informacija ovoj funkciji proučite [std::numeric_limits::min na cppreference.com](https://en.cppreference.com/w/cpp/types/numeric_limits/min)):

``` c++
float max_relative_error = std::numeric_limits<float>::min();
```

Došli smo do poziva funkcije `mat_mat_mult()` koja množi matrice na domaćinu i koja je definirana na početku datoteke kodom:

``` c++
void mat_mat_mult(T   alpha,
                  T   beta,
                  int M,
                  int N,
                  int K,
                  T*  A,
                  int As1,
                  int As2,
                  T*  B,
                  int Bs1,
                  int Bs2,
                  T*  C,
                  int Cs1,
                  int Cs2)
 {
   for(int i1 = 0; i1 < M; i1++)
   {
     for(int i2 = 0; i2 < N; i2++)
     {
       T t = 0.0;
       for(int i3 = 0; i3 < K; i3++)
       {
         t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
       }
       C[i1 * Cs1 + i2 * Cs2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
     }
   }
 }
```

U potpisu funkcije možemo vidjeti sve potrebne varijabli za skalare, matrice, njihove dimenzije, veličine i sl.

U tijelu funkcije nalaze se tri `for` petlje koje uvjetuju jedna drugu; dok svaki brojač za svaku dimenziju vrti petlju do maksimalne vrijednosti svake dimezije, provodi se uvećavanje varijable `t`, čiji je iznos definiran umnoškom retka matrice `A` i stupca matrice `B`, čije su veličine zbroj umnožaka vrijednosti brojača i veličine trenutnog elementa. Matrica `C` se množi skalarom `beta` i uvećava za skalar `alpha` pomnožen s varijablom `t` u kojoj je vrijednost umnoška retka i stupca. Time je funkcija ekvivalentna BLAS-ovoj funkciji SGEMM.

Ova funkcija je u `main()`-u pozvana kodom:

``` c++
mat_mat_mult<float>(alpha,
                    beta,
                    m,
                    n,
                    k,
                    ha.data(),
                    a_stride_1,
                    a_stride_2,
                    hb.data(),
                    b_stride_1,
                    b_stride_2,
                    hc_gold.data(),
                    1,
                    ldc);
```

Program nastavlja sa petljom `for` u kojoj se izračunava relativna greška između izračuna na domaćinu i uređaju i sprema u varijablu `relative_error` i onda se računa njegova apsolutna vrijednost. Varijabla `max_relative_error` sprema najveću vrijednost `relative_error`-a u svim iteracijama (podsjetimo se da je `?` uvjetni operator čiju dokumentaciju je moguće pronaći među [ostalim operatorima na cppreference.com](https://en.cppreference.com/w/cpp/language/operator_other)).

Zatim se i koristi vrijednost `epsilon()` iz `numeric_limits`, što je najmanja vrijednost veća od nule za neki tip podataka, u našem slučaju `float` (za više informacija o ovoj funkciji pogledajte [std::numeric_limits::epsilon na cppreference.com](https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon)) i postavlja se tolerancija `tolerance` na deset puta veću vrijednost od te.

``` c++
for(int i = 0; i < size_c; i++)
{
    float relative_error = (hc_gold[i] - hc[i]) / hc_gold[i];
    relative_error       = relative_error > 0 ? relative_error : -relative_error;
    max_relative_error
        = relative_error < max_relative_error ? max_relative_error : relative_error;
}
float eps       = std::numeric_limits<float>::epsilon();
float tolerance = 10;
```

Slijedi ispis, ovisno o ishodu operacije:

``` c++
if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
{
    rocblas_cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
}
else
{
    rocblas_cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
}
```

Naposlijetku dolazi oslobađanje prethodno alocirane memorije na uređaju (na domaćinu smo koristili `std::vector` koji ne moramo ručno dealocirati):

``` c++
CHECK_HIP_ERROR(hipFree(da));
CHECK_HIP_ERROR(hipFree(db));
CHECK_HIP_ERROR(hipFree(dc));
CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
return EXIT_SUCCESS;
```

!!! admonition "Zadatak"
    Promijenite kod da radi s matricama dvostruke preciznosti umjesto jednostruke.

---
author: Mia Doričić, Vedran Miletić
---

# rocSPARSE: ROCm SPARSE basic linear algebra subroutines

U nastavku koristimo kod iz repozitorija [rocSPARSE](https://github.com/ROCmSoftwarePlatform/rocSPARSE) ([službena dokumentacija](https://rocsparse.readthedocs.io/)).

rocSPARSE implementira podprograme bazične linearne algebre za rijetke matrice i vektore za GPU uređaje.

## Organizacija funkcionalnosti

- **Sparse Auxiliary Functions** -- pomoćne funkcije potrebne za pozive biblioteka podprograma
- **Sparse Level 1 Functions** -- operacije između vektora u rijetkom formatu i vektora u gustom formatu
- **Sparse Level 2 Functions** -- operacije između matrica u rijetkom formatu i vektora u gustom formatu
- **Sparse Level 3 Functions** -- operacije između matrice u rijetkom formatu i više vektora u gustom formatu
- **rocsparse_gemmi()** -- operacije koje manipuliraju rijetkim matricama
- **Preconditioner Functions** -- manipulacije na matrici u rijetkom formatu za dobivanje preduvjeta
- **Sparse Conversion Functions** -- operacije na matrici u rijretkom formatu za dobivanje drugačijeg formata matrice

## Formati pohranjivanja matrica

Prema [službenoj dokumentaciji](https://rocsparse.readthedocs.io/en/master/usermanual.html#storage-formats)):

- Format COO (Coordinate storage) -- matrica COO će se sortirati prema indeksima redaka i indeksima stupaca po retku
- Format CSR (Compressed Sparse Row storage) -- matrica CSR će se sortirati prema indeksima stupaca unutar svakog retka, te se svaki par indeksa mora pojaviti samo jednom
- Format BSR (Block Compressed Sparse Row storage) -- matrica BSR će se sortirati prema indeksima stupaca unutar svakog retka, međutim ako m ili n nisu ravnomjerno djeljivi s dimenzijom bloka, tada su matrice nadopunjene nulama
- Format ELL (Ellpack-Itpack storage) -- ELL matrica je pohranjena u *column-major* formatu, a redovi s elementima manjim od maksimalnog broja ne-nul elemenata po redu koji nisu nula obloženi su nulama i -1
- Format HYB (Hybrid storage) -- format HYB kombinacija je formata rijetke matrice ELL i COO. Obično se pravilni dio matrice pohranjuje u ELL formatu za pohranu, a nepravilni dio matrice u COO formatu za pohranu

## Primjer

Službeni primjer `clients/samples/example_ellmv.cpp` ([poveznica za kod](https://github.com/ROCmSoftwarePlatform/rocSPARSE/blob/develop/clients/samples/example_ellmv.cpp)) izvodi multipliciranje rijetkih matričnih vektora u formatu ELL.

Promotrimo program, odnosno funkciju `main()`:

``` c++
int main(int argc, char* argv[])
{
  // Parse command line
  if(argc < 2)
  {
      std::cerr << argv[0] << " <ndim> [<trials> <batch_size>]" << std::endl;
      return -1;
  }

  rocsparse_int ndim       = atoi(argv[1]);
  int           trials     = 200;
  int           batch_size = 1;

  if(argc > 2)
  {
      trials = atoi(argv[2]);
  }
  if(argc > 3)
  {
      batch_size = atoi(argv[3]);
  }

  // rocSPARSE handle
  rocsparse_handle handle;
  rocsparse_create_handle(&handle);

  hipDeviceProp_t devProp;
  int             device_id = 0;

  hipGetDevice(&device_id);
  hipGetDeviceProperties(&devProp, device_id);
  std::cout << "Device: " << devProp.name << std::endl;

  // Generate problem in CSR format
  std::vector<rocsparse_int> hAptr;
  std::vector<rocsparse_int> hAcol;
  std::vector<double>        hAval;

  rocsparse_int m;
  rocsparse_int n;
  rocsparse_int nnz;

  rocsparse_init_csr_laplace2d(
      hAptr, hAcol, hAval, ndim, ndim, m, n, nnz, rocsparse_index_base_zero);

  // Sample some random data
  rocsparse_seedrand();

  double halpha = random_generator<double>();
  double hbeta  = 0.0;

  std::vector<double> hx(n);
  rocsparse_init<double>(hx, 1, n, 1);

  // Matrix descriptors
  rocsparse_mat_descr descrA;
  rocsparse_create_mat_descr(&descrA);

  rocsparse_mat_descr descrB;
  rocsparse_create_mat_descr(&descrB);

  // Offload data to device
  rocsparse_int* dAptr = NULL;
  rocsparse_int* dAcol = NULL;
  double*        dAval = NULL;
  double*        dx    = NULL;
  double*        dy    = NULL;

  hipMalloc((void**)&dAptr, sizeof(rocsparse_int) * (m + 1));
  hipMalloc((void**)&dAcol, sizeof(rocsparse_int) * nnz);
  hipMalloc((void**)&dAval, sizeof(double) * nnz);
  hipMalloc((void**)&dx, sizeof(double) * n);
  hipMalloc((void**)&dy, sizeof(double) * m);

  hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
  hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
  hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice);
  hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice);

  // Convert CSR matrix to ELL format
  rocsparse_int* dBcol = NULL;
  double*        dBval = NULL;

  // Determine ELL width
  rocsparse_int ell_width;
  rocsparse_csr2ell_width(handle, m, descrA, dAptr, descrB, &ell_width);

  // Allocate memory for ELL storage format
  hipMalloc((void**)&dBcol, sizeof(rocsparse_int) * ell_width * m);
  hipMalloc((void**)&dBval, sizeof(double) * ell_width * m);

  // Convert matrix from CSR to ELL
  rocsparse_dcsr2ell(handle, m, descrA, dAval, dAptr, dAcol, descrB, ell_width, dBval, dBcol);

  // Clean up CSR structures
  hipFree(dAptr);
  hipFree(dAcol);
  hipFree(dAval);

  // Warm up
  for(int i = 0; i < 10; ++i)
  {
      // Call rocsparse ellmv
      rocsparse_dellmv(handle,
                       rocsparse_operation_none,
                       m,
                       n,
                       &halpha,
                       descrB,
                       dBval,
                       dBcol,
                       ell_width,
                       dx,
                       &hbeta,
                       dy);
  }

  // Device synchronization
  hipDeviceSynchronize();

  // Start time measurement
  double time = get_time_us();

  // ELL matrix vector multiplication
  for(int i = 0; i < trials; ++i)
  {
      for(int i = 0; i < batch_size; ++i)
      {
          // Call rocsparse ellmv
          rocsparse_dellmv(handle,
                           rocsparse_operation_none,
                           m,
                           n,
                           &halpha,
                           descrB,
                           dBval,
                           dBcol,
                           ell_width,
                           dx,
                           &hbeta,
                           dy);
      }

      // Device synchronization
      hipDeviceSynchronize();
  }

  time = (get_time_us() - time) / (trials * batch_size * 1e3);
  double bandwidth
      = static_cast<double>(sizeof(double) * (2 * m + nnz) + sizeof(rocsparse_int) * (nnz)) / time
        / 1e6;
  double gflops = static_cast<double>(2 * nnz) / time / 1e6;

  std::cout.precision(2);
  std::cout.setf(std::ios::fixed);
  std::cout.setf(std::ios::left);
  std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "nnz"
            << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
            << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::endl;
  std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << ell_width * m
            << std::setw(12) << halpha << std::setw(12) << hbeta << std::setw(12) << gflops
            << std::setw(12) << bandwidth << std::setw(12) << time << std::endl;

  // Clear up on device
  rocsparse_destroy_mat_descr(descrA);
  rocsparse_destroy_mat_descr(descrB);
  rocsparse_destroy_handle(handle);

  hipFree(dBcol);
  hipFree(dBval);
  hipFree(dx);
  hipFree(dy);

  return 0;
}
```

Krenimo od početka. U inicijalizaciji `main()` funkcije stvaraju se dvije varijable `argc` i `argv` (za više informacija što su te dvije varijable pogledajte [cppreference](https://en.cppreference.com/w/cpp/language/main_function)).

Prvo se pojavljuje uvjetovanje `if` sa postavljenim ispitivanjem je li `argc` manji od 2. U slučaju da jest, provodi se koristi se ispis na standardni izlaz za greške objektom `std::cerr` (za više infromacija o ovom objektu pogledajte [cppreference](https://en.cppreference.com/w/cpp/io/cerr)).

Dakle, u slučaju bez pojavljivanja greške, ispisuju se trenutne vrijednosti varijabli `argv[0]`, `ndim`, `trials` i `batch_size`:

``` c++
if(argc < 2)
{
    std::cerr << argv[0] << " <ndim> [<trials> <batch_size>]" << std::endl;
    return -1;
}
```

Slijedi postavljanje nove vrijednosti `rocsparse_int` varijable `ndim` na vrijednost koja se nalazi u `agrv[1]` pomoću funkcije `atoi()`, obzirom da je `argv[]` tipa `char` (za više informacija o ovoj funkciji pogledajte [cppreference](https://en.cppreference.com/w/c/string/byte/atoi)).

Također, isto se događa na `trials` i `batch_size`, te oni poprimaju vrijednosti cijelih brojeva.

``` c++
rocsparse_int ndim       = atoi(argv[1]);
int           trials     = 200;
int           batch_size = 1;
```

Nakon toga putem `if` uvjetovanja postavljamo vrijednosti na isti način kao i u prijašnjem koraku pomoću funkcije `atoi()`.

``` c++
if(argc > 2)
{
    trials = atoi(argv[2]);
}
if(argc > 3)
{
    batch_size = atoi(argv[3]);
}
```

Stvaramo rocSPARSE `handle`, na jednak način kao što smo vidjeli u rocBLAS primjerima.

``` c++
rocsparse_handle handle;
rocsparse_create_handle(&handle);
```

Sljedeći korak je stvaranje parametara koji će nam biti potrebni za dohvaćanje informacija o uređaju na kojem se vrše operacije, te ispis imena.

``` c++
hipDeviceProp_t devProp;
int             device_id = 0;

hipGetDevice(&device_id);
hipGetDeviceProperties(&devProp, device_id);
std::cout << "Device: " << devProp.name << std::endl;
```

Dalje slijedi stvaranje matrice na kojoj će se provoditi operacije. Da bismo to postigli, potrebne su nove varijable koje će definirati pokazivač po matrici (*hAptr*), stupac po kojem se vrši sortiranje (*hAcol*) i vrijednost (*hAval*).
Također, inicijaliziramo varijable u kojima će se pohranjivati vrijednosti elemenata matrice.

``` c++
std::vector<rocsparse_int> hAptr;
std::vector<rocsparse_int> hAcol;
std::vector<double>        hAval;

rocsparse_int m;
rocsparse_int n;
rocsparse_int nnz;

rocsparse_init_csr_laplace2d(
    hAptr, hAcol, hAval, ndim, ndim, m, n, nnz, rocsparse_index_base_zero);
```

Stvaramo nasumične vrijednosti koje ćemo pohraniti u matricu.

``` c++
rocsparse_seedrand();

double halpha = random_generator<double>();
double hbeta  = 0.0;

std::vector<double> hx(n);
rocsparse_init<double>(hx, 1, n, 1);
```

Nastavljamo sa stvaranjem matričnih deskriptora (strukture koje sadrže sva svojstva matrice), stvaramo iste parametre koje smo napravili za matricu A, ali u ovom slučaju za matricu B koja će služiti za konverziju formata.
Nakon toga potrebno je alocirati memoriju za novu matricu kojoj su vrijednosti trenutno postavljeni na NULL, te nakon toga kopirati podatke na uređaj.

``` c++
rocsparse_mat_descr descrA;
rocsparse_create_mat_descr(&descrA);

rocsparse_mat_descr descrB;
rocsparse_create_mat_descr(&descrB);


rocsparse_int* dAptr = NULL;
rocsparse_int* dAcol = NULL;
double*        dAval = NULL;
double*        dx    = NULL;
double*        dy    = NULL;

hipMalloc((void**)&dAptr, sizeof(rocsparse_int) * (m + 1));
hipMalloc((void**)&dAcol, sizeof(rocsparse_int) * nnz);
hipMalloc((void**)&dAval, sizeof(double) * nnz);
hipMalloc((void**)&dx, sizeof(double) * n);
hipMalloc((void**)&dy, sizeof(double) * m);

hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice);
hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice);
```

Sljedeći korak je da pripremimo varijable potrebne za konverziju iz CSR u ELL format, a to su `dBcol` i `dBval`, i njih postavljamo na NULL vrijednost.
Također, moramo odrediti i maksimalni broj ne-nul elemenata po redu, što nam je također potrebno za konverziju (za više informacija o korištenoj funkciji `rocsparse_csr2ell_width()` pogledajte [službenu dokumentaciju](https://rocsparse.readthedocs.io/en/master/usermanual.html#rocsparse-csr2ell)).

``` c++
rocsparse_int* dBcol = NULL;
double*        dBval = NULL;

rocsparse_int ell_width;
rocsparse_csr2ell_width(handle, m, descrA, dAptr, descrB, &ell_width);
```

Zatim, alociramo memoriju za ELL format za pohranu, te vršimo konverziju (za više informacija o korištenoj funkciji `rocsparse_dcsr2ell()` pogledajte [službenu dokumentaciju](https://rocsparse.readthedocs.io/en/master/usermanual.html#rocsparse-csr2ell)).

``` c++
hipMalloc((void**)&dBcol, sizeof(rocsparse_int) * ell_width * m);
hipMalloc((void**)&dBval, sizeof(double) * ell_width * m);

rocsparse_dcsr2ell(handle, m, descrA, dAval, dAptr, dAcol, descrB, ell_width, dBval, dBcol);
```

Oslobađamo memoriju koja je bila alocirana za CSR strukture, te kroz `for` petlju pozivamo funkciju `rocsparse_dellvm()`. Ta funkcija množi skalar `alfa` sa rijetkom matricom MxN u formatu ELL, i vektorom u gustom formatu, te dodaje rezultat drugom vektoru gustog formata koji se množi sa skalarom `beta` (za više informacija o ovoj funkciji pogledajte [službenu dokumentaciju](https://rocsparse.readthedocs.io/en/master/usermanual.html#rocsparse-ellmv))

``` c++
hipFree(dAptr);
hipFree(dAcol);
hipFree(dAval);

for(int i = 0; i < 10; ++i)
{

    rocsparse_dellmv(handle,
                     rocsparse_operation_none,
                     m,
                     n,
                     &halpha,
                     descrB,
                     dBval,
                     dBcol,
                     ell_width,
                     dx,
                     &hbeta,
                     dy);
}
```

Sljedeći korak je sinkronizirati uređaj, odnosno provjeriti ako su sve do sada izvedene rocSPARSE funkcije završile, te definiramo potrebne varijable za mjerenje vremena izvršavanja sljedećih operacija.

``` c++
hipDeviceSynchronize();

double time = get_time_us();
```

Sada pozivamo funkciju koju smo već pozvali, ali u ovom slučaju je unutar dvije `for` petlje, koje predstavljaju vektore koje množimo, te je njihov broj prolaza kroz petlju definiran sa varijablama `trials` i `batch_size`. Nakon množenja slijedi još jedna sinkronizacija.

``` c++
for(int i = 0; i < trials; ++i)
 {
    for(int i = 0; i < batch_size; ++i)
    {
        // Call rocsparse ellmv
        rocsparse_dellmv(handle,
                         rocsparse_operation_none,
                         m,
                         n,
                         &halpha,
                         descrB,
                         dBval,
                         dBcol,
                         ell_width,
                         dx,
                         &hbeta,
                         dy);
    }

    hipDeviceSynchronize();
}
```

Mjerimo vrijeme, definiramo na koji način će `bandwidth` i `gflops` biti izračunati te ispisujemo sve informacije koje smo dohvatili iz programa.

``` c++
time = (get_time_us() - time) / (trials * batch_size * 1e3);
double bandwidth
    = static_cast<double>(sizeof(double) * (2 * m + nnz) + sizeof(rocsparse_int) * (nnz)) / time
      / 1e6;
double gflops = static_cast<double>(2 * nnz) / time / 1e6;

std::cout.precision(2);
std::cout.setf(std::ios::fixed);
std::cout.setf(std::ios::left);
std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "nnz"
          << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
          << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::endl;
std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << ell_width * m
          << std::setw(12) << halpha << std::setw(12) << hbeta << std::setw(12) << gflops
          << std::setw(12) << bandwidth << std::setw(12) << time << std::endl;
```

Kao finalni korak, uništavamo matrične deskriptore, kao i `handle`, te oslobađamo memoriju koju smo prethodno alocirali.

``` c++
rocsparse_destroy_mat_descr(descrA);
rocsparse_destroy_mat_descr(descrB);
rocsparse_destroy_handle(handle);

hipFree(dBcol);
hipFree(dBval);
hipFree(dx);
hipFree(dy);
```

Kraj programa.

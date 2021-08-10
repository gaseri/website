---
author: Mia Doričić, Vedran Miletić
---

# rocPRIM: ROCm parallel PRIMitives

U nastavku koristimo kod iz repozitorija [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) ([službena dokumentacija](https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/)).

## Terminologija

(Sastavljeno prema [rječniku pojmova u službenoj dokumentaciji](https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/group__rocprim__glossary.html).)

- Osnova (engl. *warp*) -- odnosi se na grupu niti koje se izvršavaju istovremeno na način jedna instrukcija na više niti (engl. *single instruction, multiple thread*, kraće SIMT). Također, za njih se kaže da su valna fronta (engl. *wavefront*) na AMD-ovim grafičkim procesorima.
- Hardverska veličina osnove (engl. *hardware warp size*) -- odnosi se na broj niti u osnovi, i taj broj je zadan u hardveru. Na Nvidijinim grafičkim procesorima veličina osnove iznosi 32, a na AMD-ovim grafičkim procesorima temeljenim na arhitekturi GCN ekvivalentna veličina valne fronte iznosi 64. Na AMD-ovim grafičkim procesorima temeljenim na arhitekturi RDNA podržane su veličine 32 i 64.
- Logička veličina osnove (engl. *logical warp size*) -- odnosi se na broj niti u osnovi definiran sa strane korisnika, koji može biti jednak ili manji od broja niti definiranim hardverom.
- Identifikator staze (engl. *lane ID*) -- odnosi se na identifikator niti unutar osnove. Logički identifikator staze se odnosi na identifikator niti u logičkoj osnovi.
- Identifikator osnove (engl. *warp ID*) -- odnosi se na identifikator hardverske/logičke osnove u bloku. Garantira se jedinstvenost za svaku osnovu.
- Blok (engl. *block*) -- odnosi se na grupu niti koje se izvršavaju na jednakoj računskoj jedinici (engl. *compute unit*, kraće CU). Ove niti mogu biti indeksirane korištenjem jedne dimenzije (X), dvije dimenzije (X, Y) ili 3 dimenzije (X, Y, Z). Blok se sastoji od više osnova. U nomenklaturi koju koriste C++ AMP i HIP, blok se naziva pločica (engl. *tile*)
- Ravan identifikator (engl. flat ID) -- odnosi se na sravnjen identifikator bloka (pločice) ili niti. Ravan identifikator je 1D vrijednost stvorena iz 2D ili 3D identifikatora. Primjerice, ravan identifikator niti (X, Y) u 2D bloku niti veličine 128x4 (XxY) je Y * 128 + X.

## Pregled dostupne funkcionalnosti u rocPRIM-u

rocPRIM implementira osnovne funkcije za paralelno računanje na grafičkim procesorima AMD Radeon. **Paralelne primitive** su alati pomoću kojih možemo implementirati željene paralelne algoritme. Za svaku od funkcija postoji više varijanti ovisno o parametrima. Njih dijelimo na:

- primitive koje se odnose na blok (engl. *block-wide*)
- primitive koje se odnose na čitav uređaj (engl. *device-wide*)
- primitive koje se odnose na osnovu (engl. *warp-wide*)

### Block-wide

Neki koji se često koriste su ([službena dokumentacija](https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/group__blockmodule.html)):

- `load()` vrši učitavanje podataka iz memorije
- `reduce()` vrši operaciju smanjivanja na razne načine (npr. minimum, maksimum, zbroj, produkt)
- `scan()` vrši uključno i isključno skeniranje (tzv. [prefiksni zbroj](https://en.wikipedia.org/wiki/Prefix_sum))
- `sort()` vrši sortiranje (ključeva, parova ključeva...)
- `store()` vrši pohranu skupova podataka u memoriju

### Device-wide

Neki koji se često koriste su ([službena dokumentacija](https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/group__devicemodule.html)):

- `histogram_even()` računa histogram
- `merge()` izvodi sjedinjavanje
- `partition()` stvara particije
- `reduce(...)` vrši operaciju smanjivanja na razne načine (npr. minimum, maksimum, zbroj, produkt)
- `select(...)` odabire kontretnog primitivca

### Warp-wide

Neki koji se često koriste su ([službena dokumentacija](https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/group__warpmodule.html)):

- `reduce()` vrši operaciju smanjivanja na razne načine (npr. minimum, maksimum, zbroj, produkt)
- `scan()` vrši uključno i isključno skeniranje (tzv. [prefiksni zbroj](https://en.wikipedia.org/wiki/Prefix_sum))
- `sort()` vrši sortiranje (ključeva, parova ključeva...)

### Pomoćni alati

Neki koji se često koriste su ([službena dokumentacija](https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/group__utilsmodule__functional.html)):

- `less` gleda manju vrijednost
- `less_equal` gleda vrijednost manje ili jednako
- `greater` gleda veću vrijednost
- `greater_equal` gleda vrijednost veće ili jednako
- `equal_to` gleda vrijednost jednaku danom elementu
- `not_equal_to` gleda vrijednost različitu od danog elementa
- `plus` vrši zbrajanje
- `minus` vrši oduzimanje
- `multiplies` vrši množenje
- `maximum` postavlja najveću moguću vrijednost
- `minimum` postavlja najmanju moguću vrijednost
- `identity` ostavlja sve elemente istima, ne mijenja ništa

## Primjer korištenja

Službeni primjer `example/example_temporary_storage.cpp` ([datoteka u repozitoriju rocPRIM na GitHubu](https://github.com/ROCmSoftwarePlatform/rocPRIM/blob/develop/example/example_temporary_storage.cpp)) za prevođenje u izvršni kod zahtijeva i datoteku zaglavlja `example/example_utils.hpp` ([datoteka u repozitoriju rocPRIM na GitHubu](https://github.com/ROCmSoftwarePlatform/rocPRIM/blob/develop/example/example_utils.hpp)).

Program pokazuje korištenje 4 vrste memorije za izvođenje [prefiksne sume](https://en.wikipedia.org/wiki/Prefix_sum) (poznate i pod nazivom kumulativna suma, inkluzivna suma, sken) primjenom redukcije pojedinih elemenata na jedan element zbrajanjem.

Promotrimo funkciju `main()` od koje program kreće:

``` c++
int main()
{
  // Initializing HIP device
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDeviceProperties(&device_properties, 0));

  // Show device info
  printf("Selected device:         %s  \n", device_properties.name              );
  printf("Available global memory: %lu \n", device_properties.totalGlobalMem    );
  printf("Shared memory per block: %lu \n", device_properties.sharedMemPerBlock );
  printf("Warp size:               %d  \n", device_properties.warpSize          );
  printf("Max threads per block:   %d  \n", device_properties.maxThreadsPerBlock);

  // Running kernels
  run_example_global_memory_storage<int>(1024);
  run_example_shared_memory<int>(1024);
  run_example_union_storage_types<int>(1024);
  run_example_dynamic_shared_memory<int>(1024);
}
```

U funkcije se prvo inicijalizira HIP uređaj i provjerava da je inicijalizacija uspješna.

Zatim se objekt `device_properties` koristi za ispis svojstava uređaja kao što su ime, dostupna globalna memorija (na suvremnim grafičkim karticama to je GDDR6 ili HBM memorija koja se navodi u specifikacijama), dijeljena memorija po bloku...

Kao zadnji korak, pozivaju se sve funkcije za pokretanje svih zrna u ovom programu s parametrima koji definiraju veličinu vektora. U nastavku analiziramo svaku od pojedinih funkcija.

Funkcija za pokretanje zrna koje koristi globalnu memoriju je oblika:

``` c++
template<class T>
void run_example_global_memory_storage(size_t size)
{
  constexpr unsigned int block_size = 256;
  // Make sure size is a multiple of block_size
  auto grid_size = (size + block_size - 1) / block_size;
  size = block_size * grid_size;

  // Generate input on host and copy it to device
  std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
  // Generating expected output for kernel
  std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size);
  // For reading device output
  std::vector<T> host_output(size);

  // Device memory allocation
  T * device_input;
  T * device_output;
  HIP_CHECK(hipMalloc(&device_input, host_input.size() * sizeof(typename decltype(host_input)::value_type)));
  HIP_CHECK(hipMalloc(&device_output, host_output.size() * sizeof(typename decltype(host_output)::value_type)));

  // Writing input data to device memory
  hip_write_device_memory<T>(device_input, host_input);

  // Allocating temporary storage in global memory
  using storage_type = typename rocprim::block_scan<T, block_size>::storage_type;
  storage_type *global_storage;
  HIP_CHECK(hipMalloc(&global_storage, (grid_size * sizeof(storage_type))));

  // Launching kernel example_shared_memory
  hipLaunchKernelGGL(
    HIP_KERNEL_NAME(example_global_memory_storage<block_size, T>),
    dim3(grid_size), dim3(block_size),
    0, 0,
    device_input, device_output, global_storage
  );

  // Reading output from device
  hip_read_device_memory<T>(host_output, device_output);

  // Validating output
  OUTPUT_VALIDATION_CHECK(
    validate_device_output(host_output, host_expected_output)
  );

  HIP_CHECK(hipFree(device_input));
  HIP_CHECK(hipFree(device_output));
  HIP_CHECK(hipFree(global_storage));

  std::cout << "Kernel run_example_global_memory_storage run was successful!" << std::endl;
}
```

U deklaraciji ove funkcije možemo vidjeti da je inicijalizirana varijabla size tipa `size_t`, koju ćemo koristiti u sljedećim koracima.

Na početku funkcije možemo vidjeti da je veličina bloka postavljena na 265:

``` c++
constexpr unsigned int block_size = 256;
```

Kako nam je kasnije u kodu potrebna veličina mreže, moramo ju saznati, a to ćemo postići na sljedeći način, pritom koristeći prije spomenutu varijablu *size* koja mora biti umnožak od veličine bloka i mreže:

``` c++
auto grid_size = (size + block_size - 1) / block_size;
size = block_size * grid_size;
```

U sljedećem koraku funkcije generiraju se unos, očekivani ispis i dobiveni ispis. Oni su svi tipa `std::vector<T>` (za više informacija pogledajte [cppreference](https://en.cppreference.com/w/cpp/container/vector)):

``` c++
std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size);
std::vector<T> host_output(size);
```

Slijedi alokacija memorije uređaja. Potrebno je alocirati memoriju i za unos i za ispis. Putem `HIP_CHECK` provjeravamo hoće li alokacija biti uspješna. Alokaciju memorije za unos i ispis vršimo putem funkcije hipMalloc na sljedeći način (za više informacija o `hipMalloc()` možete pogledati [službenu dokumentaciju](https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP_API/Memory-Management.html)):

``` c++
T * device_input;
T * device_output;
HIP_CHECK(hipMalloc(&device_input, host_input.size() * sizeof(typename decltype(host_input)::value_type)));
HIP_CHECK(hipMalloc(&device_output, host_output.size() * sizeof(typename decltype(host_output)::value_type)));
```

Koristimo prethodno definirane varijable `device_input` i `host_input` za upisivanje unosa na za to predodređenu memoriju na uređaju.

Slijedi alokacija privremene pohrane podataka u globalnoj memoriji:

``` c++
using storage_type = typename rocprim::block_scan<T, block_size>::storage_type;
storage_type *global_storage;
HIP_CHECK(hipMalloc(&global_storage, (grid_size * sizeof(storage_type))));
```

Ovdje se javlja prethodno definirana funkcija rocPRIM-a, `block_scan()`. Za više informacija pogledajte podnaslov *Block-wide*. Ponovno se, slično prethodnim koracima, putem `HIP_CHECK` traži provjera uspješne alokacije memorije hipMalloc.

Pokreće se zrno koje koristi globalnu memoriju za pohranu (više informacija o funkciji `hipLaunchKernellGGL()` koja se koristi u ovom primjeru možete saznati u [službenoj dokumentaciji](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)):

``` c++
hipLaunchKernelGGL(
  HIP_KERNEL_NAME(example_global_memory_storage<block_size, T>),
  dim3(grid_size), dim3(block_size),
  0, 0,
  device_input, device_output, global_storage
);
```

Prije nego što pogledamo kojeg je zrno oblika, postoji još par koraka pri kraju ove funkcije. Sljedeće što je potrebno je čitanje ispisa sa uređaja:

``` c++
hip_read_device_memory<T>(host_output, device_output);
```

Taj je ispis potrebno validirati odnosno pokrenuti provjeru da se utvrdi je li sve u redu sa ispisom:

``` c++
OUTPUT_VALIDATION_CHECK(
  validate_device_output(host_output, host_expected_output)
);
```

Kada su svi koraci zadovoljeni, sada je vrijeme za oslobađanje memorije koju smo prethodno alocirali, i pritom provjeravamo hoće li ta akcija proći uspješno, a to činimo na sljedeći način:

``` c++
HIP_CHECK(hipFree(device_input));
HIP_CHECK(hipFree(device_output));
HIP_CHECK(hipFree(global_storage));
```

Ako je sve prošlo po planu, funkcija će ispisati da je pokretanje zrna bilo uspješno.

Uspješno smo prošli kroz prvi primjer pokretanja zrna.

Naposlijetku, zrno koje smo pokretali je oblika:

``` c++
// Kernel 4 - Using global memory for storage
template<
  const unsigned int BlockSize,
  class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void example_global_memory_storage(
  const T *input,
  T *output,
  typename rocprim::block_scan<T, BlockSize>::storage_type *global_storage)
{
  // Indexing for  this block
  unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
  // specialize block_scan for type T and block of 256 threads
  using block_scan_type = rocprim::block_scan<T, BlockSize>;
  // Variables required for performing a scan
  T input_value, output_value;

  // execute inclusive scan
  input_value = input[index];

  block_scan_type()
    .inclusive_scan(
       input_value, output_value,
       global_storage[hipBlockIdx_x],
       rocprim::plus<T>()
    );

  output[index] = output_value;
}
```

Ponovno u početnom dijelu koda stoji funkcija `block_scan`, nakon čega slijedi indeksiranje niti za blokove.

Indeks niti računamo na način:

``` c++
unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
```

Usmjerimo funkciju `block_scan` na T i block koji smo prije postavili na 256 niti.

``` c++
using block_scan_type = rocprim::block_scan<T, BlockSize>;
```

Zatim postavljamo varijable koje su nam potrebne za provođenje skeniranja putem funkcije `block_scan_type` (specificirana funkcija s parametrom `type`, kako bi se odredilo koji algoritam funkcija treba pratiti).

Skeniranje započinje, te se podatci pohranjuju u memoriju određenu za ispis.

Prošli smo jedno zrno i njegovu funkciju za pokretanje. Nastavimo dalje promatrati sljedeće funkcije koje se pozivaju u prije pokazanoj main() funkciji.

Sljedeća funkcija koju ćemo promatrati je oblika:

``` c++
template<class T>
void run_example_shared_memory(size_t size)
{
  constexpr unsigned int block_size = 256;
  // Make sure size is a multiple of block_size
  unsigned int grid_size = (size + block_size - 1) / block_size;
  size = block_size * grid_size;

  // Generate input on host and copy it to device
  std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
  // Generating expected output for kernel
  std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size);
  // For reading device output
  std::vector<T> host_output(size);

  // Device memory allocation
  T * device_input;
  T * device_output;
  HIP_CHECK(hipMalloc(&device_input, host_input.size() * sizeof(typename decltype(host_input)::value_type)));
  HIP_CHECK(hipMalloc(&device_output, host_output.size() * sizeof(typename decltype(host_output)::value_type)));

  // Writing input data to device memory
  hip_write_device_memory<T>(device_input, host_input);

  // Launching kernel example_shared_memory
  hipLaunchKernelGGL(
    HIP_KERNEL_NAME(example_shared_memory<block_size, T>),
    dim3(grid_size), dim3(block_size),
    0, 0,
    device_input, device_output
  );

  // Reading output from device
  hip_read_device_memory<T>(host_output, device_output);

  // Validating output
  OUTPUT_VALIDATION_CHECK(
    validate_device_output(host_output, host_expected_output)
  );

  HIP_CHECK(hipFree(device_input));
  HIP_CHECK(hipFree(device_output));

  std::cout << "Kernel run_example_shared_memory run was successful!" << std::endl;
}
```

Usporedite funkciju `run_example_global_memory_storage` s funkcijom `run_example_shared_memory`:

``` c++
run_example_global_memory_storage                                                        run_example_shared_memory

  hip_write_device_memory<T>(device_input, host_input);                                    hip_write_device_memory<T>(device_input, host_input);

  using storage_type = typename rocprim::block_scan<T, block_size>::storage_type;
  storage_type *global_storage;
  HIP_CHECK(hipMalloc(&global_storage, (grid_size * sizeof(storage_type))));

hipLaunchKernelGGL(                                                                      hipLaunchKernelGGL(
  HIP_KERNEL_NAME(example_global_memory_storage<block_size, T>),                           HIP_KERNEL_NAME(example_shared_memory<block_size, T>),
  dim3(grid_size), dim3(block_size),                                                       dim3(grid_size), dim3(block_size),
  0, 0,                                                                                    0, 0,
  device_input, device_output, global_storage                                              device_input, device_output
);                                                                                       );

```

Primjetiti ćete da se razlikuju jedino u dijelu gdje se "run_example_global_memory_storage" bavi globalnom memorijom. Sve ostalo je postavljeno na jednak način.

Naime, zrno koje se pokreće u funkciji `run_example_shared_memory()` je oblika:

``` c++
template<
  const unsigned int BlockSize,
  class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void example_shared_memory(const T *input, T *output)
{
  // Indexing for  this block
  unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;

  // Allocating storage in shared memory for the block
  using block_scan_type = rocprim::block_scan<T, BlockSize>;
  __shared__ typename block_scan_type::storage_type storage;

  // Variables required for performing a scan
  T input_value, output_value;

  // Execute inclusive plus scan
  input_value = input[index];

  block_scan_type()
    .inclusive_scan(
       input_value,
       output_value,
       storage,
       rocprim::plus<T>()
  );

  output[index] = output_value;
}
```

Ovo zrno možete usporediti sa zrnom "example_global_memory_storage" koje smo prvo promatrali. Primjetiti ćete da su razlike minimalne,
upravo iz razloga jer su oba zrna namijenjena za istu svrhu, no koriste dva različita tipa memorije.

Konkretno ovo zrno namjenjeno je za dijeljenu memoriju, dok je "example_global_memory_storage" namijenjeno za globalnu memoriju.

Funkcija sljedeća na redu pozivanja u main() funkciji je oblika:

``` c++
template<class T>
void run_example_union_storage_types(size_t size)
{
  constexpr unsigned int block_size = 256;
  constexpr unsigned int items_per_thread = 4;
  // Make sure size is a multiple of block_size
  auto grid_size = (size + block_size - 1) / block_size;
  size = block_size * grid_size;

  // Generate input on host and copy it to device
  std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
  // Generating expected output for kernel
  std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size, items_per_thread);
  // For reading device output
  std::vector<T> host_output(size);

  // Device memory allocation
  T * device_input;
  T * device_output;
  HIP_CHECK(hipMalloc(&device_input, host_input.size() * sizeof(typename decltype(host_input)::value_type)));
  HIP_CHECK(hipMalloc(&device_output, host_output.size() * sizeof(typename decltype(host_output)::value_type)));

  // Writing input data to device memory
  hip_write_device_memory<T>(device_input, host_input);

  // Launching kernel example_union_storage_types
  hipLaunchKernelGGL(
     HIP_KERNEL_NAME(example_union_storage_types<block_size, items_per_thread, int>),
     dim3(grid_size), dim3(block_size),
     0, 0,
     device_input, device_output
  );

  // Reading output from device
  hip_read_device_memory<T>(host_output, device_output);

  // Validating output
  OUTPUT_VALIDATION_CHECK(
     validate_device_output(host_output, host_expected_output)
  );

  HIP_CHECK(hipFree(device_input));
  HIP_CHECK(hipFree(device_output));

  std::cout << "Kernel run_example_union_storage_types run was successful!" << std::endl;
}
```

Vidjeli smo do sada dvije funkcije `run_example...` koje su podosta slične, razlikuju se u detaljima oko globalne memorije. Ova funkcija je isto tako slična, jedine razlike koje postoje su:

``` c++
...

constexpr unsigned int block_size = 256;
constexpr unsigned int items_per_thread = 4;
...

std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size, items_per_thread);
...

hipLaunchKernelGGL(
    HIP_KERNEL_NAME(example_union_storage_types<block_size, items_per_thread, int>),
    dim3(grid_size), dim3(block_size),
    0, 0,
    device_input, device_output
);
...
```

Ako pogledate malo pažljivije, ovdje je uvedena nova int varijabla; *items_per_thread*. S obzirom da se radi o unijama memorija, potrebna nam je ta varijabla za zrno koje pozivamo putem te funkcije, a ono je oblika:

``` c++
template<
  const unsigned int BlockSize,
  const unsigned int ItemsPerThread,
  class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void example_union_storage_types(const T *input, T *output)
{
  // Specialize primitives
  using block_scan_type = rocprim::block_scan<
     T, BlockSize, rocprim::block_scan_algorithm::using_warp_scan
  >;
  using block_load_type = rocprim::block_load<
     T, BlockSize, ItemsPerThread, rocprim::block_load_method::block_load_transpose
  >;
  using block_store_type = rocprim::block_store<
     T, BlockSize, ItemsPerThread, rocprim::block_store_method::block_store_transpose
  >;
  // Allocate storage in shared memory for both scan and sort operations

  __shared__ union
  {
     typename block_scan_type::storage_type scan;
     typename block_load_type::storage_type load;
     typename block_store_type::storage_type store;
  } storage;

  constexpr int items_per_block = BlockSize * ItemsPerThread;
  int block_offset = (hipBlockIdx_x * items_per_block);

  // Input/output array for block scan primitive
  T values[ItemsPerThread];

  // Loading data for this thread
  block_load_type().load(
     input + block_offset,
     values,
     storage.load
  );
  rocprim::syncthreads();

  // Perform scan
  block_scan_type()
     .inclusive_scan(
         values, // as input
         values, // as output
         storage.scan,
         rocprim::plus<T>()
     );
  rocprim::syncthreads();

  // Save elements to output
  block_store_type().store(
     output + block_offset,
     values,
     storage.store
  );
}
```

U slučaju ovog zrna, koristiti će se operacije već definirane u samom rocPRIMu:

``` c++
using block_scan_type = rocprim::block_scan<
     T, BlockSize, rocprim::block_scan_algorithm::using_warp_scan
 >;
 using block_load_type = rocprim::block_load<
     T, BlockSize, ItemsPerThread, rocprim::block_load_method::block_load_transpose
 >;
 using block_store_type = rocprim::block_store<
     T, BlockSize, ItemsPerThread, rocprim::block_store_method::block_store_transpose
 >;
```

Ovo su tzv. primitive (za više informacija o svakoj od ovih operacija pogledajte početak ovog teksta podnaslov *Block-wide*).

Slijedi alokacija prostora za pohranu u dijeljenoj memoriji za ove operacije:

``` c++
__shared__ union
{
     typename block_scan_type::storage_type scan;
     typename block_load_type::storage_type load;
     typename block_store_type::storage_type store;
} storage;

constexpr int items_per_block = BlockSize * ItemsPerThread;
int block_offset = (hipBlockIdx_x * items_per_block);
```

Stvara se input/output polje za skeniranje prije navedenih blokova, nakon čega se učitavaju dobiveni podatci u ovoj sekvenci naredbi. Pritom se pokreće i skeniranje:

``` c++
// Input/output array for block scan primitive
 T values[ItemsPerThread];

// Loading data for this thread
block_load_type().load(
    input + block_offset,
    values,
    storage.load
);
rocprim::syncthreads();

// Perform scan
block_scan_type()
    .inclusive_scan(
        values, // as input
        values, // as output
        storage.scan,
        rocprim::plus<T>()
    );
rocprim::syncthreads();

// Save elements to output
block_store_type().store(
    output + block_offset,
    values,
    storage.store
);
}
```

Na kraju skeniranja podatci se pohranjuju u memoriju koja će se kopirati natrag na domaćina.

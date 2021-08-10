---
author: Mia Doričić, Vedran Miletić
---

# rocRAND: ROCm RANDom number generator

U nastavku koristimo kod iz repozitorija [rocRAND](https://github.com/ROCmSoftwarePlatform/rocRAND) ([službena dokumentacija](https://codedocs.xyz/ROCmSoftwarePlatform/rocRAND/)).

rocRAND nudi funkcije za generiranje pseudoslučajnih i kvazislučajnih brojeva.

Podržani generatori su:

- [XORWOW](https://en.wikipedia.org/wiki/Xorshift#xorwow)
- [MRG32k3a](https://www.jstor.org/stable/171570)
- [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister) za grafičke procesore ([MTGP32](https://arxiv.org/abs/1005.4973))
- [Philox](https://dl.acm.org/doi/10.1145/2063384.2063405) (4x32, 10 rundi)
- [Sobol32](https://en.wikipedia.org/wiki/Sobol_sequence)

## Pregled dostupne funkcionalnosti u rocRAND-u

### Funkcije koje rade na domaćinu

(Za više informacija proučite [rocRAND host API](https://codedocs.xyz/ROCmSoftwarePlatform/rocRAND/group__rocrandhost.html).)

- `rocrand_create_generator()` stvara novi generator slučajnih brojeva
- `rocrand_destroy_generator()` uništava generator slučajnih brojeva
- `rocrand_initialize_generator()` incijalizira stanje generatora slučajnih brojeva

Generiranje slučajnih cijelih brojeva:

- `rocrand_generate()` stvara uniformno distribuirane 32-bitne cijele brojeve

    - `rocrand_generate_short()` stvara 16-bitne
    - `rocrand_generate_char()` stvara 8-bitne

Generiranje slučajnih brojeva s pomičnim zarezom:

- `rocrand_generate_log_normal()` stvara 32-bitne logaritamski normalno distribuirane brojeve s pomičnim zarezom

    - `rocrand_generate_log_normal_half()` stvara 16-bitne
    - `rocrand_generate_log_normal_double()` stvara 64-bitne

- `rocrand_generate_normal()` stvara 32-bitne normalno distribuirane brojeve s pomičnim zarezom

    - `rocrand_generate_normal_half()` stvara 16-bitne
    - `rocrand_generate_normal_double()` stvara 64-bitne

- `rocrand_generate_poisson()` stvara 32-bitne Poissonovo distribuirane brojeve s pomičnim zarezom

    - `rocrand_generate_poisson_half()` stvara 16-bitne
    - `rocrand_generate_poisson_double()` stvara 64-bitne

- `rocrand_generate_uniform()` stvara 32-bitne uniformno distribuirane brojeve s pomičnim zarezom

    - `rocrand_generate_uniform_half()` stvara 16-bitne
    - `rocrand_generate_uniform_double()` stvara 64-bitne

Ostale funkcije:

- `rocrand_set_stream()` postavlja trenutni tok u kojem se pokreću zrna
- `rocrand_set_seed()` postavlja sjemenku generatora slučajnih brojeva
- `rocrand_set_offset()` postavlja odmak generatora slučajnih brojeva
- `rocrand_set_quasi_random_generator_dimensions()` postavlja broj dimenzija generatora kvazislučajnih brojeva
- `rocrand_get_version()` vraća verziju biblioteke

### Funkcije koje rade na uređaju

(Za više informacija proučite [rocRAND device functions](https://codedocs.xyz/ROCmSoftwarePlatform/rocRAND/group__rocranddevice.html).)

## Primjer korištenja

Službeni primjer `benchmark/benchmark_rocrand_generate.cpp` ([datoteka u repozitoriju rocRAND na GitHubu](https://github.com/ROCmSoftwarePlatform/rocRAND/blob/develop/benchmark/benchmark_rocrand_generate.cpp)) za prevođenje u izvršni kod zahtijeva i datoteku zaglavlja `benchmark/cmdparser.hpp` ([datoteka u repozitoriju rocRAND na GitHubu](https://github.com/ROCmSoftwarePlatform/rocRAND/blob/develop/benchmark/cmdparser.hpp)). Dodatni primjer je `benchmark/benchmark_rocrand_kernel.cpp` ([datoteka u repozitoriju rocRAND na GitHubu](https://github.com/ROCmSoftwarePlatform/rocRAND/blob/develop/benchmark/benchmark_rocrand_kernel.cpp)).

Ovaj program prikazuje kako se koriste funkcije za generiranje slučajnih brojeva.

Pogledajmo funkciju `main()` od koje program kreće:

``` c++
int main(int argc, char *argv[])
{
  cli::Parser parser(argc, argv);

  const std::string distribution_desc =
    "space-separated list of distributions:" +
    std::accumulate(all_distributions.begin(), all_distributions.end(), std::string(),
      [](std::string a, std::string b) {
        return a + "\n      " + b;
      }
    ) +
    "\n      or all";
  const std::string engine_desc =
    "space-separated list of random number engines:" +
    std::accumulate(all_engines.begin(), all_engines.end(), std::string(),
      [](std::string a, std::string b) {
        return a + "\n      " + b;
      }
    ) +
    "\n      or all";

  parser.set_optional<size_t>("size", "size", DEFAULT_RAND_N, "number of values");
  parser.set_optional<size_t>("dimensions", "dimensions", 1, "number of dimensions of quasi-random values");
  parser.set_optional<size_t>("trials", "trials", 20, "number of trials");
  parser.set_optional<std::vector<std::string>>("dis", "dis", {"uniform-uint"}, distribution_desc.c_str());
  parser.set_optional<std::vector<std::string>>("engine", "engine", {"philox"}, engine_desc.c_str());
  parser.set_optional<std::vector<double>>("lambda", "lambda", {10.0}, "space-separated list of lambdas of Poisson distribution");
  parser.run_and_exit_if_error();

  std::vector<std::string> engines;
  {
    auto es = parser.get<std::vector<std::string>>("engine");
    if (std::find(es.begin(), es.end(), "all") != es.end())
    {
      engines = all_engines;
    }
    else
    {
      for (auto e : all_engines)
      {
        if (std::find(es.begin(), es.end(), e) != es.end())
          engines.push_back(e);
        }
    }
  }

  std::vector<std::string> distributions;
  {
    auto ds = parser.get<std::vector<std::string>>("dis");
    if (std::find(ds.begin(), ds.end(), "all") != ds.end())
    {
      distributions = all_distributions;
    }
    else
    {
      for (auto d : all_distributions)
      {
        if (std::find(ds.begin(), ds.end(), d) != ds.end())
          distributions.push_back(d);
      }
    }
  }

  int version;
  ROCRAND_CHECK(rocrand_get_version(&version));
  int runtime_version;
  HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
  int device_id;
  HIP_CHECK(hipGetDevice(&device_id));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device_id));

  std::cout << "rocRAND: " << version << " ";
  std::cout << "Runtime: " << runtime_version << " ";
  std::cout << "Device: " << props.name;
  std::cout << std::endl << std::endl;

  for (auto engine : engines)
  {
    rng_type_t rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
    if (engine == "xorwow")
      rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
    else if (engine == "mrg32k3a")
      rng_type = ROCRAND_RNG_PSEUDO_MRG32K3A;
    else if (engine == "philox")
      rng_type = ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
    else if (engine == "sobol32")
      rng_type = ROCRAND_RNG_QUASI_SOBOL32;
    else if (engine == "mtgp32")
      rng_type = ROCRAND_RNG_PSEUDO_MTGP32;
    else
    {
      std::cout << "Wrong engine name" << std::endl;
      exit(1);
    }

    std::cout << engine << ":" << std::endl;

    for (auto distribution : distributions)
    {
      std::cout << "  " << distribution << ":" << std::endl;
      run_benchmarks(parser, rng_type, distribution);
    }
    std::cout << std::endl;
  }

  return 0;
}
```

Program počinje inicijalizacijom varijable `parser` tipa `cli::Parser` koji je definiran u ranije spomenutoj datoteci `cmdparser.hpp`; ta će se varijabla koristiti za rasčlanjivanje raznih vrsta naredbi.

Postavljaju se konstante tipa `std::string` koje će pohranjivati informacije o razdiobama i generatorima i to tako da su informacije odvojene znakom razmaka. Posebno se postavljaju početak i kraj tog znakovnog niza pomoću funkcije `std::accumulate` (za više informacija proučite [std::accumulate na cppreference.com](https://en.cppreference.com/w/cpp/algorithm/accumulate)):

``` c++
const std::string distribution_desc =
     "space-separated list of distributions:" +
     std::accumulate(all_distributions.begin(), all_distributions.end(), std::string(),
         [](std::string a, std::string b) {
             return a + "\n      " + b;
         }
     ) +
     "\n      or all";
const std::string engine_desc =
     "space-separated list of random number engines:" +
     std::accumulate(all_engines.begin(), all_engines.end(), std::string(),
         [](std::string a, std::string b) {
             return a + "\n      " + b;
         }
     ) +
     "\n      or all";
```

Slijedi postavljanje informacija kao što su veličina, dimenzije, zadatci, razdiobe, generatori i lambda (čiju ćemo svrhu objasniti u nastavku) u rasčlanjivač naredbi.

Kao zadnja naredba stoji pokretanje i izlaz u slučaju greške.

``` c++
parser.set_optional<size_t>("size", "size", DEFAULT_RAND_N, "number of values");
parser.set_optional<size_t>("dimensions", "dimensions", 1, "number of dimensions of quasi-random values");
parser.set_optional<size_t>("trials", "trials", 20, "number of trials");
parser.set_optional<std::vector<std::string>>("dis", "dis", {"uniform-uint"}, distribution_desc.c_str());
parser.set_optional<std::vector<std::string>>("engine", "engine", {"philox"}, engine_desc.c_str());
parser.set_optional<std::vector<double>>("lambda", "lambda", {10.0}, "space-separated list of lambdas of Poisson distribution");
parser.run_and_exit_if_error();
```

Nastavljamo na dio koji puni vektore s imenima pogona za generiranje slučajnih brojeva `engines` i razdioba slučajnih brojeva `distributions`. Kao uvjet stoji da funkcija `std::find` (za više informacija proučite [std::find na cppreference.com](https://en.cppreference.com/w/cpp/algorithm/find)), sa određenim početkom i krajem, traži sve elemente u vektoru koji nisu jednaki kao zadnji, da bi na taj način upisala sve generatore iz prethodno definiranog vektora `all_engines` u vektor `engines`. Jednake naredbe stoje i za popis razdioba:

``` c++
std::vector<std::string> engines;
{
    auto es = parser.get<std::vector<std::string>>("engine");
    if (std::find(es.begin(), es.end(), "all") != es.end())
    {
        engines = all_engines;
    }
    else
    {
        for (auto e : all_engines)
        {
            if (std::find(es.begin(), es.end(), e) != es.end())
                engines.push_back(e);
        }
    }
}

std::vector<std::string> distributions;
{
    auto ds = parser.get<std::vector<std::string>>("dis");
    if (std::find(ds.begin(), ds.end(), "all") != ds.end())
    {
        distributions = all_distributions;
    }
    else
    {
        for (auto d : all_distributions)
        {
            if (std::find(ds.begin(), ds.end(), d) != ds.end())
                distributions.push_back(d);
        }
    }
}
```

Vektori znakovnih nizova `all_engines` i `all_distributions` izvan funkcije `main()` su oblika:

``` c++
const std::vector<std::string> all_engines = {
  "xorwow",
  "mrg32k3a",
  "mtgp32",
  "philox",
  "sobol32",
};

const std::vector<std::string> all_distributions = {
  "uniform-uint",
  "uniform-uchar",
  "uniform-ushort",
  "uniform-half",
  // "uniform-long-long",
  "uniform-float",
  "uniform-double",
  "normal-half",
  "normal-float",
  "normal-double",
  "log-normal-half",
  "log-normal-float",
  "log-normal-double",
  "poisson"
};
```

Slijedi provjera funkcijom `ROCRAND_CHECK` u kojoj se dobavlja verzija biblioteke putem funkcije `rocrand_get_version()` spomenute na početku. Uz ovu provjeru, slijede još tri `HIP_CHECK` koji dohvaćaju runtime verziju, identifikacijski broj uređaja i svojstava tog uređaja.

``` c++
int version;
ROCRAND_CHECK(rocrand_get_version(&version));
int runtime_version;
HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
int device_id;
HIP_CHECK(hipGetDevice(&device_id));
hipDeviceProp_t props;
HIP_CHECK(hipGetDeviceProperties(&props, device_id));

std::cout << "rocRAND: " << version << " ";
std::cout << "Runtime: " << runtime_version << " ";
std::cout << "Device: " << props.name;
std::cout << std::endl << std::endl;
```

Nakon što su se ispisale iznad navedene informacije, nastavlja se petlja `for` u kojoj se na temelju znakovnog niza određuje tip generatora koji će se koristiti. Za svaki pojedini pogon se izvodi mjerenje performansi sa svakom od razdioba, što vidimo u dodatnoj petlji for.

``` c++
for (auto engine : engines)
{
     rng_type_t rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
     if (engine == "xorwow")
         rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
     else if (engine == "mrg32k3a")
         rng_type = ROCRAND_RNG_PSEUDO_MRG32K3A;
     else if (engine == "philox")
         rng_type = ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
     else if (engine == "sobol32")
         rng_type = ROCRAND_RNG_QUASI_SOBOL32;
     else if (engine == "mtgp32")
         rng_type = ROCRAND_RNG_PSEUDO_MTGP32;
     else
     {
         std::cout << "Wrong engine name" << std::endl;
         exit(1);
     }

     std::cout << engine << ":" << std::endl;

     for (auto distribution : distributions)
     {
         std::cout << "  " << distribution << ":" << std::endl;
         run_benchmarks(parser, rng_type, distribution);
     }

     std::cout << std::endl;
 }
```

Došli smo do kraja funkcije `main()`.

Pogledajmo malo detaljnije funkciju koja se poziva u dvije petlje `for` opisane iznad, funkciju `run_benchmarks()` (uočite množinu):

``` c++
void run_benchmarks(const cli::Parser& parser,
                 const rng_type_t rng_type,
                 const std::string& distribution)
{
 if (distribution == "uniform-uint")
 {
     run_benchmark<unsigned int>(parser, rng_type,
         [](rocrand_generator gen, unsigned int * data, size_t size) {
             return rocrand_generate(gen, data, size);
         }
     );
 }
 if (distribution == "uniform-uchar")
 {
     run_benchmark<unsigned char>(parser, rng_type,
         [](rocrand_generator gen, unsigned char * data, size_t size) {
             return rocrand_generate_char(gen, data, size);
         }
     );
 }
 if (distribution == "uniform-ushort")
 {
     run_benchmark<unsigned short>(parser, rng_type,
         [](rocrand_generator gen, unsigned short * data, size_t size) {
             return rocrand_generate_short(gen, data, size);
         }
     );
 }
 if (distribution == "uniform-half")
 {
     run_benchmark<__half>(parser, rng_type,
         [](rocrand_generator gen, __half * data, size_t size) {
             return rocrand_generate_uniform_half(gen, data, size);
         }
     );
 }
 if (distribution == "uniform-float")
 {
     run_benchmark<float>(parser, rng_type,
         [](rocrand_generator gen, float * data, size_t size) {
             return rocrand_generate_uniform(gen, data, size);
         }
     );
 }
 if (distribution == "uniform-double")
 {
     run_benchmark<double>(parser, rng_type,
         [](rocrand_generator gen, double * data, size_t size) {
             return rocrand_generate_uniform_double(gen, data, size);
         }
     );
 }
 if (distribution == "normal-half")
 {
     run_benchmark<__half>(parser, rng_type,
         [](rocrand_generator gen, __half * data, size_t size) {
             return rocrand_generate_normal_half(gen, data, size, 0.0f, 1.0f);
         }
     );
 }
 if (distribution == "normal-float")
 {
     run_benchmark<float>(parser, rng_type,
         [](rocrand_generator gen, float * data, size_t size) {
             return rocrand_generate_normal(gen, data, size, 0.0f, 1.0f);
         }
     );
 }
 if (distribution == "normal-double")
 {
     run_benchmark<double>(parser, rng_type,
         [](rocrand_generator gen, double * data, size_t size) {
             return rocrand_generate_normal_double(gen, data, size, 0.0, 1.0);
         }
     );
 }
 if (distribution == "log-normal-half")
 {
     run_benchmark<__half>(parser, rng_type,
         [](rocrand_generator gen, __half * data, size_t size) {
             return rocrand_generate_log_normal_half(gen, data, size, 0.0f, 1.0f);
         }
     );
 }
 if (distribution == "log-normal-float")
 {
     run_benchmark<float>(parser, rng_type,
         [](rocrand_generator gen, float * data, size_t size) {
             return rocrand_generate_log_normal(gen, data, size, 0.0f, 1.0f);
         }
     );
 }
 if (distribution == "log-normal-double")
 {
     run_benchmark<double>(parser, rng_type,
         [](rocrand_generator gen, double * data, size_t size) {
             return rocrand_generate_log_normal_double(gen, data, size, 0.0, 1.0);
         }
     );
 }
 if (distribution == "poisson")
 {
     const auto lambdas = parser.get<std::vector<double>>("lambda");
     for (double lambda : lambdas)
     {
         std::cout << "    " << "lambda "
              << std::fixed << std::setprecision(1) << lambda << std::endl;
         run_benchmark<unsigned int>(parser, rng_type,
             [lambda](rocrand_generator gen, unsigned int * data, size_t size) {
                 return rocrand_generate_poisson(gen, data, size, lambda);
             }
         );
     }
 }
}
```

Ova je funkcija u kojoj se ispituje o kojoj je razdiobi riječ. Sastoji se od trinaest grananja naredbom `if`, svaka od kojih uspoređuje trenutnu vrijednost varijable `distribution` sa svakom vrijednosti iz liste `all_distributions`. Unutar svakog `if`-a poziva se funkcija generatora slučajnih brojeva koju ćemo detaljnije razmotriti u nastavku.

Za primjer uzmimo naredbu `if` koja se odnosi na uniformnu razdiobu prirodnih brojeva:

``` c++
if (distribution == "uniform-uint")
 {
     run_benchmark<unsigned int>(parser, rng_type,
         [](rocrand_generator gen, unsigned int * data, size_t size) {
             return rocrand_generate(gen, data, size);
         }
     );
 }
```

Da bi funkcija `run_benchmark()` (uočite jedninu) bila pozvana, mora biti zadovoljen uvjet usporedbe. Ako je uvjet zadovoljen, funkcija `run_benchmark()` poziva rocRAND funkciju `rocrand_generate_...()` s generatorom slučajnih brojeva `gen` tipa `rocrand_generator`, mjestom za pohranu podataka `data` i brojem slučajnih brojeva koji će biti generirani `size`.

U ovom slučaju `rocrand_generate()` vratiti će nepredznačenu cjelobrojnu vrijednost obzirom da razdioba tako nalaže (za više informacija o uniformnoj cjelobrojnoj razdiobi pogledajte [std::uniform_int_distribution na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution)).

Ako malo obratite pozornost na svaku od razdioba, primjetiti ćete da svaka traži drugačiji tip vrijednosti:

- `"uniform-uchar"` traži `unsigned char`:

``` c++
if (distribution == "uniform-uchar")
 {
     run_benchmark<unsigned char>(parser, rng_type,
         [](rocrand_generator gen, unsigned char * data, size_t size) {
             return rocrand_generate_char(gen, data, size);
         }
     );
 }
```

- `uniform-ushort"` traži `unsigned short`:

``` c++
if (distribution == "uniform-ushort")
 {
     run_benchmark<unsigned short>(parser, rng_type,
         [](rocrand_generator gen, unsigned short * data, size_t size) {
             return rocrand_generate_short(gen, data, size);
         }
     );
 }
```

Ako pogledate svaku sljedeću, primjetiti ćete da se pojavljuju i tipovi `"normal-half"`, `"normal-float"`, `"normal-double"` (za više informacija o normalnoj razdiobi proučite [std::normal_distribution na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/normal_distribution)). Ovdje se traži normalna razdioba 32-bitnih brojeva s pomičnim zarezom.

``` c++
if (distribution == "normal-float")
 {
     run_benchmark<float>(parser, rng_type,
         [](rocrand_generator gen, float * data, size_t size) {
             return rocrand_generate_normal(gen, data, size, 0.0f, 1.0f);
         }
     );
 }
```

Također, pojavljuju se znakovni nizovi `"log-normal-*"` s različitim tipovima koji označavaju logaritamsku normalnu razdiobu za brojeve s pomičnim zarezom (za više informacija o logaritamskoj normalnoj razdiobi proučite [std::lognormal_distribution na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/lognormal_distribution)).

``` c++
if (distribution == "log-normal-float")
 {
     run_benchmark<float>(parser, rng_type,
         [](rocrand_generator gen, float * data, size_t size) {
             return rocrand_generate_log_normal(gen, data, size, 0.0f, 1.0f);
         }
     );
 }

```

Posljednja razdioba koji vidimo jest `"poisson"`, odnosno Poissonova.

``` c++
if (distribution == "poisson")
{
     const auto lambdas = parser.get<std::vector<double>>("lambda");
     for (double lambda : lambdas)
     {
         std::cout << "    " << "lambda "
              << std::fixed << std::setprecision(1) << lambda << std::endl;
         run_benchmark<unsigned int>(parser, rng_type,
             [lambda](rocrand_generator gen, unsigned int * data, size_t size) {
                 return rocrand_generate_poisson(gen, data, size, lambda);
             }
         );
     }
}
```

Poissonova razdioba se koristi za opis rijetkih događaja, odnosno događaja koji imaju veliki uzorak i malu vjerojatnost i ovisi samo o parametru `lambda`. Promjenom parametra `lambda` mijenja se oblik razdiobe (za više informacija o Poissonovoj razdiobi proučite [std::poisson_distribution na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/poisson_distribution)).

Pogledajmo sada detaljnije funkciju `run_benchmark`:

``` c++
void run_benchmark(const cli::Parser& parser,
                const rng_type_t rng_type,
                generate_func_type<T> generate_func)
{
 const size_t size0 = parser.get<size_t>("size");
 const size_t trials = parser.get<size_t>("trials");
 const size_t dimensions = parser.get<size_t>("dimensions");
 const size_t size = (size0 / dimensions) * dimensions;

 T * data;
 HIP_CHECK(hipMalloc((void **)&data, size * sizeof(T)));

 rocrand_generator generator;
 ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

 rocrand_status status = rocrand_set_quasi_random_generator_dimensions(generator, dimensions);
 if (status != ROCRAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
 {
     ROCRAND_CHECK(status);
 }

 // Warm-up
 for (size_t i = 0; i < 5; i++)
 {
     ROCRAND_CHECK(generate_func(generator, data, size));
 }
 HIP_CHECK(hipDeviceSynchronize());

 // Measurement
 auto start = std::chrono::high_resolution_clock::now();
 for (size_t i = 0; i < trials; i++)
 {
     ROCRAND_CHECK(generate_func(generator, data, size));
 }
 HIP_CHECK(hipDeviceSynchronize());
 auto end = std::chrono::high_resolution_clock::now();
 std::chrono::duration<double, std::milli> elapsed = end - start;

 std::cout << std::fixed << std::setprecision(3)
           << "      "
           << "Throughput = "
           << std::setw(8) << (trials * size * sizeof(T)) /
                 (elapsed.count() / 1e3 * (1 << 30))
           << " GB/s, Samples = "
           << std::setw(8) << (trials * size) /
                 (elapsed.count() / 1e3 * (1 << 30))
           << " GSample/s, AvgTime (1 trial) = "
           << std::setw(8) << elapsed.count() / trials
           << " ms, Time (all) = "
           << std::setw(8) << elapsed.count()
           << " ms, Size = " << size
           << std::endl;

 ROCRAND_CHECK(rocrand_destroy_generator(generator));
 HIP_CHECK(hipFree(data));
}
```

Ako pogledate pomnije u kod ovog programa, primjetiti ćete da je na početku samog koda definiran predložak:

``` c++
template<typename T>
using generate_func_type = std::function<rocrand_status(rocrand_generator, T *, size_t)>;
```

U deklaraciji funkcije `run_benchmark` stoji `generate_func` koji je proizašao upravo iz tog definiranog predloška, kojeg se koristi u nastavku koda.

Slijedi definicija varijabli `size0`, `trials`, `dimensions` i `size` gdje su svi dobiveni pomoću Parser dohvaćanja, osim veličine `size` koja je umnožak vrijednosti dimenzije i rezultata dijeljenja `size0` i `dimensions`.

``` c++
const size_t size0 = parser.get<size_t>("size");
const size_t trials = parser.get<size_t>("trials");
const size_t dimensions = parser.get<size_t>("dimensions");
const size_t size = (size0 / dimensions) * dimensions;
```

Nastavljamo sa HIP_CHECK provjerom alocirane memorije, i ROCRAND_CHECK provjerom generatora, i tipa generiranog broja.

``` c++
rocrand_generator generator;
ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));
```

Nakon toga definiramo `status`, čija će vrijednost biti jednaka rezultatu rocRAND funkcije `rocrand_set_quasi_random_generator_dimensions` koji je zapravo broj dimenzija generatora quasi-slučajnih brojeva. (Za više informacija o ovoj funkciji možete pogledati na početak ovog teksta):

``` c++
rocrand_status status = rocrand_set_quasi_random_generator_dimensions(generator, dimensions);
```

Slijedi provjera; ako vrijednost dobivena u `status` nije quasi-nasumičan broj, pokreće se ROCRAND_CHECK provjera za `status`.

``` c++
if (status != ROCRAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
{
    ROCRAND_CHECK(status);
}
```

Sljedeće se pokreće petlja `for` u kojoj se provodi ROCRAND_CHECK provjera generirane funkcije.

``` c++
for (size_t i = 0; i < 5; i++)
 {
     ROCRAND_CHECK(generate_func(generator, data, size));
 }
 HIP_CHECK(hipDeviceSynchronize());
```

Započinje mjerenje vremena izvršavanja provjere u `for` petlji koja ovaj puta broji do vrijednosti `trials`, nakon čega slijedi HIP_CHECK provjera sinkronizacije uređaja:

``` c++
for (size_t i = 0; i < trials; i++)
 {
     ROCRAND_CHECK(generate_func(generator, data, size));
 }

HIP_CHECK(hipDeviceSynchronize());
 auto end = std::chrono::high_resolution_clock::now();
 std::chrono::duration<double, std::milli> elapsed = end - start;
```

Nakon svega slijedi ispis svih dobivenih rezultata tokom ove funkcije u obliku definiranom putem `std::setprecision` (za više informacija o funkciji pogledajte [std::setprecision na cppreference.com](https://en.cppreference.com/w/cpp/io/manip/setprecision)):

``` c++
std::cout << std::fixed << std::setprecision(3)
           << "      "
           << "Throughput = "
           << std::setw(8) << (trials * size * sizeof(T)) /
                 (elapsed.count() / 1e3 * (1 << 30))
           << " GB/s, Samples = "
           << std::setw(8) << (trials * size) /
                 (elapsed.count() / 1e3 * (1 << 30))
           << " GSample/s, AvgTime (1 trial) = "
           << std::setw(8) << elapsed.count() / trials
           << " ms, Time (all) = "
           << std::setw(8) << elapsed.count()
           << " ms, Size = " << size
           << std::endl;
```

Za kraj funkcije uništava se generator, i oslobađa se prethodno alocirana memorija.

``` c++
ROCRAND_CHECK(rocrand_destroy_generator(generator));
HIP_CHECK(hipFree(data));
```

Uspješno smo prošli primjer programa koji koristi funkcije biblioteke rocRAND.

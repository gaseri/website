---
author: Vedran Miletić
---

# Generiranje pseudoslučajnih brojeva u C++ aplikacijama

Od verzije standarda C++11 moguće je generirati (pseudo)slučajne brojeve korištenjem standardne biblioteke ([detaljna dokumentacija na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random)).

Za početak ćemo uključiti zaglavlje `random`:

``` c++
#include <random>
```

Inicijalizirat ćemo generator `rd` tipa `std::random_device` ([dokumentacija na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/random_device)) koji će nam trebati za sijanje pogona za generirane slučajnih brojeva:

``` c++
std::random_device rd;
```

Inicijalizirat ćemo dva pogona za generiranje slučajnih brojeva. Prvi je tipa `std::default_random_engine` i implementacije standardne biblioteke definira po želi koji će od mogućih pogona biti korišten i prilikom inicijalizacije traži samo jednu vrijednost sjemenke slučajnih brojeva koja se tada generira pomoću generatora `rd`; drugi tipa `std::mt19937` i implementira (32-bitni) Mersenneov Twister i može koristiti više vrijednosti sjemenki pa inicijaliziramo varijablu `seed` tipa `std::seed_seq` koja te vrijednosti sadrži:

``` c++
std::default_random_engine engine1(rd());

std::seed_seq seed{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
std::mt19937 engine2(seed);
```

Detalji o dostupnim predefiniranim pogonima i njihovim parametrima mogu se naći u dokumentaciji osnovnih pogona:

- [std::linear_congruential_engine na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/linear_congruential_engine),
- [std::mersenne_twister_engine na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine),
- [std::subtract_with_carry_engine na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/subtract_with_carry_engine),

te dokumentaciji adaptera pogona:

- [std::discard_block_engine na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/discard_block_engine),
- [std::independent_bits_engine na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/independent_bits_engine),
- [std::shuffle_order_engine na cppreference.com](https://en.cppreference.com/w/cpp/numeric/random/shuffle_order_engine).

Inicijalizirati ćemo za korištenje uniformnu cjelobrojnu distribuciju u rasponu od 1 do 10 i normalnu distribuciju sa srednjom vrijednosti 5.0 i standardnom devijacijom 1.8:

``` c++
std::uniform_int_distribution<> uniform_dist(1, 10);
std::normal_distribution<> normal_dist(5.0, 1.8);
```

Slučajne vrijednosti generiramo korištenjem distribucije i pogona na način:

``` c++
int value1 = uniform_dist(engine1);
fmt::print("Randomly-chosen value (uniform integer distribution): {}\n", value1);

double value2 = normal_dist(engine2);
fmt::print("Randomly-chosen value (normal distribution): {}\n", value2);
```

Cjelokupan kod je oblika:

``` c++
#include <fmt/format.h>

#include <random>

int main()
{
    std::random_device rd;

    std::default_random_engine engine1(rd());

    std::seed_seq seed{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    std::mt19937 engine2(seed);

    std::uniform_int_distribution<> uniform_dist(1, 10);
    std::normal_distribution<> normal_dist(5.0, 2.0);

    int value1 = uniform_dist(engine1);
    fmt::print("Randomly-chosen value (uniform integer distribution): {}\n", value1);

    double value2 = normal_dist(engine2);
    fmt::print("Randomly-chosen value (normal distribution): {}\n", value2);

    return 0;
}
```

!!! admonition "Zadatak"
    Promijenite kod primjera tako da:

    - koristi pogon za generiranje slučajnih brojeva `knuth_b` umjesto zadanog i 64-bitni Mersenneov Twister umjesto 32-bitnog,
    - generira uniformno distribuirane realne brojeve umjesto cijelih (raspon po želji) te koristi logaritamsku normalnu razdiobu umjesto normalne (parametri po želji),
    - generira i ispisuje 10000 slučajnih brojeva te računa prosječnu vrijednost za obje distribucije.

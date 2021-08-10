---
author: Vedran Miletić
---

# Mjerenje brzine izvođenja C++ aplikacija

Pamćenjem vrijednosti vremena zidnog sata prije početka i nakon završetka izvođenja (dijela) koda možemo izmjeriti brzinu izvođenja. To ćemo izvesti korištenjem biblioteke `chrono` koja je dio standarda C++11 ([std::chrono na cppreference.com](https://en.cppreference.com/w/cpp/chrono)).

``` c++
#include <chrono>
#include <iostream>

int main()
{
  auto start = std::chrono::steady_clock::now();

  // kod čije vrijeme izvođenja mjerimo dolazi ovdje

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
}
```

Umjesto `std::cout` za ispis možemo koristiti i biblioteku [{fmt}](https://fmt.dev/) na način:

``` c++
#include <fmt/format.h>

#include <chrono>

int main()
{
  auto start = std::chrono::steady_clock::now();

  // kod čije vrijeme izvođenja mjerimo dolazi ovdje

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  fmt::print("elapsed time: {}s\n", elapsed_seconds.count());
}
```

Biblioteka {fmt} [od verzije 6.0.0 nadalje](https://github.com/fmtlib/fmt/blob/master/ChangeLog.rst#600---2019-08-26) može oblikovati trajanje vremena u odgovarajućim vremenskim jedinicama (u ovom slučaju sekundama) pa je nepotrebno pozivati metodu `count()` i ručno navoditi jedinice. Kako bi biblioteka {fmt} oblikovala vrijeme u odgovarajućim jedinicama, dovoljno je uključiti zaglavlje `fmt/chrono.h` i izravno koristiti vrijednost tipa `std::chrono::duration<>` ([std::chrono::duration na cppreference.com](https://en.cppreference.com/w/cpp/chrono/duration)):

``` c++
#include <fmt/format.h>

#include <chrono>

int main()
{
  auto start = std::chrono::steady_clock::now();

  // kod čije vrijeme izvođenja mjerimo dolazi ovdje

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  fmt::print("elapsed time: {}\n", elapsed_seconds);
}
```

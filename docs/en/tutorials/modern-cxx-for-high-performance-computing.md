---
marp: true
theme: default
class: default
math: katex
author: Vedran Miletić and Henri Menke
title: 'Modern C++ for High-Performance Computing: Concepts, Tools, and Optimization Strategies'
description: 'Meet MPCDF seminar June 2025'
keywords: debugging, sanitizers, compilers
abstract: |
  'C, C++, and Fortran have long been and are likely to remain the pillars of high-performance computing, each offering unique strengths in numerical computation, memory management, and parallel execution, making them indispensable in scientific simulations, large-scale data processing, and computational research. While Fortran excels in numerical computing and C provides low-level control, C++ builds upon both by offering powerful abstractions, modern memory management, and rich standard libraries, enabling developers to write high-performance, maintainable, and scalable HPC applications with greater flexibility and efficiency. This talk explores key techniques and tooling that empower developers: core language optimizations, memory management strategies, parallelization and acceleration approaches, alongside practical insights into profiling, benchmarking, and debugging libraries and tools.'
---

# Modern C++ for High-Performance Computing: Concepts, Tools, and Optimization Strategies

## [Vedran](https://vedran.miletic.net/) [Miletic](https://www.miletic.net/) and [Henri Menke](https://www.henrimenke.de/), HPC Application Support division, MPCDF

![MPG MPCDF logos](https://www.mpcdf.mpg.de/assets/institutes/headers/mpcdf-desktop-en-bc2a89605e5cb6effc55ad732f103d71afb8c7060ecaa95c5fb93987e4c8acbd.svg)

### [Max Planck](https://www.mpg.de/) [Computing & Data Facility](https://www.mpcdf.mpg.de/), [Meet MPCDF](https://www.mpcdf.mpg.de/services/training), [12. June 2025](https://datashare.mpcdf.mpg.de/s/p5Ny8pHqD0zTLxz)

---

<!--
paginate: true
header: Modern C++ for High-Performance Computing: Concepts, Tools, and Optimization Strategies || Vedran Miletic, Henri Menke
footer: Max Planck Computing & Data Facility || Meet MPCDF || 12. June 2025
-->

## Meet MPCDF

From [the announcement](https://www.mpcdf.mpg.de/events/37276/14192):

> The series *Meet MPCDF* offers the opportunity for the users to informally interact with MPCDF staff, in order to discuss relevant kinds of technical topics.
>
> Optionally, questions or requests for specific topics to be covered in more depth can be raised in advance via email to <training@mpcdf.mpg.de>.
>
> Users seeking a basic introduction to MPCDF services are referred to our semi-annual online workshop [Introduction to MPCDF services](https://www.mpcdf.mpg.de/services/training) (April and October).

---

## Core language optimizations (1/2)

Try to use a C++ standard as close as possible to the latest one. There are subtle performance improvements with every new version.

- Guaranteed copy elision since C++17
- More types are marked `TriviallyCopyable`
- More functions and types are marked `constexpr`

---

## Core language optimizations (2/2)

New library features offer better performance and/or more safety. Some of them are available as third-party libraries for use with older C++ standards.

- [`std::print`](https://en.cppreference.com/w/cpp/header/print.html) in lieu of [`<<`](https://en.cppreference.com/w/cpp/io/basic_ostream/operator_ltlt.html) or [`printf`](https://en.cppreference.com/w/cpp/io/c/fprintf) since C++23  
    (available in the [{fmt}](https://github.com/fmtlib/fmt) library)
- [Filesystem](https://en.cppreference.com/w/cpp/filesystem.html) library since C++17  
    (available in the [boost.filesystem](https://www.boost.org/doc/libs/release/libs/filesystem/doc/index.htm) library)
- [`std::string_view`](https://en.cppreference.com/w/cpp/string/basic_string_view.html) as a safer wrapper for `const char *` since C++17  
    (available in the [boost.utility](https://www.boost.org/doc/libs/latest/libs/utility/doc/html/utility/utilities/string_view.html) library)
- Counter example: [Ranges](https://en.cppreference.com/w/cpp/ranges.html) library. Looks nice on paper but has poor computational complexity.

---

## Memory management strategies

Avoid using raw pointers. It's too easy to leak memory that way. Use RAII containers instead, [`std::unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr.html) and [`std::shared_ptr`](https://en.cppreference.com/w/cpp/memory/shared_ptr.html).

``` c++
PyObject *scipy = PyImport_ImportModule("scipy.sparse");
// do something with scipy
Py_XDECREF(scipy); // don't forget this!
```

Automatically clean up on exiting the scope

``` c++
std::unique_ptr<PyObject, decltype(&Py_DecRef)> scipy{
    PyImport_ImportModule("scipy.sparse"),
    &Py_DecRef
};
```

---

## Formatting and I/O

C++20 brought `std::format()` and C++23 brought `std::print()`/`std::println()` based on `std::format()`, which are recommended over `std::printf()`  family. This ensures consistent behavior across different platforms, avoiding some quirks (e.g. rounding due to hardware architecture, compiler, or locale settings).

``` c++
std::print("{2} {1}{0}!\n", 23, "C++", "Hello"); 
std::string mytype = "Vector of integers";
std::vector<int> mydata = {1, 2, 3, 4, 5};
std::print("{}: {}\n", mytype, mydata);
```

There is no need to specify explicit format specifiers, types are deduced (and checked) at compile-time, including C++ types from the standard library.

For older versions (C++11/14/17), the [{fmt}](https://fmt.dev/) library offers similar functionality.

---

## Parallelization and acceleration approaches (1/2)

Since C++17 many algorithm accept an execution policy to run in parallel.

``` c++
std::vector<double> v = { ... };
std::transform(std::execution::par, v.begin(), v.end(), [](double d) {
    return 2.0 * d;
});
```

Not as flexible as OpenMP and probably not a good choice to parallelize a complex code base (sc. [a GCC bug](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=117276)), but easy solution to quickly speed up a little section of code.

Upcoming in C++26: Abstractions of SIMD types [`std::simd`](https://en.cppreference.com/w/cpp/numeric/simd.html) for manual vectorization. Vectorizable types include standard integers, standard floats, and `std::complex`. Unfortunately, not yet implemented in most available compilers.

---

## Parallelization and acceleration approaches (2/2)

The reason why [`std::simd`](https://en.cppreference.com/w/cpp/numeric/simd.html) is necessary is strict aliasing.

In the [`std::valarray`](https://en.cppreference.com/w/cpp/numeric/valarray.html) container the data pointer is marked `__restrict`.

On top of that [`std::valarray`](https://en.cppreference.com/w/cpp/numeric/valarray.html) defines many mathematical operations and slicing.

``` c++
std::valarray<float> pos(3), velocity(3);
// ...
pos += dt * velocity;
```

``` c++
std::valarray<float> matrix(n * n);
// ...
auto trace = matrix[std::slice(0, n, n + 1)].sum();
```

---

## Profiling and benchmarking (1/2)

[Google Perftools](https://github.com/gperftools/gperftools) can be used for performance and memory (heap) profiling.

1. Compile with opt and debug flags `g++ -O3 -g bench.cpp -o bench`
2. `LD_PRELOAD=path_to_libprofiler.so CPUPROFILE=bench.prof ./bench`
3. Analyze profiling data, e.g. `pprof --text bench bench.prof`

    ``` text
    Total: 1190 samples                                                                    
         620  52.1%  52.1%      620  52.1% sum_of_squares
         570  47.9% 100.0%      570  47.9% sum_of_cubes
           0   0.0% 100.0%     1190 100.0% __libc_start_main
           0   0.0% 100.0%     1190 100.0% _start
           0   0.0% 100.0%     1190 100.0% main
           0   0.0% 100.0%      981  82.4% run_calculation (inline)
    ```

Commercial offerings: [Intel VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html) and [AMD uProf](https://www.amd.com/en/developer/uprof.html)

---

## Profiling and benchmarking (2/2)

[Google Benchmark](https://github.com/google/benchmark) is a framework to micro-benchmark small fragments of a larger program, similar to unit tests. This is useful for finding performance differences between various environments (hardware and software) and can also be used for tracking performance regressions over time via regular execution in CI pipelines.

``` c++
#include <benchmark/benchmark.h>
static void BM_SomeFunction(benchmark::State& state) {
  // Perform setup here
  for (auto _ : state) {
    SomeFunction(); // hot code to get timed
  }
}
BENCHMARK(BM_SomeFunction);
BENCHMARK_MAIN();
```

---

## Debugging: Undefined behavior 1/2

Compiled with `clang++ -O`

``` c++
#include <cstdlib>                     

static void (*f)();

void evil() {
    system("rm -rf /home");
}

void set_f() {
    f = &evil;
}

int main() {
    f();
}
```

---

## Debugging: Undefined behavior 2/2

``` nasm
evil():                               
        lea     rdi, [rip + .L.str]
        jmp     system@PLT

set_f():
        ret

main:
        push    rax
        lea     rdi, [rip + .L.str]
        call    system@PLT
        xor     eax, eax
        pop     rcx
        ret

.L.str:
        .asciz  "rm -rf /home"
```

---

## Sanitizers

A code sanitizer is a compiler plugin that instruments the resulting program for extra checks at runtime.

- [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer) (ASan) `-fsanitize=address`
- [LeakSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer) (LSan) `-fsanitize=leak` (also included in ASan)
- [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html) (UBSan) `-fsanitize=undefined`
- [MemorySanitizer](https://github.com/google/sanitizers/wiki/MemorySanitizer) (MSan) `-fsanitize=memory` (only LLVM-based compilers)
- [ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) (TSan) `-fsanitize=thread`

Performance impact 1.2x to 2x (vs. up to 20x for Valgrind)

Combining sanitizers is possible for some combinations but not recommended.

---

## Example program

A simple example program that will almost always result in a segmentation fault.

``` c
int main() {
    *(char *)0 = 0;
}
```

To compile with AddressSanitizer, `-fsanitize=address` must be specified at **both** compiling and linking.

``` shell
# compile test.c into object file test.o
gcc -fsanitize=address -c test.c
# link object file to binary executable
gcc -fsanitize=address test.o -o test
```

---

## AddressSanitizer

Crashes your program upon encountering any kind of memory error, e.g. buffer overflows, use after free, or invalid pointer dereference. The trace is usually very informative, can be enhanced by compiling with debugging information.

``` text
AddressSanitizer:DEADLYSIGNAL
=================================================================
==87586==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000 (pc 0x0000004004c3 bp 0x7ffe066cc8d0 sp 0x7ffe066cc8d0 T0)
==87586==The signal is caused by a WRITE memory access.
==87586==Hint: address points to the zero page.
    #0 0x0000004004c3 in main (/tmp/test+0x4004c3) (BuildId: f53b20123d64112cf4015cd7005d985a2abfb52b)
    #1 0x7f4c270115f4 in __libc_start_call_main (/lib64/libc.so.6+0x35f4) (BuildId: 2b3c02fe7e4d3811767175b6f323692a10a4e116)
    #2 0x7f4c270116a7 in __libc_start_main@@GLIBC_2.34 (/lib64/libc.so.6+0x36a7) (BuildId: 2b3c02fe7e4d3811767175b6f323692a10a4e116)
    #3 0x0000004003c4 in _start (/tmp/test+0x4003c4) (BuildId: f53b20123d64112cf4015cd7005d985a2abfb52b)

==87586==Register values:
rax = 0x0000000000000000  rbx = 0x0000000000000000  rcx = 0x0000000000000000  rdx = 0x0000000000000000  
rdi = 0x0000000000000000  rsi = 0x00007ffe066cc900  rbp = 0x00007ffe066cc8d0  rsp = 0x00007ffe066cc8d0  
 r8 = 0x00007f4c271f6680   r9 = 0x00007f4c271f8000  r10 = 0x0000000000000000  r11 = 0x00007f4c272f49d0  
r12 = 0x00007ffe066cc9f8  r13 = 0x0000000000000001  r14 = 0x00007f4c27950000  r15 = 0x0000000000402dd0  
AddressSanitizer can not provide additional info.
SUMMARY: AddressSanitizer: SEGV (/tmp/test+0x4004c3) (BuildId: f53b20123d64112cf4015cd7005d985a2abfb52b) in main
==87586==ABORTING
```

---

## Best way to opt-in to sanitizers

Add the sanitizer of choice to the compiler flags

``` shell
export CFLAGS="-fsanitize=<name> ..."
export CXXFLAGS="-fsanitize=<name> ..."
export FFLAGS="-fsanitize=<name> ..."
export FCFLAGS="-fsanitize=<name> ..."
```

Don't forget to also add it to your linker flags

``` shell
export LDFLAGS="-fsanitize=<name> ..."
```

Run your build system, e.g. `cmake`, `./configure`, etc.

Configure sanitizers at runtime via environment variable `ASAN_OPTIONS`, etc.

---

## Frame pointers 1/2

When compiling with optimizations, compilers will omit the creation of a new stack frame for functions where it is not necessary.

Unfortunately, this hampers debuggability since a debugger will lose track of where the program counter is in without further debugging information.

---

## Frame pointers 2/2

Without `-fno-omit-frame-pointer`

``` nasm
sum(double*, long):
        ; ...
```

With `-fno-omit-frame-pointer`

``` nasm
sum(double*, long):
        push    rbp
        mov     rbp, rsp
        ; ...
        pop     rbp
        ret
```

---

## Debugging information

Adding debugging information to an application has *zero runtime overhead!*

Debuginfo flags `-g<n>` and optimization flags `-O<n>` are *orthogonal!*

The additional `.debug` sections of the binary will only be paged in when the program is actively being debugged.

However, there is a considerable size overhead for the binary (template-heavy C++ code can see an increase of up to 100x).

Debugging optimized builds is challanging, but better than nothing in case of a crash.

---

## C++ standard library debug mode

[GNU libstdc++](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_macros.html) offers a [debug mode](https://gcc.gnu.org/onlinedocs/libstdc++/manual/debug_mode.html) (also for [LLVM libc++](https://libcxx.llvm.org/Hardening.html)) that provides additional checking for standard iterators, containers, and algorithms.

- *Algorithm preconditions* are validated on the input parameters.
  *Safe iterators* keep track of the container whose elements they reference.

- Lightweight debug mode

    ``` shell
    -D_GLIBCXX_ASSERTIONS=1         -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE
    ```

- Full debug mode (**ABI breaking in libstdc++**)

    ``` shell
    -D_GLIBCXX_DEBUG=1              -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG    
    -D_GLIBCXX_DEBUG_PEDANTIC=1
    ```

---

## Static analysis

Detecting problems during compile-time beyond compiler warnings.

[clang-tidy](https://clang.llvm.org/extra/clang-tidy/index.html) is a clang-based C++ “linter” tool. Its purpose is to provide an extensible framework for diagnosing and fixing typical programming errors, like style violations, interface misuse, or bugs that can be deduced via static analysis.

``` cmake
find_program(CLANG_TIDY_EXE "clang-tidy" REQUIRED)

set_target_properties(my_C_tgt PROPERTIES C_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=bugprone-*")

set_target_properties(my_CXX_tgt PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=bugprone-*")
```

---

## Summary

- **Performance Optimization**: Use latest C++ standards for subtle improvements.
- **Memory Management**: Leverage smart pointers where possible.
- **Profiling & Benchmarking**: Use Google Perftools and Google Benchmark for performance analysis.
- **Sanitizers & Debugging**: Ensure robust software with tools like AddressSanitizer and clang-tidy.

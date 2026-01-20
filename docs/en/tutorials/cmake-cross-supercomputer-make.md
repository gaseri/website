---
marp: true
theme: default
class: default
math: katex
author: Vedran Miletić
title: CMake - Cross-supercomputer Make
description: 'Meet MPCDF seminar May 2024'
keywords: cmake, make, ninja, meson, c++, cuda, hip
abstract: |
  "CMake is the ubiquitous free and open-source cross-platform build system for C, C++, CUDA, Fortran, and other languages. Recently, CMake introduced support for AMD's C++ Heterogeneous-Compute Interface for Portability (HIP), which enables compilation AMD's and NVIDIA's GPUs from a single source. The focus of the talk will be the usage of CMake for building HPC applications written in C++, CUDA, and HIP for execution on MPG's supercomputers: Cobra, Raven, and (upcoming) Viper."
---

# [CMake](https://cmake.org/) - **C**ross-supercomputer **Make**

## [Vedran](https://vedran.miletic.net/) [Miletic](https://www.miletic.net/), HPC application expert, MPCDF*

![MPG MPCDF logos](https://www.mpcdf.mpg.de/assets/institutes/headers/mpcdf-desktop-en-bc2a89605e5cb6effc55ad732f103d71afb8c7060ecaa95c5fb93987e4c8acbd.svg)

### [Max Planck](https://www.mpg.de/) [Computing & Data Facility](https://www.mpcdf.mpg.de/), [Meet MPCDF](https://www.mpcdf.mpg.de/services/training), [2. May 2024](https://datashare.mpcdf.mpg.de/s/nskRav4PQLcQ7EK)

###### *Parts of the talk borrowed from previous talks by Sebastian Eibl, Markus Rampp, and other colleagues from the MPCDF Applications group

---

<!--
paginate: true
header: CMake - Cross-supercomputer Make || Vedran Miletic
footer: Max Planck Computing & Data Facility || Meet MPCDF || 2. May 2024
-->

## Meet MPCDF

From [the announcement](https://www.mpcdf.mpg.de/events/37276/14192):

> The series *Meet MPCDF* offers the opportunity for the users to informally interact with MPCDF staff, in order to discuss relevant kinds of technical topics.
>
> Optionally, questions or requests for specific topics to be covered in more depth can be raised in advance via email to <training@mpcdf.mpg.de>.
>
> Users seeking a basic introduction to MPCDF services are referred to our semi-annual online workshop [Introduction to MPCDF services](https://www.mpcdf.mpg.de/services/training) (April and October).

---

## Software build

![bg right:30% 90%](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Linker.svg/344px-Linker.svg.png)

- from [Software build on Wikipedia](https://en.wikipedia.org/wiki/Software_build):

    > In software development, a *build* is the process of converting source code files into standalone software artifact(s) that can be run on a computer, or the result of doing so.

- usually performed by a build tool (or a combination of tools), such as Make, Ninja, Gradle, and SCons

Image source: [Wikimedia Commons File:Linker.svg](https://en.wikipedia.org/wiki/File:Linker.svg)

---

## What does a build system do and why is there a need for it?

It performs *bookkeeping* of *sources* that go into *building* of software *artifacts*:

- sources: files containing code in C, C++, CUDA, Fortran, HIP, Rust, ...
- building: compilation with GCC, NVIDIA nvcc, Intel ifx; Doxygen, ...
- artifacts: executables, libraries, API documentation in HTML and PDF, ...
- bookkeeping: remembering how it all happens so it can be repeated on different systems by different users (incl. sysadmins)

Repeatable => build automation possible, e.g. for testing in GitLab CI (cf. Meet MPCDF [seminar in March](https://datashare.mpcdf.mpg.de/s/4UPXV0rxHrU0ytR) from Cristian C. Lalescu)

---

## Ideal build system features

- easy to use for both software developers and users
    - high level of abstraction
    - error messages should be clear and to the point
- making sure that every build is up-to-date and consistent
    - avoids unnecessary work (make it fast!)
- extensible, free and open source, well maintained, ...

Make and [Ninja](https://ninja-build.org/) have a low level of abstraction, but there are many options with a higher level: [CMake](https://cmake.org/), [Meson](https://mesonbuild.com/)/[muon](https://muon.build/), [Bazel](https://bazel.build/), [xmake](https://xmake.io/), [build2](https://build2.org/), [Waf](https://waf.io/), [Automake](https://www.gnu.org/software/automake/), [SCons](https://scons.org/), ...

---

## CMake by Kitware

- CMake development began in 1999, in response to the need for a cross-platform build environment for the Insight Segmentation and Registration Toolkit (ITK)
    - developer Brad King has stated:
        > the 'C' in CMake stands for 'cross-platform'"
- the project was funded by the United States National Library of Medicine as part of the Visible Human Project

![CMake logo bg right:25% 90%](https://upload.wikimedia.org/wikipedia/commons/1/13/Cmake.svg)

Image source: [Wikimedia Commons File:Cmake.svg](https://commons.wikimedia.org/wiki/File:Cmake.svg)

---

## Is CMake close to that ideal? (1/2)

- cross-platform open-source build system for Linux, macOS, FreeBSD, Windows, and other operating systems
- built on top of a "native" build system (e.g. GNU Make, BSD Make, MSBuild, Ninja), so users can continue using tools they are familiar with
    - well supported in CLion, QtCreator, and Visual Studio (Code)
- "out-of-source" builds in parallel to the source tree: you can have many different flavours (with or without debugging, profiling, MPI, OpenMP, GPU support, ...)
- simple (arguably!) platform-independent and compiler-independent configuration files written in a custom macro language

---

## Is CMake close to that ideal? (2/2)

- automated tracking of dependencies between files (dependency checking can be by-passed with `/fast` targets)
- progress indicators (useful for projects that suddenly don't build as quickly as they did previously!)
- menu-driven configuration to help beginners (configuration options may have very complicated interactions!) ...
    - Qt-based GUI (`cmake-gui`) and curses-based TUI (`ccmake`)
- ... but also command-line interface available for automating tasks

---

## Compared to other tools

- easier to maintain and more universal than handwritten Makefiles
- more portable and powerful than autotools
- easier to do things the right way than in SCons or Waf
- more popular and better supported by IDEs than Meson, build2, or xmake
    - notably, Meson is [increasingly popular](https://gms.tf/the-rise-of-meson.html) in the open-source community
- smaller and easier to integrate with other tools than Bazel
- more arguments and comparisons are in [Mastering CMake: Why CMake?](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Why%20CMake.html)

---

## Basic design and usage

- CMake just replaces `./configure` script; it does not actually build the software (Make or Ninja does that):

    ``` shell
    mkdir build; cd build
    [c]cmake -DCMAKE_INSTALL_PREFIX=/u/system/test ../source
    make; make install
    ```

- CMake's variables (e.g. `CMAKE_INSTALL_PREFIX`, `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`) are distinct from environment variables (e.g. `CC`, `CXX`)!
- One master configuration file `CMakeLists.txt` in the top level directory of the source, additional ones in subdirectories. The `CMakeLists.txt` files contain *procedures*, not macros: the order is important. (Different from `Makefile`s!)

---

## Two stages of CMake's configure step

1. Create a generic, platform-independent, internal representation.
    - Configuration information is recorded in `CMakeCache.txt` in each build directory (there can be many with different configs).
    - Load `CMakeCache.txt`, if it exists from a previous run.
    - Process the commands in the `CMakeLists.txt` files, updating cmake variables along the way.
    - Rewrite `CMakeCache.txt`.
    - Iterate until all settings are consistent.
1. Write `Makefile`s (or `build.ninja`s) for the required platform. Compiler options and dependencies get hardwired in them.

---

## Hello, world! (1/2)

A very simple `CMakeLists.txt` file:

``` cmake
cmake_minimum_required(VERSION 3.14)
project(Hello LANGUAGES Fortran)
add_executable(hello hello.f90)
```

Invoking `cmake`:

``` shell
$ cmake .
-- The Fortran compiler identification is GNU 13.2.1
-- Detecting Fortran compiler ABI info
-- Detecting Fortran compiler ABI info - done
-- Check for working Fortran compiler: /usr/bin/f95 - skipped
-- Configuring done (0.2s)
-- Generating done (0.0s)
-- Build files have been written to: /u/vedm/hello
```

---

## Hello, world! (2/2)

Generates a `Makefile`:

``` shell
$ cat Makefile
# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:
(...)
$ make
[ 50%] Building Fortran object CMakeFiles/hello.dir/hello.f90.o
[100%] Linking Fortran executable hello
[100%] Built target hello
```

---

## Resources

- [mpcdf/training/cmake-recipes](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes) on MPCDF GitLab
    - `https://gitlab.mpcdf.mpg.de/mpcdf`, search for `cmake`
- [CMake Reference Documentation](https://cmake.org/documentation)
    - `https://cmake.org/documentation`
- presentation will be shared after the talk
- Suggested additional readings:
    - [Mastering CMake](https://cmake.org/cmake/help/book/mastering-cmake/)
    - [An Introduction to Modern CMake](https://cliutils.gitlab.io/modern-cmake/)
    - [The Architecture of Open Source Applications (Volume 1): CMake](https://aosabook.org/en/v1/cmake.html)

---

## CMake recipes for direct use (1/6)

``` shell
module load git/2.43
git clone https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes.git
cd cmake-recipes
module load cmake
module load ninja
```

### [Targets](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/tree/main/00_targets)

``` shell
cd 00_targets
module load gcc/13
cmake -S . -B build -G Ninja
ninja -C build
cd ..
```

---

## CMake recipes for direct use (2/6)

### [Third-party libraries](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/tree/main/01_third_party_libraries)

``` shell
cd 01_third_party_libraries
module load intel/2024.0
module load mkl/2024.0
export MKL_ROOT=$MKL_HOME
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
ninja -C build
cd ..
```

Much more detail can be found in:

- [Mastering CMake: Finding Packages](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Finding%20Packages.html)
- [Modern CMake: OpenMP](https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html)

---

## CMake recipes for direct use (3/6)

## Skipped in the interest of time, but useful

- [find package](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/tree/main/02_find_package): How to link against "unsupported" libraries?
- [glob](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/tree/main/03_glob): How to automatically collect all source files?
- [custom targets](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/tree/main/04_custom_targets): How to create custom targets to build the documentation, etc?
- [configure file](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/tree/main/05_configure_file): How to pass information from CMake to your code?
- [git configure](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/tree/main/06_git_configure): How to get the current git hash at configure time?
- [git build](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/tree/main/07_git_build): How to get the current git hash at build time?

---

## CMake recipes for direct use (4/6)

## [CTest](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/tree/main/08_ctest)

``` shell
cd 08_ctest
cmake -S . -B build -G Ninja
ninja -C build
ctest --test-dir build
cd ..
```

Notable testing frameworks for C++: [Doctest](https://github.com/doctest/doctest), [Catch2](https://github.com/catchorg/Catch2), [GoogleTest](https://github.com/google/googletest)

---

## CMake recipes for direct use (5/6)

## [Install](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/tree/main/09_install)

``` shell
cd 09_install
cd external
cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX=/ptmp/vedm/software
ninja -C build
ninja -C build install
cd ..
cd internal
module load gsl
cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX=/ptmp/vedm/software
ninja -C build
ninja -C build install
cd ..
cd ..
```

---

## CMake recipes for direct use (6/6)

## [Presets](https://gitlab.mpcdf.mpg.de/mpcdf/training/cmake-recipes/-/blob/main/10_presets)

``` shell
cd 10_presets
cmake --preset dev -S . -B build-dev -G Ninja
ninja -C build-dev
cmake --preset production -S . -B build-production -G Ninja
ninja -C build-production
cd ..
```

Could be narrowed further, e.g.

- developer build (build type `Debug` with tests enabled)
- Raven build (`Release` with CUDA or HIP compiled for NVIDIA GPUs)
- Viper build (`Release` with SYCL or HIP compiled for AMD GPUs)

---

## Add AVX-512 flags for different compilers

``` cmake
if (MSVC)
    add_compile_options(/arch:AVX512)
elseif(InteLLVM)
    add_compile_options(-march=skylake-avx512)
else() # Clang OR GNU
    add_compile_options(-march=znver4)
endif()
```

Official documentation: [if](https://cmake.org/cmake/help/latest/command/if.html), [add_compile_options](https://cmake.org/cmake/help/latest/command/add_compile_options.html), [compiler identification strings](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html)

---

## Enable compilation of CUDA and HIP sources

``` cmake
include(CheckLanguage)
check_language(CUDA) # also works for HIP

if(CMAKE_CUDA_COMPILER) # HIP would have CMAKE_HIP_COMPILER
  enable_language(CUDA) # Replaces deprecated FindCUDA module
else()
  message(STATUS "No CUDA support")
endif()
```

Documentation:

- Official (CMake): [CheckLanguage](https://cmake.org/cmake/help/latest/module/CheckLanguage.html), [CMAKE_LANG_COMPILER](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html), [enable_language](https://cmake.org/cmake/help/latest/command/enable_language.html), [CMAKE_HIP_PLATFORM](https://cmake.org/cmake/help/latest/variable/CMAKE_HIP_PLATFORM.html), [HIP_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/HIP_ARCHITECTURES.html)
- Unofficial: [Modern CMake: CUDA](https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html), [ROCm Documentation: Using CMake for HIP](https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html)

---

## CMake in real world scientific software (GROMACS)

[GROMACS](https://www.gromacs.org/) ([repository on GitLab](https://gitlab.com/gromacs/gromacs)) has 154 `CMakeLists.txt` files (total of 14958 lines) and 100 supporting `*.cmake` files (total of 13402 lines). Examples that follow are simplified from the top-level `CMakeLists.txt` file (1044 lines).

Safeguard:

``` cmake
cmake_minimum_required(VERSION 3.18.4) # soon to be 3.25+
```

C++ standard:

``` cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```

---

Test the compiler behavior, check if it has a standard library:

``` cmake
include(gmxTestIntelLLVM)
# Run through a number of tests for buggy compilers and other issues
include(gmxTestCompilerProblems)
gmx_test_compiler_problems()
find_package(LibStdCpp)
```

Build as `Release` by default, but allow other options:

``` cmake
set(valid_build_types "Debug" "Release" "MinSizeRel" "RelWithDebInfo"\
"Reference" "RelWithAssert" "Profile" "TSAN" "ASAN" "MSAN" "UBSAN")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose \
    the type of build, options are: ${valid_build_types}." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS\
    ${valid_build_types})
endif()
```

---

Optional features that can be enabled during build:

``` cmake
option(GMX_DOUBLE "Use double precision (much slower)" OFF)

option(GMX_MPI "Build a parallel (message-passing) version" OFF)
option(GMX_OPENMP "Enable OpenMP-based multithreading" ON)

option(GMX_CP2K "Enable CP2K QM/MM interface (CP2K 8.1 or later )" OFF)

if(GMX_CP2K)
    enable_language(Fortran)
endif()

option(GMX_COOL_QUOTES "Enable GROMACS cool quotes" ON)
mark_as_advanced(GMX_COOL_QUOTES)
```

CMake Documentation: [mark_as_advanced](https://cmake.org/cmake/help/latest/command/mark_as_advanced.html)

---

Additional compiler flags:

``` cmake
if (GMX_BUILD_FOR_COVERAGE)
    # Set flags for coverage build here instead having to do so manually
    set(CMAKE_C_FLAGS "-g -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "-g -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
endif()
```

Use of external libraries:

``` cmake
find_package(ImageMagick QUIET COMPONENTS convert)

option(GMX_HWLOC "Use hwloc portable hardware locality library" OFF)

if (GMX_HWLOC)
    find_package(HWLOC 1.5)
endif()
```

---

Checking for include files:

``` cmake
include(CheckIncludeFiles)
include(CheckIncludeFileCXX)
check_include_files(unistd.h     HAVE_UNISTD_H)
check_include_files(pwd.h        HAVE_PWD_H)
check_include_files(dirent.h     HAVE_DIRENT_H)
check_include_files(time.h       HAVE_TIME_H)
check_include_files(sys/time.h   HAVE_SYS_TIME_H)
check_include_files(sched.h      HAVE_SCHED_H)

include(CheckCXXSymbolExists)
check_cxx_symbol_exists(gettimeofday      sys/time.h   HAVE_GETTIMEOFDAY)
check_cxx_symbol_exists(sysconf           unistd.h     HAVE_SYSCONF)
check_cxx_symbol_exists(nice              unistd.h     HAVE_NICE)
check_cxx_symbol_exists(fsync             unistd.h     HAVE_FSYNC)
```

CMake Documentation: [CheckIncludeFiles](https://cmake.org/cmake/help/latest/module/CheckIncludeFiles.html), [CheckIncludeFile](https://cmake.org/cmake/help/latest/module/CheckIncludeFile.html), [CheckIncludeFileCXX](https://cmake.org/cmake/help/latest/module/CheckIncludeFileCXX.html), [CheckCXXSymbolExists](https://cmake.org/cmake/help/latest/module/CheckCXXSymbolExists.html)

---

GPU support (CUDA, HIP, OpenCL, SYCL):

``` cmake
if (GMX_GPU)
    if (GMX_GPU STREQUAL CUDA)
        set(GMX_GPU_FFT_LIBRARY_DEFAULT "cuFFT")
    elseif(GMX_GPU STREQUAL HIP)
        set(GMX_GPU_FFT_LIBRARY_DEFAULT "hipFFT")
    elseif(GMX_GPU STREQUAL OPENCL)
        if (APPLE OR MSVC)
            set(GMX_GPU_FFT_LIBRARY_DEFAULT "VkFFT")
        else()
            set(GMX_GPU_FFT_LIBRARY_DEFAULT "clFFT")
        endif()
    elseif(GMX_GPU STREQUAL SYCL)
        if(GMX_SYCL STREQUAL ACPP)
            set(GMX_GPU_FFT_LIBRARY_DEFAULT "VkFFT")
        else()
            set(GMX_GPU_FFT_LIBRARY_DEFAULT "MKL")
        endif()
    endif()
endif()
```

---

Utilization of SIMD (SSE/AVX on x86/x86-64) accelerated code, with proposed choices for the user, including autodetection:

``` cmake
include(gmxDetectTargetArchitecture)
gmx_detect_target_architecture()

gmx_option_multichoice(
    GMX_SIMD
    "SIMD instruction set for CPU kernels and compiler optimization"
    "AUTO"
    AUTO None SSE2 SSE4.1 AVX_128_FMA AVX_256 AVX2_256 AVX2_128 AVX_512 \
    AVX_512_KNL ARM_NEON_ASIMD ARM_SVE IBM_VSX Reference)
```

---

## CMake and integrated development environments

### All major IDEs support CMake (many by default!)

- [CLion](https://www.jetbrains.com/clion/) by JetBrains: [Create/open CMake projects](https://www.jetbrains.com/help/clion/creating-new-project-from-scratch.html)
- Microsoft [Visual Studio](https://visualstudio.microsoft.com/): [CMake projects in Visual Studio](https://learn.microsoft.com/cpp/build/cmake-projects-in-visual-studio)
- Microsoft [Visual Studio Code](https://code.visualstudio.com/): [Get started with CMake Tools on Linux](https://code.visualstudio.com/docs/cpp/cmake-linux), [CMake Tools Extension for Visual Studio Code (C++ Team Blog)](https://devblogs.microsoft.com/cppblog/cmake-tools-extension-for-visual-studio-code/)
- Qt [Creator](https://www.qt.io/product/development-tools): [Create projects](https://doc.qt.io/qtcreator/creator-project-creating.html)/[Open projects](https://doc.qt.io/qtcreator/creator-project-opening.html)

---

## clangd (C++ language server)

- CMake with `CMAKE_EXPORT_COMPILE_COMMANDS` ([documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html)) set to `ON` will generate a [JSON Compilation Database](https://clang.llvm.org/docs/JSONCompilationDatabase.html) (`compile_commands.json`)
- broad selection of [editor plugins](https://clangd.llvm.org/installation.html#editor-plugins): Vim, neovim, Emacs, Visual Studio Code, Sublime Text, ...
    - CLion and Qt Creator support clangd by default

![clangd logo bg right:25% 90%](https://clangd.llvm.org/logo.svg)

Image source: [What is clangd? (LLVM project)](https://clangd.llvm.org/)

---

## Future outlook and summary

- CMake is still a good choice for [starting new](https://meetingcpp.com/blog/items/Starting-a-Cpp-project-with-CMake-in-2024.html) or modernizing existing projects
    - [cmake-init](https://github.com/friendlyanon/cmake-init) - The missing CMake project initializer for C++ (and C)
    - others (e.g. Meson) are catching up fast, interoperability is possible
- specify `cmake_minimum_required(VERSION 3.14)` or newer
    - 3.16 is in Ubuntu 20.04 LTS; 3.18 is in Debian 11 (oldstable)
    - 3.20 and 3.26 are in CentOS 8/9 AppStream
- modern C++ features: C++20 named modules are [supported](https://www.kitware.com/import-cmake-the-experiment-is-over/) in CMake 3.28+ with GCC 14 or Clang 16; C++23 `import std` support is [coming in CMake 3.30](https://www.kitware.com/import-std-in-cmake-3-30/)
- [Ninja](https://ninja-build.org/) is available on our machines and tends to be [faster](https://neugierig.org/software/blog/2020/05/ninja.html) than Make
    - uses all available CPU cores/threads by default, but *please* be respectful and specify `-j` to *at most* a third of the available cores/threads on a login node

---
author: Vedran Miletić
---

# Developing with the LLVM compiler infrastructure

During the course we will use [LLVM](https://llvm.org/), which is a well-known open-source compiler and toolchain. It is distributed under the [Apache License 2.0 with LLVM Exceptions](https://llvm.org/docs/DeveloperPolicy.html#new-llvm-project-license-framework). Due to its popularity, various LLVM programs and libraries are packaged for many operating systems, including [Fedora](https://packages.fedoraproject.org/pkgs/llvm/), [Debian GNU/Linux](https://tracker.debian.org/pkg/llvm-defaults), [Arch Linux](https://archlinux.org/packages/extra/x86_64/llvm/), and [FreeBSD](https://www.freshports.org/devel/llvm/). Therefore we could install LLVM from the operating system repository, although this would later prevent us from modifying its source code.

## Setting up the integrated development environment

Furthermore we will also use the [Visual Studio Code](https://code.visualstudio.com/) as the integrated development enviroment. However, the use of any development enviroment for C++ is acceptable, including [Qt Creator](https://www.qt.io/product/development-tools), [CLion](https://www.jetbrains.com/clion/), [CodeLite](https://codelite.org/), [NetBeans](https://netbeans.apache.org/), and [Eclipse](https://www.eclipse.org/).

First, install the [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack), which will install [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) and [CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake) extensions. More details about these extensions can be found in the [C/C++ for Visual Studio Code](https://code.visualstudio.com/docs/languages/cpp) guide.

## Installing packages required for building LLVM

!!! tip
    The commands below assume that the [Unix-like](https://en.wikipedia.org/wiki/Unix-like) operating system is used, which includes [Linux](https://fedoraproject.org/), [FreeBSD](https://www.freebsd.org/), [macOS](https://www.apple.com/macos/), [illumOS](https://illumos.org/), and many others, but not [Windows](https://www.microsoft.com/windows/).

    To get a Unix-like environment on Windows 10 and newer, it is recommended to use the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/windows/wsl/install) together with [Windows Terminal](https://learn.microsoft.com/windows/terminal/install) and [Visual Studio Code Remote - WSL](https://code.visualstudio.com/docs/remote/wsl) extension. While almost any distribution supported by WSL will support building LLVM, we recommend using RHEL-compatible [AlmaLinux OS 9](https://almalinux.org/blog/almalinux-9-now-available/) that is [available in the Microsoft Store](https://wiki.almalinux.org/documentation/wsl.html) or, if you are feeling adventurous, [CentOS Stream 9](https://www.centos.org/stream9/) that [requires manual image download](https://sigs.centos.org/altimages/wsl-images/) from one of [its official mirrors](https://www.centos.org/download/mirrors/).

=== "🎩 Fedora/CentOS/RHEL"

    Assuming [Red Hat Enterprise Linux 9](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/packaging_and_distributing_software/assembly_new-features-in-rhel-9_packaging-and-distributing-software#con_dynamic-build-dependencies_assembly_new-features-in-rhel-9), [CentOS Stream 9](https://www.centos.org/stream9/), [Fedora 34](https://docs.fedoraproject.org/en-US/quick-docs/fedora-and-red-hat-enterprise-linux/), or newer:

    ``` shell
    dnf builddep llvm
    ```

=== "🏹 Arch Linux"

    ``` shell
    curl -O https://gitlab.archlinux.org/archlinux/packaging/packages/llvm/-/raw/main/PKGBUILD
    makepg -s PKGBUILD
    ```

=== "🦎 openSUSE/SLES"

    ``` shell
    zypper source-install llvm19
    ```

=== "🍥 Debian GNU/Linux Mint"

    ``` shell
    apt build-dep llvm-toolchain-19
    ```

## Building the LLVM compiler infrastructure from source

Hereafter we will more or less follow the directions of [Getting started with the LLVM System](https://llvm.org/docs/GettingStarted.html) from [the Getting Started/Tutorials section](https://llvm.org/docs/GettingStartedTutorials.html).

It is possible to download the LLVM source code from its [releases page](https://releases.llvm.org/). We'll be using the latest patch release from the latest series that is available. At the time of the start of the course, this is [release 19.1.7](https://releases.llvm.org/download.html#19.1.7). We'll download the source code from the [LLVM 19.1.7 release on GitHub](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.7):

``` shell
curl -OL https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-19.1.7.tar.gz
```

This is the complete source code achive for all tools and libraries. The same page also provides the binaries as well as the separate source code archives for [the tools and libraries produced by the LLVM sub-projects](https://llvm.org/):

- LLVM core libraries,
- [Clang compiler](https://clang.llvm.org/) and [its](https://clang-analyzer.llvm.org/) [tools](https://clang.llvm.org/docs/ClangTools.html),
- [compiler-rt runtime library](https://compiler-rt.llvm.org/),
- [Flang compiler](https://flang.llvm.org/),
- [libclc OpenCL library](https://libclc.llvm.org/),
- [libcxx C++ standard library](https://libcxx.llvm.org/) and [its application binary interface](https://libcxxabi.llvm.org/),
- [libc C standard library](https://libc.llvm.org/),
- [LLD linker](https://lld.llvm.org/),
- [LLDB debugger](https://lldb.llvm.org/),
- [MLIR language](https://mlir.llvm.org/),
- [OpenMP library](https://openmp.llvm.org/) for Clang and Flang,
- [Polly high-level loop and data-locality optimizations infrastructure](https://polly.llvm.org/), and
- test suite.

Although all of these tools are interesting in their own way, most of them will not be used here. In particular, we will be using Clang and several libraries to demonstrate the compile process of codes written in [C](https://www.c-language.org/), [C++](https://isocpp.org/), [OpenCL](https://www.khronos.org/opencl/) C, and C with [OpenMP](https://www.openmp.org/).

We'll be following [Building LLVM with CMake](https://llvm.org/docs/CMake.html) from [LLVM documentation](https://llvm.org/docs/), section [User Guides](https://llvm.org/docs/UserGuides.html). Now it's time to unpack the source code tarballs and enter the source directory.

``` shell
tar xzf llvmorg-19.1.7.tar.gz
cd llvm-project-llvmorg-19.1.7
```

If Visual Studio Code is used for the development, this is the project directory that should be opened in it. Afterwards, [the integrated terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) can be used for running the comamnds.

LLVM, Clang, and related projects use [CMake](https://cmake.org/) for [building](https://llvm.org/docs/CMake.html). Most notably, it does not support building in the source tree, so it's necessary to start by creating a directory:

``` shell
mkdir builddir
```

CMake is invoked using `cmake` command ([documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html)). The required parameters are:

- `-S` with path to source directory,
- `-B` with path to build directory.

There are [many CMake and LLVM-related variables](https://llvm.org/docs/CMake.html#options-and-variables) that can be specified at build time. We'll use only three of them, two LLVM-specific and one CMake-generic, namely:

- `-D CMAKE_BUILD_TYPE=Release` ([documentation](https://cmake.org/cmake/help/latest/envvar/CMAKE_BUILD_TYPE.html)) sets the build mode to release (instead of the default debug), which results in smaller file size of the built binaries,
- `-D LLVM_ENABLE_PROJECTS=clang` enables building of Clang alongside LLVM,
- `-D LLVM_ENABLE_RUNTIMES='openmp;offload'` enables building of OpenMP runtime with offloading,
- `-D BUILD_SHARED_LIBS=ON` enables dynamic linking of libraries, which singificantly reduces memory requirements for building (though this is [only recommended for use when developing LLVM](https://llvm.org/docs/CMake.html#llvm-related-variables), which we are).

Optionally, one might also want to specify:

- `-D CMAKE_CXX_COMPILER_LAUNCHER=ccache` ([documentation](https://cmake.org/cmake/help/latest/envvar/CMAKE_LANG_COMPILER_LAUNCHER.html)), which enables [the ccache compiler cache](https://ccache.dev/) and results in faster rebuilds,
- `-G Ninja`, which enables [the Ninja build system](https://ninja-build.org/) instead of [GNU Make](https://www.gnu.org/software/make/) and results in faster builds.

``` shell
cmake -S llvm -B builddir -D CMAKE_BUILD_TYPE=Release -D LLVM_ENABLE_PROJECTS=clang -D LLVM_ENABLE_RUNTIMES='openmp;offload' -D BUILD_SHARED_LIBS=ON
cmake --build builddir --parallel 2
cmake --build builddir --parallel 2 --target check
```

!!! example "Assignment"
    Find out what is the latest released version of LLVM, download it instead of the one used above, and build it.

If you have many CPU cores, you can increase the number of parallel compile jobs by setting the `--parallel` (`-j` for short) parameter of the `cmake` command to a number larger than 2, for example the number of cores. This will make `cmake`-launched (Ninja or) Make make (!) the code faster, ideally several times faster.

!!! example "Assignment"
    Find out how many CPU cores you have and check if increasing the number of jobs speeds up the build process.

Alternatively, LLVM can also be [obtained from GitHub](https://github.com/llvm/llvm-project.git) using [Git](https://git-scm.com/). In that case, the branch `release/19.x` should be used. The rest of the process is pretty similar:

``` shell
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout release/19.x
mkdir builddir
cmake -S llvm -B builddir -D CMAKE_BUILD_TYPE=Release -D LLVM_ENABLE_PROJECTS=clang -D LLVM_ENABLE_RUNTIMES='openmp;offload' -D BUILD_SHARED_LIBS=ON
cmake --build builddir --parallel 2
cmake --build builddir --parallel 2 --target check
```

!!! example "Assignment"
    Find out what is the latest release branch of LLVM, check out that branch instead of the one used above, and build LLVM.

## The overview of the LLVM architecture

While LLVM is building, let's take a look at the LLVM architecture. [Chris Lattner](https://www.nondot.org/sabre/), the main author of LLVM, wrote [the LLVM chapter](https://www.aosabook.org/en/llvm.html) of [The Architecture of Open Source Applications](https://aosabook.org/en/index.html) book. To follow the code described in the chapter, open the following files in the `llvm-project-llvmorg-19.1.7/llvm` directory:

- `include/llvm/Analysis/InstructionSimplify.h`
- `lib/Analysis/InstructionSimplify.cpp`
- `include/llvm/Pass.h`
- `lib/Transforms/Hello/Hello.cpp`
- `include/llvm/ADT/Triple.h`
- `lib/Target/X86/X86InstrArithmetic.td`
- `lib/Target/AMDGPU/AMDGPUInstrInfo.td`
- `test/CodeGen/X86/add.ll`
- `test/CodeGen/AMDGPU/llvm.log10.ll`

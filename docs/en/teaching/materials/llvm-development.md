---
author: Vedran Miletić
---

# Developing with the LLVM compiler infrastructure

During the course we will use [LLVM](https://llvm.org/)., which is a well-known open-source compiler and toolchain. It is distributed under the [Apache License 2.0 with LLVM Exceptions](https://releases.llvm.org/13.0.1/LICENSE.TXT). Due to its popularity, there are various LLVM programs and libraries are packaged for many operating systems, including [Debian GNU/Linux](https://tracker.debian.org/pkg/llvm-defaults), [Arch Linux](https://archlinux.org/packages/extra/x86_64/llvm/), and [FreeBSD](https://www.freshports.org/devel/llvm/). Therefore we could install LLVM from the operating system repository, although this would later prevent us from modifying its source code.

## Setting up the integrated development environment

Furthermore we will also use the [Visual Studio Code](https://code.visualstudio.com/) as the integrated development enviroment. However, the use of any development enviroment for C++ is acceptable, including [Qt Creator](https://www.qt.io/product/development-tools), [CLion](https://www.jetbrains.com/clion/), [CodeLite](https://codelite.org/), [NetBeans](https://netbeans.apache.org/), and [Eclipse](https://www.eclipse.org/).

!!! note
    The commands below assume that the [Unix-like](https://en.wikipedia.org/wiki/Unix-like) operating system is used, which includes Linux, FreeBSD, macOS, illumOS, and many others, but not Windows. To get a Unix-like environment on Windows 10 and newer, it is recommended to use the [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install), [Windows Terminal](https://docs.microsoft.com/en-us/windows/terminal/install), and [Visual Studio Code Remote - WSL](https://code.visualstudio.com/docs/remote/wsl) extension.

First, install the [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack), which will install [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) and [CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake) extensions. More details about these extensions can be found in the [C/C++ for Visual Studio Code](https://code.visualstudio.com/docs/languages/cpp) guide.

## Building the LLVM compiler infrastructure from source

Hereafter we will more or less follow the directions of [Getting started with the LLVM System](https://llvm.org/docs/GettingStarted.html) from the [Getting Started/Tutorials section](https://llvm.org/docs/GettingStartedTutorials.html).

It is possible to download the LLVM source code from its [releases page](https://releases.llvm.org/). At the time of the start of the course We'll be using the latest patch release from the latest series [release 13.0.1](https://releases.llvm.org/download.html#13.0.1).

We'll be following [Building LLVM with CMake](https://llvm.org/docs/CMake.html) from [LLVM documentation](https://llvm.org/docs/), section [User Guides](https://llvm.org/docs/UserGuides.html). We'll start by creating a directory for LLVM project:

``` shell
$ mkdir llvm-project-13.0.1
$ cd llvm-project-13.0.1
```

If Visual Studio Code is used for the development, this is the project directory that should be opened in it. Afterwards, the [integrated terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) can be used for running the comamnds.

We'll download the source code from the [LLVM 13.0.1 release on GitHub](https://github.com/llvm/llvm-project/releases/tag/llvmorg-13.0.1):

``` shell
$ curl -OL https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.1/llvm-13.0.1.src.tar.xz
```

Observe the `.src` in the name, indicating that we're downloading the source code. The same page also provides the binaries as well as the source code for the [tools and libraries produced by the LLVM sub-projects](https://llvm.org/):

- Clang compiler and its tools,
- compiler-rt runtime library,
- Flang compiler,
- libclc OpenCL library,
- libcxx C++ standard library and its application binary interface,
- lld linker,
- lldb debugger
- OpenMP library for Clang and Flang,
- Polly high-level loop and data-locality optimizations infrastructure, and
- test suite.

Although all of these tools are interesting in their own way, most of them will not be used here. In particular, we will be using Clang to demonstrate the compile process. We'll download it just like LLVM:

``` shell
$ curl -OL https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.1/clang-13.0.1.src.tar.xz
```

Now it's time to unpack the source code tarballs.

``` shell
$ tar xJf llvm-13.0.1.src.tar.xz
$ tar xJf clang-13.0.1.src.tar.xz
```

LLVM, Clang, and related projects use [CMake for building](https://llvm.org/docs/CMake.html). Most notably, it does not support building in the source tree, so it's necessary to start by creating a directory:

``` shell
$ mkdir builddir
$ cd builddir
```

There are [many CMake and LLVM-related variables](https://llvm.org/docs/CMake.html#options-and-variables) that can be specified at build time. We'll use only three of them, one CMake and two LLVM-related, specifically:

- `-DCMAKE_BUILD_TYPE=Release` sets the build mode to release (instead of the default debug), which results in smaller file size of the built binaries
- `-DBUILD_SHARED_LIBS=ON` enables dynamic linking of libraries, which singificantly reduces memory requirements for building and results in smaller file size of the built binaries
- `-DLLVM_ENABLE_PROJECTS=clang` enables building of Clang alongside LLVM

``` shell
$ cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_PROJECTS=clang ../llvm
$ make -j 2
$ make -j 2 check
```

!!! admonition "Assignment"
    Find out what is the latest released version of LLVM, download it instead of the one used above, and build it.

If you have many CPU cores, you can increase the number of parallel compile jobs by setting the `-j` parameter of the `make` command to a number larger than 2, for example the number of cores. This will make `make` make (!) the code faster, ideally several times faster.

!!! admonition "Assignment"
    Find out how many CPU cores you have and check if increasing the number of jobs speeds up the build process.

Alternatively, LLVM can also be [obtained from GitHub](https://github.com/llvm/llvm-project.git) using [Git](https://git-scm.com/). In that case, the branch `release/13.x` should be used. The rest of the process is pretty similar:

``` shell
$ git clone https://github.com/llvm/llvm-project.git
$ cd llvm-project
$ git checkout release/13.x
$ mkdir builddir
$ cd builddir
$ cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_PROJECTS=clang ../llvm
$ make -j 2
$ make -j 2 check
```

!!! admonition "Assignment"
    Find out what is the latest release branch of LLVM, check out that branch instead of the one used above, and build LLVM.

## The overview of the LLVM architecture

While LLVM is building, let's take a look at the LLVM architecture. [Chris Lattner](https://www.nondot.org/sabre/), the main author of LLVM, wrote the [LLVM](https://www.aosabook.org/en/llvm.html) chapter of [The Architecture of Open Source Applications](https://aosabook.org/en/index.html) book. To follow the code described in the chapter, open the following files in the `llvm-project-13.0.1/llvm-13.0.1` directory:

- `include/llvm/Analysis/InstructionSimplify.h`
- `lib/Analysis/InstructionSimplify.cpp`
- `include/llvm/Pass.h`
- `lib/Transforms/Hello/Hello.cpp`
- `include/llvm/ADT/Triple.h`
- `lib/Target/X86/X86InstrArithmetic.td`
- `lib/Target/AMDGPU/AMDGPUInstrInfo.td`
- `test/CodeGen/X86/add.ll`
- `test/CodeGen/AMDGPU/llvm.log10.ll`

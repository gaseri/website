---
author: Vedran Miletić
---

# The OpenMP offloading with the Clang compiler

Let's get into our build directory:

``` shell
cd builddir
```

We'll use a little symlink-powered hack to ease Clang's job of finding the OpenMP library:

``` shell
cd lib
ln -s ../runtimes/runtimes-bins/openmp/runtime/src/libomp.so
cd ..
```

We'll use the examples from [the OpenMP Application Programming Interface Examples book, Version 5.2.2 (April 2024)](https://www.openmp.org/wp-content/uploads/openmp-examples-5.2.2-final.pdf). Since copying code from PDF tends to be a suboptimal experience, we'll use [the examples from GitHub repository](https://github.com/OpenMP/Examples/tree/v5.2.2), specifically `devices/sources/target.1.c`:

``` c
/*
* @@name: target.1
* @@type: C
* @@operation: compile
* @@expect: success
* @@version: omp_4.0
*/
extern void init(float*, float*, int);
extern void output(float*, int);
void vec_mult(int N)
{
   int i;
   float p[N], v1[N], v2[N];
   init(v1, v2, N);
   #pragma omp target
   #pragma omp parallel for private(i)
   for (i=0; i<N; i++)
     p[i] = v1[i] * v2[i];
   output(p, N);
}
```

While Clang [doesn't support all possible OpenMP 5 features](https://clang.llvm.org/docs/OpenMPSupport.html), it supports enough to be able to compile this example.

Naively trying to compile this code will result in linking errors due to missing symbols:

``` shell
./bin/clang target.1.c
```

``` text
/usr/bin/ld: /lib/x86_64-linux-gnu/Scrt1.o: in function `_start':
(.text+0x17): undefined reference to `main'
/usr/bin/ld: /tmp/target-7a0854.o: in function `vec_mult':
target.1.c:(.text+0x76): undefined reference to `init'
/usr/bin/ld: target.1.c:(.text+0xc3): undefined reference to `output'
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

Since we want to inspect the object code (and the generated assembly inside it) and don't care about linking at this point, we will use the `-c` flag:

``` shell
./bin/clang -c target.1.c
```

The symbols in the generated `target.1.o` file can be inspected using [llvm-objdump](https://llvm.org/docs/CommandGuide/llvm-objdump.html) with the `--syms` parameter:

``` shell
./bin/llvm-objdump --syms target.1.o
```

``` text
target.1.o:     file format elf64-x86-64

SYMBOL TABLE:
0000000000000000 l    df *ABS*  0000000000000000 target.1.c
0000000000000000 l    d  .text  0000000000000000 .text
0000000000000000 g     F .text  00000000000000d3 vec_mult
0000000000000000         *UND*  0000000000000000 init
0000000000000000         *UND*  0000000000000000 output
```

There is a notable lack of `__omp_`- and `__kmpc_`-prefixed symbols, which would be expected in OpenMP-enabled build. This is because OpenMP support is not enabled automatically, but has to be done via a `-openmp` parameter ([documentation](https://openmp.llvm.org/CommandLineArgumentReference.html#general-command-line-arguments)):

``` shell
./bin/clang -c target.1.c -fopenmp
```

``` shell
./bin/llvm-objdump --syms target.1.o
```

``` text
target.1.o:     file format elf64-x86-64

SYMBOL TABLE:
0000000000000000 l    df *ABS*  0000000000000000 target.1.c
0000000000000000 l    d  .text  0000000000000000 .text
00000000000000e0 l     F .text  0000000000000075 __omp_offloading_809_d4b5d_vec_mult_l15
0000000000000160 l     F .text  0000000000000153 __omp_offloading_809_d4b5d_vec_mult_l15.omp_outlined
0000000000000000 l    d  .rodata.str1.1 0000000000000000 .rodata.str1.1
0000000000000000 l    d  .data.rel.ro   0000000000000000 .data.rel.ro
0000000000000000 g     F .text  00000000000000d1 vec_mult
0000000000000000         *UND*  0000000000000000 init
0000000000000000         *UND*  0000000000000000 output
0000000000000000         *UND*  0000000000000000 __kmpc_fork_call
0000000000000000         *UND*  0000000000000000 __kmpc_for_static_init_4
0000000000000000         *UND*  0000000000000000 __kmpc_for_static_fini
```

While offloading symbols are present, no images for a target architectures are present:

``` shell
./bin/llvm-objdump --offloading target.1.o
```

``` text
target.1.o:     file format elf64-x86-64
```

To enable generation of images, offloading requires specification of target architecture with `-fopenmp-targets` and `--offload-arch` parameters ([documentation](https://openmp.llvm.org/CommandLineArgumentReference.html#offload-command-line-arguments)):

``` shell
./bin/clang -c target.1.c -fopenmp -fopenmp-targets=amdgcn-amd-hsa --offload-arch=gfx942 -nogpulib
```

GFX942, which we requested with the `gfx942` value for `--offload-arch` parameter, is the architecture for [AMD Instinct MI300A](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300a.html). Additionally, observe the presence of `-nogpulib` parameter, which is helpful to avoid the requirement for [AMD ROCm](https://www.amd.com/en/products/software/rocm.html) installation on the system.

After printing the list of offloading images, we can see that the one for GFX942 is present:

``` shell
./bin/llvm-objdump --offloading target.1.o
```

``` text
target.1.o:     file format elf64-x86-64

OFFLOADING IMAGE [0]:
kind            llvm ir
arch            gfx942
triple          amdgcn-amd-amdhsa
producer        openmp
```

It is possible to generate code for multiple offload architectures, which results in multiple images. Let's, as an example, use *Kaveri* (`kaveri` or `gfx700` value for `--offload-arch` parameter), which happened to be [the first APU to support unified memory](https://www.anandtech.com/show/7677/amd-kaveri-review-a8-7600-a10-7850k/6):

``` shell
./bin/clang -c target.1.c -fopenmp -fopenmp-targets=amdgcn-amd-hsa --offload-arch=gfx942,kaveri -nogpulib
```

``` shell
./bin/llvm-objdump --offloading target.1.o
```

``` text
target.1.o:     file format elf64-x86-64

OFFLOADING IMAGE [0]:
kind            llvm ir
arch            gfx700
triple          amdgcn-amd-amdhsa
producer        openmp

OFFLOADING IMAGE [1]:
kind            llvm ir
arch            gfx942
triple          amdgcn-amd-amdhsa
producer        openmp
```

!!! example "Assignment"
    Find another OpenMP target offloading example and compile it for Radeon VII GPU and Instinct MI250X Accelerator. Refer to [the User Guide for AMDGPU Backend](https://llvm.org/docs/AMDGPUUsage.html) to figure out the architecture names.

!!! tip
    [The related question](https://openmp.llvm.org/SupportAndFAQ.html#q-can-openmp-offloading-compile-for-multiple-architectures) in the [LLVM/OpenMP FAQ](https://openmp.llvm.org/SupportAndFAQ.html) explains this in more details and covers non-AMD architectures.

Enabling a feature, such as [HSA XNACK](https://niconiconi.neocities.org/tech-notes/xnack-on-amd-gpus/), can be specified after the architecture, separated by a colon:

``` shell
./bin/clang -c target.1.c -fopenmp -fopenmp-targets=amdgcn-amd-hsa --offload-arch=gfx942:xnack+ -nogpulib
```

``` shell
vmiletic@atlas:~/workspace/llvm-project/builddir$ ./bin/llvm-objdump --offloading target.1.o
```

``` text
target.1.o:     file format elf64-x86-64

OFFLOADING IMAGE [0]:
kind            llvm ir
arch            gfx942:xnack+
triple          amdgcn-amd-amdhsa
producer        openmp
```

Using `-` instead of `+` would, of course, disable a feature. Notable limitation is that, if any target architecture specifies a feature, it has to be specified (as enabled or disabled) in all target architectures.

!!! example "Assignment"
    Learn about other supported features of GFX942 in [the User Guide for AMDGPU Backend](https://llvm.org/docs/AMDGPUUsage.html) and check if you can enable them using the above approach.

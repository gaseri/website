---
author: Vedran Miletić
---

# The Clang compiler

[Clang](https://clang.llvm.org/) is a compiler for C-like programming languages, including C, C ++, Objective C/C++, OpenCL C, and CUDA C/C++. It is part of the [LLVM project](https://llvm.org/).

Before continuing, let us make sure that Clang has been successfully compiled:

``` shell
$ ./bin/clang --version
clang version 13.0.1 (https://github.com/llvm/llvm-project.git 75e33f71c2dae584b13a7d1186ae0a038ba98838)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /home/vedranm/workspace/Development/llvm-project/builddir/./bin
```

The `Target:` line specifies the default target triple containing the processor architecture, the processor sub-architecture (optional), the processor vendor, the operating system, the environment, and the object format (optional). In the absence of additional parameters spoecifying the architecture, we will compile the code for the host processor architecture ([x86-64](https://en.wikipedia.org/wiki/X86-64) in our case, also known as [AMD64](https://www.amd.com/system/files/TechDocs/24592.pdf) and [Intel EM64T](https://www.intel.com/content/www/us/en/support/articles/000005898/processors.html)), the host operating system (Linux in our case), and the host environment (GNU). Notice how the host processor vendor is unknown, as Clang compiled for x86-64 can be executed on both Intel and AMD processors.

## Creating executable files from the C/C++ source code

Clang can be used for compiling C and C++ source files to executable files. For example, the C source file `example1.c` containing:

``` c
#include <stdio.h>

int main(void)
{
    printf("Hello from C\n");
}
```

can be compiled to `example1` executable file using the `clang` command and `-o` parameter:

``` shell
$ ./bin/clang example1.c -o example1
```

Executing gives:

``` shell
$ ./example1
Hello from C
```

Therefore, the compilation and the execution were successful.

Trying to compile the C++ source file `example2.cpp` containing:

``` cpp
#include <iostream>

int main()
{
    std::cout << "Hello from C++\n";
}
```

using the analogous command will fail:

``` shell
$ ./bin/clang example2.cpp -o example2
/usr/bin/ld: /tmp/example-4609c2.o: in function `main':
example2.cpp:(.text+0x6): undefined reference to `std::cout'
/usr/bin/ld: example2.cpp:(.text+0x19): undefined reference to `std::basic_ostream<char, std::char_traits<char> >& std::operator<< <std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*)'
/usr/bin/ld: /tmp/example-4609c2.o: in function `__cxx_global_var_init':
example2.cpp:(.text.startup+0xf): undefined reference to `std::ios_base::Init::Init()'
/usr/bin/ld: example2.cpp:(.text.startup+0x15): undefined reference to `std::ios_base::Init::~Init()'
clang-13: error: linker command failed with exit code 1 (use -v to see invocation)
```

The failure occurs at the linking stage; Clang treats the the source file language as C-like instead of specifically C++ and therefore does not link the resulting executable to the C++ standard library. This can be done manually by using the `-l` parameter:

``` shell
$ ./bin/clang example2.cpp -o example2 -l stdc++
```

Although this approach almost always work, the recommended method is to specify C++ as the source file language using the `-x` parameter and let Clang take care of linking the standard library:

``` shell
$ ./bin/clang example2.cpp -o example2 -x c++
```

Executing `example2` shows that the compilation was successful:

``` shell
$ ./example2
Hello from C++
```

For convenience, Clang also provides `clang++` command which is equivalent to `clang -x c++`. Therefore the C++ source file can also be compiled with:

``` shell
$ ./bin/clang++ example2.cpp -o example2
```

!!! admonition "Assignment"
    Select the C or the C++ example and modify it to include other code, e.g. value assignments to variables and control flow (`if`, `for`, or `while`). Pay attention to Clang's reporting of errors and warnings during compilation.

Although Clang is an excellent compiler for practical applications (including large applications such as [LibreOffice](https://wiki.documentfoundation.org/Development/Building_LibreOffice_with_Clang) and [Linux](https://docs.kernel.org/kbuild/llvm.html)), from now on we will focus on using Clang to translate the source code into assembly code or the LLVM intermediate representation (IR). In both case,  Clang will produce the output code after the specified compilation stage, but will not link it to libraries. Therefore, Clang will not create an executable file as it has done so far and we will focus on studying the results by reading the output instead of executing it.

## Compiling the source code into the target machine assembly

When we run Clang with the `-S` parameter, it will translate the C/C++ source code into the assembly code (for the x86-64 architecture) and output that code into the file with the same name as the input file and the extension `.s`:

``` shell
$ ./bin/clang example1.c -S
```

The file `example1.s` contains the following assembly code:

``` asm
    .text
    .file   "example1.c"
    .globl  main                            # -- Begin function main
    .p2align    4, 0x90
    .type   main,@function
main:                                   # @main
    .cfi_startproc
# %bb.0:
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register %rbp
    movabsq $.L.str, %rdi
    movb    $0, %al
    callq   printf
    xorl    %eax, %eax
    popq    %rbp
    .cfi_def_cfa %rsp, 8
    retq
.Lfunc_end0:
    .size   main, .Lfunc_end0-main
    .cfi_endproc
                                        # -- End function
    .type   .L.str,@object                  # @.str
    .section    .rodata.str1.1,"aMS",@progbits,1
.L.str:
    .asciz  "hello, world\n"
    .size   .L.str, 14

    .ident  "clang version 13.0.1 (https://github.com/llvm/llvm-project.git 75e33f71c2dae584b13a7d1186ae0a038ba98838)"
    .section    ".note.GNU-stack","",@progbits
    .addrsig
    .addrsig_sym printf
```

The output file name can be specified using the `-o` parameter. Specifying `-` as the name prints the resulting assembly code to the standard output:

``` shell
$ ./bin/clang example1.c -S -o -
    .text
    .file   "example1.c"
    .globl  main                            # -- Begin function main
    .p2align    4, 0x90
    .type   main,@function
main:                                   # @main
    .cfi_startproc
# %bb.0:
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register %rbp
    movabsq $.L.str, %rdi
    movb    $0, %al
    callq   printf
    xorl    %eax, %eax
    popq    %rbp
    .cfi_def_cfa %rsp, 8
    retq
.Lfunc_end0:
    .size   main, .Lfunc_end0-main
    .cfi_endproc
                                        # -- End function
    .type   .L.str,@object                  # @.str
    .section    .rodata.str1.1,"aMS",@progbits,1
.L.str:
    .asciz  "hello, world\n"
    .size   .L.str, 14

    .ident  "clang version 13.0.1 (https://github.com/llvm/llvm-project.git 75e33f71c2dae584b13a7d1186ae0a038ba98838)"
    .section    ".note.GNU-stack","",@progbits
    .addrsig
    .addrsig_sym printf
```

Optimization level can be specified by using some variant of the `-O` parameter: `-O0`, `-O1`, `-O2`, `-O3`, `-Ofast`, `-Os`, `-Oz`, `-Og`, `-O`, and `-O4`. [Official documentation](https://clang.llvm.org/docs/CommandGuide/clang.html#cmdoption-o0) says:

> - `-O0` -- Means "no optimization": this level compiles the fastest and generates the most debuggable code.
> - `-O1` -- Somewhere between -O0 and -O2.
> - `-O2` -- Moderate level of optimization which enables most optimizations.
> - `-O3` -- Like `-O2`, except that it enables optimizations that take longer to perform or that may generate larger code (in an attempt to make the program run faster).
> - `-Ofast` -- Enables all the optimizations from `-O3` along with other aggressive optimizations that may violate strict compliance with language standards.
> - `-Os` -- Like `-O2` with extra optimizations to reduce code size.
> - `-Oz` -- Like `-Os` (and thus `-O2`), but reduces code size further.
> - `-Og` -- Like `-O1`. In future versions, this option might disable different optimizations in order to improve debuggability.
> - `-O` -- Equivalent to `-O1`.
> - `-O4` and higher -- Currently equivalent to `-O3`.

!!! admonition "Assignment"
    Add some dead code of your choice to the example and find out which optimization levels eliminate it.

Other targets can be specified using the `-target` parameter. To compile the code for MIPS on GNU/Linux:

``` shell
$ ./bin/clang example1.c -S -target mips-unknown-linux-gnu
```

The resulting `example1.s` contains:

``` asm
    .text
    .abicalls
    .option pic0
    .section    .mdebug.abi32,"",@progbits
    .nan    legacy
    .text
    .file   "example1.c"
    .globl  main                            # -- Begin function main
    .p2align    2
    .type   main,@function
    .set    nomicromips
    .set    nomips16
    .ent    main
main:                                   # @main
    .frame  $fp,32,$ra
    .mask   0xc0000000,-4
    .fmask  0x00000000,0
    .set    noreorder
    .set    nomacro
    .set    noat
# %bb.0:
    addiu   $sp, $sp, -32
    sw  $ra, 28($sp)                    # 4-byte Folded Spill
    sw  $fp, 24($sp)                    # 4-byte Folded Spill
    move    $fp, $sp
    sw  $zero, 20($fp)
    lui $1, %hi($.str)
    addiu   $4, $1, %lo($.str)
    jal printf
    nop
    addiu   $2, $zero, 0
    move    $sp, $fp
    lw  $fp, 24($sp)                    # 4-byte Folded Reload
    lw  $ra, 28($sp)                    # 4-byte Folded Reload
    addiu   $sp, $sp, 32
    jr  $ra
    nop
    .set    at
    .set    macro
    .set    reorder
    .end    main
$func_end0:
    .size   main, ($func_end0)-main
                                        # -- End function
    .type   $.str,@object                   # @.str
    .section    .rodata.str1.1,"aMS",@progbits,1
$.str:
    .asciz  "hello, world\n"
    .size   $.str, 14

    .ident  "clang version 13.0.1 (https://github.com/llvm/llvm-project.git 75e33f71c2dae584b13a7d1186ae0a038ba98838)"
    .section    ".note.GNU-stack","",@progbits
    .addrsig
    .addrsig_sym printf
    .text
```

!!! admonition "Assignment"
    Find out if the choice of the operating system and the environment affects the resulting assembly code.

!!! admonition "Assignment"
    Find out if this code can be compiled to assembly code for ARM CPUs, RISC-V CPUs, AMD HD2XXX-HD6XXX and GCN GPUs, and WebAssembly.

Finally, we can compile the C++ source code the same way:

``` shell
$ ./bin/clang example2.cpp -S
```

The resulting assembly file `example2.s` is slightly more complicated:

``` asm
    .text
    .file   "example2.cpp"
    .section    .text.startup,"ax",@progbits
    .p2align    4, 0x90                         # -- Begin function __cxx_global_var_init
    .type   __cxx_global_var_init,@function
__cxx_global_var_init:                  # @__cxx_global_var_init
    .cfi_startproc
# %bb.0:
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register %rbp
    movabsq $_ZStL8__ioinit, %rdi
    callq   _ZNSt8ios_base4InitC1Ev
    movabsq $_ZNSt8ios_base4InitD1Ev, %rdi
    movabsq $_ZStL8__ioinit, %rsi
    movabsq $__dso_handle, %rdx
    callq   __cxa_atexit
    popq    %rbp
    .cfi_def_cfa %rsp, 8
    retq
.Lfunc_end0:
    .size   __cxx_global_var_init, .Lfunc_end0-__cxx_global_var_init
    .cfi_endproc
                                        # -- End function
    .text
    .globl  main                            # -- Begin function main
    .p2align    4, 0x90
    .type   main,@function
main:                                   # @main
    .cfi_startproc
# %bb.0:
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register %rbp
    movabsq $_ZSt4cout, %rdi
    movabsq $.L.str, %rsi
    callq   _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
    xorl    %eax, %eax
    popq    %rbp
    .cfi_def_cfa %rsp, 8
    retq
.Lfunc_end1:
    .size   main, .Lfunc_end1-main
    .cfi_endproc
                                        # -- End function
    .section    .text.startup,"ax",@progbits
    .p2align    4, 0x90                         # -- Begin function _GLOBAL__sub_I_example2.cpp
    .type   _GLOBAL__sub_I_example2.cpp,@function
_GLOBAL__sub_I_example2.cpp:            # @_GLOBAL__sub_I_example2.cpp
    .cfi_startproc
# %bb.0:
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register %rbp
    callq   __cxx_global_var_init
    popq    %rbp
    .cfi_def_cfa %rsp, 8
    retq
.Lfunc_end2:
    .size   _GLOBAL__sub_I_example2.cpp, .Lfunc_end2-_GLOBAL__sub_I_example2.cpp
    .cfi_endproc
                                        # -- End function
    .type   _ZStL8__ioinit,@object          # @_ZStL8__ioinit
    .local  _ZStL8__ioinit
    .comm   _ZStL8__ioinit,1,1
    .hidden __dso_handle
    .type   .L.str,@object                  # @.str
    .section    .rodata.str1.1,"aMS",@progbits,1
.L.str:
    .asciz  "Hello, world!\n"
    .size   .L.str, 15

    .section    .init_array,"aw",@init_array
    .p2align    3
    .quad   _GLOBAL__sub_I_example2.cpp
    .ident  "clang version 13.0.1 (https://github.com/llvm/llvm-project.git 75e33f71c2dae584b13a7d1186ae0a038ba98838)"
    .section    ".note.GNU-stack","",@progbits
    .addrsig
    .addrsig_sym __cxx_global_var_init
    .addrsig_sym __cxa_atexit
    .addrsig_sym _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
    .addrsig_sym _GLOBAL__sub_I_example2.cpp
    .addrsig_sym _ZStL8__ioinit
    .addrsig_sym __dso_handle
    .addrsig_sym _ZSt4cout
```

Note the presence of the [C++ symbol name mangling](https://en.wikipedia.org/wiki/Name_mangling#C++), e.g. `_ZSt4cout` is the mangled name of `std::cout`. For demangling the less obvious mangled names, LLVM provides the [llvm-cxxfilt](https://llvm.org/docs/CommandGuide/llvm-cxxfilt.html) symbol name demangler:

``` shell
$ ./bin/llvm-cxxfilt _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
std::basic_ostream<char, std::char_traits<char> >& std::operator<<<std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*)
```

## Compiling the source code into the LLVM intermediate representation

Just like assembly, Clang can output the LLVM intermediate representation (IR), which is more convenient for the study of optimization (i.e. analysis and transformation) passes. To produce LLVM IR, we will use `-emit-llvm` parameter in addition to the `-S` parameter:

``` shell
$ ./clang example1.c -S -emit-llvm
```

This will create the `example1.ll` file containing:

``` llvm
; ModuleID = 'example1.c'
source_filename = "example1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [14 x i8] c"hello, world\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str, i64 0, i64 0))
  ret i32 0
}

declare dso_local i32 @printf(i8*, ...) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 13.0.1 (https://github.com/llvm/llvm-project.git 75e33f71c2dae584b13a7d1186ae0a038ba98838)"}
```

!!! admonition "Assignment"
    Find out if the LLVM IR differs for different taget processors and, if it does, in what way.

!!! admonition "Assignment"
    Compare the LLVM IR produced by different optimization levels and see if you can observe the optimizations performed.

From now on, we will treat Clang's code generation as a black box and modify only the optimization steps in LLVM. A good starting point to delve into the inner workings of Clang's code generation is the [LLVM Target-Independent Code Generator section](https://llvm.org/docs/CodeGenerator.html) contained in the the official documentation.

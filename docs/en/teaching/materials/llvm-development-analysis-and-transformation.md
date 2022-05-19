---
author: Vedran Miletić
---

# Implementing an analysis and transformation pass in the LLVM compiler infrastructure

The LLVM documentation contains the [Getting Started/Tutorials section](https://llvm.org/docs/GettingStartedTutorials.html), which offers two [tutorials](https://llvm.org/docs/tutorial/index.html), one on implementing a language frontend and the other on building a just-in-time (JIT) compiler. While the latter tutorial [covers optimization](https://llvm.org/docs/tutorial/BuildingAJIT2.html), it does so in a specific context that is not required here.

We will be following [The LLVM Compiler Framework and Infrastructure Tutorial](https://llvm.org/pubs/2004-09-22-LCPCLLVMTutorial.html) presented by Chris Lattner and Vikram Adve at the LCPC'04 Mini Workshop on Compiler Research Infrastructures in Septembed 2004. While the presentation is a bit older, the information in it is still very much relevant and a good augmentation of the available LLVM documentation.

## Writing an LLVM pass

The LLVM documentation contains the guidance for writing a pass in two ways:

- [using the legacy pass manager](https://llvm.org/docs/WritingAnLLVMPass.html) and
- [using the new pass manager](https://llvm.org/docs/NewPassManager.html) (which has [additional documentation on its usage](https://llvm.org/docs/NewPassManager.html)).

Since the legacy pass manager will eventually be removed from LLVM and the new pass manager is much easier to use, and it requires less boilerplate code, we will be following the latter guide.

The `HelloWorldPass` described in the [Basic code required](https://llvm.org/docs/WritingAnLLVMNewPMPass.html#basic-code-required) section is already a part of the LLVM source code and it got compiled when we initially set up the development environment. We will be modifying it from now on, but first let's use `llvm-stress` to create the file containing LLVM IR that will be used for testing the pass:

``` shell
$ ./bin/llvm-stress -o example-stress.ll
```

The pass implemented in the `HelloWorldPass` class is registered as `helloworld` in the `llvm/lib/Passes/PassRegistry.def` file:

``` cpp
FUNCTION_PASS("helloworld", HelloWorldPass())
```

so it can be called by adding the parameter `-passes` with the value `helloworld` to `opt`:

``` shell
$ ./bin/opt example-stress.ll -S -passes helloworld -o example-stress-opt.ll
autogen_SD0
```

We can observe the function name in the output. The resulting LLVM IR file can be compared to the starting file with `llvm-diff` and there will be no difference as the pass only performs analysis (specifically, finding functions and printing their names) and does not perform any transformation.

!!! admonition "Assignment"
    Change the optimization pass so that it also prints the the number of operands for each function and the function type (signature); the [API documentation of the Function class](https://llvm.org/doxygen/classllvm_1_1Function.html) is a good place to look for a way to obtain this information.

!!! admonition "Assignment"
    Change the optimization pass so that it also prints the number of times each function was called. Amend the `example-stress.ll` file with the function `manualgen_SD0` calling the function `autogen_SD0`:

    ``` llvm
    define void @manualgen_SD0(i8* %0, i32* %1, i64* %2, i32 %3, i64 %4, i8 %5) {
      call void (i8*, i32*, i64*, i32, i64, i8) @autogen_SD0(i8* %0, i32* %1, i64* %2, i32 %3, i64 %4, i8 %5)
      ret void
    }
    ```

    and use this file to verify the correct number is printed

!!! admonition "Assignment"
    Change the optimization pass so that it also prints the number of usages (reads and writes) for each of the function arguments.

!!! admonition "Assignment"
    Change the optimization pass so that it also prints the number of instructions and then interates over the instruction list (note that the instructions reside inside of the basic blocks). For each of the instructions, print the type and the operands, if any.

Finally, before starting with a specific project implemented using the LLVM libraries, the required reading is the [LLVM Programmer’s Manual](https://llvm.org/docs/ProgrammersManual.html).

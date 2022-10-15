---
author: Vedran Miletić
---

# The LLVM intermediate representation

The [LLVM intermediate representation (IR)](https://en.wikipedia.org/wiki/LLVM#Intermediate_representation) is a low-level assembly-like programming language. LLVM frontends convert input into LLVM IR, which is then converted into the target instruction set by the backend. LLVM IR is available in three formats:

- a human-readable assembly format, which we shall deal first
- an in-memory format suitable for frontends, which we are likely to encounter when writing code in LLVM, and
- a dense bitcode format for serialization, into which the human-readable assembly format is easily converted.

For the introduction to LLVM IR, we will follow [Modern Intermediate Representations (IR)](https://llvm.org/devmtg/2017-06/1-Davis-Chisnall-LLVM-2017.pdf) by [David](https://www.microsoft.com/en-us/research/people/dachisna/) [Chisnall](https://www.cl.cam.ac.uk/~dc552/) presented at the [HPC summer school 2017](https://llvm.org/devmtg/2017-06/). For more information on particular types, intrinsics, and special instructions, please see the [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html).

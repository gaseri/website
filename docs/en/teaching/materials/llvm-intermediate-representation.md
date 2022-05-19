---
author: Vedran Miletić
---

# The LLVM intermediate representation

The [LLVM intermediate representation (IR)](https://en.wikipedia.org/wiki/LLVM#Intermediate_representation) is the low-level programming language similar to assembly. The LLVM frontends convert the input to LLVM IR, which is then converted to the target instruction set by the backend. LLVM IR appears three formats:

- a human-readable assembly format, which we will see first
- an in-memory format suitable for frontends, which we will encounter when writing code in LLVM, and
- a dense bitcode format for serializing, which the human-readable assembly format is easily converted to.

For the introduction to LLVM IR, we'll be following [Modern Intermediate Representations (IR)](https://llvm.org/devmtg/2017-06/1-Davis-Chisnall-LLVM-2017.pdf) by [David](https://www.microsoft.com/en-us/research/people/dachisna/) [Chisnall](https://www.cl.cam.ac.uk/~dc552/) presented at the [HPC summer school 2017](https://llvm.org/devmtg/2017-06/). For more information about the particular types, intrinsics, and instructions, we'll be consulting the [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html).

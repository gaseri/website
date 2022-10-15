---
author: Vedran Miletić
---

# The LLVM tools

In addition to the use of LLVM via the Clang compiler and the Mesa 3d graphics library, it is also possible to use LLVM via [several command-line tools bundled with it](https://llvm.org/docs/CommandGuide/index.html). Among these We have already used `llc` and will also be using `llvm-cxxfilt`:

- [llvm-mca](#llvm-mca), the LLVM machine code analyzer
- [lli](#lli), the LLVM just-in-time compiler or interpreter
- [llvm-diff](#llvm-diff), the structural `diff` tool for files containing LLVM intermediate representations
- [llvm-stress](#llvm-stress), the random generator of files containing LLVM intermediate representation
- [opt](#opt), the LLVM optimizer

We will be working with files containing the LLVM intermediate representation (IR). Many LLVM IR examples are available in the [VirtualMachine/ir-examples repository on GitHub](https://github.com/Virtual-Machine/ir-examples) that we obtained earlier. We will use both the [C source code](https://github.com/Virtual-Machine/ir-examples/tree/master/c) and the [LLVM IR source code](https://github.com/Virtual-Machine/ir-examples/tree/master/ll) from that repository.

In the following examples we will place the `ir-examples` directory  in the `builddir` directory of the `llvm-project`. Once placed in the `builddir` directory, we can achieve this using cURL:

``` shell
$ curl -OL https://github.com/Virtual-Machine/ir-examples/archive/refs/heads/master.zip
$ unzip master.zip
$ mv ir-examples-master ir-examples
```

or Git:

``` shell
$ git clone https://github.com/Virtual-Machine/ir-examples.git
```

## llvm-mca

[llvm-mca](https://llvm.org/docs/CommandGuide/llvm-mca.html) predicts the performance of machine code based on the scheduling models of processors in LLVM.

It takes assembly as the input. For example, to predict the performance of `test1.c` when executed on [AMD Jaguar microarchitecture](https://en.wikipedia.org/wiki/Jaguar_(microarchitecture)) CPUs, we should use `llvm-mca` with the `-mcpu` parameter set to `btver2` (for what it's worth, `btver1` is [AMD Bobcat](https://en.wikipedia.org/wiki/Bobcat_(microarchitecture)), while `bdver1`, `bdver2`, `bdver3`, and `bdver4` are [AMD](https://www.amd.com/en/processors) [Bulldozer](https://en.wikipedia.org/wiki/Bulldozer_(microarchitecture)), [Piledriver](https://en.wikipedia.org/wiki/Piledriver_(microarchitecture)), [Steamroller](https://en.wikipedia.org/wiki/Steamroller_(microarchitecture)), and [Excavator](https://en.wikipedia.org/wiki/Excavator_(microarchitecture)) (respectively)):

``` shell
$ ./bin/clang ir-examples/c/test1.c -O2 -target x86_64-unknown-unknown -S -o - | ./bin/llvm-mca -mcpu=btver2
warning: found a return instruction in the input assembly sequence.
note: program counter updates are ignored.
Iterations:        100
Instructions:      1000
Total Cycles:      803
Total uOps:        1000

Dispatch Width:    2
uOps Per Cycle:    1.25
IPC:               1.25
Block RThroughput: 5.0


Instruction Info:
[1]: #uOps
[2]: Latency
[3]: RThroughput
[4]: MayLoad
[5]: MayStore
[6]: HasSideEffects (U)

[1]    [2]    [3]    [4]    [5]    [6]    Instructions:
 1      1     1.00           *            pushq %rbp
 1      1     0.50                        movq  %rsp, %rbp
 1      1     0.50                        leal  5(%rdi), %eax
 1      3     1.00    *                   popq  %rbp
 1      4     1.00                  U     retq
 1      1     1.00           *            pushq %rbp
 1      1     0.50                        movq  %rsp, %rbp
 1      0     0.50                        xorl  %eax, %eax
 1      3     1.00    *                   popq  %rbp
 1      4     1.00                  U     retq


Resources:
[0]   - JALU0
[1]   - JALU1
[2]   - JDiv
[3]   - JFPA
[4]   - JFPM
[5]   - JFPU0
[6]   - JFPU1
[7]   - JLAGU
[8]   - JMul
[9]   - JSAGU
[10]  - JSTC
[11]  - JVALU0
[12]  - JVALU1
[13]  - JVIMUL


Resource pressure per iteration:
[0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]
2.50   2.50    -      -      -      -      -     4.00    -     2.00    -      -      -      -

Resource pressure by instruction:
[0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]   Instructions:
 -      -      -      -      -      -      -      -      -     1.00    -      -      -      -     pushq %rbp
0.49   0.51    -      -      -      -      -      -      -      -      -      -      -      -     movq  %rsp, %rbp
0.50   0.50    -      -      -      -      -      -      -      -      -      -      -      -     leal  5(%rdi), %eax
 -      -      -      -      -      -      -     1.00    -      -      -      -      -      -     popq  %rbp
0.50   0.50    -      -      -      -      -     1.00    -      -      -      -      -      -     retq
 -      -      -      -      -      -      -      -      -     1.00    -      -      -      -     pushq %rbp
0.52   0.48    -      -      -      -      -      -      -      -      -      -      -      -     movq  %rsp, %rbp
 -      -      -      -      -      -      -      -      -      -      -      -      -      -     xorl  %eax, %eax
 -      -      -      -      -      -      -     1.00    -      -      -      -      -      -     popq  %rbp
0.49   0.51    -      -      -      -      -     1.00    -      -      -      -      -      -     retq
```

We can observe the prediction of 1.25 instructions per cycle on average during the program exection (`IPC`) (higher is better) and the approximately 50% usage of arithmetic logic units (ALUs) in the `Resource pressure` section (lower is better).

!!! admonition "Assignment"
    Compare the instructions per cycle and resurce pressure when `control_flow.c` is executed on AMD Jaguar and AMD Piledriver CPUs.

## lli

[lli](https://llvm.org/docs/CommandGuide/lli.html) is the interpreter that enables direct execution of the programs written in LLVM IR:

``` shell
$ ./bin/lli ir-examples/ll/switch.ll
1.2 + 1.4 = 2.6
```

!!! admonition "Assignment"
    Modify the `switch.c` file so that it sums three numbers instead of two. Compile to LLVM IR and compare the resulting file with the `switch.ll` from the repository. Check if the resulting file can be executed with `lli`.

## llvm-diff

[llvm-diff](https://llvm.org/docs/CommandGuide/llvm-diff.html) is the structural comparison tool for LLVM IR files. Usage:

``` shell
$ ./bin/llvm-diff example1.ll example2.ll
```

!!! admonition "Assignment"
    Use `diff` and then `llvm-diff` to compare the two LLVM IR files from the previous assignment. Observe what `llvm-diff` takes into account that `diff` does not.

## llvm-stress

[llvm-stress](https://llvm.org/docs/CommandGuide/llvm-stress.html) is the random LLVM IR generator.

``` shell
$ ./bin/llvm-stress -o example-stress.ll
```

The file `example-stress.ll` contains:

``` llvm
; ModuleID = '/tmp/autogen.bc'
source_filename = "/tmp/autogen.bc"

define void @autogen_SD0(i8* %0, i32* %1, i64* %2, i32 %3, i64 %4, i8 %5) {
BB:
  %A4 = alloca <8 x double>, align 64
  %A3 = alloca <2 x i16>, align 4
  %A2 = alloca <4 x i8>, align 4
  %A1 = alloca <16 x i1>, align 16
  %A = alloca <16 x i1>, align 16
  %L = load <16 x i1>, <16 x i1>* %A, align 16
  store i8 %5, i8* %0, align 1
  %E = extractelement <2 x i8> zeroinitializer, i32 0
  %Shuff = shufflevector <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i32> <i32 undef, i32 7, i32 1, i32 undef>
  %I = insertelement <1 x i8> zeroinitializer, i8 21, i32 0
  %B = fmul double 0x18DF23FE11DD527C, 0xAE97BFB633957A34
  %Se = sext i8 77 to i16
  %Sl = select <8 x i1> zeroinitializer, <8 x i1> zeroinitializer, <8 x i1> zeroinitializer
  %Cmp = icmp ugt i16 18761, %Se
  br label %CF

CF:                                               ; preds = %CF, %CF85, %CF86, %BB
  %L5 = load i8, i8* %0, align 1
  store i8 21, i8* %0, align 1
  %E6 = extractelement <2 x i16> zeroinitializer, i32 0
  %Shuff7 = shufflevector <2 x i8> zeroinitializer, <2 x i8> zeroinitializer, <2 x i32> <i32 undef, i32 3>
  %I8 = insertelement <4 x i32> zeroinitializer, i32 32529, i32 3
  %B9 = sub i64 %4, 251161
  %Tr = trunc <4 x i32> %I8 to <4 x i16>
  %Sl10 = select i1 %Cmp, <2 x i16>* %A3, <2 x i16>* %A3
  %Cmp11 = icmp ne <1 x i8> zeroinitializer, zeroinitializer
  %L12 = load <2 x i16>, <2 x i16>* %Sl10, align 4
  store <2 x i16> %L12, <2 x i16>* %Sl10, align 4
  %E13 = extractelement <2 x i64> zeroinitializer, i32 1
  %Shuff14 = shufflevector <4 x i32> zeroinitializer, <4 x i32> %Shuff, <4 x i32> <i32 2, i32 undef, i32 6, i32 0>
  %I15 = insertelement <4 x i32> zeroinitializer, i32 %3, i32 0
  %B16 = and i8 %E, 21
  %FC = uitofp <4 x i32> %I15 to <4 x double>
  %Sl17 = select i1 true, i16 -1979, i16 16293
  %Cmp18 = fcmp ueq double 0xAE97BFB633957A34, 0x99ABD3CAB0A9A048
  br i1 %Cmp18, label %CF, label %CF83

CF83:                                             ; preds = %CF83, %CF92, %CF
  %L19 = load <2 x i16>, <2 x i16>* %Sl10, align 4
  store <2 x i16> zeroinitializer, <2 x i16>* %Sl10, align 4
  %E20 = extractelement <8 x i1> %Sl, i32 1
  br i1 %E20, label %CF83, label %CF92

CF92:                                             ; preds = %CF83
  %Shuff21 = shufflevector <8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <8 x i32> <i32 undef, i32 14, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10>
  %I22 = insertelement <16 x i1> zeroinitializer, i1 true, i32 14
  %B23 = mul <1 x i8> %I, zeroinitializer
  %FC24 = fptoui <4 x double> %FC to <4 x i16>
  %Sl25 = select i1 true, i8 %E, i8 %B16
  %Cmp26 = icmp ule i32 0, 132849
  br i1 %Cmp26, label %CF83, label %CF85

CF85:                                             ; preds = %CF92
  %L27 = load i8, i8* %0, align 1
  store <2 x i16> %L12, <2 x i16>* %Sl10, align 4
  %E28 = extractelement <2 x i64> zeroinitializer, i32 1
  %Shuff29 = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 6, i32 undef, i32 2, i32 4>
  %I30 = insertelement <8 x i32> %Shuff21, i32 132849, i32 0
  %B31 = and i8 77, 21
  %Tr32 = trunc <4 x i32> %I8 to <4 x i16>
  %Sl33 = select i1 true, i8 %B31, i8 %5
  %Cmp34 = icmp ule <2 x i16> %L19, %L12
  %L35 = load <2 x i16>, <2 x i16>* %Sl10, align 4
  store i8 %L5, i8* %0, align 1
  %E36 = extractelement <2 x i1> %Cmp34, i32 0
  br i1 %E36, label %CF, label %CF82

CF82:                                             ; preds = %CF82, %CF89, %CF91, %CF85
  %Shuff37 = shufflevector <16 x i1> %I22, <16 x i1> zeroinitializer, <16 x i32> <i32 7, i32 undef, i32 undef, i32 undef, i32 15, i32 17, i32 undef, i32 21, i32 23, i32 25, i32 27, i32 undef, i32 31, i32 1, i32 3, i32 5>
  %I38 = insertelement <4 x double> %FC, double 0xAE97BFB633957A34, i32 1
  %FC39 = sitofp i8 %L5 to double
  %Sl40 = select i1 true, <16 x i1> %I22, <16 x i1> %I22
  %Cmp41 = icmp uge i8 %Sl25, 21
  br i1 %Cmp41, label %CF82, label %CF89

CF89:                                             ; preds = %CF82
  %L42 = load i8, i8* %0, align 1
  store <2 x i16> %L12, <2 x i16>* %Sl10, align 4
  %E43 = extractelement <16 x i1> %Shuff37, i32 11
  br i1 %E43, label %CF82, label %CF88

CF88:                                             ; preds = %CF88, %CF89
  %Shuff44 = shufflevector <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i32> <i32 6, i32 0, i32 2, i32 undef>
  %I45 = insertelement <8 x i1> %Sl, i1 true, i32 0
  %B46 = srem <1 x i8> %I, %B23
  %Tr47 = fptrunc double %FC39 to float
  %Sl48 = select <8 x i1> %I45, <8 x i1> %I45, <8 x i1> zeroinitializer
  %Cmp49 = icmp eq i64 251161, %E28
  br i1 %Cmp49, label %CF88, label %CF91

CF91:                                             ; preds = %CF88
  %L50 = load <2 x i16>, <2 x i16>* %Sl10, align 4
  store i8 21, i8* %0, align 1
  %E51 = extractelement <16 x i1> zeroinitializer, i32 10
  br i1 %E51, label %CF82, label %CF86

CF86:                                             ; preds = %CF91
  %Shuff52 = shufflevector <8 x i1> %Sl48, <8 x i1> %I45, <8 x i32> <i32 13, i32 15, i32 1, i32 3, i32 5, i32 undef, i32 9, i32 11>
  %I53 = insertelement <4 x i32> zeroinitializer, i32 0, i32 3
  %FC54 = uitofp <2 x i16> zeroinitializer to <2 x float>
  %Sl55 = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i16> %Tr32, <4 x i16> %Tr32
  %Cmp56 = icmp ult <16 x i1> %L, %Shuff37
  %L57 = load <2 x i16>, <2 x i16>* %Sl10, align 4
  store <2 x i16> %L12, <2 x i16>* %Sl10, align 4
  %E58 = extractelement <4 x i16> %Tr32, i32 0
  %Shuff59 = shufflevector <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i32> <i32 3, i32 5, i32 7, i32 1>
  %I60 = insertelement <16 x i1> zeroinitializer, i1 true, i32 5
  %B61 = add <2 x i8> %Shuff7, zeroinitializer
  %Tr62 = trunc i64 %B9 to i1
  br i1 %Tr62, label %CF, label %CF81

CF81:                                             ; preds = %CF81, %CF90, %CF87, %CF86
  %Sl63 = select i1 true, <2 x i16> zeroinitializer, <2 x i16> %L50
  %Cmp64 = icmp sgt <4 x i32> %I8, %Shuff
  %L65 = load <16 x i1>, <16 x i1>* %A, align 16
  store i8 21, i8* %0, align 1
  %E66 = extractelement <2 x i16> %Sl63, i32 1
  %Shuff67 = shufflevector <2 x i64> zeroinitializer, <2 x i64> zeroinitializer, <2 x i32> <i32 2, i32 undef>
  %I68 = insertelement <2 x i64> zeroinitializer, i64 %4, i32 0
  %B69 = urem i64 251161, 251161
  %FC70 = fptosi double %B to i8
  %Sl71 = select i1 %E36, i1 true, i1 %Cmp49
  br i1 %Sl71, label %CF81, label %CF90

CF90:                                             ; preds = %CF81
  %Cmp72 = icmp slt i16 18437, %E58
  br i1 %Cmp72, label %CF81, label %CF87

CF87:                                             ; preds = %CF90
  %L73 = load i8, i8* %0, align 1
  store i64 %B9, i64* %2, align 4
  %E74 = extractelement <16 x i1> zeroinitializer, i32 2
  br i1 %E74, label %CF81, label %CF84

CF84:                                             ; preds = %CF87
  %Shuff75 = shufflevector <4 x i32> %I15, <4 x i32> %I53, <4 x i32> <i32 5, i32 7, i32 1, i32 3>
  %I76 = insertelement <8 x i1> zeroinitializer, i1 %Tr62, i32 7
  %B77 = fsub double 0x99ABD3CAB0A9A048, 0xAE97BFB633957A34
  %FC78 = uitofp i32 0 to double
  %Sl79 = select i1 true, i8 %L42, i8 %L73
  %Cmp80 = icmp ult <8 x i1> %Shuff52, %Sl
  store i8 %FC70, i8* %0, align 1
  store <2 x i16> %L12, <2 x i16>* %Sl10, align 4
  store <2 x i16> %L12, <2 x i16>* %Sl10, align 4
  store i8 %E, i8* %0, align 1
  store i8 %L42, i8* %0, align 1
  ret void
}
```

If we desire a smaller example file, we can specify the number of desired instructions using the `-size` parameter:

``` shell
$ ./bin/llvm-stress -size 8 -o example-stress-size8.ll
```

The file `example-stress-size8.ll` contains:

``` llvm
; ModuleID = '/tmp/autogen.bc'
source_filename = "/tmp/autogen.bc"

define void @autogen_SD0(i8* %0, i32* %1, i64* %2, i32 %3, i64 %4, i8 %5) {
BB:
  %A4 = alloca <8 x double>, align 64
  %A3 = alloca <2 x i16>, align 4
  %A2 = alloca <4 x i8>, align 4
  %A1 = alloca <16 x i1>, align 16
  %A = alloca <16 x i1>, align 16
  store i8 21, i8* %0, align 1
  store <8 x double> <double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF>, <8 x double>* %A4, align 64
  store i8 77, i8* %0, align 1
  store i8 77, i8* %0, align 1
  ret void
}
```

It is also possible to specify the `-seed` parameter with the value used for seeding the random generator.

!!! admonition "Assignment"
    Check if the results of the subsequent runs of `llvm-stress` are the same (you can use `llvm-diff` to compare the files). Check if the result changes when different seed values are specified. (**Hint:** Rename functions if necessary for comparison.)

## opt

[opt](https://llvm.org/docs/CommandGuide/opt.html) is used for performing optimization (i.e. analysis and transform) passes.

Availables passes can be printed using the `-print-passes` parameter:

``` shell
$ ./bin/opt -print-passes
Module passes:
  always-inline
  attributor
  annotation2metadata
  openmp-opt
  called-value-propagation
  canonicalize-aliases
  cg-profile
  constmerge
  cross-dso-cfi
  deadargelim
  elim-avail-extern
  extract-blocks
  forceattrs
  function-import
  function-specialization
  globaldce
  globalopt
  globalsplit
  hotcoldsplit
  hwasan
  khwasan
  inferattrs
  inliner-wrapper
  inliner-wrapper-no-mandatory-first
  insert-gcov-profiling
  instrorderfile
  instrprof
  internalize
  invalidate<all>
  ipsccp
  iroutliner
  print-ir-similarity
  loop-extract
  lowertypetests
  metarenamer
  mergefunc
  name-anon-globals
  no-op-module
  objc-arc-apelim
  partial-inliner
  pgo-icall-prom
  pgo-instr-gen
  pgo-instr-use
  print-profile-summary
  print-callgraph
  print
  print-lcg
  print-lcg-dot
  print-must-be-executed-contexts
  print-stack-safety
  print<module-debuginfo>
  rel-lookup-table-converter
  rewrite-statepoints-for-gc
  rewrite-symbols
  rpo-function-attrs
  sample-profile
  scc-oz-module-inliner
  loop-extract-single
  strip
  strip-dead-debug-info
  pseudo-probe
  strip-dead-prototypes
  strip-debug-declare
  strip-nondebug
  strip-nonlinetable-debuginfo
  synthetic-counts-propagation
  verify
  wholeprogramdevirt
  dfsan
  asan-module
  msan-module
  tsan-module
  kasan-module
  sancov-module
  memprof-module
  poison-checking
  pseudo-probe-update
Module analyses:
  callgraph
  lcg
  module-summary
  no-op-module
  profile-summary
  stack-safety
  verify
  pass-instrumentation
  asan-globals-md
  inline-advisor
  ir-similarity
  globals-aa
Module alias analyses:
  globals-aa
CGSCC passes:
  argpromotion
  invalidate<all>
  function-attrs
  attributor-cgscc
  inline
  openmp-opt-cgscc
  coro-split
  no-op-cgscc
CGSCC analyses:
  no-op-cgscc
  fam-proxy
  pass-instrumentation
Function passes:
  aa-eval
  adce
  add-discriminators
  aggressive-instcombine
  assume-builder
  assume-simplify
  alignment-from-assumptions
  annotation-remarks
  bdce
  bounds-checking
  break-crit-edges
  callsite-splitting
  consthoist
  constraint-elimination
  chr
  coro-early
  coro-elide
  coro-cleanup
  correlated-propagation
  dce
  dfa-jump-threading
  div-rem-pairs
  dse
  dot-cfg
  dot-cfg-only
  early-cse
  early-cse-memssa
  ee-instrument
  fix-irreducible
  make-guards-explicit
  post-inline-ee-instrument
  gvn-hoist
  gvn-sink
  helloworld
  infer-address-spaces
  instcombine
  instcount
  instsimplify
  invalidate<all>
  irce
  float2int
  no-op-function
  libcalls-shrinkwrap
  lint
  inject-tli-mappings
  instnamer
  loweratomic
  lower-expect
  lower-guard-intrinsic
  lower-constant-intrinsics
  lower-matrix-intrinsics
  lower-matrix-intrinsics-minimal
  lower-widenable-condition
  guard-widening
  load-store-vectorizer
  loop-simplify
  loop-sink
  lowerinvoke
  lowerswitch
  mem2reg
  memcpyopt
  mergeicmps
  mergereturn
  nary-reassociate
  newgvn
  jump-threading
  partially-inline-libcalls
  lcssa
  loop-data-prefetch
  loop-load-elim
  loop-fusion
  loop-distribute
  loop-versioning
  objc-arc
  objc-arc-contract
  objc-arc-expand
  pgo-memop-opt
  print
  print<assumptions>
  print<block-freq>
  print<branch-prob>
  print<da>
  print<divergence>
  print<domtree>
  print<postdomtree>
  print<delinearization>
  print<demanded-bits>
  print<domfrontier>
  print<func-properties>
  print<inline-cost>
  print<inliner-size-estimator>
  print<loops>
  print<memoryssa>
  print<phi-values>
  print<regions>
  print<scalar-evolution>
  print<stack-safety-local>
  print-alias-sets
  print-predicateinfo
  print-mustexecute
  print-memderefs
  reassociate
  redundant-dbg-inst-elim
  reg2mem
  scalarize-masked-mem-intrin
  scalarizer
  separate-const-offset-from-gep
  sccp
  sink
  slp-vectorizer
  slsr
  speculative-execution
  sroa
  strip-gc-relocates
  structurizecfg
  tailcallelim
  unify-loop-exits
  vector-combine
  verify
  verify<domtree>
  verify<loops>
  verify<memoryssa>
  verify<regions>
  verify<safepoint-ir>
  verify<scalar-evolution>
  view-cfg
  view-cfg-only
  transform-warning
  asan
  kasan
  msan
  kmsan
  tsan
  memprof
Function passes with params:
  loop-unroll<O0;O1;O2;O3;full-unroll-max=N;no-partial;partial;no-peeling;peeling;no-profile-peeling;profile-peeling;no-runtime;runtime;no-upperbound;upperbound>
  msan<recover;kernel;track-origins=N>
  simplifycfg<no-forward-switch-cond;forward-switch-cond;no-switch-to-lookup;switch-to-lookup;no-keep-loops;keep-loops;no-hoist-common-insts;hoist-common-insts;no-sink-common-insts;sink-common-insts;bonus-inst-threshold=N>
  loop-vectorize<no-interleave-forced-only;interleave-forced-only;no-vectorize-forced-only;vectorize-forced-only>
  mldst-motion<no-split-footer-bb;split-footer-bb>
  gvn<no-pre;pre;no-load-pre;load-pre;no-split-backedge-load-pre;split-backedge-load-pre;no-memdep;memdep>
  print<stack-lifetime><may;must>
Function analyses:
  aa
  assumptions
  block-freq
  branch-prob
  domtree
  postdomtree
  demanded-bits
  domfrontier
  func-properties
  loops
  lazy-value-info
  da
  inliner-size-estimator
  memdep
  memoryssa
  phi-values
  regions
  no-op-function
  opt-remark-emit
  scalar-evolution
  stack-safety-local
  targetlibinfo
  targetir
  verify
  pass-instrumentation
  divergence
  basic-aa
  cfl-anders-aa
  cfl-steens-aa
  objc-arc-aa
  scev-aa
  scoped-noalias-aa
  tbaa
Function alias analyses:
  basic-aa
  cfl-anders-aa
  cfl-steens-aa
  objc-arc-aa
  scev-aa
  scoped-noalias-aa
  tbaa
Loop passes:
  canon-freeze
  dot-ddg
  invalidate<all>
  licm
  lnicm
  loop-flatten
  loop-idiom
  loop-instsimplify
  loop-interchange
  loop-rotate
  no-op-loop
  print
  loop-deletion
  loop-simplifycfg
  loop-reduce
  indvars
  loop-unroll-and-jam
  loop-unroll-full
  print-access-info
  print<ddg>
  print<iv-users>
  print<loopnest>
  print<loop-cache-cost>
  loop-predication
  guard-widening
  loop-bound-split
  loop-reroll
  loop-versioning-licm
Loop passes with params:
  simple-loop-unswitch<nontrivial;no-nontrivial;trivial;no-trivial>
Loop analyses:
  no-op-loop
  access-info
  ddg
  iv-users
  pass-instrumentation
```

Without any parameters `opt` will output bitcode:

``` shell
$ ./bin/opt example-stress-size8.ll
WARNING: You're attempting to print out a bitcode file.
This is inadvisable as it may cause display problems. If
you REALLY want to taste LLVM bitcode first-hand, you
can force output with the `-f' option.
```

To get LLVM IR as the output, `-S` parameter should be used:

``` shell
$ ./bin/opt example-stress-size8.ll -S -o example-stress-size8-opt.ll
```

The `example-stress-size8-opt.ll` file contains the same contents as `example-stress-size8.ll` since no optimization passes were specified:

``` llvm
; ModuleID = 'example-stress.ll'
source_filename = "/tmp/autogen.bc"

define void @autogen_SD0(i8* %0, i32* %1, i64* %2, i32 %3, i64 %4, i8 %5) {
BB:
  %A4 = alloca <8 x double>, align 64
  %A3 = alloca <2 x i16>, align 4
  %A2 = alloca <4 x i8>, align 4
  %A1 = alloca <16 x i1>, align 16
  %A = alloca <16 x i1>, align 16
  store i8 21, i8* %0, align 1
  store <8 x double> <double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF>, <8 x double>* %A4, align 64
  store i8 77, i8* %0, align 1
  store i8 77, i8* %0, align 1
  ret void
}
```

An interesting pass to try is Aggressive Dead Code Elimination (ADCE) ([documentation](https://llvm.org/docs/Passes.html#adce-aggressive-dead-code-elimination)). It is enabled by adding `-passes` parameter with the value `adce`:

``` shell
$ ./bin/opt example-stress-size8.ll -S -passes adce -o example-stress-size8-adce.ll
```

The contents file `example-stress-size8-adce.ll` are visibly different from `example-stress-size8.ll`:

``` llvm
; ModuleID = 'example-stress.ll'
source_filename = "/tmp/autogen.bc"

define void @autogen_SD0(i8* %0, i32* %1, i64* %2, i32 %3, i64 %4, i8 %5) {
BB:
  %A4 = alloca <8 x double>, align 64
  store i8 21, i8* %0, align 1
  store <8 x double> <double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF>, <8 x double>* %A4, align 64
  store i8 77, i8* %0, align 1
  store i8 77, i8* %0, align 1
  ret void
}
```

Using `llvm-diff` we can see that 4 allocations were removed:

``` shell
$ ./bin/llvm-diff example-stress.ll example-stress-opt.ll
in function autogen_SD0:
  in block %BB:
    in instruction store to %A4 / store to %A4:
      operands %A4 and %A4 differ
    <   %A4 = alloca <8 x double>, align 64
    <   %A3 = alloca <2 x i16>, align 4
    <   %A2 = alloca <4 x i8>, align 4
    <   %A1 = alloca <16 x i1>, align 16
```

Adding the parameter `-time-passes` will cause `opt` to print the pass execution timing report after the optimization is finished:

``` shell
$ ./bin/opt example-stress-size8.ll -S -passes adce -time-passes -o example-stress-size8-adce.ll
===-------------------------------------------------------------------------===
                      ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0002 seconds (0.0002 wall clock)

   ---User Time---   --System Time--   --User+System--   ---Wall Time---  --- Name ---
   0.0001 ( 42.8%)   0.0000 ( 39.1%)   0.0001 ( 42.3%)   0.0001 ( 42.7%)  PrintModulePass
   0.0000 ( 28.3%)   0.0000 ( 26.1%)   0.0001 ( 28.0%)   0.0001 ( 28.3%)  ADCEPass
   0.0000 ( 12.7%)   0.0000 ( 13.0%)   0.0000 ( 12.7%)   0.0000 ( 13.0%)  VerifierPass
   0.0000 (  7.8%)   0.0000 ( 13.0%)   0.0000 (  8.5%)   0.0000 (  8.0%)  VerifierAnalysis
   0.0000 (  8.4%)   0.0000 (  8.7%)   0.0000 (  8.5%)   0.0000 (  8.0%)  PostDominatorTreeAnalysis
   0.0002 (100.0%)   0.0000 (100.0%)   0.0002 (100.0%)   0.0002 (100.0%)  Total

===-------------------------------------------------------------------------===
                                LLVM IR Parsing
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0003 seconds (0.0003 wall clock)

   ---User Time---   --System Time--   --User+System--   ---Wall Time---  --- Name ---
   0.0003 (100.0%)   0.0000 (100.0%)   0.0003 (100.0%)   0.0003 (100.0%)  Parse IR
   0.0003 (100.0%)   0.0000 (100.0%)   0.0003 (100.0%)   0.0003 (100.0%)  Total
```

!!! admonition "Assignment"
    Check the behaviour of this optimization pass on other files created by `llvm-stress`. I partocular, check whether a similar amount of code is deleted and try to explain why or why not by inspecting the files.

!!! admonition "Assignment"
    Many optimization passes are not applicable to many situations. Pick any file created by `llvm-stress` and find two optimization passes that modify the code and two that do not (i.e. the code is invariant in relation to these optimizations).

Optimizations are generally noncommutative, that is, the [order of optimization passes is important](https://llvm.org/docs/Frontend/PerformanceTips.html#pass-ordering). For example, consider the following optimization of  the`example-stress.ll` file in which loop unrolling is performed first, and code sinking second:

``` shell
$ ./bin/opt example-stress.ll -S -passes loop-unroll,sink -o example-stress-loop-unroll-sink.ll
```

and the optimization where code sinking is performed first, and loop unrolling second:

``` shell
$ ./bin/opt example-stress.ll -S -passes sink,loop-unroll -o example-stress-sink-loop-unroll.ll
```

The results of these optimizations are different:

``` shell
./bin/llvm-diff example-stress-loop-unroll-sink.ll example-stress-sink-loop-unroll.ll
in function autogen_SD0:
  in block %CF:
    >   %I15 = insertelement <4 x i32> zeroinitializer, i32 %3, i32 0
    >   %FC = uitofp <4 x i32> %I15 to <4 x double>
  in block %CF83.preheader:
    >   %FC24 = fptoui <4 x double> %FC to <4 x i16>
    <   %I15 = insertelement <4 x i32> zeroinitializer, i32 %3, i32 0
    <   %B16 = and i8 %E, 21
    <   %FC = uitofp <4 x i32> %I15 to <4 x double>
    <   %Shuff21 = shufflevector <8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <8 x i32> <i32 undef, i32 14, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10>
    <   %I22 = insertelement <16 x i1> zeroinitializer, i1 true, i32 14
    <   %B23 = mul <1 x i8> %I, zeroinitializer
    <   %FC24 = fptoui <4 x double> %FC to <4 x i16>
    <   %Sl25 = select i1 true, i8 %E, i8 %B16
  in block %CF85:
    >   %B16 = and i8 %E, 21
    >   %Shuff21 = shufflevector <8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <8 x i32> <i32 undef, i32 14, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10>
    >   %I22 = insertelement <16 x i1> zeroinitializer, i1 true, i32 14
    >   %B23 = mul <1 x i8> %I, zeroinitializer
    >   %Sl25 = select i1 true, i8 %E, i8 %B16
        %L27 = load i8, i8* %0, align 1
        store <2 x i16> %L12, <2 x i16>* %Sl10, align 4
    >   %E28 = extractelement <2 x i64> zeroinitializer, i32 1
        %Shuff29 = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 6, i32 undef, i32 2, i32 4>
    >   %I30 = insertelement <8 x i32> %Shuff21, i32 132849, i32 0
    <   %I30 = insertelement <8 x i32> %Shuff21, i32 132849, i32 0
  in block %CF82.preheader:
    <   %E28 = extractelement <2 x i64> zeroinitializer, i32 1
  in block %CF82:
    >   %I38 = insertelement <4 x double> %FC, double 0xAE97BFB633957A34, i32 1
    >   %Sl40 = select i1 true, <16 x i1> %I22, <16 x i1> %I22
    >   %Cmp41 = icmp uge i8 %Sl25, 21
    >   br i1 %Cmp41, label %CF82.backedge, label %CF89
    <   %I38 = insertelement <4 x double> %FC, double 0xAE97BFB633957A34, i32 1
    <   %Sl40 = select i1 true, <16 x i1> %I22, <16 x i1> %I22
    <   %Cmp41 = icmp uge i8 %Sl25, 21
    <   br i1 %Cmp41, label %CF82.backedge, label %CF89
```

!!! admonition "Assignment"
    Check whether two optimizations that were changing the code chosen in the last assignment can be performed on that code in any order.

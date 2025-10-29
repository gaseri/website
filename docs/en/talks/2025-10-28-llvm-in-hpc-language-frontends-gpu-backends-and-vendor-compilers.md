---
marp: true
theme: default
class: default
author: Vedran Miletić
title: LLVM in HPC – language frontends, GPU backends, and vendor compilers
description: LLVM Meetup in Munich – October 28th, 2025
keywords: compilers, supercomputing
---

# LLVM 🐉 in HPC 🧮 – language frontends, GPU 🎨 backends, and vendor compilers

## [Vedran](https://vedran.miletic.net/) [Miletić](https://www.miletic.net/)

### HPC application expert, Max Planck Computing and Data Facility (MPCDF)

![MPG MPCDF logos](https://www.mpcdf.mpg.de/assets/institutes/headers/mpcdf-desktop-en-bc2a89605e5cb6effc55ad732f103d71afb8c7060ecaa95c5fb93987e4c8acbd.svg)

#### [LLVM Meetup in Munich](https://www.meetup.com/llvm-social-munich/events/311492134/) – [October 28th, 2025](https://discourse.llvm.org/t/llvm-meetup-in-munich-october-28th-2025/88554)

---

<!-- paginate: true -->

## Short background

- former postdoc (Heidelberg Institute for Theoretical Studies)
    - [contributed to Mesa/Clang/LLVM/libclc](../people/principal-investigator.md#open-source-software-contributions) to enable running several OpenCL applications on AMD GPUs (pre-ROCm FOSS stack)
- former junior professor (University of Rijeka, Croatia)
    - taught [Code optimization](../teaching/courses/CO.md) course, inspired in part by [Optimising Compilers](https://www.cl.cam.ac.uk/teaching/2021/OptComp/) (Timothy Jones, Tom Stuart, and Alan Mycroft, University of Cambridge)
- currently mainly working with software running on [AMD Instinct MI300A](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300a.html) APUs at Max Planck, but also other NVIDIA- and Intel-powered machines

---

<!-- class: invert -->

## MPCDF supercomputer Viper, Garching

Image source: [Viper-GPU User Guide](https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide.html)

![Viper bg](https://docs.mpcdf.mpg.de/_images/viper-gpu-2025.jpg)

---

<!-- class: default -->

## Languages for HPC applications

![bg left:45%](https://upload.wikimedia.org/wikipedia/commons/3/39/C_Hello_World_Program.png)

- C, C++, Fortran... Python, Julia, R...
    - Clang, [Flang(-new)](https://blog.llvm.org/posts/2025-03-11-flang-new/)
- OpenCL: portability over ease of use, features, and performance
- OpenMP: ease of use over performance
- CUDA/HIP: full hardware capability
    - [Clang](https://llvm.org/docs/CompileCudaWithLLVM.html)/[Clang](https://clang.llvm.org/docs/HIPSupport.html)
- SYCL: (hopefully) portable full hardware capability

Image source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:C_Hello_World_Program.png)

---

## Why don't GPUs just use C/C++?

- [2024 LLVM Dev Mtg - A C++ Toolchain for Your GPU (Joseph Huber)](https://youtu.be/4TxGWis1mws)
    - [2023 LLVM Dev Mtg - The LLVM C Library for GPUs](https://youtu.be/_LLGc48GYHc)
    - [DOOM on AMDGPU](https://youtu.be/E7X1yvyVml4)

![bg right:55%](https://i3.ytimg.com/vi/E7X1yvyVml4/maxresdefault.jpg)

Image source: [YouTube](https://youtu.be/E7X1yvyVml4)

---

## Backends

- NVPTX
- AMDGPU
- (SPIR-V)
- (DirectX)
- common features: GPU-specific intrinsics, address space management, kernel metadata

![GPU bg left:63%](https://upload.wikimedia.org/wikipedia/commons/8/88/AMD_HD5470_GPU.JPG)

Image source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:AMD_HD5470_GPU.JPG)

---

## NVPTX backend

From [Wikipedia Parallel Thread Execution page](https://en.wikipedia.org/wiki/Parallel_Thread_Execution):

> **Parallel Thread Execution** (PTX or NVPTX) is a low-level parallel thread execution virtual machine and instruction set architecture used in Nvidia's Compute Unified Device Architecture (CUDA) programming environment. The LLVM-based Nvidia CUDA Compiler (NVCC) translates code written in OpenCL C and CUDA C/C++ into PTX instructions (an IL), and the graphics driver contains a compiler which translates PTX instructions into executable binary code, which can run on the processing cores of Nvidia graphics processing units (GPUs).

- used in Clang with CUDA and OpenMP offloading
- [SPIR-V support in AMD ROCm](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.0.0/conceptual/spirv.html) aims to provide similar functionality

---

## AMDGPU backend

- supports (R600), GCN, RDNA, and CDNA generations
- assumptions for [GFX942](https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942)/Instinct MI300A:

> - Each agent has multiple shader arrays (SA).
> - Each SA has multiple compute units (CU).
> - Each CU has multiple SIMDs that execute wavefronts.
> - The wavefronts for a single work-group are executed in the same CU but may be executed by different SIMDs.
> - Each CU has a single LDS memory shared by the wavefronts of the work-groups executing on it.
> - (...)

---

## AMDGPU backend: features

From `llvm/lib/Target/AMDGPU/AMDGPU.td`:

``` c
// Unless +-flat-for-global is specified, turn on FlatForGlobal for
// all OS-es on VI and newer hardware to avoid assertion failures due
// to missing ADDR64 variants of MUBUF instructions.
// FIXME: moveToVALU should be able to handle converting addr64 MUBUF
// instructions.

def FeatureFlatForGlobal : SubtargetFeature<"flat-for-global",
  "FlatForGlobal",
  "true",
  "Force to generate flat instruction for global"
>;
```

---

## AMDGPU backend: generations

``` c
def FeatureGFX9 : GCNSubtargetFeatureGeneration<"GFX9",
  "gfx9",
  [FeatureFP64,
   FeatureWavefrontSize64, FeatureFlatAddressSpace,
   FeatureGCN3Encoding, FeatureCIInsts, Feature16BitInsts,
   FeatureSMemRealTime, FeatureScalarStores, FeatureInv2PiInlineImm,
   FeatureApertureRegs, FeatureGFX9Insts, FeatureVOP3P, FeatureVGPRIndexMode,
   FeatureFastFMAF32, FeatureDPP, FeatureIntClamp,
   FeatureSDWA, FeatureSDWAOmod, FeatureSDWAScalar, FeatureSDWASdst,
   FeatureFlatInstOffsets, FeatureFlatGlobalInsts, FeatureFlatScratchInsts,
   FeatureAddNoCarryInsts, FeatureGFX8Insts, FeatureGFX7GFX8GFX9Insts,
   FeatureScalarFlatScratchInsts, FeatureScalarAtomics, FeatureR128A16,
   FeatureA16, FeatureSMemTimeInst, FeatureFastDenormalF32, FeatureSupportsXNACK,
   FeatureUnalignedBufferAccess, FeatureUnalignedScratchAccess,
   FeatureUnalignedDSAccess, FeatureNegativeScratchOffsetBug, FeatureGWS,
   FeatureDefaultComponentZero,FeatureVmemWriteVgprInOrder, FeatureMemToLDSLoad
  ]
>;
```

---

## AMDGPU backend: lowering 1/2

From `llvm/lib/Target/AMDGPU/AMDGPUISelLowering.cpp`:

``` c++
AMDGPUTargetLowering::AMDGPUTargetLowering(const TargetMachine &TM,
                                           const AMDGPUSubtarget &STI)
    : TargetLowering(TM), Subtarget(&STI) {
  // ...
  setOperationAction(
      {ISD::FLOG, ISD::FLOG10, ISD::FEXP, ISD::FEXP2, ISD::FEXP10}, MVT::f32,
      Custom);
  // ...
  setOperationAction({ISD::FLOG10, ISD::FLOG, ISD::FEXP, ISD::FEXP10}, MVT::f16,
                     Custom);
  // ..
}
```

---

## AMDGPU backend: lowering 2/2

``` c++
SDValue AMDGPUTargetLowering::LowerOperation(SDValue Op,
                                             SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    // ...
  case ISD::FLOG10:
    return LowerFLOGCommon(Op, DAG);
  // ...
  }
}
```

---

## AMDGPU backend: logarithm 1/4

``` c++
SDValue AMDGPUTargetLowering::LowerFLOGCommon(SDValue Op,
                                              SelectionDAG &DAG) const {
  // ...
  const auto &Options = getTargetMachine().Options;
  if (VT == MVT::f16 || Flags.hasApproximateFuncs()) {
    if (VT == MVT::f16 && !Subtarget->has16BitInsts()) {
      // Log and multiply in f32 is good enough for f16.
      X = DAG.getNode(ISD::FP_EXTEND, DL, MVT::f32, X, Flags);
    }
    SDValue Lowered = LowerFLOGUnsafe(X, DL, DAG, IsLog10, Flags);
    if (VT == MVT::f16 && !Subtarget->has16BitInsts()) {
      return DAG.getNode(ISD::FP_ROUND, DL, VT, Lowered,
                         DAG.getTargetConstant(0, DL, MVT::i32), Flags);
    }
    return Lowered;
  }
  // ...
```

---

## AMDGPU backend: logarithm 2/4

``` c++
SDValue AMDGPUTargetLowering::LowerFLOGUnsafe(SDValue Src, const SDLoc &SL,
                                              SelectionDAG &DAG, bool IsLog10,
                                              SDNodeFlags Flags) const {
  EVT VT = Src.getValueType();
  unsigned LogOp =
      VT == MVT::f32 ? (unsigned)AMDGPUISD::LOG : (unsigned)ISD::FLOG2;

  double Log2BaseInverted =
      IsLog10 ? numbers::ln2 / numbers::ln10 : numbers::ln2;
```

---

## AMDGPU backend: logarithm 3/4

``` c++
  if (VT == MVT::f32) {
    auto [ScaledInput, IsScaled] = getScaledLogInput(DAG, SL, Src, Flags);
    if (ScaledInput) {
      SDValue LogSrc = DAG.getNode(AMDGPUISD::LOG, SL, VT, ScaledInput, Flags);
      SDValue ScaledResultOffset =
          DAG.getConstantFP(-32.0 * Log2BaseInverted, SL, VT);

      SDValue Zero = DAG.getConstantFP(0.0f, SL, VT);

      SDValue ResultOffset = DAG.getNode(ISD::SELECT, SL, VT, IsScaled,
                                         ScaledResultOffset, Zero, Flags);

      SDValue Log2Inv = DAG.getConstantFP(Log2BaseInverted, SL, VT);

      if (Subtarget->hasFastFMAF32())
        return DAG.getNode(ISD::FMA, SL, VT, LogSrc, Log2Inv, ResultOffset,
                           Flags);
      SDValue Mul = DAG.getNode(ISD::FMUL, SL, VT, LogSrc, Log2Inv, Flags);
      return DAG.getNode(ISD::FADD, SL, VT, Mul, ResultOffset);
    }
  }
```

---

## AMDGPU backend: logarithm 4/4

``` c++
  SDValue Log2Operand = DAG.getNode(LogOp, SL, VT, Src, Flags);
  SDValue Log2BaseInvertedOperand = DAG.getConstantFP(Log2BaseInverted, SL, VT);

  return DAG.getNode(ISD::FMUL, SL, VT, Log2Operand, Log2BaseInvertedOperand,
                     Flags);
}
```

---

## AMDGPU backend: test 1/2

From `llvm/test/CodeGen/AMDGPU/llvm.log10.ll`:

``` amdgpu
GFX900-SDAG-LABEL: s_log10_f32:
GFX900-SDAG:       ; %bb.0:
GFX900-SDAG-NEXT:    s_load_dword s6, s[4:5], 0x2c
GFX900-SDAG-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
GFX900-SDAG-NEXT:    v_mov_b32_e32 v0, 0x800000
GFX900-SDAG-NEXT:    v_mov_b32_e32 v1, 0x411a209b
GFX900-SDAG-NEXT:    v_mov_b32_e32 v2, 0
GFX900-SDAG-NEXT:    s_waitcnt lgkmcnt(0)
GFX900-SDAG-NEXT:    v_cmp_lt_f32_e32 vcc, s6, v0
GFX900-SDAG-NEXT:    s_and_b64 s[2:3], vcc, exec
GFX900-SDAG-NEXT:    s_cselect_b32 s2, 32, 0
GFX900-SDAG-NEXT:    v_cndmask_b32_e32 v0, 0, v1, vcc
GFX900-SDAG-NEXT:    v_mov_b32_e32 v1, s2
GFX900-SDAG-NEXT:    v_ldexp_f32 v1, s6, v1
; ...
```

---

## AMDGPU backend: test 2/2

``` amdgpu
GFX900-SDAG-NEXT:    v_log_f32_e32 v1, v1
GFX900-SDAG-NEXT:    s_mov_b32 s2, 0x3e9a209a
GFX900-SDAG-NEXT:    s_mov_b32 s3, 0x3284fbcf
GFX900-SDAG-NEXT:    v_mul_f32_e32 v3, 0x3e9a209a, v1
GFX900-SDAG-NEXT:    v_fma_f32 v4, v1, s2, -v3
GFX900-SDAG-NEXT:    v_fma_f32 v4, v1, s3, v4
GFX900-SDAG-NEXT:    s_mov_b32 s2, 0x7f800000
GFX900-SDAG-NEXT:    v_add_f32_e32 v3, v3, v4
GFX900-SDAG-NEXT:    v_cmp_lt_f32_e64 vcc, |v1|, s2
GFX900-SDAG-NEXT:    v_cndmask_b32_e32 v1, v1, v3, vcc
GFX900-SDAG-NEXT:    v_sub_f32_e32 v0, v1, v0
GFX900-SDAG-NEXT:    global_store_dword v2, v0, s[0:1]
GFX900-SDAG-NEXT:    s_endpg
```

---

## Vendor compilers: AMD

![AMD height:100px](https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg)

- AMD-LLVM (FOSS)
    - AMD's fork of LLVM: stays close to usptream, improved OpenMP, heterogenous debugging and address sanitization, hipcc wrapper, ...
- [AMD Optimizing C/C++ and Fortran Compilers (AOCC)](https://www.amd.com/en/developer/aocc.html) (not FOSS)
    - focus on CPU optimizations (primarily for Epyc), used together with [AMD Optimizing CPU Libraries (AOCL)](https://www.amd.com/en/developer/aocl.html)
    - version 5.0, from a year ago, is based on LLVM 17

Image source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:AMD_Logo.svg)

---

## Vendor compilers: Intel

![Intel height:100px](https://upload.wikimedia.org/wikipedia/commons/8/85/Intel_logo_2023.svg)

- a.k.a. IntelLLVM, Intel's fork of LLVM
- oneAPI DPC++ compiler: C, C++, SYCL, OpenMP offload (FOSS)
- custom Fortran compiler (not FOSS)
- oneAPI Math Kernel Library (oneMKL) (not FOSS)
    - has a FOSS implementation named oneMath

Image source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Intel_logo_2023.svg)

---

## Vendor compilers: NVIDIA

![NVIDIA height:100px](https://upload.wikimedia.org/wikipedia/commons/a/a4/NVIDIA_logo.svg)

- NVIDIA HPC compilers, successor to the PGI compilers, based on LLVM
- C, C++, Fortran
    - replace GCC's role in the CUDA stack
- CUDA, OpenACC, OpenMP
- not FOSS

Image source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:NVIDIA_logo.svg)

---

## Why should the application developers care about the compiler?

- (lack of) standards and features support
- (lack of) performance optimizations
- (lack of) warnings about bad code
- (lack of) descriptive error messages
- from this perspective:
    - the addition of Clang/LLVM as a production-ready C/C++ compiler to the FOSS ecosystem a decade ago did great things for code quality and
    - provided vendors with a common and reliable platform to build custom hardware and software-dependant optimizations on top

---

## Thank you for your attention

- Social: Twitter/X [@vedranmiletic](https://x.com/vedranmiletic) ; LinkedIn [vedranmiletic](https://www.linkedin.com/in/vedranmiletic/)
- **E-mail:** <vedran@miletic.net>
- *Web site:* <https://vedran.miletic.net/>

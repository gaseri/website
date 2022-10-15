---
author: Vedran Miletić
---

# The usage of the LLVM libraries in the Mesa 3D graphics library

The [Mesa 3D graphics library](https://www.mesa3d.org/) represents the de facto standard open-source implementation of [OpenGL](https://www.opengl.org/), [OpenGL ES](https://www.khronos.org/opengles/), [OpenCL](https://www.khronos.org/opencl/), [Vulkan](https://www.vulkan.org/), and [other open standards](https://www.khronos.org/developers). Mesa offers [several hardware drivers](https://docs.mesa3d.org/systems.html), including the drivers for several generations of AMD Radeon GPUs. In the following we will focus on [RadeonSI](https://www.phoronix.com/scan.php?page=search&q=RadeonSI), Mesa's OpenGL, OpenGL ES, and OpenCL driver for Graphics Core Next (GCN) and Radeon DNA (RDNA) GPUs. For what it's worth, Mesa's GCN and RDNA Vulkan driver is called [RADV](https://www.phoronix.com/scan.php?page=search&q=RADV); a more detailed overview of the driver structure can be found in the [State of open source AMD GPU drivers](https://2019.dorscluc.org/talk/29/) presentation ([recording](https://youtu.be/cWyBkWkzcYM)).

## Overview of the Graphics Core Next (GCN) and Radeon DNA (RDNA) architecture generations

Generations of the [Graphics Core Next (GCN)](https://en.wikipedia.org/wiki/Graphics_Core_Next) architecture are:

- 1st generation GCN (GFX6): Southern Islands ([Radeon HD 7000](https://en.wikipedia.org/wiki/Radeon_HD_7000_series) and [Radeon HD 8000 series](https://en.wikipedia.org/wiki/Radeon_HD_8000_series); PlayStation 4, Xbox One)
- 2nd generation GCN (GFX7): Sea Islands ([Radeon R5 / R7 / R9 200](https://en.wikipedia.org/wiki/Radeon_Rx_200_series) and [Radeon R5 / R7 / R9 300 series](https://en.wikipedia.org/wiki/Radeon_Rx_300_series))
- 3rd generation GCN (GFX8): Volcanic Islands ([Radeon R9 285](https://en.wikipedia.org/wiki/Radeon_Rx_200_series#Radeon_R9_285), [Radeon R9 380, and R9 Fury](https://en.wikipedia.org/wiki/Radeon_Rx_300_series))
- 4th generation GCN (GFX8): Arctic Islands ([Radeon RX 400 series](https://en.wikipedia.org/wiki/Radeon_RX_400_series), [Radeon RX 500 series](https://en.wikipedia.org/wiki/Radeon_RX_500_series), and [Radeon 600 series](https://en.wikipedia.org/wiki/Radeon_600_series); PlayStation 4 Pro, Xbox One X)
- 5th generation GCN (GFX9): Vega ([Radeon RX Vega series and VII](https://en.wikipedia.org/wiki/Radeon_RX_Vega_series))

Generations of the [Radeon DNA (RDNA)](https://en.wikipedia.org/wiki/RDNA_(microarchitecture)) architecture are:

- 1st generation RDNA (GFX10): Navi ([Radeon RX 5000 series](https://en.wikipedia.org/wiki/Radeon_RX_5000_series))
- 2nd generation RDNA (GFX10): Big Navi ([Radeon RX 6000 series](https://en.wikipedia.org/wiki/Radeon_RX_6000_series); PlayStation 5, Xbox Series X and S)

## Running the OpenCL programs using Clover

[Gallium](https://docs.mesa3d.org/gallium/index.html) is Mesa's driver API that enables drivers for different hardware to share device-agnostic parts of the code. RadeonSI is one of the drivers using Gallium API; [Nouveau](https://nouveau.freedesktop.org/) is another, offering support for [NVIDIA GPUs](https://nouveau.freedesktop.org/CodeNames.html). Both drivers can use Mesa's [Gallium frontends](https://docs.mesa3d.org/gallium/distro.html#gallium-frontends) (also known as state trackers), which implement various standards for 3D graphics, compute, and video decoding acceleration. We are specifically interested in Clover, which is the Gallium frontend for OpenCL.

!!! note
    While Clover on RadeonSI can run many OpenCL programs, it is not a complete implementation of the OpenCL standard; the detailed list of the supported extensions can be found on the [Mesa drivers matrix](https://mesamatrix.net/#OpenCL). There is an ongoing community effort to [improve the Clover frontend](https://cgit.freedesktop.org/mesa/mesa/log/?qt=grep&q=clover) and the RadeonSI driver as well. For a 2016/2017 overview of the work required to make Clover and RadeonSI usable for the scientific computing applications, see the presentations [LLVM AMDGPU for High Performance Computing: are we competitive yet?](https://www.llvm.org/devmtg/2017-03/2017/02/20/accepted-sessions.html#31) ([slides](https://llvm.org/devmtg/2017-03//assets/slides/llvm_admgpu_for_high_performance_computing_are_we_compatitive_yet.pdf), [recording](https://youtu.be/r2Chmg85Xik?list=PL_R5A0lGi1AD12EbUChEnD3s51oqfZLe3)) and [Towards fully open source GPU accelerated molecular dynamics simulation](https://www.llvm.org/devmtg/2016-03/#lightning6) ([slides](https://llvm.org/devmtg/2016-03/Lightning-Talks/miletic-gromacs-amdgpu.pdf), [recording](https://youtu.be/TkanbGAG_Fo?t=23m47s&list=PL_R5A0lGi1ADuZKWUJOVOgXr2dRW06e55)) by the author of these exercises.

We will start by running [clinfo](https://github.com/Oblomov/clinfo), a simple OpenCL program that prints out all known properties of all OpenCL platforms and devices in the system. Using the `--version` parameter we'll make sure that the `clinfo` command is working properly and that a recent version is being used:

``` shell
$ clinfo --version
clinfo version 3.0.21.02.21
```

When `--list` parameter is specified, `clinfo` will print the list of OpenCL platforms and devices on each of the platforms:

``` shell
$ clinfo --list
Platform #0: Clover
 `-- Device #0: AMD Radeon RX 6800 (SIENNA_CICHLID, DRM 3.44.0, 5.16.12-zen1-1-zen, LLVM 13.0.1)
```

We can see that we have only one platform (Clover) and only one device (Radeon RX 6800, 2nd generation RDNA GPU codenamed [Sienna Cichild](https://videocardz.com/newz/amd-sienna-cichlid-confirmed-as-navi-21-navy-flounder-is-navi-22)). Running `clinfo` command without parameters will make it print the platform and device properties:

``` shell
$ clinfo
Number of platforms                               1
  Platform Name                                   Clover
  Platform Vendor                                 Mesa
  Platform Version                                OpenCL 1.1 Mesa 21.3.7
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_icd
  Platform Extensions function suffix             MESA

  Platform Name                                   Clover
Number of devices                                 1
  Device Name                                     AMD Radeon RX 6800 (SIENNA_CICHLID, DRM 3.44.0, 5.16.12-zen1-1-zen, LLVM 13.0.1)
  Device Vendor                                   AMD
  Device Vendor ID                                0x1002
  Device Version                                  OpenCL 1.1 Mesa 21.3.7
  Device Numeric Version                          0x401000 (1.1.0)
  Driver Version                                  21.3.7
  Device OpenCL C Version                         OpenCL C 1.1
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Device Available                                Yes
  Compiler Available                              Yes
  Max compute units                               60
  Max clock frequency                             2475MHz
  Max work item dimensions                        3
  Max work item sizes                             256x256x256
  Max work group size                             256
  Preferred work group size multiple (kernel)     64
  Preferred / native vector sizes
    char                                                16 / 16
    short                                                8 / 8
    int                                                  4 / 4
    long                                                 2 / 2
    half                                                 0 / 0        (n/a)
    float                                                4 / 4
    double                                               2 / 2        (cl_khr_fp64)
  Half-precision Floating-point support           (n/a)
  Single-precision Floating-point support         (core)
    Denormals                                     No
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 No
    Round to infinity                             No
    IEEE754-2008 fused multiply-add               No
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  No
  Double-precision Floating-point support         (cl_khr_fp64)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
  Address bits                                    64, Little-Endian
  Global memory size                              17179869184 (16GiB)
  Error Correction support                        No
  Max memory allocation                           13743895347 (12.8GiB)
  Unified memory for Host and Device              No
  Minimum alignment for any data type             128 bytes
  Alignment of base address                       32768 bits (4096 bytes)
  Global Memory cache type                        None
  Image support                                   No
  Local memory type                               Local
  Local memory size                               32768 (32KiB)
  Max number of constant args                     16
  Max constant buffer size                        67108864 (64MiB)
  Max size of kernel argument                     1024
  Queue properties
    Out-of-order execution                        No
    Profiling                                     Yes
  Profiling timer resolution                      0ns
  Execution capabilities
    Run OpenCL kernels                            Yes
    Run native kernels                            No
    ILs with version                              (n/a)
  Built-in kernels with version                   (n/a)
  Device Extensions                               cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_fp64 cl_khr_extended_versioning
  Device Extensions with Version                  cl_khr_byte_addressable_store                                    0x400000 (1.0.0)
                                                  cl_khr_global_int32_base_atomics                                 0x400000 (1.0.0)
                                                  cl_khr_global_int32_extended_atomics                             0x400000 (1.0.0)
                                                  cl_khr_local_int32_base_atomics                                  0x400000 (1.0.0)
                                                  cl_khr_local_int32_extended_atomics                              0x400000 (1.0.0)
                                                  cl_khr_int64_base_atomics                                        0x400000 (1.0.0)
                                                  cl_khr_int64_extended_atomics                                    0x400000 (1.0.0)
                                                  cl_khr_fp64                                                      0x400000 (1.0.0)
                                                  cl_khr_extended_versioning                                       0x400000 (1.0.0)

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  Clover
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   Success [MESA]
  clCreateContext(NULL, ...) [default]            Success [MESA]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_DEFAULT)  Success (1)
    Platform Name                                 Clover
    Device Name                                   AMD Radeon RX 6800 (SIENNA_CICHLID, DRM 3.44.0, 5.16.12-zen1-1-zen, LLVM 13.0.1)
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  Success (1)
    Platform Name                                 Clover
    Device Name                                   AMD Radeon RX 6800 (SIENNA_CICHLID, DRM 3.44.0, 5.16.12-zen1-1-zen, LLVM 13.0.1)
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  Success (1)
    Platform Name                                 Clover
    Device Name                                   AMD Radeon RX 6800 (SIENNA_CICHLID, DRM 3.44.0, 5.16.12-zen1-1-zen, LLVM 13.0.1)

ICD loader properties
  ICD loader Name                                 OpenCL ICD Loader
  ICD loader Vendor                               OCL Icd free software
  ICD loader Version                              2.3.1
  ICD loader Profile                              OpenCL 3.0
```

We can see LLVM 13.0.1 mentioned in several places. Clang and LLVM are used by Clover for compiling the OpenCL C code to assembly code for the gfx1030 processor, which is a part of the Radeon RX 6800 GPU. The resulting assembly code is then linked with [libclc](https://libclc.llvm.org/) that contains the implementations of the fundamental OpenCL C data types and functions. Finally, the resulting code after linking is executed by RadeonSI on the Radeon 6800 GPU.

!!! admonition "Assignment"
    Compare the output of `clinfo` on your machine to the output shown above. (If you do not posses a GPU with an OpenCL driver, use [Portable Computing Language](http://portablecl.org/) ([GitHub](https://github.com/pocl/pocl)) to run OpenCL on the CPU.)

### Using the environment variables

Mesa supports [many environment variables](https://docs.mesa3d.org/envvars.html) which can be used for debugging purposes as well as learning how the compilation. [Clover frontend environment variables](https://docs.mesa3d.org/envvars.html#clover-environment-variables) are:

> - `CLOVER_EXTRA_BUILD_OPTIONS` allows specifying additional compiler and linker options. Specified options are appended after the options set by the OpenCL program in `clBuildProgram`.
> - `CLOVER_EXTRA_COMPILE_OPTIONS` allows specifying additional compiler options. Specified options are appended after the options set by the OpenCL program in `clCompileProgram`.
> - `CLOVER_EXTRA_LINK_OPTIONS` allows specifying additional linker options. Specified options are appended after the options set by the OpenCL program in `clLinkProgram`.

[RadeonSI driver environment variable](https://docs.mesa3d.org/envvars.html#radeonsi-driver-environment-variables) `AMD_DEBUG` has several interesting options, including `preoptir`, which prints the LLVM intermediate representation before initial optimizations, and `gisel`, which enables the LLVM [global instruction selector](https://llvm.org/devmtg/2015-10/slides/Colombet-GlobalInstructionSelection.pdf).

## Compiling the OpenCL programs with Clang

For convenience, we will be compiling the OpenCL programs with standalone Clang. An example OpenCL kernel that performs vector addition is as follows:

``` c
__kernel void vector_add(__global const int *a, __global const int *b, __global int *c) {

    // Get the global identifier of the thread
    int i = get_global_id(0);

    // Perform addition on elements at index i
    c[i] = a[i] + b[i];
}
```

Save this kernel in a file named `vecadd.cl`. In order to compile it, we will use the following parameters we have not used before:

- `-x cl` tells Clang to treat the input file as being written in OpenCL ([documentation](https://clang.llvm.org/docs/CommandGuide/clang.html#cmdoption-x))
- `-D cl_clang_storage_class_specifiers` enables the usage of the OpenCL C [storage-class specifiers/qualifiers](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/storageQualifiers.html) (`typedef`, `static`, and `extern`; `auto` and `register` are not supported) ([documentation](https://clang.llvm.org/docs/CommandGuide/clang.html#cmdoption-d-macroname))
- `-isystem libclc/generic/include` adds the system include directories for libclc ([documentation](https://clang.llvm.org/docs/CommandGuide/clang.html#cmdoption-i-directory))
- `-include clc/clc.h` includes the `clc/clc.h` file ([documentation](https://clang.llvm.org/docs/CommandGuide/clang.html#cmdoption-include))

``` shell
$ ./bin/clang -x cl vecadd.cl -S -target amdgcn--amdhsa -D cl_clang_storage_class_specifiers -isystem libclc/generic/include -include clc/clc.h
```

!!! note
    The parameter `-emit-llvm` can be used in addition to `-S` to make Clang write the LLVM intermediate representation instead of the assembly, just like we have used it previously.

The resulting file is named `vecadd.s`. Let's take a look at its contents:

``` asm
    .text
    .amdgcn_target "amdgcn-unknown-amdhsa--gfx700"
    .protected    vector_add              ; -- Begin function vector_add
    .globl    vector_add
    .p2align    8
    .type    vector_add,@function
vector_add:                             ; @vector_add
; %bb.0:
    s_mov_b32 s32, 0
    s_mov_b32 s33, 0
    s_mov_b32 flat_scratch_lo, s7
    s_add_i32 s6, s6, s9
    s_lshr_b32 flat_scratch_hi, s6, 8
    s_add_u32 s0, s0, s9
    s_addc_u32 s1, s1, 0
    s_load_dwordx4 s[36:39], s[4:5], 0x0
    s_load_dwordx2 s[34:35], s[4:5], 0x4
    s_getpc_b64 s[4:5]
    s_add_u32 s4, s4, _Z13get_global_idj@rel32@lo+4
    s_addc_u32 s5, s5, _Z13get_global_idj@rel32@hi+12
    v_mov_b32_e32 v0, 0
    s_swappc_b64 s[30:31], s[4:5]
    v_mov_b32_e32 v1, v0
    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v3, s37
    v_mov_b32_e32 v5, s39
    v_ashr_i64 v[0:1], v[0:1], 30
    v_add_i32_e32 v2, vcc, s36, v0
    v_addc_u32_e32 v3, vcc, v3, v1, vcc
    v_add_i32_e32 v4, vcc, s38, v0
    v_addc_u32_e32 v5, vcc, v5, v1, vcc
    flat_load_dword v2, v[2:3]
    flat_load_dword v3, v[4:5]
    v_mov_b32_e32 v4, s35
    s_waitcnt vmcnt(0)
    v_add_i32_e32 v2, vcc, v3, v2
    v_add_i32_e32 v0, vcc, s34, v0
    v_addc_u32_e32 v1, vcc, v4, v1, vcc
    flat_store_dword v[0:1], v2
    s_endpgm
    .section    .rodata,#alloc
    .p2align    6
    .amdhsa_kernel vector_add
        .amdhsa_group_segment_fixed_size 0
        .amdhsa_private_segment_fixed_size 16384
        .amdhsa_kernarg_size 80
        .amdhsa_user_sgpr_private_segment_buffer 1
        .amdhsa_user_sgpr_dispatch_ptr 0
        .amdhsa_user_sgpr_queue_ptr 0
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_user_sgpr_dispatch_id 0
        .amdhsa_user_sgpr_flat_scratch_init 1
        .amdhsa_user_sgpr_private_segment_size 0
        .amdhsa_system_sgpr_private_segment_wavefront_offset 1
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 0
        .amdhsa_system_sgpr_workgroup_id_z 0
        .amdhsa_system_sgpr_workgroup_info 0
        .amdhsa_system_vgpr_workitem_id 0
        .amdhsa_next_free_vgpr 6
        .amdhsa_next_free_sgpr 40
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 3
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_dx10_clamp 1
        .amdhsa_ieee_mode 1
        .amdhsa_exception_fp_ieee_invalid_op 0
        .amdhsa_exception_fp_denorm_src 0
        .amdhsa_exception_fp_ieee_div_zero 0
        .amdhsa_exception_fp_ieee_overflow 0
        .amdhsa_exception_fp_ieee_underflow 0
        .amdhsa_exception_fp_ieee_inexact 0
        .amdhsa_exception_int_div_zero 0
    .end_amdhsa_kernel
    .text
.Lfunc_end0:
    .size    vector_add, .Lfunc_end0-vector_add
                                        ; -- End function
    .section    .AMDGPU.csdata
; Kernel info:
; codeLenInByte = 152
; NumSgprs: 44
; NumVgprs: 6
; ScratchSize: 16384
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 6
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
    .hidden    _Z13get_global_idj
    .ident    "clang version 13.0.1"
    .section    ".note.GNU-stack"
    .addrsig
    .amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .is_const:       true
        .offset:         0
        .size:           8
        .type_name:      'int*'
        .value_kind:     global_buffer
      - .address_space:  global
        .is_const:       true
        .offset:         8
        .size:           8
        .type_name:      'int*'
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .type_name:      'int*'
        .value_kind:     global_buffer
      - .offset:         24
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         32
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .language:       OpenCL C
    .language_version:
      - 1
      - 2
    .max_flat_workgroup_size: 256
    .name:           vector_add
    .private_segment_fixed_size: 16384
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         vector_add.kd
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-unknown-amdhsa--gfx700
amdhsa.version:
  - 1
  - 1
...

    .end_amdgpu_metadata
```

Common GCN and RDNA assembly instructions can be divided into two groups, scalar (names starting with `s_`) and vector (names starting with `v_`). Scalar instructions use scalar general-purpose registers (SGPRs, named `s1`, `s2` etc.), while vector instructions use (VGPRs, named `v1`, `v2` etc.).

!!! admonition "Assignment"
    The assembly code above contains, among other, the assembly code produced by from the OpenCL C code lines `int i = get_global_id(0);` and `c[i] = a[i] + b[i];`. Figure out which lines in the assembly code correspond to each of the two lines and classify them as scalar or vector instructions.

!!! admonition "Assignment"
    Modify the OpenCL C kernel to compute the sum of the three vectors instead of two and compile it into the assembly code. Compare the two resulting assembly codes in terms of code size and register usage.

!!! tip
    For details on how to compile with Clang for different GPUs, a good starting point is the article titled [Compile OpenCL Kernel into LLVM-IR or Nvidia PTX](https://blog.modest-destiny.com/posts/compile-opencl-kernel-into-llvm-ir-or-nvidia-ptx/) written by Adam Basfop Cavendish and posted on his blog named [Modest Destiny](https://blog.modest-destiny.com/).

### llc

[llc](https://llvm.org/docs/CommandGuide/llc.html) compiles the LLVM intermediate representation into the assembly language of any of the processor architectures supported by LLVM.

The list of supported architectures can be obtained using the `--version` parameter:

``` shell
$ ./bin/llc --version
LLVM (http://llvm.org/):
  LLVM version 13.0.1
  Optimized build.
  Default target: x86_64-unknown-linux-gnu
  Host CPU: skylake

  Registered Targets:
    aarch64    - AArch64 (little endian)
    aarch64_32 - AArch64 (little endian ILP32)
    aarch64_be - AArch64 (big endian)
    amdgcn     - AMD GCN GPUs
    arm        - ARM
    arm64      - ARM64 (little endian)
    arm64_32   - ARM64 (little endian ILP32)
    armeb      - ARM (big endian)
    avr        - Atmel AVR Microcontroller
    bpf        - BPF (host endian)
    bpfeb      - BPF (big endian)
    bpfel      - BPF (little endian)
    hexagon    - Hexagon
    lanai      - Lanai
    mips       - MIPS (32-bit big endian)
    mips64     - MIPS (64-bit big endian)
    mips64el   - MIPS (64-bit little endian)
    mipsel     - MIPS (32-bit little endian)
    msp430     - MSP430 [experimental]
    nvptx      - NVIDIA PTX 32-bit
    nvptx64    - NVIDIA PTX 64-bit
    ppc32      - PowerPC 32
    ppc32le    - PowerPC 32 LE
    ppc64      - PowerPC 64
    ppc64le    - PowerPC 64 LE
    r600       - AMD GPUs HD2XXX-HD6XXX
    riscv32    - 32-bit RISC-V
    riscv64    - 64-bit RISC-V
    sparc      - Sparc
    sparcel    - Sparc LE
    sparcv9    - Sparc V9
    systemz    - SystemZ
    thumb      - Thumb
    thumbeb    - Thumb (big endian)
    wasm32     - WebAssembly 32-bit
    wasm64     - WebAssembly 64-bit
    x86        - 32-bit X86: Pentium-Pro and above
    x86-64     - 64-bit X86: EM64T and AMD64
    xcore      - XCore
```

Note the `amdgcn` entry in the list of the LLVM's registered target architectures, i.e. AMD's graphics processors based on the [Graphics Core Next architecture (GCN)](https://www.amd.com/en/technologies/gcn). As the instructions of the [Radeon DNA (RDNA)](https://www.amd.com/en/technologies/rdna) architecture are very similar to the instructions of the GCN architecture, the same LLVM backend is also used for RDNA, despite the name perhaps suggesting otherwise. The situation is similar with older graphics processors: the `r600` backend supports the R600 architecture (marketing names Radeon HD 2000 and Radeon HD 3000) as well as R700 (Radeon HD 4000 series), Evergreen (Radeon HD 5000 series), and Northern Islands (Radeon HD 6000 series) architectures.

Each architecture generation has a number of processors. The [official documentation of the LLVM AMDGPU backend](https://llvm.org/docs/AMDGPUUsage.html) contains the list of [processors](https://llvm.org/docs/AMDGPUUsage.html#processors) and [features](https://llvm.org/docs/AMDGPUUsage.html#target-features). In addition, the list of the supported processors and features of the target can be obtained using the `llc` command with `-march` and `-mattr` parameters:

``` shell
./bin/llc -march=amdgcn -mattr=help
Available CPUs for this target:

  bonaire     - Select the bonaire processor.
  carrizo     - Select the carrizo processor.
  fiji        - Select the fiji processor.
  generic     - Select the generic processor.
  generic-hsa - Select the generic-hsa processor.
  gfx1010     - Select the gfx1010 processor.
  gfx1011     - Select the gfx1011 processor.
  gfx1012     - Select the gfx1012 processor.
  gfx1013     - Select the gfx1013 processor.
  gfx1030     - Select the gfx1030 processor.
  gfx1031     - Select the gfx1031 processor.
  gfx1032     - Select the gfx1032 processor.
  gfx1033     - Select the gfx1033 processor.
  gfx1034     - Select the gfx1034 processor.
  gfx1035     - Select the gfx1035 processor.
  gfx600      - Select the gfx600 processor.
  gfx601      - Select the gfx601 processor.
  gfx602      - Select the gfx602 processor.
  gfx700      - Select the gfx700 processor.
  gfx701      - Select the gfx701 processor.
  gfx702      - Select the gfx702 processor.
  gfx703      - Select the gfx703 processor.
  gfx704      - Select the gfx704 processor.
  gfx705      - Select the gfx705 processor.
  gfx801      - Select the gfx801 processor.
  gfx802      - Select the gfx802 processor.
  gfx803      - Select the gfx803 processor.
  gfx805      - Select the gfx805 processor.
  gfx810      - Select the gfx810 processor.
  gfx900      - Select the gfx900 processor.
  gfx902      - Select the gfx902 processor.
  gfx904      - Select the gfx904 processor.
  gfx906      - Select the gfx906 processor.
  gfx908      - Select the gfx908 processor.
  gfx909      - Select the gfx909 processor.
  gfx90a      - Select the gfx90a processor.
  gfx90c      - Select the gfx90c processor.
  hainan      - Select the hainan processor.
  hawaii      - Select the hawaii processor.
  iceland     - Select the iceland processor.
  kabini      - Select the kabini processor.
  kaveri      - Select the kaveri processor.
  mullins     - Select the mullins processor.
  oland       - Select the oland processor.
  pitcairn    - Select the pitcairn processor.
  polaris10   - Select the polaris10 processor.
  polaris11   - Select the polaris11 processor.
  stoney      - Select the stoney processor.
  tahiti      - Select the tahiti processor.
  tonga       - Select the tonga processor.
  tongapro    - Select the tongapro processor.
  verde       - Select the verde processor.

Available features for this target:

  16-bit-insts                          - Has i16/f16 instructions.
  DumpCode                              - Dump MachineInstrs in the CodeEmitter.
  a16                                   - Support gfx10-style A16 for 16-bit coordinates/gradients/lod/clamp/mip image operands.
  add-no-carry-insts                    - Have VALU add/sub instructions without carry out.
  aperture-regs                         - Has Memory Aperture Base and Size Registers.
  architected-flat-scratch              - Flat Scratch register is a readonly SPI initialized architected register.
  atomic-fadd-insts                     - Has buffer_atomic_add_f32, buffer_atomic_pk_add_f16, global_atomic_add_f32, global_atomic_pk_add_f16 instructions.
  auto-waitcnt-before-barrier           - Hardware automatically inserts waitcnt before barrier.
  ci-insts                              - Additional instructions for CI+.
  cumode                                - Enable CU wavefront execution mode.
  dl-insts                              - Has v_fmac_f32 and v_xnor_b32 instructions.
  dot1-insts                            - Has v_dot4_i32_i8 and v_dot8_i32_i4 instructions.
  dot2-insts                            - Has v_dot2_i32_i16, v_dot2_u32_u16 instructions.
  dot3-insts                            - Has v_dot8c_i32_i4 instruction.
  dot4-insts                            - Has v_dot2c_i32_i16 instruction.
  dot5-insts                            - Has v_dot2c_f32_f16 instruction.
  dot6-insts                            - Has v_dot4c_i32_i8 instruction.
  dot7-insts                            - Has v_dot2_f32_f16, v_dot4_u32_u8, v_dot8_u32_u4 instructions.
  dpp                                   - Support DPP (Data Parallel Primitives) extension.
  dpp-64bit                             - Support DPP (Data Parallel Primitives) extension.
  dpp8                                  - Support DPP8 (Data Parallel Primitives) extension.
  ds-src2-insts                         - Has ds_*_src2 instructions.
  dumpcode                              - Dump MachineInstrs in the CodeEmitter.
  enable-ds128                          - Use ds_{read|write}_b128.
  enable-prt-strict-null                - Enable zeroing of result registers for sparse texture fetches.
  extended-image-insts                  - Support mips != 0, lod != 0, gather4, and get_lod.
  fast-denormal-f32                     - Enabling denormals does not cause f32 instructions to run at f64 rates.
  fast-fmaf                             - Assuming f32 fma is at least as fast as mul + add.
  flat-address-space                    - Support flat address space.
  flat-for-global                       - Force to generate flat instruction for global.
  flat-global-insts                     - Have global_* flat memory instructions.
  flat-inst-offsets                     - Flat instructions have immediate offset addressing mode.
  flat-scratch-insts                    - Have scratch_* flat memory instructions.
  flat-segment-offset-bug               - GFX10 bug where inst_offset is ignored when flat instructions access global memory.
  fma-mix-insts                         - Has v_fma_mix_f32, v_fma_mixlo_f16, v_fma_mixhi_f16 instructions.
  fmaf                                  - Enable single precision FMA (not as fast as mul+add, but fused).
  fp64                                  - Enable double precision operations.
  full-rate-64-ops                      - Most fp64 instructions are full rate.
  g16                                   - Support G16 for 16-bit gradient image operands.
  gcn3-encoding                         - Encoding format for VI.
  get-wave-id-inst                      - Has s_get_waveid_in_workgroup instruction.
  gfx10                                 - GFX10 GPU generation.
  gfx10-3-insts                         - Additional instructions for GFX10.3.
  gfx10-insts                           - Additional instructions for GFX10+.
  gfx10_a-encoding                      - Has BVH ray tracing instructions.
  gfx10_b-encoding                      - Encoding format GFX10_B.
  gfx7-gfx8-gfx9-insts                  - Instructions shared in GFX7, GFX8, GFX9.
  gfx8-insts                            - Additional instructions for GFX8+.
  gfx9                                  - GFX9 GPU generation.
  gfx9-insts                            - Additional instructions for GFX9+.
  gfx90a-insts                          - Additional instructions for GFX90A+.
  half-rate-64-ops                      - Most fp64 instructions are half rate instead of quarter.
  image-gather4-d16-bug                 - Image Gather4 D16 hardware bug.
  image-store-d16-bug                   - Image Store D16 hardware bug.
  inst-fwd-prefetch-bug                 - S_INST_PREFETCH instruction causes shader to hang.
  int-clamp-insts                       - Support clamp for integer destination.
  inv-2pi-inline-imm                    - Has 1 / (2 * pi) as inline immediate.
  lds-branch-vmem-war-hazard            - Switching between LDS and VMEM-tex not waiting VM_VSRC=0.
  lds-misaligned-bug                    - Some GFX10 bug with multi-dword LDS and flat access that is not naturally aligned in WGP mode.
  ldsbankcount16                        - The number of LDS banks per compute unit..
  ldsbankcount32                        - The number of LDS banks per compute unit..
  load-store-opt                        - Enable SI load/store optimizer pass.
  localmemorysize0                      - The size of local memory in bytes.
  localmemorysize32768                  - The size of local memory in bytes.
  localmemorysize65536                  - The size of local memory in bytes.
  mad-mac-f32-insts                     - Has v_mad_f32/v_mac_f32/v_madak_f32/v_madmk_f32 instructions.
  mad-mix-insts                         - Has v_mad_mix_f32, v_mad_mixlo_f16, v_mad_mixhi_f16 instructions.
  mai-insts                             - Has mAI instructions.
  max-private-element-size-16           - Maximum private access size may be 16.
  max-private-element-size-4            - Maximum private access size may be 4.
  max-private-element-size-8            - Maximum private access size may be 8.
  mfma-inline-literal-bug               - MFMA cannot use inline literal as SrcC.
  mimg-r128                             - Support 128-bit texture resources.
  movrel                                - Has v_movrel*_b32 instructions.
  negative-scratch-offset-bug           - Negative immediate offsets in scratch instructions with an SGPR offset page fault on GFX9.
  negative-unaligned-scratch-offset-bug - Scratch instructions with a VGPR offset and a negative immediate offset that is not a multiple of 4 read wrong memory on GFX10.
  no-data-dep-hazard                    - Does not need SW waitstates.
  no-sdst-cmpx                          - V_CMPX does not write VCC/SGPR in addition to EXEC.
  nsa-clause-bug                        - MIMG-NSA in a hard clause has unpredictable results on GFX10.1.
  nsa-encoding                          - Support NSA encoding for image instructions.
  nsa-max-size-13                       - The maximum non-sequential address size in VGPRs..
  nsa-max-size-5                        - The maximum non-sequential address size in VGPRs..
  nsa-to-vmem-bug                       - MIMG-NSA followed by VMEM fail if EXEC_LO or EXEC_HI equals zero.
  offset-3f-bug                         - Branch offset of 3f hardware bug.
  packed-fp32-ops                       - Support packed fp32 instructions.
  packed-tid                            - Workitem IDs are packed into v0 at kernel launch.
  pk-fmac-f16-inst                      - Has v_pk_fmac_f16 instruction.
  promote-alloca                        - Enable promote alloca pass.
  r128-a16                              - Support gfx9-style A16 for 16-bit coordinates/gradients/lod/clamp/mip image operands, where a16 is aliased with r128.
  register-banking                      - Has register banking.
  s-memrealtime                         - Has s_memrealtime instruction.
  s-memtime-inst                        - Has s_memtime instruction.
  scalar-atomics                        - Has atomic scalar memory instructions.
  scalar-flat-scratch-insts             - Have s_scratch_* flat memory instructions.
  scalar-stores                         - Has store scalar memory instructions.
  sdwa                                  - Support SDWA (Sub-DWORD Addressing) extension.
  sdwa-mav                              - Support v_mac_f32/f16 with SDWA (Sub-DWORD Addressing) extension.
  sdwa-omod                             - Support OMod with SDWA (Sub-DWORD Addressing) extension.
  sdwa-out-mods-vopc                    - Support clamp for VOPC with SDWA (Sub-DWORD Addressing) extension.
  sdwa-scalar                           - Support scalar register with SDWA (Sub-DWORD Addressing) extension.
  sdwa-sdst                             - Support scalar dst for VOPC with SDWA (Sub-DWORD Addressing) extension.
  sea-islands                           - SEA_ISLANDS GPU generation.
  sgpr-init-bug                         - VI SGPR initialization bug requiring a fixed SGPR allocation size.
  shader-cycles-register                - Has SHADER_CYCLES hardware register.
  si-scheduler                          - Enable SI Machine Scheduler.
  smem-to-vector-write-hazard           - s_load_dword followed by v_cmp page faults.
  southern-islands                      - SOUTHERN_ISLANDS GPU generation.
  sramecc                               - Enable SRAMECC.
  sramecc-support                       - Hardware supports SRAMECC.
  tgsplit                               - Enable threadgroup split execution.
  trap-handler                          - Trap handler support.
  trig-reduced-range                    - Requires use of fract on arguments to trig instructions.
  unaligned-access-mode                 - Enable unaligned global, local and region loads and stores if the hardware supports it.
  unaligned-buffer-access               - Hardware supports unaligned global loads and stores.
  unaligned-ds-access                   - Hardware supports unaligned local and region loads and stores.
  unaligned-scratch-access              - Support unaligned scratch loads and stores.
  unpacked-d16-vmem                     - Has unpacked d16 vmem instructions.
  unsafe-ds-offset-folding              - Force using DS instruction immediate offsets on SI.
  vcmpx-exec-war-hazard                 - V_CMPX WAR hazard on EXEC (V_CMPX issue ONLY).
  vcmpx-permlane-hazard                 - TODO: describe me.
  vgpr-index-mode                       - Has VGPR mode register indexing.
  vmem-to-scalar-write-hazard           - VMEM instruction followed by scalar writing to EXEC mask, M0 or SGPR leads to incorrect execution..
  volcanic-islands                      - VOLCANIC_ISLANDS GPU generation.
  vop3-literal                          - Can use one literal in VOP3.
  vop3p                                 - Has VOP3P packed instructions.
  vscnt                                 - Has separate store vscnt counter.
  wavefrontsize16                       - The number of threads per wavefront.
  wavefrontsize32                       - The number of threads per wavefront.
  wavefrontsize64                       - The number of threads per wavefront.
  xnack                                 - Enable XNACK support.
  xnack-support                         - Hardware supports XNACK.

Use +feature to enable a feature, or -feature to disable it.
For example, llc -mcpu=mycpu -mattr=+feature1,-feature2
```

!!! tip
    A good starting point for further study of AMDGPU backend is [Tom Stellard](https://www.stellard.net/tom/blog/)'s [A Detailed Look at the R600 Backend](https://llvm.org/devmtg/2013-11/#talk7) ([slides](https://llvm.org/devmtg/2013-11/slides/Stellard-R600.pdf), [recording](https://youtu.be/hz1jFSi1fEY?list=PL_R5A0lGi1AA4GNONa4vof63jalYbs-MG)), presented at [2013 LLVM Developers' Meeting](https://llvm.org/devmtg/2013-11/). While focused on R600 and not GCN, many of the points made in the talk still hold.

!!! admonition "Assignment"
    Use Clang to compile the OpenCL C code from the previous example to LLVM intermediate representation and then use `llc` to compile it for `gfx700`, `fiji`, and `gfx1030` processors. Compare the resulting assembly codes in terms of the code size, types of instructions used as well as register pressure.

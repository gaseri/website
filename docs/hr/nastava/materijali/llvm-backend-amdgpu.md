---
author: Vedran Miletić
---

# Prevođenje OpenCL C koda u asemblerski kod arhitektura AMD GCN i RDNA

## Arhitekture i procesori

Korištenjem već ranije spomenutog LLVM-ovog alata llc možemo saznati koje su podržane ciljne arhitekture.

``` shell
$ llc --version
LLVM (http://llvm.org/):
  LLVM version 11.0.0

  Optimized build.
  Default target: x86_64-pc-linux-gnu
  Host CPU: core2

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

Uočimo da u popisu registriranih ciljnih arhitektura u koje LLVM može prevesti kod postoji `amdgcn`, odnosno AMD-ovi grafički procesori temeljeni na arhitekturi Graphics Core Next (GCN). Kako su instrukcije AMD-ove arhitekture Radeon DNA (RDNA) vrlo slične instrukcijama arhitekture GCN, isti pozadinski dio program prevoditelja koristi se i za RDNA, iako naziv sugerira drugačije. Vrijedi spomenuti da je slična situacija kod starijih grafičkih procesora pa `r600` osim Radeona R600 (marketinški serije HD 2000 i HD 3000) podržava i R700 (serija HD 4000), Evergreen (serija HD 5000) i Northern Islands (serija HD 6000).

Generacije arhitekture GCN su redom:

- GCN1 (GFX6): Southern Islands (serije HD 7000 i HD 8000; PlayStation 4, Xbox One)
- GCN2 (GFX7): Sea Islands (serije R5/R7/R9 200 i R5/R7/R9 300)
- GCN3 (GFX8): Volcanic Islands (R9 285, R9 380, R9 Fury)
- GCN4 (GFX8): Arctic Islands (serije RX 400, RX 500, RX 600; PlayStation 4 Pro, Xbox One X)
- GCN5 (GFX9): Vega (serija RX Vega i VII)

Generacije arhitekture RDNA su redom:

- RDNA1 (GFX10): Navi (serija RX 5000)
- RDNA2 (GFX11): Big Navi (serija RX 6000; PlayStation 5, Xbox Series X i S)

Popis podržanih procesora i značajki možemo također dobiti korištenjem naredbe llc.

``` shell
$ llc -march=amdgcn -mattr=help
Available CPUs for this target:

  bonaire     - Select the bonaire processor.
  carrizo     - Select the carrizo processor.
  fiji        - Select the fiji processor.
  generic     - Select the generic processor.
  generic-hsa - Select the generic-hsa processor.
  gfx1010     - Select the gfx1010 processor.
  gfx1011     - Select the gfx1011 processor.
  gfx1012     - Select the gfx1012 processor.
  gfx1030     - Select the gfx1030 processor.
  gfx600      - Select the gfx600 processor.
  gfx601      - Select the gfx601 processor.
  gfx700      - Select the gfx700 processor.
  gfx701      - Select the gfx701 processor.
  gfx702      - Select the gfx702 processor.
  gfx703      - Select the gfx703 processor.
  gfx704      - Select the gfx704 processor.
  gfx801      - Select the gfx801 processor.
  gfx802      - Select the gfx802 processor.
  gfx803      - Select the gfx803 processor.
  gfx810      - Select the gfx810 processor.
  gfx900      - Select the gfx900 processor.
  gfx902      - Select the gfx902 processor.
  gfx904      - Select the gfx904 processor.
  gfx906      - Select the gfx906 processor.
  gfx908      - Select the gfx908 processor.
  gfx909      - Select the gfx909 processor.
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
  verde       - Select the verde processor.

Available features for this target:

  16-bit-insts                - Has i16/f16 instructions.
  DumpCode                    - Dump MachineInstrs in the CodeEmitter.
  a16                         - Support gfx10-style A16 for 16-bit coordinates/gradients/lod/clamp/mip image operands.
  add-no-carry-insts          - Have VALU add/sub instructions without carry out.
  aperture-regs               - Has Memory Aperture Base and Size Registers.
  atomic-fadd-insts           - Has buffer_atomic_add_f32, buffer_atomic_pk_add_f16, global_atomic_add_f32, global_atomic_pk_add_f16 instructions.
  auto-waitcnt-before-barrier - Hardware automatically inserts waitcnt before barrier.
  ci-insts                    - Additional instructions for CI+.
  code-object-v3              - Generate code object version 3.
  cumode                      - Enable CU wavefront execution mode.
  dl-insts                    - Has v_fmac_f32 and v_xnor_b32 instructions.
  dot1-insts                  - Has v_dot4_i32_i8 and v_dot8_i32_i4 instructions.
  dot2-insts                  - Has v_dot2_f32_f16, v_dot2_i32_i16, v_dot2_u32_u16, v_dot4_u32_u8, v_dot8_u32_u4 instructions.
  dot3-insts                  - Has v_dot8c_i32_i4 instruction.
  dot4-insts                  - Has v_dot2c_i32_i16 instruction.
  dot5-insts                  - Has v_dot2c_f32_f16 instruction.
  dot6-insts                  - Has v_dot4c_i32_i8 instruction.
  dpp                         - Support DPP (Data Parallel Primitives) extension.
  dpp8                        - Support DPP8 (Data Parallel Primitives) extension.
  ds-src2-insts               - Has ds_*_src2 instructions.
  dumpcode                    - Dump MachineInstrs in the CodeEmitter.
  enable-ds128                - Use ds_{read|write}_b128.
  enable-prt-strict-null      - Enable zeroing of result registers for sparse texture fetches.
  fast-denormal-f32           - Enabling denormals does not cause f32 instructions to run at f64 rates.
  fast-fmaf                   - Assuming f32 fma is at least as fast as mul + add.
  flat-address-space          - Support flat address space.
  flat-for-global             - Force to generate flat instruction for global.
  flat-global-insts           - Have global_* flat memory instructions.
  flat-inst-offsets           - Flat instructions have immediate offset addressing mode.
  flat-scratch-insts          - Have scratch_* flat memory instructions.
  flat-segment-offset-bug     - GFX10 bug, inst_offset ignored in flat segment.
  fma-mix-insts               - Has v_fma_mix_f32, v_fma_mixlo_f16, v_fma_mixhi_f16 instructions.
  fmaf                        - Enable single precision FMA (not as fast as mul+add, but fused).
  fp64                        - Enable double precision operations.
  g16                         - Support G16 for 16-bit gradient image operands.
  gcn3-encoding               - Encoding format for VI.
  get-wave-id-inst            - Has s_get_waveid_in_workgroup instruction.
  gfx10                       - GFX10 GPU generation.
  gfx10-3-insts               - Additional instructions for GFX10.3.
  gfx10-insts                 - Additional instructions for GFX10+.
  gfx10_b-encoding            - Encoding format GFX10_B.
  gfx7-gfx8-gfx9-insts        - Instructions shared in GFX7, GFX8, GFX9.
  gfx8-insts                  - Additional instructions for GFX8+.
  gfx9                        - GFX9 GPU generation.
  gfx9-insts                  - Additional instructions for GFX9+.
  half-rate-64-ops            - Most fp64 instructions are half rate instead of quarter.
  inst-fwd-prefetch-bug       - S_INST_PREFETCH instruction causes shader to hang.
  int-clamp-insts             - Support clamp for integer destination.
  inv-2pi-inline-imm          - Has 1 / (2 * pi) as inline immediate.
  lds-branch-vmem-war-hazard  - Switching between LDS and VMEM-tex not waiting VM_VSRC=0.
  lds-misaligned-bug          - Some GFX10 bug with misaligned multi-dword LDS access in WGP mode.
  ldsbankcount16              - The number of LDS banks per compute unit..
  ldsbankcount32              - The number of LDS banks per compute unit..
  load-store-opt              - Enable SI load/store optimizer pass.
  localmemorysize0            - The size of local memory in bytes.
  localmemorysize32768        - The size of local memory in bytes.
  localmemorysize65536        - The size of local memory in bytes.
  mad-mac-f32-insts           - Has v_mad_f32/v_mac_f32/v_madak_f32/v_madmk_f32 instructions.
  mad-mix-insts               - Has v_mad_mix_f32, v_mad_mixlo_f16, v_mad_mixhi_f16 instructions.
  mai-insts                   - Has mAI instructions.
  max-private-element-size-16 - Maximum private access size may be 16.
  max-private-element-size-4  - Maximum private access size may be 4.
  max-private-element-size-8  - Maximum private access size may be 8.
  mfma-inline-literal-bug     - MFMA cannot use inline literal as SrcC.
  mimg-r128                   - Support 128-bit texture resources.
  movrel                      - Has v_movrel*_b32 instructions.
  no-data-dep-hazard          - Does not need SW waitstates.
  no-sdst-cmpx                - V_CMPX does not write VCC/SGPR in addition to EXEC.
  no-sram-ecc-support         - Hardware does not support SRAM ECC.
  no-xnack-support            - Hardware does not support XNACK.
  nsa-encoding                - Support NSA encoding for image instructions.
  nsa-to-vmem-bug             - MIMG-NSA followed by VMEM fail if EXEC_LO or EXEC_HI equals zero.
  offset-3f-bug               - Branch offset of 3f hardware bug.
  pk-fmac-f16-inst            - Has v_pk_fmac_f16 instruction.
  promote-alloca              - Enable promote alloca pass.
  r128-a16                    - Support gfx9-style A16 for 16-bit coordinates/gradients/lod/clamp/mip image operands, where a16 is aliased with r128.
  register-banking            - Has register banking.
  s-memrealtime               - Has s_memrealtime instruction.
  s-memtime-inst              - Has s_memtime instruction.
  scalar-atomics              - Has atomic scalar memory instructions.
  scalar-flat-scratch-insts   - Have s_scratch_* flat memory instructions.
  scalar-stores               - Has store scalar memory instructions.
  sdwa                        - Support SDWA (Sub-DWORD Addressing) extension.
  sdwa-mav                    - Support v_mac_f32/f16 with SDWA (Sub-DWORD Addressing) extension.
  sdwa-omod                   - Support OMod with SDWA (Sub-DWORD Addressing) extension.
  sdwa-out-mods-vopc          - Support clamp for VOPC with SDWA (Sub-DWORD Addressing) extension.
  sdwa-scalar                 - Support scalar register with SDWA (Sub-DWORD Addressing) extension.
  sdwa-sdst                   - Support scalar dst for VOPC with SDWA (Sub-DWORD Addressing) extension.
  sea-islands                 - SEA_ISLANDS GPU generation.
  sgpr-init-bug               - VI SGPR initialization bug requiring a fixed SGPR allocation size.
  si-scheduler                - Enable SI Machine Scheduler.
  smem-to-vector-write-hazard - s_load_dword followed by v_cmp page faults.
  southern-islands            - SOUTHERN_ISLANDS GPU generation.
  sram-ecc                    - Enable SRAM ECC.
  trap-handler                - Trap handler support.
  trig-reduced-range          - Requires use of fract on arguments to trig instructions.
  unaligned-buffer-access     - Support unaligned global loads and stores.
  unaligned-scratch-access    - Support unaligned scratch loads and stores.
  unpacked-d16-vmem           - Has unpacked d16 vmem instructions.
  unsafe-ds-offset-folding    - Force using DS instruction immediate offsets on SI.
  vcmpx-exec-war-hazard       - V_CMPX WAR hazard on EXEC (V_CMPX issue ONLY).
  vcmpx-permlane-hazard       - TODO: describe me.
  vgpr-index-mode             - Has VGPR mode register indexing.
  vmem-to-scalar-write-hazard - VMEM instruction followed by scalar writing to EXEC mask, M0 or SGPR leads to incorrect execution..
  volcanic-islands            - VOLCANIC_ISLANDS GPU generation.
  vop3-literal                - Can use one literal in VOP3.
  vop3p                       - Has VOP3P packed instructions.
  vscnt                       - Has separate store vscnt counter.
  wavefrontsize16             - The number of threads per wavefront.
  wavefrontsize32             - The number of threads per wavefront.
  wavefrontsize64             - The number of threads per wavefront.
  xnack                       - Enable XNACK support.

Use +feature to enable a feature, or -feature to disable it.
For example, llc -mcpu=mycpu -mattr=+feature1,-feature2
```

## Prevođenje koda za različite procesore

Prevedimo kod:

``` shell
$ clang -target amdgcn -S kernel.cl
```

Zadani procesor je `gfx600`, odnosno `tahiti` (Radeon serija HD 7900). Dobiveni kod je oblika:

``` ca65
        .text
        .section        .AMDGPU.config
        .long   47176
        .long   11469125
        .long   47180
        .long   133
        .long   47200
        .long   4194304
        .long   4
        .long   0
        .long   8
        .long   0
        .text
        .globl  multiply_by_two
        .p2align        8
        .type   multiply_by_two,@function
multiply_by_two:
        s_mov_b32 s32, 0
        s_mov_b32 s33, 0
        s_mov_b32 s40, SCRATCH_RSRC_DWORD0
        s_mov_b32 s41, SCRATCH_RSRC_DWORD1
        s_mov_b32 s42, -1
        s_mov_b32 s43, 0xe8f000
        s_add_u32 s40, s40, s3
        s_addc_u32 s41, s41, 0
        s_getpc_b64 s[2:3]
        s_add_u32 s2, s2, get_global_id@gotpcrel32@lo+4
        s_addc_u32 s3, s3, get_global_id@gotpcrel32@hi+4
        s_load_dwordx2 s[4:5], s[2:3], 0x0
        s_load_dwordx4 s[36:39], s[0:1], 0x9
        v_mov_b32_e32 v0, 0
        s_mov_b64 s[0:1], s[40:41]
        s_mov_b64 s[2:3], s[42:43]
        s_waitcnt lgkmcnt(0)
        s_swappc_b64 s[30:31], s[4:5]
        v_ashrrev_i32_e32 v1, 31, v0
        s_mov_b32 s3, 0xf000
        v_lshl_b64 v[0:1], v[0:1], 2
        s_mov_b32 s2, 0
        s_mov_b64 s[0:1], s[36:37]
        buffer_load_dword v2, v[0:1], s[0:3], 0 addr64
        s_waitcnt vmcnt(0)
        v_add_f32_e32 v2, v2, v2
        s_mov_b64 s[0:1], s[38:39]
        buffer_store_dword v2, v[0:1], s[0:3], 0 addr64
        s_endpgm
.Lfunc_end0:
        .size   multiply_by_two, .Lfunc_end0-multiply_by_two

        .ident  "Debian clang version 11.0.1-2"
        .section        ".note.GNU-stack"
        .amd_amdgpu_isa "amdgcn----gfx600
```

Promotrimo funkciju `multiply_by_two`. Vidimo niz skalarnih (prefix `s_`) i vektorskih (prefiks `v_`) instrukcija. Funkcija množenja s 2 ovdje postaje instrukcija zbrajanja broja sa samim sobom, `v_add_f32_e32` koja djeluje na vektorskom registru opće namjene `v2`. Ta transformacija množenja s dva u zbrajanje je posljedica optimizacije koja zna da je zbrajanje manje zahtjevna operacija od množenja. Uočimo još dvije instrukcije:

- instrukciju `buffer_load_dword` iznad, koja učitava podatke iz globalne memorije grafičkog procesora u vektorski registar `v2`,
- instrukciju `buffer_store_dword` ispod, koja sprema podatke iz vektorskog registra `v2` u globalnu memoriju.

Usporedimo dobiveni kod za grafički procesor `gfx803`, odnosno `fiji` (Radeon R9 Fury).

``` shell
$ clang -target amdgcn -mcpu=gfx803 -S kernel.cl
```

``` ca65
        .text
        .section        .AMDGPU.config
        .long   47176
        .long   11469189
        .long   47180
        .long   133
        .long   47200
        .long   4194304
        .long   4
        .long   0
        .long   8
        .long   0
        .text
        .globl  multiply_by_two
        .p2align        8
        .type   multiply_by_two,@function
multiply_by_two:
        s_mov_b32 s40, SCRATCH_RSRC_DWORD0
        s_mov_b32 s41, SCRATCH_RSRC_DWORD1
        s_mov_b32 s42, -1
        s_mov_b32 s43, 0xe80000
        s_add_u32 s40, s40, s3
        s_addc_u32 s41, s41, 0
        s_load_dwordx4 s[36:39], s[0:1], 0x24
        s_getpc_b64 s[0:1]
        s_add_u32 s0, s0, get_global_id@gotpcrel32@lo+4
        s_addc_u32 s1, s1, get_global_id@gotpcrel32@hi+4
        s_load_dwordx2 s[4:5], s[0:1], 0x0
        s_mov_b64 s[0:1], s[40:41]
        s_mov_b64 s[2:3], s[42:43]
        v_mov_b32_e32 v0, 0
        s_mov_b32 s32, 0
        s_mov_b32 s33, 0
        s_waitcnt lgkmcnt(0)
        s_swappc_b64 s[30:31], s[4:5]
        v_ashrrev_i32_e32 v1, 31, v0
        v_lshlrev_b64 v[0:1], 2, v[0:1]
        v_mov_b32_e32 v3, s37
        v_add_u32_e32 v2, vcc, s36, v0
        v_addc_u32_e32 v3, vcc, v3, v1, vcc
        flat_load_dword v2, v[2:3]
        v_mov_b32_e32 v4, s39
        v_add_u32_e32 v0, vcc, s38, v0
        v_addc_u32_e32 v1, vcc, v4, v1, vcc
        s_waitcnt vmcnt(0) lgkmcnt(0)
        v_add_f32_e32 v2, v2, v2
        flat_store_dword v[0:1], v2
        s_endpgm
.Lfunc_end0:
        .size   multiply_by_two, .Lfunc_end0-multiply_by_two

        .ident  "Debian clang version 11.0.1-2"
        .section        ".note.GNU-stack"
        .amd_amdgpu_isa "amdgcn----gfx803"
```

Uočimo kako je instrukcija `v_add_f32_e32` ostala ista, ali kako su instrukcije `flat_load_dword` i `flat_store_dword` za rad s generičkim adresnim prostorom zamijenile instrukcije s prefiksom `buffer_`; naime, u trećoj generaciji arhitekture GCN uklonjene su specifične instrukcije za učitavanje iz globalne memorije i umjesto njih koriste se instrukcije za rad s generičkim adresnim prostorom.

U nastavku ćemo koristiti grafički procesor `gfx900` (Radeon RX Vega). Vega uvodi nove instrukcije za rad s globalnom memorijom `global_load_dword` i `global_store_dword`:

``` shell
$ clang -target amdgcn -mcpu=gfx900 -S kernel.cl
```

``` ca65
        .text
        .section        .AMDGPU.config
        .long   47176
        .long   11469189
        .long   47180
        .long   133
        .long   47200
        .long   4194304
        .long   4
        .long   0
        .long   8
        .long   0
        .text
        .globl  multiply_by_two
        .p2align        8
        .type   multiply_by_two,@function
multiply_by_two:
        s_mov_b32 s40, SCRATCH_RSRC_DWORD0
        s_mov_b32 s41, SCRATCH_RSRC_DWORD1
        s_mov_b32 s42, -1
        s_mov_b32 s43, 0xe00000
        s_add_u32 s40, s40, s3
        s_addc_u32 s41, s41, 0
        s_load_dwordx4 s[36:39], s[0:1], 0x24
        s_getpc_b64 s[0:1]
        s_add_u32 s0, s0, get_global_id@gotpcrel32@lo+4
        s_addc_u32 s1, s1, get_global_id@gotpcrel32@hi+4
        s_load_dwordx2 s[4:5], s[0:1], 0x0
        s_mov_b64 s[0:1], s[40:41]
        s_mov_b64 s[2:3], s[42:43]
        v_mov_b32_e32 v0, 0
        s_mov_b32 s32, 0
        s_mov_b32 s33, 0
        s_waitcnt lgkmcnt(0)
        s_swappc_b64 s[30:31], s[4:5]
        v_ashrrev_i32_e32 v1, 31, v0
        v_lshlrev_b64 v[0:1], 2, v[0:1]
        v_mov_b32_e32 v3, s37
        v_add_co_u32_e32 v2, vcc, s36, v0
        v_addc_co_u32_e32 v3, vcc, v3, v1, vcc
        global_load_dword v2, v[2:3], off
        v_mov_b32_e32 v4, s39
        v_add_co_u32_e32 v0, vcc, s38, v0
        v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
        s_waitcnt vmcnt(0)
        v_add_f32_e32 v2, v2, v2
        global_store_dword v[0:1], v2, off
        s_endpgm
.Lfunc_end0:
        .size   multiply_by_two, .Lfunc_end0-multiply_by_two

        .ident  "Debian clang version 11.0.1-2"
        .section        ".note.GNU-stack"
        .amd_amdgpu_isa "amdgcn----gfx900"
```

Iako nam neće trebati toliko detaljno poznavanje instrukcija, [čitava specifikacija pete generacije arhitekture GCN](https://rocmdocs.amd.com/en/latest/GCN_ISA_Manuals/testdocbook.html) dostupna je kao dio [službene dokumentacije ROCm-a](https://rocmdocs.amd.com/).

## Nivoi optimizacije

Clang [podržava 4 nivoa optimizacije](https://developers.redhat.com/blog/2019/08/05/customize-the-compilation-process-with-clang-optimization-options): `-O0`, `-O1`, `-O2` i `-O3`.

Iskoristimo nivo optimizacije `-O0`:

``` shell
$ clang -target amdgcn -mcpu=gfx900 -O0 -S kernel.cl
```

Rezultirajući kod je oblika:

``` ca65
        .text
        .section        .AMDGPU.config
        .long   47176
        .long   11469125
        .long   47180
        .long   133
        .long   47200
        .long   4202496
        .long   4
        .long   0
        .long   8
        .long   0
        .text
        .globl  multiply_by_two
        .p2align        8
        .type   multiply_by_two,@function
multiply_by_two:
        s_mov_b32 s32, 0x800
        s_mov_b32 s33, 0
        s_mov_b32 s36, SCRATCH_RSRC_DWORD0
        s_mov_b32 s37, SCRATCH_RSRC_DWORD1
        s_mov_b32 s38, -1
        s_mov_b32 s39, 0xe00000
        s_add_u32 s36, s36, s3
        s_addc_u32 s37, s37, 0
        s_load_dwordx2 s[2:3], s[0:1], 0x24
        s_load_dwordx2 s[0:1], s[0:1], 0x2c
        s_waitcnt lgkmcnt(0)
        s_mov_b32 s4, s3
        v_mov_b32_e32 v0, s4
        buffer_store_dword v0, off, s[36:39], s33 offset:12
        v_mov_b32_e32 v0, s2
        buffer_store_dword v0, off, s[36:39], s33 offset:8
        s_mov_b32 s2, s1
        v_mov_b32_e32 v0, s2
        buffer_store_dword v0, off, s[36:39], s33 offset:20
        v_mov_b32_e32 v0, s0
        buffer_store_dword v0, off, s[36:39], s33 offset:16
        s_getpc_b64 s[6:7]
        s_add_u32 s6, s6, get_global_id@gotpcrel32@lo+4
        s_addc_u32 s7, s7, get_global_id@gotpcrel32@hi+4
        s_load_dwordx2 s[6:7], s[6:7], 0x0
        s_mov_b64 s[8:9], s[36:37]
        s_mov_b64 s[10:11], s[38:39]
        v_mov_b32_e32 v0, 0
        s_mov_b64 s[0:1], s[8:9]
        s_mov_b64 s[2:3], s[10:11]
        s_waitcnt lgkmcnt(0)
        s_swappc_b64 s[30:31], s[6:7]
        buffer_store_dword v0, off, s[36:39], s33 offset:24
        buffer_load_dword v0, off, s[36:39], s33 offset:8
        buffer_load_dword v1, off, s[36:39], s33 offset:12
        s_waitcnt vmcnt(1)
        v_mov_b32_e32 v2, v0
        s_waitcnt vmcnt(0)
        v_mov_b32_e32 v3, v1
        buffer_load_dword v0, off, s[36:39], s33 offset:24
        s_waitcnt vmcnt(0)
        v_ashrrev_i32_e64 v1, 31, v0
        v_mov_b32_e32 v4, v0
        v_mov_b32_e32 v5, v1
        s_mov_b32 s4, 2
        v_lshlrev_b64 v[4:5], s4, v[4:5]
        v_mov_b32_e32 v0, v2
        v_mov_b32_e32 v1, v4
        v_mov_b32_e32 v2, v5
        v_add_co_u32_e64 v0, s[6:7], v0, v1
        v_addc_co_u32_e64 v1, s[6:7], v3, v2, s[6:7]
        v_mov_b32_e32 v6, v0
        v_mov_b32_e32 v7, v1
        global_load_dword v0, v[6:7], off
        s_waitcnt vmcnt(0)
        v_add_f32_e64 v0, v0, v0
        buffer_load_dword v1, off, s[36:39], s33 offset:16
        buffer_load_dword v2, off, s[36:39], s33 offset:20
        s_waitcnt vmcnt(1)
        v_mov_b32_e32 v6, v1
        s_waitcnt vmcnt(0)
        v_mov_b32_e32 v7, v2
        v_mov_b32_e32 v1, v6
        v_mov_b32_e32 v2, v4
        v_add_co_u32_e64 v1, s[6:7], v1, v2
        v_addc_co_u32_e64 v2, s[6:7], v7, v5, s[6:7]
        v_mov_b32_e32 v8, v1
        v_mov_b32_e32 v9, v2
        global_store_dword v[8:9], v0, off
        s_endpgm
.Lfunc_end0:
        .size   multiply_by_two, .Lfunc_end0-multiply_by_two

        .ident  "Debian clang version 11.0.1-2"
        .section        ".note.GNU-stack"
        .amd_amdgpu_isa "amdgcn----gfx900"
```

Optimizacija koju imamo iznad slična je `-O1` i višim nivoima:

``` shell
$ clang -target amdgcn -mcpu=gfx900 -O1 -S kernel.cl
```

Rezultirajući kod je oblika:

``` ca65
        .text
        .section        .AMDGPU.config
        .long   47176
        .long   11469189
        .long   47180
        .long   133
        .long   47200
        .long   4194304
        .long   4
        .long   0
        .long   8
        .long   0
        .text
        .globl  multiply_by_two
        .p2align        8
        .type   multiply_by_two,@function
multiply_by_two:
        s_mov_b32 s32, 0
        s_mov_b32 s40, SCRATCH_RSRC_DWORD0
        s_mov_b32 s41, SCRATCH_RSRC_DWORD1
        s_mov_b32 s42, -1
        s_mov_b32 s43, 0xe00000
        s_add_u32 s40, s40, s3
        s_addc_u32 s41, s41, 0
        s_load_dwordx4 s[36:39], s[0:1], 0x24
        s_getpc_b64 s[0:1]
        s_add_u32 s0, s0, get_global_id@gotpcrel32@lo+4
        s_addc_u32 s1, s1, get_global_id@gotpcrel32@hi+4
        s_load_dwordx2 s[4:5], s[0:1], 0x0
        s_mov_b64 s[0:1], s[40:41]
        s_mov_b64 s[2:3], s[42:43]
        v_mov_b32_e32 v0, 0
        s_waitcnt lgkmcnt(0)
        s_swappc_b64 s[30:31], s[4:5]
        v_ashrrev_i32_e32 v1, 31, v0
        v_lshlrev_b64 v[0:1], 2, v[0:1]
        v_mov_b32_e32 v3, s37
        v_add_co_u32_e32 v2, vcc, s36, v0
        v_addc_co_u32_e32 v3, vcc, v3, v1, vcc
        v_mov_b32_e32 v4, s39
        global_load_dword v2, v[2:3], off
        s_waitcnt vmcnt(0)
        v_add_f32_e32 v2, v2, v2
        v_add_co_u32_e32 v0, vcc, s38, v0
        v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
        global_store_dword v[0:1], v2, off
        s_endpgm
.Lfunc_end0:
        .size   multiply_by_two, .Lfunc_end0-multiply_by_two

        .ident  "Debian clang version 11.0.1-2"
        .section        ".note.GNU-stack"
        .amd_amdgpu_isa "amdgcn----gfx900"
```

Uočimo kako je optimizirani kod bitno kraći i kako, primjerice, eliminira pet instrukcija `buffer_store_dword` i isto toliko instrukcija `buffer_load_dword`.

## Vektorske instrukcije i tipovi podataka

Uočimo kako su instrukcije i registri koje smo dosad sreli vektorski, odnosno izvode ih sve niti koje izvode zrno. Ista je situacija kad kod ne koristi polja, primjerice:

``` c
__kernel void a_b_c()
{
    int a = 3;
    int b = 4;
    int c = a + b;
}
```

Prevođenjem ovog koda s uključenim optimizacijama nećemo dobiti ništa konkretno jer će program prevoditelj uočiti da se varijabla `c` nigdje ne koristi i eliminirati je, a zatim eliminirati varijable `a` i `b`. Iskoristimo parametar `-O0`:

``` shell
$ clang -target amdgcn -mcpu=gfx900 -O0 -S kernel.cl
```

Dobivamo kod:

``` ca65
        .text
        .section        .AMDGPU.config
        .long   47176
        .long   11469056
        .long   47180
        .long   129
        .long   47200
        .long   4096
        .long   4
        .long   0
        .long   8
        .long   0
        .text
        .globl  a_b_c
        .p2align        8
        .type   a_b_c,@function
a_b_c:
        s_mov_b32 s33, 0
        s_mov_b32 s4, SCRATCH_RSRC_DWORD0
        s_mov_b32 s5, SCRATCH_RSRC_DWORD1
        s_mov_b32 s6, -1
        s_mov_b32 s7, 0xe00000
        s_add_u32 s4, s4, s1
        s_addc_u32 s5, s5, 0
        v_mov_b32_e32 v0, 3
        buffer_store_dword v0, off, s[4:7], s33 offset:4
        v_mov_b32_e32 v0, 4
        buffer_store_dword v0, off, s[4:7], s33 offset:8
        buffer_load_dword v0, off, s[4:7], s33 offset:4
        buffer_load_dword v1, off, s[4:7], s33 offset:8
        s_waitcnt vmcnt(0)
        v_add_u32_e64 v0, v0, v1
        buffer_store_dword v0, off, s[4:7], s33 offset:12
        s_endpgm
.Lfunc_end0:
        .size   a_b_c, .Lfunc_end0-a_b_c

        .ident  "Debian clang version 11.0.1-2"
        .section        ".note.GNU-stack"
        .amd_amdgpu_isa "amdgcn----gfx900"
```

Uočimo postavljanje vrijednosti 3 i 4 u vektorski registar `v0` instrukcijom `v_mov_b32_e32`, a zatim uočimo zbrajanje instrukcijom `v_add_u32_e64` koja nije ista kao instrukcija korištena ranije. Naime, ovdje se radi o cjelobrojnim tipovima, a ranije smo koristili brojeve s pomičnim zarezom pa su i instrukcije za rad s njima odvojene.

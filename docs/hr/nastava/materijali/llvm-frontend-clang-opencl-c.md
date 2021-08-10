---
author: Vedran Miletić
---

# Prevođenje OpenCL C koda program prevoditeljem Clang

[Clang](https://clang.llvm.org/) je program prevoditelj za programske jezike slične C-u (C, C++, Objective C/C++, OpenCL C i CUDA C/C++) i dio je [projekta LLVM](https://llvm.org/).

Mi ćemo se u nastavku ograničiti na prevođenje jezika OpenCL C u LLVM-ovu srednju reprezentaciju (koja se zatim može prevesti u asemblerski kod grafičkog ili osnovnog procesora), ali proces prevođenja jezika CUDA C/C++ i HIP bio bi vrlo sličan.

## Izvorni kod zrna

Uzmimo da imamo jednostavno zrno u OpenCL C-u koje sve elemente u polju `in` postavlja na dvostruku vrijednost i sprema u polje `out`. Kod je oblika:

``` c
__kernel void multiply_by_two(__global float *in, __global float *out)
{
    int index = get_global_id(0);
    out[index] = 2.0f * in[index];
}
```

Uočimo poziv funkcije `get_global_id()`; kad bismo ovaj kod zrna prevodili unutar programa, ta funkcija bi postojala u zaglavlju upravljačkog programa grafičkog procesora i bila bi uključena u proces prevođenja. Radi jednostavnosti, ovdje ćemo umjesto uključivanja zaglavlja sami deklarirati tu funkciju:

``` c
int get_global_id(int index);

__kernel void multiply_by_two(__global float *in, __global float *out)
{
    int index = get_global_id(0);
    out[index] = 2.0f * in[index];
}
```

Spremimo taj kod u datoteku `kernel.cl`.

Prvo se uvjerimo da imamo Clang instaliran i provjerimo njegovu verziju.

``` shell
$ clang --version
Debian clang version 11.0.1-2
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /usr/bin
```

## Prevođenje u asemblerski kod

U nastavku nas zanima proces prevođenja zrna, a ne njegovo izvođenje. Clang možemo pokrenuti korištenjem parametra `-S` tako da izvede samo prevođenje programskog koda u asemblerski kod, ali ne i povezivanje s bibliotekama i izradu izvršne datoteke. Bez dodatnih parametara kojima bismo naveli arhitekturu dobivamo asemblerski kod za arhitekturu x86-64 (AMD64 i Intel EM64T), koju vidimo navedenu iznad u dijelu `Target`.

Pokretanjem naredbe

``` shell
$ clang -S kernel.cl
```

dobivamo datoteku `kernel.s` sadržaja:

``` asm
        .text
        .file   "kernel.cl"
        .globl  multiply_by_two                 # -- Begin function multiply_by_two
        .p2align        4, 0x90
        .type   multiply_by_two,@function
multiply_by_two:                        # @multiply_by_two
        .cfi_startproc
# %bb.0:
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        pushq   %r14
        pushq   %rbx
        .cfi_offset %rbx, -32
        .cfi_offset %r14, -24
        movq    %rsi, %r14
        movq    %rdi, %rbx
        xorl    %edi, %edi
        callq   get_global_id
        cltq
        movss   (%rbx,%rax,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
        addss   %xmm0, %xmm0
        movss   %xmm0, (%r14,%rax,4)
        popq    %rbx
        popq    %r14
        popq    %rbp
        .cfi_def_cfa %rsp, 8
        retq
.Lfunc_end0:
        .size   multiply_by_two, .Lfunc_end0-multiply_by_two
        .cfi_endproc
                                        # -- End function
        .ident  "Debian clang version 11.0.1-2"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
```

Želimo li neku drugu arhitekturu, npr. MIPS, možemo je navesti parametrom `-target`:

``` shell
$ clang -target mips -S kernel.cl
```

Datoteka `kernel.s` je sada sadržaja:

```
        .text
        .abicalls
        .option pic0
        .section        .mdebug.abi32,"",@progbits
        .nan    legacy
        .module fp=xx
        .module nooddspreg
        .text
        .file   "kernel.cl"
        .globl  multiply_by_two                 # -- Begin function multiply_by_two
        .p2align        2
        .type   multiply_by_two,@function
        .set    nomicromips
        .set    nomips16
        .ent    multiply_by_two
multiply_by_two:                        # @multiply_by_two
        .frame  $fp,32,$ra
        .mask   0xc0030000,-4
        .fmask  0x00000000,0
        .set    noreorder
        .set    nomacro
        .set    noat
# %bb.0:
        addiu   $sp, $sp, -32
        sw      $ra, 28($sp)                    # 4-byte Folded Spill
        sw      $fp, 24($sp)                    # 4-byte Folded Spill
        sw      $17, 20($sp)                    # 4-byte Folded Spill
        sw      $16, 16($sp)                    # 4-byte Folded Spill
        move    $fp, $sp
        move    $16, $5
        move    $17, $4
        jal     get_global_id
        addiu   $4, $zero, 0
        sll     $1, $2, 2
        lwxc1   $f0, $1($17)
        add.s   $f0, $f0, $f0
        swxc1   $f0, $1($16)
        move    $sp, $fp
        lw      $16, 16($sp)                    # 4-byte Folded Reload
        lw      $17, 20($sp)                    # 4-byte Folded Reload
        lw      $fp, 24($sp)                    # 4-byte Folded Reload
        lw      $ra, 28($sp)                    # 4-byte Folded Reload
        jr      $ra
        addiu   $sp, $sp, 32
        .set    at
        .set    macro
        .set    reorder
        .end    multiply_by_two
$func_end0:
        .size   multiply_by_two, ($func_end0)-multiply_by_two
                                        # -- End function
        .ident  "Debian clang version 11.0.1-2"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
        .text
```

Prevođenjem u asemblerski kod za grafičke procesore bavit ćemo se malo kasnije pa ćemo i analizirati instrukcije koje se nalaze u kodu.

## Prevođenje u srednju reprezentaciju

Clang možemo zaustaviti i prije nego proizvede asemblerski kod, odnosno reći mu da izvede samo prevođenje koda u [LLVM-ovu srednju reprezentaciju](https://llvm.org/docs/LangRef.html) dodavanjem parametra `-emit-llvm`.

``` shell
$ clang -S -emit-llvm kernel.cl
```

Dobivamo datoteku `kernel.ll` sadržaja:

``` llvm
; ModuleID = 'kernel.cl'
source_filename = "kernel.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: convergent norecurse nounwind uwtable
define dso_local spir_kernel void @multiply_by_two(float* nocapture readonly %0, float* nocapture %1) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
%3 = tail call i32 @get_global_id(i32 0) #2
%4 = sext i32 %3 to i64
%5 = getelementptr inbounds float, float* %0, i64 %4
%6 = load float, float* %5, align 4, !tbaa !7
%7 = fmul float %6, 2.000000e+00
%8 = getelementptr inbounds float, float* %1, i64 %4
store float %7, float* %8, align 4, !tbaa !7
ret void
}

; Function Attrs: convergent
declare dso_local i32 @get_global_id(i32) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{!"Debian clang version 11.0.1-2"}
!3 = !{i32 1, i32 1}
!4 = !{!"none", !"none"}
!5 = !{!"float*", !"float*"}
!6 = !{!"", !""}
!7 = !{!8, !8, i64 0}
!8 = !{!"float", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
```

Uočimo da srednja reprezentacija sadrži atribute kao što su `"target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false"` koji su specifični za x86-64 i bit će korišteni kod prevođenja u asemblerski kod. Srednju reprezentaciju za MIPS možemo dobiti naredbom `clang -target mips -S -emit-llvm kernel.cl` i ona je oblika:

``` llvm
; ModuleID = 'kernel.cl'
source_filename = "kernel.cl"
target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips"

; Function Attrs: convergent norecurse nounwind
define dso_local spir_kernel void @multiply_by_two(float* nocapture readonly %0, float* nocapture %1) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %3 = tail call i32 @get_global_id(i32 signext 0) #2
  %4 = getelementptr inbounds float, float* %0, i32 %3
  %5 = load float, float* %4, align 4, !tbaa !7
  %6 = fmul float %5, 2.000000e+00
  %7 = getelementptr inbounds float, float* %1, i32 %3
  store float %6, float* %7, align 4, !tbaa !7
  ret void
}

; Function Attrs: convergent
declare dso_local i32 @get_global_id(i32 signext) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="mips32r2" "target-features"="+fpxx,+mips32r2,+nooddspreg,-noabicalls" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="mips32r2" "target-features"="+fpxx,+mips32r2,+nooddspreg,-noabicalls" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{!"Debian clang version 11.0.1-2"}
!3 = !{i32 1, i32 1}
!4 = !{!"none", !"none"}
!5 = !{!"float*", !"float*"}
!6 = !{!"", !""}
!7 = !{!8, !8, i64 0}
!8 = !{!"float", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
```

Srednjom reprezentacijom se nećemo dalje baviti, ali je načelno vrlo korisna za analizu načina rada optimizacija jer ima jednostavniju sintaksu od asemblerskih jezika. Ona se može prevesti u asemblerski kod [LLVM-ovim alatom llc](https://llvm.org/docs/CommandGuide/llc.html).

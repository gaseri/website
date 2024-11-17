---
author: Antonela Čerkez, Rafael Teodor Mirić, Matea Turalija
---

# MIPS Instruction Set Cheat Sheet

The MIPS Instruction Set Architecture is a reduced instruction set computer (RISC) architecture that is widely used in various computing systems. It provides a set of instructions that define the operations that a MIPS processor can perform. These instructions are designed to be simple, efficient, and easy to understand.

The MIPS Instruction Set offers a variety of instructions for data manipulation, arithmetic and logical operations, memory access, control flow, and input/output operations. Each instruction is represented by a specific binary code and performs a specific operation when executed by the processor.

## Arithmetic Instructions

Some of these instructions have an unsigned version, which can be obtained by appending `u` to the opcode (e.g., addu, subu...)

| Instruction | Syntax | Opcode | Description |
| ----------- | ------ | ------ | ----------- |
| `add(u)` | `add(u) Rdest, Rsrc1, Rsrc2` | 100000 (100001) | Adds two registers `Rsrc1` and `Rsrc2` into register `Rdest` |
| `addi` | `addi Rdest, Rsrc1, Imm` | 001000 (001001) | Adds the immediate specified (`Imm`, a constant number) in the instruction to a value in register `Rsrc1` into register `Rdest` |
| `sub(u)` | `sub(u) Rdest, Rsrc1, Rsrc2` | 100010 (100011) | Put the difference of the integers from register `Rsrc1` and `Rsrc2` into register `Rdest` |
| `subi(u)` | `subi(u) Rdest, Rsrc1, Imm` | | Put the difference of the integers from register `Rsrc1` and a sign-extended immediate value `Imm` into register `Rdest` |
| `mul(u)` | `mul(u) Rdest, Rsrc1, Rsrc2` | | Multiply `Rsrc1` and `Rsrc2` into register `Rdest` |
| `mult(u)` | `mult(u) Rsrc1, Rsrc2` | 011000 (011001) | Multiply `Rsrc1` and `Rsrc2`. Less significant bits are saved to register `lo`, and more significant ones in register `hi` |
| `div(u)` | `div(u) Rsrc1, Rsrc2` | 011010 (011011) | Divide `Rscr1` and `Rsrc2` into register `Rdest` |
| `rem` | `rem Rdest, Rsrc1, Rsrc2` | | Put the remainder from dividing the integer in register `Rsrc1` by the integer in `Rsrc2` into register `Rdest` |
| `abs` | `abs Rdest, Rsrc` | | Put the absolute value of integer from register `Rsrc` into register `Rdest` |
| `neg` | `neg Rdest, Rsrc` | | Put the negative of the integer from register `Rsrc` into register `Rdest` |
| `rol` | `rol Rdest, Rsrc1, Rsrc2` | | Rotate the contents of register `Rsrc1` left by the distance indicated by `Rsrc2` and put the result in register `Rdest` |
| `ror` | `ror Rdest, Rsrc1, Rsrc2` | | Rotate the contents of register `Rsrc1` right by the distance indicated by `Rsrc2` and put the result in register `Rdest` |

## Logic Instructions

| Instruction | Syntax | Opcode | Description |
| ----------- | ------ | ------ | ----------- |
| `not` | `not Rdest, Rsrc` | | Put the bitwise logical negation of the integer from register `Rsrc` into register `Rdest` |
| `and` | `and Rdest, Rsrc1, Rsrc2` | 100100 | Put the logical AND of the integers from register `Rsrc1` and `Rsrc2` into register `Rdest` |
| `andi` | `andi Rdest, Rsrc1, Imm` | 001100 | Put the logical AND of the integers from register `Rsrc1` and immediate value `Imm` into register `Rdest` |
| `or` | `or Rdest, Rsrc1, Rsrc2` | 100101 | Put the logical OR of the integers from register `Rsrc1` and `Rsrc2` into register `Rdest` |
| `ori` | `ori Rdest, Rsrc1, Imm` | 001101 | Put the logical OR of the integers from register `Rsrc1` and immediate value `Imm` into register `Rdest` |
| `nor` | `nor Rdest, Rsrc1, Rsrc2` | 100111 | Put the logical NOR of the integers from register `Rsrc1` and `Rsrc2` to register `Rdest` |
| `xor` | `xor Rdest, Rsrc1, Rsrc2` | 100110 | Put the logical XOR of the integers from register `Rsrc1` and `Rsrc2` to register `Rdest` |
| `xori` | `xori Rdest, Rsrc1, Imm` | 001110 | Put the logical XOR of the integers from register `Rsrc1` and immediate value `Imm` into register `Rdest` |

## Comparison Instructions

| Instruction | Syntax | Opcode | Description |
| ----------- | ------ | ------ | ----------- |
| `seq` | `seq Rdest, Rsrc1, Rsrc2` | pseudoinstruction | Sets register `Rdest` to 1 if `Rsrc1 = Rsrc2`, else it is set to 0 |
| `sne` | `sne Rdest, Rsrc1, Rsrc2` | | Sets `Rdest` to 1 if `Rsrc1 != Rsrc2`, else it is set to 0 |
| `sge(u)` | `sge(u) Rdest, Rsrc1, Rsrc2` | pseudoinstruction | Sets register `Rdest` to 1 if `Rsrc1 >= Rsrc2`, else it is set to 0 |
| `sgt(u)` | `sgt(u) Rdest, Rsrc1, Rsrc2` | pseudoinstruction | Sets register `Rdest` to 1 if `Rsrc1 > Rsrc2`, else it is set to 0 |
| `sle(u)` | `sle(u) Rdest, Rsrc1, Rsrc2` | pseudoinstruction | Sets register `Rdest` to 1 if `Rsrc1 <= Rsrc2`, else it is set to 0 |
| `slt(u)` | `slt(u) Rdest, Rsrc1, Rsrc2` | 101010 (101001) | Sets `Rdest` to 1 if `Rsrc1 < Rsrc2`, else it is set to 0 |
| `slti(u)` | `slti(u) Rdest, Rsrc1, Imm` | 001010 (001001) | Sets `Rdest` to 1 if `Rsrc1 < Imm`, else it is set to 0 |

## Branch and Jump Instructions

| Instruction | Syntax | Opcode | Description |
| ----------- | ------ | ------ | ----------- |
| `b` | `b lab` | pseudoinstruction | Unconditionally branch to the instruction at label `lab` |
| `beq` | `beq Rsrc1, Rsrc2, label` | 000100 | Conditionally branch to the instruction at the label if contents of register `Rsrc1 = Rsrc2` |
| `bge` | `bge Rsrc1, Rsrc2, label` | pseudoinstruction | Conditionally branch to the instruction at the label if the contents of register `Rscr1 >= Rsrc2` |
| `bgt` | `bgt Rsrc1, Rsrc2, label` | pseudoinstruction | Conditionally branch to the instruction at the label if the contents of register `Rsrc1 > Rsrc2` |
| `ble` | `ble Rsrc1, Rsrc2, label` | pseudoinstruction | Conditionally branch to the instrucion at the label if the contents of register `Rsrc1 <= Rsrc2` |
| `blt` | `blt Rsrc1, Rsrc2, label` | pseudoinstruction | Conditionally branch to the instruction at the label if contents of register `Rsrc1 < Rsrc2` |
| `bne` | `bne Rsrc1, Rsrc2, label` | 000101 | Conditionally branch to the instruction at the label if contents of register `Rsrc1 != Rsrc2` |
| `bnez` | `bnez Rsrc1, label` | pseudoinstruction | Conditionally branch on the instruction at the label if contents of register `Rsrc1 != 0` |
| `beqz` | `beqz, Rsrc1, label` | pseudoinstruction | Conditionally branch on the instruction at the label if contents of register `Rsrc1 = 0` |
| `bgez` | `bgez, Rsrc1, label` | pseudoinstruction | Conditionally branch on the instruction at the label if contents of register `Rsrc1 >= 0` |
| `bgtz` | `bgtz, Rsrc1, label` | 000111 | Conditionally branch on the instruction at the label if contents of register `Rsrc1 > 0` |
| `blez` | `blez, Rsrc1, label` | 000110 | Conditionally branch on the instruction at the label if contents of register `Rsrc1 <= 0` |
| `bltz` | `bltz, Rsrc1, label` | pseudoinstruction | Conditionally branch on the instruction at the label if contents of register `Rsrc1 < 0` |
| `jal` | `jal label` | 000011 | Unconditionally jump to the instruction at the label. Save the address of the next instruction in register `$ra` |
| `jr` | `jr Rsrc` | 001000 | Unconditionally jump to the instruction whoose address is in register `Rsrc` |

## Instructions for Loading and Saving Values

| Instruction | Syntax | Opcode | Description |
| ----------- | ------ | ------ | ----------- |
| `move` | `move Rdest, Rsrc` | pseudoinstruction | Move the contents of `Rsrc` to `Rdest` |
| `li` | `li Rdest, imm` | pseudoinstruction | Move the immediate value `imm` into register `Rdest` |
| `la` | `la Rdest, addr` | pseudoinstruction | Load computed address, not the contents of location, into register `Rdest` |
| `lb` | `lb Rdest, addr` | 100000 | Load the byte at address into register `Rdest` |
| `lh` | `lh Rdest, addr` | 100001 | Load the 16-bit quantity (halfword) at address into register `Rdest` |
| `lw` | `lw Rdest, addr` | 100011 | Load the 32-bit quantity (word) at address into register `Rdest` |
| `sb` | `sb Rsrc, addr` | 101000 | Store the low byte from register `Rsrc` at `addr` |
| `sh` | `sh Rsrc, addr` | 101001 | Store the low halfword from register `Rsrc` at `addr` |
| `sw` | `sw Rsrc, addr` | 101011 | Store the word from register `Rsrc` at `addr` |

## System Calls

| Service | Code in `$v0` | Arguments | Result |
| ------- | ------------- | --------- | ------ |
| print_int | 1 | `$a0` = integer | |
| print_float | 2 | `$f12` = float | |
| print_double | 3 | `$f12` = double | |
| print_string | 4 | `$a0` = string address | |
| read_int | 5 | | int to `$v0` |
| read_float | 6 | | float to `$f0` |
| read_double | 7 | | double to `$f0` |
| read_string | 8 | `$a0` = cache address, `$a1` = buffer size | |
| sbrk | 9 | | address in `$v0` |
| exit | 10 | `$a0` = integer | |
| print_character | 11 | `$a0` = character | |
| read_character | 12 | | character (in `$v0`) |
| open | 13 | `$a0` = filename, `$a1` = flags, `$a2` = mode | file descriptor (in `$v0`) |
| read | 14 | `$a0` = file descriptor, `$a1` = buffer, `$a2` = count | bytes read (in `$v0`) |
| write | 15 | `$a0` = file descriptor,`$a1` = buffer, `$a2` = count | bytes written (in `$v0`) |
| close | 16 | `$a0` = file descriptor | 0 (in `$v0`) |
| exit2 | 17 | `$a0` = value | |

## Registers

| Name | Number | Description |
| ---- | ------ | ----------- |
| `zero` | `0` | Constant `0` |
| `at` | `1` | Reserved for assembler |
| `v0, v1` | `2, 3` | Results of subroutine |
| `a0-a3` | `4–7` | Arguments 1–4 |
| `t0–t7` | `8–15` | Temporary, their values are not saved during the call of subroutine |
| `s0–s7` | `16–23` | Saved temporary, their values are saved during the call of subroutine |
| `t8, t9` | `24, 25` | Temporary, their values are not saved |
| `k0, k1` | `26, 27` | Reserved for OS kernel |
| `gp` | `28` | Pointer to global area |
| `sp` | `29` | Stack pointer |
| `fp` | `30` | Frame pointer (if necessary) |
| `ra` | `31` | Return address |

## MIPS Assembler Directives

| Name | Description |
| ---- | ----------- |
| .align | Align next data item on specified byte boundary |
| .ascii | Store the string in the Data segment (without NULL terminator) |
| .asciiz | Store the string in the Data segment and add NULL terminator |
| .byte | Store listed values as 8 bit types |
| .data | Subsequent items stored in Data segment at next available address |
| .double | Store the listed values as double precision floating point |
| .end_macro | End macro definitions |
| .eqv | Substitute second operand for first |
| .extern | Declare the listed lable and byte lenght to a global data field |
| .float | Store the listed values as single precision floating points |
| .globl | Declare the listed values as global to enable referencing them from other files |
| .half | Store the listed values as 16 bit halfwords on halfword boundary |
| .include | Insert the contents of specified file (Filename in quotes) |
| .kdata | Subsequent items stored in Kernel Data segment at next available address |
| .ktext | Subsequent items stored in Kernel Text segment at next available address |
| .macro | Begin macro definition |
| .set | Set assembler variables |
| .space | Reserve specified number of bytes in Data segment |
| .text | Subsequent itmes stored in Text segment at next available address |
| .word | Store the listed values as 32 bit words on word boundary |

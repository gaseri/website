---
marp: true
author: Vedran Miletić
title: Code optimization
description: code optimization
keywords: code optimization
theme: uncover
class: _invert
paginate: true
---

# Code optimization: course introduction

## Dr. Vedran Miletić, Sen. Lect., vmiletic@inf.uniri.hr

### [vedran.miletic.net](https://vedran.miletic.net/) | [group.miletic.net](https://group.miletic.net/hr/)

#### YUFE course, 2021/2022 academic year

---

## Vedran Miletić

- 2009\. M. Ed. in math and CS
- 2015\. Ph. D. in computer science
- 2015--2018\. PostDoc in scientific computing, specifically computational biochemistry
- Interests: parallelization and other performance optimization techniques in scientific software

---

## Learning outcomes (1/2)

- I1. Analyse the properties enabling code transformation and represent the code using a flowchart.
- I2. Show the differences between local and global optimization and identify where each of them applies.
- I3. Perform a conventional data flow analysis, register allocation by graph colouring and common subexpression elimination.

---

## Learning outcomes (2/2)

- I4. Describe mode of operation of higher-level optimization and apply existing optimizations.
- I5. Describe differences between higher-level optimizations and target architecture-specific optimizations.
- I6. Choose instructions.
- I7. Analyse the problem of optimization phase sequence.

---

## Course content (1/3)

- Overview of programming language optimizing compiler. Optimization per elements. Analysis of properties enabling transformation. Flowchart and representation of program concepts. Problem of optimization phase sequence.
- Types of optimization. Local optimization: peephole optimization, instruction scheduling. Global optimization: common subexpressions, code changes. Interprocedural optimization. Call graph.

---

## Course content (2/3)

- Conventional data flow analysis. Algorithms on graphs, sets of live and available variables. Register allocation by graph colouring. Common subexpression elimination. Spilling to memory; use of temporary expressions introduced during common subexpression elimination. Data flow anomalies. Static single assignment form.

---

## Course content (3/3)

- Overview of higher-level optimizations. Pointer analysis and pseudonym analysis.
- Target architecture-specific optimization. Choice of instruction. Instruction scheduling and related problem of optimization phase sequence.

---

## Activities: Written commentary

- short written commentary on a given topic
- e.g. compare the target-dependent optimizations available in GCC and LLVM compilers
- 2 during the semester
- max. 10 points in total

---

## Activities: Quiz

- online quiz on the topics from the lectures
- e.g. explain the difference between analysis and transformation stages of optimization
- 2 during the semester
- max. 20 points in total

---

## Activities: Homework

- exercises related to the lectures
- e.g. perform common sub-expression elimnation on a given code
- 2 during the semester
- max. 30 points in total

---

## Activities: Summary and requirements

- 10 (Written commentary) + 20 (Quiz) + 30 (Homework) = 60, min. 30 required to be able to take the course project

---

## Activities: Project

- implementation of given optimization techniques in the LLVM compiler
- max. 40 points in total, min. 20 required to pass the course

---

## Points to grades

- 60 (pre-Project) + 40 (Project) = 100 points
- min. 30 + 20 = 50 points
- 90--100 points => grade 5
- 75--89,9 points => grade 4
- 60--74,9 points => grade 3
- 50--59,9 points => grade 2

---

## Mandatory literature

1. Cooper, K. D. & Torczon, L. Engineering a compiler. (Elsevier/Morgan Kaufmann, 2011).
1. Holub, A. I. Compiler design in C. (Prentice Hall, 1990). (e-book is available for free download from the author's site [holub.com/compiler/](https://holub.com/compiler/))
1. Scripts, presentations and other learning material available in the e-course.

---

## Presentations

- For lectures we will be reusing the presentations authored by Timothy Jones, Tom Stuart, and Alan Mycroft (University of Cambridge) for the Optimising Compilers course
    - special thanks to Domagoj Stolfa (Univ. Cambridge, [domagoj.stolfa.org](http://domagoj.stolfa.org)) for initiating this partnership

---

## Required software

- Visual Studio Code ([code.visualstudio.com](https://code.visualstudio.com/))
    - Markdown editing: `ms-vscode.wordcount`, `DavidAnson.vscode-markdownlint`
    - Graphviz preview: `joaompinto.vscode-graphviz`
- If you're on Windows, install Windows Subsystem for Linux (WSL) ([docs.microsoft.com/en-us/windows/wsl/install](https://docs.microsoft.com/en-us/windows/wsl/install)) and VS Code extension `ms-vscode-remote.remote-wsl`
- VS Code C++ extension: `ms-vscode.cpptools`

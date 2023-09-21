---
marp: true
author: Vedran Miletić, Matea Turalija
title: Molecular dynamics simulation for exascale supercomputing era -- scientific research and software engineering challenges
description: Invited lecture at the Computational Chemistry Day in Zagreb, Croatia, 16th of September, 2023.
keywords: molecular dynamics, simulation, exascale, supercomputing
theme: default
class: _invert
paginate: true
abstract: |
  The new era in supercomputers started last year with the arrival of the first batch of exascale supercomputers to the TOP500 list. Compared to the previous-generation petascale supercomputers, such computers require the use of much more complex software models and algorithms to make optimal use of all available computing resources. In the last ten years, several partnerships and projects have been launched to develop and adapt software for the "exascale era", examples of which are the German Software for Exascale Computing (SPPEXA), the American Exascale Computing Project (ECP), the European High-Performance Computing Joint Undertaking (EuropHPC JU), and Croatian High performance scalable algorithms for future heterogeneous distributed computing systems (HybridScale).

  Specifically, in the SPPEXA programme [1], progress has been made with adapting many software packages for the expected exascale supercomputer architectures, including the popular open-source molecular dynamics (MD) simulation software GROMACS [2]. The present approach is focused on developing and implementing the fast multipole method (FMM) in place of particle mesh Ewald (PME). Our group has joined this effort in collaboration with Max Planck Institute for Multidisciplinary Sciences (MPI-NAT) in Goettingen, Germany with the goals of expanding the range of supported simulation box shapes and extending the accelerator support to smart network interface cards, also known as data processing units.

  The talk will cover the architectural changes of exascale supercomputers and their impact on scientific software development. The specific focus will be on potential for further development of the molecular dynamics simulation algorithms in GROMACS and the expected impact of these developments on the larger ecosystem of computational biochemistry tools.
references:
  - \[1\] SPPEXA. http://www.sppexa.de/.
  - \[2\] GROMACS. https://www.gromacs.org/.
---

# Molecular dynamics simulation for exascale supercomputing era: scientific research and software engineering challenges

## [*Vedran*](https://vedran.miletic.net/) [*Miletić*](https://www.miletic.net/) 😎, [Matea Turalija](https://mateaturalija.github.io/) 😎

### 😎  **G**roup for **A**pplications and **S**ervices on **E**xascale **R**esearch **I**nfrastructure (GASERI), Faculty of Informatics and Digital Technologies (FIDIT), University of Rijeka

#### Invited lecture at the [Computational Chemistry Day](https://www.compchemday.org/) in Zagreb, Croatia, 16th of September, 2023.

---

## First off, thanks to the organizers for the invitation

![Computational Chemistry Day 2023 Logo](https://www.compchemday.org/wp-content/uploads/2023/04/CCD2023_Logo.png)

Image source: [Computational Chemistry Day Logo](https://www.compchemday.org/)

---

## Faculty of Informatics and Digital Technologies (FIDIT)

- founded in 2008. as *Department of Informatics* at University of Rijeka
- became *Faculty of Informatics and Digital Technologies* in 2022.
- [several laboratories](https://www.inf.uniri.hr/znanstveni-i-strucni-rad/laboratoriji) and research groups; scientific interests:
    - computer vision, pattern recognition
    - natural language processing, machine translation
    - e-learning, digital transformation
    - **scientific software parallelization, high performance computing**

![FIDIT logo bg right:35% 95%](https://upload.wikimedia.org/wikipedia/commons/1/14/FIDIT-logo.svg)

---

## **G**roup for **A**pplications and **S**ervices on **E**xascale **R**esearch **I**nfrastructure (GASERI)

- focus: optimization of computational biochemistry applications for running on modern exascale supercomputers
- vision: high performance algorithms readily available
    - for academic and industrial use
    - in state-of-the-art open-source software
- members: [Dr. Vedran](https://vedran.miletic.net/) [Miletić](https://www.miletic.net/) (PI), [Matea Turalija](https://mateaturalija.github.io/) (PhD student), [Milan Petrović](https://milanxpetrovic.github.io/) (teaching assistant)

![GASERI logo](../../images/gaseri-logo-text.png)

---

## How did we end up here, in computational chemistry?

- Prof. Dr. [Branko Mikac](https://www.fer.unizg.hr/branko.mikac)'s optical network simulation group at FER, Department of Telecommunications
- What to do after finishing the Ph.D. thesis? 🤔
    - [NVIDIA CUDA Teaching Center](../../hr/partnerstva-i-suradnje.md#obrazovni-centar-za-grafičke-procesore-gpu-education-center-bivši-cuda-nastavni-centar-cuda-teaching-center) (later: GPU Education Center)
    - research in late [Prof. Željko Svedružić](https://svedruziclab.github.io/principal-investigator.html)’s Biomolecular Structure and Function Group and Group (BioSFGroup)
- postdoc in [Prof. Frauke Gräter](https://www.h-its.org/people/prof-dr-frauke-grater/)'s Molecular Biomechanics (MBM) group at Heidelberg Institute for Theoretical Studies (HITS)
    - collaboration with GROMACS molecular dynamics simulation software developers from KTH, MPI-NAT, UMich, CU, and others

---

## High performance computing

- a **supercomputer** is a computer with a high level of performance as compared to a general-purpose computer
    - also called high performance computer (HPC)
- measure: **floating-point operations per second** (FLOPS)
    - PC -> teraFLOPS; Bura -> 100 teraFLOPS, Supek -> 1,2 petaFLOPS
    - modern HPC -> 1 do 10 petaFLOPS, top 1194 petaFLOPS (~1,2 exaFLOPS)
    - future HPC -> 2+ exaFLOPS
- nearly exponential growth of FLOPS over time (source: [Wikimedia Commons File:Supercomputers-history.svg](https://commons.wikimedia.org/wiki/File:Supercomputers-history.svg))

---

![bg 80% Computing power of the top 1 supercomputer each year, measured in FLOPS](https://upload.wikimedia.org/wikipedia/commons/8/81/Supercomputers-history.svg)

---

## Languages used in scientific software development are also evolving to utilize the new hardware

- [C++](https://isocpp.org/): C++11 (C++0x), C++14, C++17, C++20, C++23
    - Parallel standard library
    - Task blocks
- [Fortran](https://fortran-lang.org/): Fortran 2008, Fortran 2018 (2015), Fortran 2023
    - Coarray Fortran
    - `DO CONCURRENT` for loops
    - `CONTIGUOUS` storage layout
- New languages: Python, Julia, Rust, ...

---

## More heterogeneous architectures require complex programming models

- different types of accelerators
    - GPUs (half, single, double precision), TPUs/TCGPUs, FPGAs
    - in-network and in-storage computation (e.g. [Blue](https://store.nvidia.com/en-us/networking/store/?page=1&limit=9&locale=en-us&search=bluefield)[Field](https://www.pny.com/professional/explore-our-products/networking-solutions/nvidia-bluefield-data-processing-units) [DPU](https://www.nvidia.com/en-us/networking/products/data-processing-unit/))
- several projects to adjust existing software for the exascale era
    - [Software for Exascale Computing](http://www.sppexa.de/) (SPPEXA)
    - [Exascale Computing Project](https://www.exascaleproject.org/) (ECP)
    - [European High-Performance Computing Joint Undertaking](https://eurohpc-ju.europa.eu/index_en) (EuropHPC JU)

---

## SPPEXA project GROMEX

- full project title: Unified Long-range Electrostatics and Dynamic Protonation for Realistic Biomolecular Simulations on the Exascale
- principal investigators:
    - [Helmut Grubmüller](https://www.mpinat.mpg.de/grubmueller) (Max Planck Institute for Biophysical Chemistry, now Multidisciplinary Sciences, MPI-NAT)
    - [Holger Dachsel](https://www.fz-juelich.de/profile/dachsel_h) (Jülich Supercomputing Centre, JSC)
    - [Berk Hess](https://www.kth.se/profile/hess) (Stockholm University, SU)
- building on previous work on molecular dynamics simulation performance evaluation: [Best bang for your buck: GPU nodes for GROMACS biomolecular simulations](https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.24030) and [More bang for your buck: Improved use of GPU nodes for GROMACS 2018](https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.26011)

---

## Parallelization for a large HPC requires domain decomposition

![DD cells](https://manual.gromacs.org/current/_images/dd-cells.png)

Image source for both figures: [Parallelization (GROMACS Manual)](https://manual.gromacs.org/current/reference-manual/algorithms/parallelization-domain-decomp.html)

![bg right 70% DD triclinical](https://manual.gromacs.org/current/_images/dd-tric.png)

---

## Domain decomposition and particle mesh Ewald

![MPMD PME width:950px](https://manual.gromacs.org/current/_images/mpmd-pme.png)

Image source for both figures: [Parallelization (GROMACS Manual)](https://manual.gromacs.org/current/reference-manual/algorithms/parallelization-domain-decomp.html)

This approachs still does not scale indefinitely. Is there an alternative?

---

## GROMEX, part of SPPEXA

> The particle mesh Ewald method (PME, currently state of the art in molecular simulation) does not scale to large core counts as it suffers from a communication bottleneck, and does not treat titratable sites efficiently.
>
> The fast multipole method (FMM) will enable an efficient calculation of long-range interactions on massively parallel exascale computers, including alternative charge distributions representing various forms of titratable sites.

[SPPEXA Projects - Phase 2 (2016 - 2018)](http://www.sppexa.de/general-information/projects-phase-2.html)

---

## FlexFMM

![FlexFMM Logo height:180px](https://www.mpinat.mpg.de/4371009/header_image-1676374403.webp)

Image source: [Max Planck Institute for Multidisciplinary Sciences FlexFMM](https://www.mpinat.mpg.de/grubmueller/sppexa)

- continuation of SPPEXA (2022 - 2025)
- project partners:
    - JSC: [Computational time optimized exascale simulations for biomedical applications](https://www.fz-juelich.de/en/ias/jsc/projects/flexfmm)
    - MPI-NAT: [FMM-based electrostatics for biomolecular simulations at constant pH](https://www.mpinat.mpg.de/grubmueller/sppexa)
- our group is collaborating on the project via our colleagues at MPI-NAT

---

## Our GROMACS developments: generalized FMM

- molecular dynamics simulations are periodic with various [simulation box types](https://manual.gromacs.org/current/reference-manual/algorithms/periodic-boundary-conditions.html#some-useful-box-types): cubic, rhombic dodecahedron; present design and implementation of the fast multipole method supports *only cubic* boxes
    - many useful applications (materials, interfaces) fit well into rectangular cuboid boxes, not cubic -> **Matea's PhD thesis research**

![Cuboid height:200px](https://upload.wikimedia.org/wikipedia/commons/7/70/Cuboid_no_label.svg)

Image source: [Wikimedia Commons File:Cuboid no label.svg](https://commons.wikimedia.org/wiki/File%3ACuboid_no_label.svg)

- it is possible to also support rhombic dodecahedron: ~30% less volume => ~30% less computation time per step required

---

## Our GROMACS developments: NVIDIA BlueField

![NVIDIA Logo height:100px](https://upload.wikimedia.org/wikipedia/commons/a/a4/NVIDIA_logo.svg)

Image source: [Wikimedia Commons File:NVIDIA logo.svg](https://commons.wikimedia.org/wiki/File%3ANVIDIA_logo.svg)

- funded by NVIDIA, inspired by custom-silicon Anton 2 supercomputer's hardware and software architecture
- heterogeneous parallelism presently uses NVIDIA/AMD/Intel GPUs with CUDA/SYCL, also use NVIDIA BlueField DPUs with DOCA
- first publication came out last year: Turalija, M., Petrović, M. & Kovačić, B. [Towards General-Purpose Long-Timescale Molecular Dynamics Simulation on Exascale Supercomputers with Data Processing Units](https://ieeexplore.ieee.org/document/9803537)
- DOCA 2.0 improved RDMA support, which eases our efforts

---

## Our GROMACS developments: weight factor expressions and generalized flow simulation (1/4)

![100 ps molecular dynamics simulation of water. width:700px](https://upload.wikimedia.org/wikipedia/commons/f/f4/MD_water.gif)

Image source: [Wikimedia Commons File:MD water.gif](https://commons.wikimedia.org/wiki/File:MD_water.gif)

---

## Our GROMACS developments: weight factor expressions and generalized flow simulation (2/4)

- Flow is a movement of solvent atoms
- Pulling all solvent atoms works when no other molecules except water are present in the simulation
- Just a slice of solvent molecules should be pulled to allow the solvent atoms to interact with the biomolecule(s) without being "dragged away"

---

## Our GROMACS developments: weight factor expressions and generalized flow simulation (3/4)

![A rectangular water box with ~32,000 water molecules is shown. width:900px](https://www.cell.com/cms/attachment/e85de1e5-aa83-4584-b94b-e201beafb7f6/gr1_lrg.jpg)

Image source: Biophys. J. 116(6), 621–632 (2019). [doi:10.1016/j.bpj.2018.12.025](https://doi.org/10.1016/j.bpj.2018.12.025)

---

## Our GROMACS developments: weight factor expressions and generalized flow simulation (4/4)

- Atom weight = dynamic weight factor computed from the expression **x** weight factor specified in the parameters file **x** atom mass-derived weight factor
- Dynamic weight factor (and atom weight) recomputed in each simulation step
- Weight factor expression variables:
    - Atom position in 3D (`x`, `y`, `z`)
    - Atom velocity in 3D (`vx`, `vy`, `vz`)
- Examples:
    - Atom weight factor is sum of squares of positions: `x^2 + y^2 + z^2`
    - Atom weight factor is a linear combination of velocities: `1.75 * vx + 1.5 * vy + 1.25 * vz`

---

## Our *potential* GROMACS developments

- Monte Carlo ([Davide Mercadante](https://lab.mercadante.net/), University of Auckland)
    - many efforts over the years, none with broad acceptance
    - should be rethought, and then designed and implemented from scratch with exascale in mind
- polarizable simulations using the classical Drude oscillator model ([Justin Lemkul](https://www.thelemkullab.com/), Virginia Tech)
    - should be parallelized for multi-node execution
- other drug design tools such as Random Acceleration Molecular Dynamics ([Rebecca Wade](https://www.h-its.org/people/prof-dr-rebecca-wade/), Heidelberg Institute for Theoretical Studies and Daria Kokh, Cancer Registry of Baden-Württemberg)

---

## Affiliation changes (1/2)

- starting October, Matea will be joining [Faculty of Medicine](https://medri.uniri.hr/), University of Rijeka
- PhD topic staying as planned and presented here

Image source: [Wikimedia Commons File:Medicinski fakultet Rijeka 0710 1.jpg](https://commons.wikimedia.org/wiki/File%3AMedicinski_fakultet_Rijeka_0710_1.jpg)

![Faculty of Medicine bg right:60%](https://upload.wikimedia.org/wikipedia/commons/2/25/Medicinski_fakultet_Rijeka_0710_1.jpg)

---

## Affiliation changes (2/2)

- several days ago, I joined [Max Planck](https://www.mpg.de) [Computing and Data Facility](https://www.mpcdf.mpg.de/) in Garching near Munich, Germany as a part of [HPC Application Support Division](https://www.mpcdf.mpg.de/services/application-support)
- focus areas:
    - improving functionality and performance of lambda dynamics (free energy calculations)
    - developing fast multipole method implementation

Image source: [Wikimedia Commons File:110716031-TUM.JPG](https://commons.wikimedia.org/wiki/File%3A110716031-TUM.JPG)

![TUM Campus bg right:40%](https://upload.wikimedia.org/wikipedia/commons/2/2e/110716031-TUM.JPG)

---

## Thank you for your attention

![GASERI Logo](../../images/gaseri-logo-animated.webp)

GASERI website: [group.miletic.net](../../index.md)

![Vedran bg left](https://vedran.miletic.net/images/vm.jpg)

![Matea bg right](https://mateaturalija.github.io/images/profile.jpg)

---
marp: true
author: Vedran Miletić
title: The challenges of the upcoming exascale supercomputing era in computational biochemistry
description: Research Class
keywords: GASERI, GROMACS
theme: gaia
class: gaia
paginate: true
abstract: |
  The beginning of the new era in supercomputers is expected in June 2022 with the arrival of the first batch of exascale supercomputers to the TOP500 list. Compared to existing petascale supercomputers, such computers require the use of more complex software models and algorithms to make optimal use of all available computing resources. In the last ten years, several partnerships and projects have been launched to develop and adapt software for the "exascale era", examples of which are the German Software for Exascale Computing (SPPEXA), the American Exascale Computing Project (ECP), the European European High-Performance Computing Joint Undertaking (EuropHPC JU), and Croatian High performance scalable algorithms for future heterogeneous distributed computing systems (HybridScale). Specifically, in the SPPEXA programme, progress has been made with adapting many software packages for the expected exascale supercomputer architectures, including the popular open-source molecular dynamics (MD) simulation software GROMACS.

  The talk will cover what is known about the architectures of supercomputers that are expected to appear on the list during 2022 as well as hardware and software developments in the area of high performance computing by AMD, Intel, NVIDIA (Mellanox), ARM, and other companies. The specific focus will be on potential for further development of the molecular dynamics simulation algorithms in GROMACS and the expected impact of these developments on the larger ecosystem of computational biochemistry tools.
---

# The challenges of the upcoming exascale supercomputing era in computational biochemistry

## Dr. Vedran Miletić ([group.miletic.net](https://group.miletic.net/))

### 😎 Group for Applications and Services on Exascale Research Infrastructure, Faculty of Informatics and Digital Technologies, University of Rijeka

#### Research Class, FIDIT, UniRi, 26th January 2022

---

## Stream and recording check

- OBS
- BBB

---

<!-- 2 min -->
## Dr. Vedran Miletić's previous research work

- Dr. Branko Mikac's group at FER Dept. of Telecommunications
- What to do after finishing the Ph.D. thesis? 🤔
    - NVIDIA CUDA Teaching Center (later: GPU Education Center)
    - research in Dr. Željko Svedružić’s Biomolecular Structure and Function Group and Group (BioSFGroup)
- postdoc in Dr. Frauke Gräter's Molecular Biomechanics (MBM) group at Heidelberg Institute for Theoretical Studies
    - collaboration with GROMACS developers from KTH, Max Planck Institute for Biophysical Chemistry (now: Multidisciplinary Sciences), and University of Virginia

---

<!-- 1 min -->
## RxTx

- returned from Heidelberg, became a Senior Lecturer
    - 90% working hours teaching (courses + Bura supercomputer), 10% administration, **0% research**
- started RxTx ([www.rxtx.tech](https://www.rxtx.tech/))
    - collaboration with Patrik Nikolić ([www.nikoli.ch](https://nikoli.ch/), former student researcher in BioSFGroup)
    - vision: *advancing the pharmaceutical drug research by improving the scientific software behind the scenes*
    - developed open-source high-throughput virtual screening engine RxDock (until the promotion to assistant professor)

---

<!-- 2 min -->
## Group for Applications and Services on Exascale Research Infrastructure (GASERI)

- **The main interest:** the application of exascale computing to solve problems in computational biochemistry
- **The goal:** design better-performing algorithms and offer their implementations for academic and industrial use to
    - study the existing molecular systems faster
    - study the existing molecular systems in more detail
    - study larger molecular systems

---

<!-- 2 min -->
## Introduction

- a **supercomputer** is a computer with a high level of performance as compared to a general-purpose computer
    - also called high performance computer (HPC)
- measure: **floating-point operations per second** (FLOPS)
    - PC -> teraFLOPS; Bura -> 100 teraFLOPS
    - modern HPC -> 1 do 10 petaFLOPS, top 442 petaFLOPS
    - future exascalar HPC -> 1+ exaFLOPS
- nearly exponential growth of FLOPS over time (source: [Wikimedia Commons File:Supercomputers-history.svg](https://commons.wikimedia.org/wiki/File:Supercomputers-history.svg))

---

<!-- 1 min -->
![bg 80% Computing power of the top 1 supercomputer each year, measured in FLOPS](https://upload.wikimedia.org/wikipedia/commons/8/81/Supercomputers-history.svg)

---

<!-- 4 min -->
## More heterogeneous architectures require complex programming models

- different types of accelerators
    - GPUs (half, single, double precision), TPUs/TCGPUs, FPGAs
    - in-network and in-storage computation (e.g. [Blue](https://store.nvidia.com/en-us/networking/store/?page=1&limit=9&locale=en-us&search=bluefield)[Field](https://www.pny.com/professional/explore-our-products/networking-solutions/nvidia-bluefield-data-processing-units) [DPU](https://www.nvidia.com/en-us/networking/products/data-processing-unit/))
- several projects to adjust existing software for the exascale era
    - Software for Exascale Computing (SPPEXA)
    - Exascale Computing Project (ECP)
    - European High-Performance Computing Joint Undertaking (EuropHPC JU)

---

<!-- 2 min -->
## SPPEXA project GROMEX

- full title: Unified Long-range Electrostatics and Dynamic Protonation for Realistic Biomolecular Simulations on the Exascale
- principal investigators:
    - Helmut Grubmüller (Max Planck Institute for Biophysical Chemistry, now Multidisciplinary Sciences)
    - Holger Dachsel (Jülich Supercomputing Centre)
    - Berk Hess (Stockholm University)
- molecular dynamics visualization: [Electron transport chain](https://youtu.be/LQmTKxI4Wn4)

---

<!-- 1 min -->
## GROMEX

> The particle mesh Ewald method (PME, currently state of the art in molecular simulation) does not scale to large core counts as it suffers from a communication bottleneck, and does not treat titratable sites efficiently.
>
> The fast multipole method (FMM) will enable an efficient calculation of long-range interactions on massively parallel exascale computers, including alternative charge distributions representing various forms of titratable sites.

[SPPEXA Projects - Phase 2 (2016 - 2018)](http://www.sppexa.de/general-information/projects-phase-2.html)

---

<!-- 5 min -->
## Planned GROMACS developments (1/2)

- heterogeneous parallelism presently uses GPUs, could be expanded to also use DPUs
    - custom-silicon Anton 2 supercomputer's hardware and software architecture could be an inspiration
    - identification of packets that do not need to be delivered to all receivers and force reductions
    - NVIDIA already offers [free developer kits](https://developer.nvidia.com/converged-accelerator-developer-kit) to [interested parties](https://developer.nvidia.com/converged-accelerator-developer-kit-interest) for similar purposes

---

<!-- 4 min -->
## Planned GROMACS developments (2/2)

- molecular dynamics simulations are periodic
- [simulation box types](https://manual.gromacs.org/current/reference-manual/algorithms/periodic-boundary-conditions.html#some-useful-box-types): cubic, rhombic dodecahedron
- present design and implementation of the fast multipole method only supports cubic boxes
    - it is possible to also support rhombic dodecahedron: ~30% less volume => ~30% less computation time per step required
- potentially apply for HrZZ UIP (if announced)

---

<!-- 3 min -->
## Potential GROMACS developments

- Monte Carlo (Davide Mercadante, University of Auckland)
    - many efforts over the years, none with broad acceptance
    - should be rethought, and then designed and implemented from scratch with exascale in mind
- polarizable simulations using the classical Drude oscillator model (Justin Lemkul, Virginia Tech)
    - should be parallelized for multi-node execution
- other drug design tools such as Random Acceleration Molecular Dynamics (Rebecca Wade, Heidelberg Institute for Theoretical Studies and Daria Kokh, Cancer Registry of Baden-Württemberg)

---

<!-- 3 min -->
## Interesting developments in the broader computational biochemistry ecosystem

- RDKit
- RxDock 😇
- data science: KNIME
- applied artificial intelligence, machine learning, neural networks, and deep learning
    - e.g. [Deep Docking: A Deep Learning Platform for Augmentation of Structure Based Drug Discover](https://pubs.acs.org/doi/10.1021/acscentsci.0c00229)
    - AlphaFold

---

<!-- 2 min -->
## RDKit and RxDock

- RDKit, the open-source chemoinformatics toolkit
    - official blog frequently talks about [molecular](https://greglandrum.github.io/rdkit-blog/posts/2021-05-21-similarity-search-thresholds.html) [fingerprints](https://greglandrum.github.io/rdkit-blog/posts/2022-01-04-number-of-unique-fp-bits.html)
    - [database cartridge for PostgreSQL](https://www.rdkit.org/docs/Cartridge.html) offers scalable molecular storage and retrieval
- RxDock predicts binding modes of small molecules to proteins and nucleic acids

- in the late 2021. we submitted the study of [36 million molecules](https://www.emolecules.com/) binding to SARS-CoV-2 main protease

---

<!--  2 min-->
## KNIME

- [analytics platform](https://www.knime.com/software-overview)
- set of Lego-like blocks that can be connected via GUI
    - replaces scripting, easy to use for non-programmers
- state of the art of computational biochemistry methods:
    - [Schrödinger on KNIME Hub](https://hub.knime.com/schroedinger/extensions/com.schrodinger.knime.feature.nodes/latest)
    - [Vernalis](https://hub.knime.com/vernalis/extensions/com.vernalis.knime.feature/latest)
    - [RDKit](https://hub.knime.com/search?q=RDKit)

---

<!-- 2 min -->
## AlphaFold

- protein structure != protein sequence
    - sequence: 100 EUR and 20 minutes
    - structure: [O(100 000) EUR and many years](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)
- earlier computational solutions: [Folding@home](https://foldingathome.org/)
- enabled by the evolution of GPUs and developments in AI
- [Forbes calls it The Most Important Achievement In AI—Ever](https://www.forbes.com/sites/robtoews/2021/10/03/alphafold-is-the-most-important-achievement-in-ai-ever/): 'Critical Assessment of Protein Structure Prediction co-founder and long-time protein folding expert John Moult put the AlphaFold achievement in historical context: "This is the first time a serious scientific problem has been solved by AI."'

---

<!-- 3 min -->
## Potential development: HTVSDB

- web interface and REST API to a molecular database and molecular docking service
- open-source software so it could be hosted locally by other research groups at other universities
- unique features: molecular recommendation, federation
- based on RDKit, RxDock, and potentially AlphaFold
- long-term evolution on a best-effort basis

---

<!-- 2 min -->
![bg Drug Discovery and Development Pipeline](https://www.frontiersin.org/files/Articles/542271/fphar-11-00733-HTML/image_m/fphar-11-00733-g001.jpg)

---

**Figure source:** Cui W, Aouidate A, Wang S, Yu Q, Li Y and Yuan S (2020) [Discovering Anti-Cancer Drugs via Computational Methods.](https://www.frontiersin.org/articles/10.3389/fphar.2020.00733/full) Front. Pharmacol. 11:733. doi: 10.3389/fphar.2020.00733

---

<!-- 1 min -->
## Unified vision and specific applications

- high-throughput virtual screening and molecular dynamics simulations could be offered as a service to Croatian, regional, and EU research groups
    - methods -> algorithms -> applications
- e.g. industry/academic group has a molecular target
    - RxDock, RDKit (HTVSDB, KNIME/Python automation): millions of molecules -> tens of molecules
    - GROMACS (KNIME/Python automation) -> tens of molecules -> several molecules

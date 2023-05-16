---
marp: true
author: Vedran Miletić
theme: uncover
class: _invert
paginate: true
---

# Bura HPC

## Using supercomputing resources at University of Rijeka for academic and industrial goals

### Vedran Miletić, [Faculty of Informatics and Digital Technologies](https://www.inf.uniri.hr/), [University of Rijeka](https://uniri.hr/)

---

## Link to these notes

### `miletic.net/dsc2023`

---

## FIDIT

- founded in 2008. as Department of Informatics at UniRi
- became **F**aculty of **I**nformatics and **Di**gital **T**echnologies in 2022.
- research (computer vision, pattern recognition, natural language processing, machine translation, e-learning, digital transformation, parallel applications on supercomputers) and teaching

![FIDIT logo bg right:30% 90%](https://upload.wikimedia.org/wikipedia/commons/1/14/FIDIT-logo.svg)

---

## GASERI

- **G**roup for **A**pplications and **S**ervices on **E**xascale **R**esearch **I**nfrastructure
- focus: optimization of computational biochemistry applications for running on modern exascale supercomputers
- vision: high performance algorithms readily available for academic and industrial use in state-of-the-art open-source software

![GASERI logo bg right:20% 90%](../../images/gaseri-logo.png)

---

## Details about Bura HPC

- hosted at and maintained by [Center for Advanced Computing and Modelling](https://cnrm.uniri.hr/) at [University of Rijeka](https://uniri.hr/)
- [Top 500](https://www.top500.org/) in [November 2015](https://www.top500.org/lists/top500/2015/11/) ranked it in [441th place](https://www.top500.org/lists/top500/list/2015/11/?page=5)
- [computing resources](https://cnrm.uniri.hr/bura/)
- [cooling system and efficiency](https://cnrm.uniri.hr/cooling-system-and-efficiency/), [data center](https://cnrm.uniri.hr/data-center/)
- [software](https://cnrm.uniri.hr/software/)
- [access](https://cnrm.uniri.hr/applications/)
- [tutorials](https://cnrm.uniri.hr/tutorials/)

---

## Accessing Bura HPC

### [bura.uniri.hr](https://bura.uniri.hr/)

---

## Slurm workload manager

- [quick start user guide](https://slurm.schedmd.com/quickstart.html)
- [sinfo](https://slurm.schedmd.com/sinfo.html)
- [squeue](https://slurm.schedmd.com/squeue.html)
- [scontrol](https://slurm.schedmd.com/scontrol.html)
- [srun](https://slurm.schedmd.com/srun.html)
- [sbatch](https://slurm.schedmd.com/sbatch.html)
- [scancel](https://slurm.schedmd.com/scancel.html)

---

## GROMACS molecular dynamics simulation software

- [popular](https://scholar.google.com/scholar?q=gromacs) open-source [molecular dynamics](https://en.wikipedia.org/wiki/Molecular_dynamics) simulation [software](https://en.wikipedia.org/wiki/Category:Molecular_dynamics_software) focused on supercomputer usage
- examples: [freezing of ice](https://www.youtube.com/watch?v=ZAsUIqv3xb8), [bacterial cytoplasm](https://www.youtube.com/watch?v=5JcFgj2gHx8)
- [paralelization techniques](https://manual.gromacs.org/current/reference-manual/algorithms/parallelization-domain-decomp.html)

---

## Input files for GROMACS

- we'll be using [a free GROMACS benchmark set](https://www.mpinat.mpg.de/grubmueller/bench) prepared by [Cartsen Kutzner](https://www.mpinat.mpg.de/grubmueller/kutzner)
    - credit: [Theoretical and Computational Biophysics Research Group of Helmut Grubmüller](https://www.mpinat.mpg.de/grubmueller), [Max Planck Institute for Multidisciplinary Sciences](https://www.mpinat.mpg.de/en)
- mirrored by our group
    - `https://files.group.miletic.net/Bura-HPC/`
    - we'll use `wget` or `curl` to download them to Bura

---

## Further reading

- [projects](https://cnrm.uniri.hr/projects/)
- [publications](https://cnrm.uniri.hr/publications/)

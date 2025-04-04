---
author: Vedran Miletić
---

# Projects

## DPU offload of force reduction calculations in molecular dynamics simulations

**Project Type:** Research

**Teaching Impact:** N/A

**Results Timeframe:** More than 12 months

**Industry Segment (Vertical):** Healthcare & Life Sciences; HPC / Supercomputing

**Interest Area:** Data Center / Cloud Computing

**Project Abstract/Description:**

[GROMACS](https://www.gromacs.org/) is the leading open-source molecular dynamics software package. Thanks to [Szilard Pall (KTH)](https://www.kth.se/profile/pszilard) and other developers, it has been providing an option to use CUDA for the offload of force calculation on GPUs for a decade now. Specialized supercomputers for molecular dynamics use custom network cards with accelerators for force calculations. The goal is to use commercial off-the-shelf components, specifically [BlueField](https://www.nvidia.com/en-us/networking/products/data-processing-unit/) to enable wide industry and academic use of DPUs for force reduction calculations in GROMACS. We see an in-network acceleration to have significant potential in the identification of packets that do not need to be delivered to all receivers and force reductions; a similar approach is used by Anton 2 ([doi:10.1109/IPDPS.2015.42](https://doi.org/10.1109/IPDPS.2015.42)). Potentially, DPUs could also accelerate 3D FFTs. Our research would be a proof-of-concept for the next generation of DPUs as it would greatly benefit from lower latency, specifically when using [GPUDirect RDMA](https://network.nvidia.com/products/GPUDirect-RDMA/) on DPUs to obtain data from GPUs.

**Hardware Grant:** 2 x [NVIDIA BlueField-2 E-Series 100GbE Crypto Enabled](https://store.nvidia.com/en-us/networking/store/product/MBF2M516A-CEEOT/NVIDIAMBF2M516ACEEOTBlueField2ESeriesDPU100GbECryptoEnabled/)

## National Competence Centres in the framework of EuroHPC (EuroCC)

The aims of the EuroCC project are:

- to set up a network of National Competence Centres in high-performance computing (HPC) across Europe in 31 participating, member, and associated states,
- to provide a broad service portfolio tailored to the respective national needs of industry, academia, and public administrations.

The underlying motivation is the increase of the national HPC competencies and the usability of HPC technologies accross Europe.

More information:

- [The EuroCC and CASTIEL Projects (EuroCC ACCESS)](https://www.eurocc-access.eu/the-projects/)
- [O projektu EuroCC (Hrvatski centar kompetencija za računarstvo visokih performansi, HR HPC CC)](https://www.hpc-cc.hr/o-projektu-eurocc) (in Croatian)

The project has received funding from the [European High-Performance Computing Joint Undertaking (EuroHPC JU)](https://eurohpc-ju.europa.eu/) under European Union's [Horizon 2020](https://ec.europa.eu/programmes/horizon2020/en/home) research and innovation funding programme (grant agreement ID: [951732](https://cordis.europa.eu/project/id/951732)).

---
author: Vedran Miletić
---

# Osnovni pojmovi paralelnog, distribuiranog i heterogenog računarstva

## Paralelna obrada

Paralelizacija čini da se više radnji, operacija ili proračuna izvodi istovremeno. **Radnje, operacije ili proračuni moraju biti takvi da je paralelizacija moguća.** Ako to nije slučaj, vrijeme izvođenja paralelne varijante biti će jednako serijskoj, ili čak duže.

Paralelizacija se može izvesti na tri razine:

- procesi
- [procesne niti](https://en.wikipedia.org/wiki/Thread_(computer_science)) (engl. *threads*)
- [koprogrami](https://en.wikipedia.org/wiki/Coroutines) (engl. *coroutines*), [vlakna](https://en.wikipedia.org/wiki/Fiber_(computer_science)) (engl. *fibers*) ili [zelene niti](https://en.wikipedia.org/wiki/Green_threads) (engl. *green threads*)

Općenito, tri zakona govore o mogućnosti kraćenja vremena izvođenja korištenjem paralelizacije:

- [Littleov zakon](https://en.wikipedia.org/wiki/Little's_law),
- [Amdahlov zakon](https://en.wikipedia.org/wiki/Amdahl's_law),
- [Gustafsonov zakon](https://en.wikipedia.org/wiki/Gustafson's_law).

## Primjene paralelne obrade

Tipovi problema na koje se često primjenjuju metode paralelnog računarstva (preuzeto sa [Wikipedijine stranice o paralelnom računanju](https://en.wikipedia.org/wiki/Parallel_computing)):

- gusta i rijetka linearna algebra,
- spektralne metode (kao što je Cooley-Tukey brza Fourierova transformacija),
- problemi međudjelovanja čestica (kao što je Barnes-Hutova simulacija),
- problemi kvadratične rešetke (kao što su Boltzmannove metode rešetke),
- problemi poligonalna rešetke (kao što se nalaze u analizi konačnih elemenata),
- Monte Carlo simulacije,
- kombinatorna logika (kao što su brute-force kriptografske tehnike),
- obilazak grafa (kao što su algoritmi pretraživanja),
- dinamičko programiranje,
- metode grananja i ograničavanja,
- multi-start metaheuristike,
- grafički modeli (kao što je traženje skrivenih Markovljevih modela i konstrukcija Bayesovih mreža),
- simulacije konačnih automata.

## Razvoj paralelnih računala

Evolucija računalnih sustava prema [AMD](https://en.wikipedia.org/wiki/Advanced_Micro_Devices)-u uključuje tri odvojene ere (preuzeto iz prezentacije [AMD Fusion Fund Overview](https://www.slideshare.net/AMD/amd-fusion-fund-media-presentation)):

- **era jednojezgrenih sustava** traje otprilike do 2004.

    - **primjeri:** AMD Athlon XP i stariji, Intel Pentium 4 i stariji
    - **razvoj omogućuju:** [Mooreov zakon](https://en.wikipedia.org/wiki/Moore's_law), povećanje napona, [Dennardova teorija smanjivanja MOSFET-a](https://en.wikipedia.org/wiki/MOSFET#MOSFET_scaling)
    - **razvoj ograničavaju:** potrošnja energije, složenost arhitekture
    - **programski alati:** [asembler](https://en.wikipedia.org/wiki/Assembly_language) -> [C](https://en.wikipedia.org/wiki/C_(programming_language)) i [C++](https://en.wikipedia.org/wiki/C++) -> [Java](https://en.wikipedia.org/wiki/Java_(programming_language)) i [Python](https://en.wikipedia.org/wiki/Python_(programming_language))

- **era višejezgrenih sustava** traje otprilike do 2011.

    - **primjeri:** AMD Phenom serija, AMD FX serija, Intel Core serija
    - **razvoj omogućuju:** paralelizacija softvera, Mooreov zakon, [SMP](https://en.wikipedia.org/wiki/Symmetric_multiprocessing) arhitektura
    - **razvoj ograničavaju:** potošnja energije, (ne)paralelnost softvera, skalabilnost
    - **programski alati:** [pthreads](https://en.wikipedia.org/wiki/POSIX_Threads) -> [OpenMP](https://en.wikipedia.org/wiki/OpenMP) -> [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)

- **era heterogenih sustava** je [trenutno](https://gpuopen.com/professional-compute/) [hit](https://www.slideshare.net/npinto/cs264-01-introduction) [tema](https://code.google.com/p/stanford-cs193g-sp2010/) u [akademskom](https://people.maths.ox.ac.uk/gilesm/cuda/) [istraživačkom](https://research.nvidia.com/) i [nastavnom](https://www.coursera.org/learn/opencl-fpga-introduction) [okruženju](https://www.pmf.unizg.hr/math/predmet/uupr)

    - **primjeri:** AMD Llano i Trinity (A, E i C serije), Intel Sandy Bridge i Ivy Bridge (Core i3/i5/i7-2000 i 3000 serije), NVIDIA Tegra
    - **razvoj omogućuju:** [programabilni Shaderi](https://en.wikipedia.org/wiki/Shader), masivna paralelizacija softvera, energetski efikasni GPU-i, [GPGPU](https://en.wikipedia.org/wiki/GPGPU)
    - **razvoj ograničavaju:** načini programiranja, pretek zbog komunikacije
    - **programski alati:** [NVIDIA Cg](https://en.wikipedia.org/wiki/Cg_(programming_language)), [Microsoft HLSL](https://en.wikipedia.org/wiki/High_Level_Shader_Language) -> [NVIDIA CUDA](https://en.wikipedia.org/wiki/CUDA), [Microsoft DirectCompute](https://en.wikipedia.org/wiki/DirectCompute) -> [OpenCL](https://en.wikipedia.org/wiki/OpenCL)

## Distribuirano računarstvo

### Arhitektura distribuiranih sustava

!!! todo
    Ovaj dio treba napisati.

[Cluster dio Bure](https://cnrm.uniri.hr/hr/bura/) je primjer distribuiranog sustava sastavljenog od više računala povezanih mrežom.

### Standard Message Passing Interface (MPI)

- sučelje za paralelizaciju aplikacija zasnovano na razmjeni poruka (engl. *message passing*)
- procesi komuniciraju putem cijevi, nema dijeljene memorije

    - jednostavno raspodijeliti procese za izvođenje na više računala
    - **donekle** sličan način rada kao modul `multiprocessing` u Pythonu

- aplikacija se pokreće korištenjem pomoćnih alata kao MPI **posao** (engl. *MPI job*)

    - svaki posao se sastoji od više procesa na jednom ili više računala
    - kod većih sustava alati kao [HTCondor](https://research.cs.wisc.edu/htcondor/) služe za redanje više MPI poslova za izvođenje

- otvoreni standard, specifikacija dostupna na [MPI Forumu](https://www.mpi-forum.org/)

    - zadnja verzija standarda je MPI-3
    - najkorištenije značajke dio (koje ćemo i mi koristiti) su dio i MPI-1 verzije standarda; novosti iz MPI-2 se nešto rjeđe koriste

- podržan u mnogim jezicima: C, C++ ([Boost.MPI](https://www.boost.org/doc/libs/release/libs/mpi/)), Fortran, Java ([MPJ](http://mpjexpress.org/)), Python, Perl, Ruby, ...
- dvije implementacije se aktivno razvijaju; podrška za MPI-2 je postoji već dugo vremena, podrška za MPI-3 je dostupna odnedavno

    - [Open MPI](https://www.open-mpi.org/) (najpopularnija implementacija, nasljednik LAM/MPI)
    - [MPICH2 i MPICH 3](https://www.mpich.org/) (također vrlo popularna implementacija, nasljednik MPICH)
    - velika prednost: standardizirano sučelje => **kompatibilnost na razini izvornog koda**

- dvije implementacije su nekad bile značajne; kompletna podrška za MPI-1, djelomična podrška za MPI-2

    - [LAM/MPI](https://blogs.cisco.com/performance/a-farewell-to-lammpi)
    - [MPICH1](https://www.mpich.org/), koji je postao temelj za [MVAPICH](https://mvapich.cse.ohio-state.edu/)

- korišten u znanosti i istraživanju, dostupno puno tutoriala

    - [Lawrence Livermore National Laboratory tutorial](https://hpc-tutorials.llnl.gov/mpi/)
    - [Argonne National Laboratory tutorial](https://www.mcs.anl.gov/research/projects/mpi/tutorial/)
    - [Kansas University tutorial](http://condor.cc.ku.edu/~grobe/docs/intro-MPI-C.shtml)
    - [Torsten Hoefler (ETH Zürich) tutorials](https://htor.inf.ethz.ch/teaching/mpi_tutorials/)
    - [Google pretraga za "MPI tutorial" daje još mnogo rezultata...](https://www.google.com/search?q=mpi+tutorial)

## Heterogeno računarstvo

### Arhitektura heterogenih sustava

Heterogeno računalo (kakvo postoji u [GPGPU dijelu Bure](https://cnrm.uniri.hr/hr/bura/)) ima dva dijela:

- domaćin (engl. *host*), u našem slučaju osnovni procesor
- uređaj (engl. *device*), u našem slučaju grafički procesor

U budućnosti se očekuje hardver i s tim programske paradigme kod kojih će memorija domaćina i uređaja biti dijeljena i način programiranja će zbog toga biti biti nešto pojednostavljen, ali svi koncepti koje u nastavku opisujemo, kao i način razmišljanja koji koristimo, i dalje će vrijediti.

Osnvoni procesor je dobar za serijsku obradu, a grafički procesor je dobar za masivnu paralelnu obradu podataka. [Heterogeni sustav kombinira oba](https://youtu.be/LIcFJn1TO50).

### HSA Foundation

[AMD](https://www.amd.com/), [ARM](https://www.arm.com/), [Imagination](https://www.imgtec.com/), [MediaTek](https://www.mediatek.com/) i [Texas Instruments](https://www.ti.com/) su [12. lipnja 2012. godine u gradu Bellevu u saveznoj državi Washington osnovali HSA Foundation](https://hsafoundation.com/2012/06/12/hello-hsa-foundation/). Zakladi su se vremeno pridružili [Vivante](https://hsafoundation.com/2012/08/29/vivante-joins-the-hsa-foundation-as-a-member/), [Sonics](https://hsafoundation.com/2012/08/30/sonics-joins-hsa-foundation-to-help-drive-open-standard-for-next-generation-heterogeneous-computing/), [Apical, MulticoreWare, Symbio](https://hsafoundation.com/2012/08/31/hsa-foundation-announces-six-new-members/), [Arteris](https://hsafoundation.com/2012/08/31/4263-2/), [Qualcomm](https://hsafoundation.com/2012/10/03/hsa-foundation-announces-qualcomm-as-newest-founder-member/), [DMP](https://hsafoundation.com/2012/10/30/dmp-joins-heterogeneous-system-architecture-hsa-foundation-to-contribute-its-expertise-in-3d-graphics-and-common-compute/), [LG Electronics](https://hsafoundation.com/2012/10/31/hsa-foundation-announces-lg-electronics-as-newest-member/), [Ceva](https://hsafoundation.com/2012/11/14/ceva-and-tensilica-are-new-hsa-foundation-members/) i [Tensilica](https://hsafoundation.com/2013/03/18/tensilica-joins-hsa-foundation-to-help-establish-standards-for-embedded-heterogeneous-computing/).

Iako **heterogeno računarstvo** (engl. *heterogeneous computing*) ne počinje s **heterogenom sustavskom arhitekturom** (engl. *Heterogeneous System Architecture*, HSA), ona je danas vjerojatno najbolji primjer istoga. Na stranicama [zaklade HSA Foundation](https://hsafoundation.com/) može se pronaći [opis temeljnih značajki HSA](https://hsafoundation.com/2012/08/31/what-is-heterogeneous-system-architecture-hsa/) i [argumentacija zašto HSA predstavlja evoluciju računarstva](https://hsafoundation.com/2012/08/28/hsa-represents-the-evolution-of-computing/). [Prva specifikacija](https://hsafoundation.com/standards/) je [objavljena 29. svibnja 2013. godine](https://hsafoundation.com/2013/05/29/hsa-foundation-announces-first-specification/).

### Tehnologije NVIDIA Compute Unified Device Architecture (CUDA) i OpenCL

!!! todo
    Ovaj dio treba napisati.

### Sadašnjost i budućnost heterogenih sustava

Pored toga, AMD opisuje razvoje heterogene arhitekture sustava kroz 4 etape (preuzeto iz AMD-ove prezentacije [HSA Overview](https://www.slideshare.net/hsafoundation/hsa-overview)):

- **fizička integracija** (2011. godine, Llano) -- CPU i GPU nalaze se na jednom siliciju; CPU i GPU dijele jedinicu za upravljanje memorijom
- **optimizacija platforme** (2012. godina, Trinity i 2013. godina, [Richland](https://youtu.be/MQcjEA3it90)) -- CPU i GPU dijele cjelokupnu količinu memorije, GPU može alocirati koliko god je potrebno; CPU i GPU imaju zajedničko dinamičko upravljanje energijom
- **arhitekturalna integracija** (2014. godina, Kaveri) -- CPU i GPU vide unificirani memorijski prostor, pokazivači se mogu prosljeđivati u oba smjera; GPU može pristupati CPU međuspremnicima
- **sustavska integracija** (2015. godina, Carrizo) -- GPU multitasking, specifično mogućnost da se izvede context switch između grafičkih i compute aplikacija; GPU pre-emption, specifično mogućnost da se zaustavi proces koji se dugo izvodi radi procesa koji će se izvoditi kraće, prioriteti izvođenja aplikacija

Komercijalni čip koji sadrži CPU i GPU na jednom siliciju, zasnovan na heterogenoj sustavskoj arhitekturi, AMD naziva [APU](https://en.wikipedia.org/wiki/AMD_Accelerated_Processing_Unit). Intel i NVIDIA, unatoč tome što imaju vrlo slične čipove, zasad ovaj termin nisu prihvatili.

---
author: Vedran Miletić, Martina Ašenbrener Katić, Željko Svedružić, Patrik Nikolić
---

# Application form

## Thematic HRZZ Call for Scientific Research Proposals

### Infectious Diseases Caused by Corona Viruses and the Social and Educational Aspects of the Pandemic

#### (Call identifier: IP-CORONA-2020-04)

##### Application Form[^1]

## SARS-CoV-2 high-throughput virtual screening campaign and target protein inhibitors drug design

### COVIDOCK

#### Dr. Martina Ašenbrener Katić

## Basic Information about the Project Proposal

### Principal Investigator (name, address, e-mail)

|  |  |
| --- | --- |
| Full name | Martina Ašenbrener Katić |
| University/Institution | University of Rijeka Department of Informatics |
| Address | Radmile Matejčić 2, 51000 Rijeka, Croatia |
| E-mail | <masenbrener@inf.uniri.hr> |
| Phone | +385 51 584 703 |

### Project Proposal

|  |  |
| --- | --- |
| Title | SARS-CoV-2 high-throughput virtual screening campaign and target protein inhibitors drug design |
| Keywords | database modeling, high-throughput virtual screening, molecular dynamics, high performance computing |
| Research field | Computational and medicinal chemistry |
| Total budget requested (EUR) | 196 540,91 |
| Total budget requested (HRK) | 1 494 975,66 |

### Research topics

Please mark the topic(s) you are applying for:

- [ ] **1. Immune response and development of novel diagnostic approaches to COVID-19**
    - [ ] 1.1. Existing diagnostic methods for COVID-19 are still not sufficiently developed and more sensitive, specific, precise, simpler, cheaper and faster methods are urgently needed.
    - [ ] 1.2. In order to understand viral mutations and their significance for immune control as well as the tendency to accumulate mutations, including specific variations for this micro-region, sequencing of the viral genome needs to be performed.
    - [ ] 1.3. Research of the immune response, pathogenesis as well as clinical outcomes of SARS-CoV-2 infections.
- [ ] **2. Development of new vaccines, treatments as well as drugs and agents for the COVID-19 inhibition**
    - [x] 2.1. Developing novel antiviral agents against SARS-CoV-2 and designing drugs for COVID-19 therapy.
    - [ ] 2.2. Investigating protective capacity of different immuno-therapeutic antiviral agents against SARS-CoV-2 in experimental animal models.
    - [ ] 2.3. Developing experimental vaccines and vaccine vectors for COVID-19 and studying their protective effect.
- [ ] **3. Social aspects of the COVID-19 pandemic**
    - [ ] 3.1. Psyhosocial consequences of emergency measures in the control of the COVID-19 pandemic, resilience and risk factors for mental health and well-being of the general and vulnerable population.
- [ ] 3.2. Crisis communication related to the COVID-19 pandemic for alleviating fear, stereotyping and uncertainty, strengthening responsible individual and group behaviour and trust in public authorities.
- [ ] **4. Educational aspects of the pandemic**
    - [ ] 4.1. Effects of changing paradigms of leaning and evaluating processes, successful models for achieving distance-learning outcomes in primary, secondary and higher education. Technological prerequisites and development of human resources for successful organisation and implementation of remote teaching.
    - [ ] 4.2. Use of big data, learning analytics and especially data from virtual environments for advancing teaching and as the grounds for informed, strategic decision-making in education.

### Summary of the Project Proposal

**Background:** In the last twenty years, eight scientists have received Nobel prices in Chemistry for the development of computational methods for the analysis of structure and function of biomolecules. Our university has recently purchased for 4.7 million euros a supercomputer with EU funds. The supercomputer is primarily used for the studies of the structure and function of biomolecules and drug-design.

**Approach:** The available structures of viral proteins and RNA molecules are used to screen and design a new set of drugs using the commercial databases and molecular docking protocols. The selected results are then described to finer details using molecular dynamics and quantum mechanics protocols. In a slightly more challenging approach, the structures of the viral proteins are not readily available. We can design the structures using bioinformatics and homology modeling tools that require very high computational power which we have. The prepared molecular structures can be used in standard screening, docking, and molecular dynamics protocols. Experimental results can be used in pharmacophore modeling protocols to optimize the initial results. Based on our earlier experiences we can target proteases, enzymes in DNA and RNA metabolism, biological membranes and protein--protein interactions.

Screening for and design of new drugs are the major aims of our work. Prior to the screening campaign, we will develop new open-source computational infrastructure, with two major outcomes. A new database containing all commercially available small-molecule ligands will be developed and deployed on the University of Rijeka. A docking server with a web-based user interface will be developed and interfaced with the compound database. The docking server will use the database for sourcing of the molecules for the high-throughput virtual screening.

Our approach offers major advantages that can bypass the problems that have traditionally plagued the pharmaceutical industry: our protocols are faster, cheaper, versatile, and offer minimal risks. The computational methods allow us to define different families of potential drug candidates that can be sorted based on price, ADME properties, time of development, risk of development, the likelihood of binding to the target site, and the mechanism of action. Such categorization can allow the creation of multiple drug-design strategies that can be adapted to the different demands and risk levels. Most notably we are developing new drugs using commercial databases. This allows us to buy the lead compounds for affordable prices that can bypass expensive and slow organic synthesis protocols.

**Our team:** The major advantage of our work is that we have a well-integrated team of computer science experts and experimental biochemistry with experience in drug design. The strong expertise in computational science allows us to implement the newest computational methods on the most sophisticated machines as soon as they become available. The approach makes us financially more efficient, we use open source software that does not require expensive software license fees. The financial expenses are primarily directed to maximize the hardware and the manpower that can directly work on the drug-design.

Computer and information scientists in our group have been recently trained in the newest methodology in the premier European and Croatian institutions. The computational and experimental biochemists in our group have well-documented successful work experience in small and big pharmaceutical industries, in academic and clinical laboratories. Broad experience on different laboratory problems and in different work environments is the major requirement for success in the early stages of drug development. They will be in charge of selecting realistic targets and evaluation of the impact of the obtained computational results on the drug-design efforts in the wet lab.

A number of superb groups in computational biochemistry exists, as well as a number of superb experimental biochemistry laboratories. However, the communication between the two worlds remains a limiting factor, especially at the level of most advanced cutting edge technologies, and diverse background of this team will alleviate said issue.

**Conclusions:** This call for funding aims to fund projects that can in a short time collect the maximal possible amount of information to understand COVID-19 physiology and design drugs specific for SARS-CoV-2 targets. We can address all challenges in a fast and efficient manner using the high-performance computing, a large set of cutting-edge software tools, commercial compound databases, and a unique mix of complementary expertise in computational and medicinal
biochemistry.

## Section A -- Project Proposal

### a. Research objectives

#### 1. High-throughput virtual screening campaign of SARS-CoV-2 targets with commercially available compounds

A novel coronavirus (SARS-CoV-2) has caused 1 353 361 confirmed cases and 79 235 confirmed deaths globally, according to the COVID-19 situation report from WHO on April 8^th^, 2020^[@ref1](#ref1)^, and the numbers are still growing. Similar to the SARS-CoV outbreak in 2002, SARS-CoV-2 causes severe respiratory problems -- fever, coughing, difficulties in breathing and shortage of breath are common symptoms. Human-to-human transmission occurs rapidly, and even human-to-feline contact was observed. To treat infected patients and slow this pandemic down, rapid and efficient development of highly specific antiviral drugs is of the highest urgency^[@ref2](#ref2)^.

SARS-Cov-2 binds to a specific enzyme found on the surface of human cells. The enzyme, named angiotensin-converting enzyme 2, or ACE2 for short, lines endothelial cells in the lungs' blood vessels. The enzyme happens to be the same target for SARS-CoV, the virus that caused SARS.

Like other kinds of coronaviruses, SARS-CoV-2 forms a series of "spike" proteins that protrude from its core, giving it the shape of the sun's corona. Those spike proteins latch onto ACE2, which allows the virus to fuse its membrane to the cell membrane. Once inside, SARS-CoV-2 acts like any other virus -- it hijacks the human cell, forcing the cell to produce many copies of the virus, spurring an infection.

A team of researchers designed a Michael acceptor inhibitor N3 using computer-aided drug design (CADD), which can specifically inhibit multiple CoV M^pro^s, including those from SARS-CoV and MERS-CoV^[@ref3; @ref4; @ref5; @ref6](#ref2)^. They constructed a homology model of COVID-10 M^pro^ and used molecular docking to see if N3 could target this new CoV M^pro^. A docking pose showed that it could fit inside the substrate-binding pocket. To assess the efficacy of N3 for COVID-19 M^pro^, a kinetic analysis was performed^[@ref3](#ref3)^. A progress curve showed that it is a time-dependent irreversible inhibitor of this enzyme. The shape of this curve supports the mechanism of two-step irreversible inactivation. The inhibitor first associates with COVID-19 M^pro^ with a dissociation constant Ki; then, a stable covalent bond is formed between N3 and M^pro^ (E-I). The evaluation of this time-dependent inhibition requires both the equilibrium-binding constant Ki (designated as k2/k1) and the inactivation rate constant for covalent bond formation k3. However, N3 exhibits very potent inhibition of COVID-19 M^pro^, such that measurement of Ki and k3 proved difficult. When very rapid inactivation occurs, kobs/\[I\] was utilized to evaluate the inhibition as an approximation of the pseudo-second-order rate constant (k3/Ki)^[@ref3](#ref3)^. The value of kobs/\[I\] of N3 for COVID-19 M^pro^ was determined to be 11,300±880 M^-1^s^-1^, suggesting this Michael acceptor has potent inhibition.

To elucidate the inhibitory mechanism of this compound, they determined the crystal structure of COVID-19 M^pro^ in complex with N3 to 2.1-Å resolution^[@ref7](#ref7)^. The crystal structure is available in the PDB protein databank as 6LU7 entry^[@ref8](#ref8)^ (shown on Fig. 1). All residues (residues 1--306) are visible in electron density maps. Each protomer is composed of three domains. Domains I (residues 8--101) and II (residues 102--184) have an antiparallel β-barrel structure. Domain III (residues 201--303) contains five α-helices arranged into a largely antiparallel globular cluster and is connected to domain II through a long loop region (residues 185--200). COVID-19 M^pro^ has a Cys--His catalytic dyad, and the substrate-binding site is located in a cleft between Domain I and II. These features are similar to those of other M^pros^ reported previously^[@ref4; @ref6; @ref9; @ref10; @ref11](#ref4)^. The electron density map shows that N3 binds in the substrate-binding pocket in an extended conformation (Fig. 2, Fig. 3), with the inhibitor backbone atoms forming an antiparallel sheet with residues 164--168 of the long strand~155-168~ on one side, and with residues 189--191 of the loop linking domains II and III.

![Figure 1: 6LU7 - The crystal structure of COVID-19 main protease in complex with an inhibitor N3.]

![Figure 2: COVID-19 main protease and N3 2D interaction diagram.]

![Figure 3: COVID-19 main protease binding site.]

Here we detail the specific interactions of N3 with M^pro^. The electron density shows that the Sγ atom of C145-A forms a covalent bond (1.8-Å) with the Cβ of the vinyl group, confirming that the Michael addition has occurred. The S1 subsite has an absolute requirement for Gln at the P1 position. The side chains of F140-A, N142-A, E166-A, H163-A, H172-A, S1-B (from protomer B), and main chains of F140-A and L141-A are involved in S1 subsite formation, which also includes two ordered water molecules (named W1 and W2). The lactam at P1 inserts into the S1 subsite and forms a hydrogen bond with H163-A. The side chain of Leu at the P2 site deeply inserts into the hydrophobic S2 subsite, which consists of the side chains of H41-A, M49-A, Y54-A, M165-A, and the alkyl portion of the side chain of D187-A. The side chain of Val at P3 is solvent-exposed, indicating that this site can tolerate a wide range of functional groups. The side chain of Ala at P4 side is surrounded by the side chains of M165-A, L167-A, F185-A, Q192-A and the main chain of Q189-A, all of which form a small hydrophobic pocket. P5 makes van der Waals contacts with P168-A and the backbone of residues 190--191. The bulky benzyl group extends into the S1 site, possibly forming van der Waals interactions with T24-A and T25-A. Also, N3 forms multiple hydrogen bonds with the main chain of the residues in the substrate-binding pocket, which also helps lock the inhibitor inside the substrate-binding pocket (Fig. 1, Extended Data Fig. 2).

The structure of SARC-CoV-2 M^pro^ in complex with N3 provides a model for rapidly identifying lead inhibitors to target SARS-CoV-2 M^pro^ through *in silico* screening. To achieve this, the existing and new molecular docking software will be combined to screen the compounds efficiently.

#### 2. Virtual screening of bioactive peptide and synthetic peptides against SARS-CoV-2 and ACE2 protein-protein interface

Recent crystallographic studies of the SARS-COV-2 receptor-binding domain (RBD) and full-lengths human ACE2 receptor revealed key amino acid residues at the contact interface between the two proteins and provide valuable structural information that can be leveraged for the development of disruptors specific for the SARS-CoV-2/ACE2 protein-protein interaction (PPI)^[@ref12; @ref13](#ref12)^. Small-molecule inhibitors are effective in binding to targeted binding sites, such as the aforementioned SARS-CoV-2 and N3 complex binding site, but are less effective at disrupting extended protein binding interfaces^[@ref14](#ref14)^. Peptides, on the other hand, offer a synthetically accessible solution to disrupt protein-protein interactions by binding at interface regions containing multiple contact "hot spots"^[@ref15](#ref15)^. Analyzing the RBD-ACE2 co-crystal structure, we found that SARS-CoV-2-RBD/ACE2 interface spans a large elongated surface area, as is common for protein-protein interactions.

We hypothesize that disruption of the viral SARS-CoV-2-RBD and host ACE2 interaction with peptide-based binders will prevent virus entry into human cells, offering a novel opportunity for therapeutic intervention. To accomplish this, all proven bioactive peptides will be screened on the SARS-CoV-2-RBD domain using the aforementioned methodology, and the best peptides will serve as a model for the development of novel, synthetic peptides that will specifically target SARS-CoV-2-RBD surface. The peptides will then be evaluated using molecular dynamic simulations to observe specific polar and non-polar protein--protein interactions between the functional domains of SARS-CoV-2-RBD and ACE2.

#### 3. All-atom molecular dynamic simulations of single- and double-anti-viral therapy effect of SARS-CoV-2 binding to ACE2 target

The co-crystal structure of SARS-CoV-2-RBD with ACE2 (PDB: 6M17, Fig. 4) will be used for the initial structure for molecular dynamic (MD) simulation for peptide binders development. Standard MD simulation procedures will be used during this project, particularly conformational exploration simulations of protein-protein docking, model refinement and testing of homology modeled SARS-CoV-2 proteins, addition and removal of ligands from HTVS campaign, mutation and modification of specific peptide chains, application of mechanical force to characterize strength of protein-protein interactions, and conformational changes of protein-protein supramolecular complex with various ligands and synthetic peptides designed to disrupt said complex.

![Figure 4: (Top left) SARS-CoV-2-RBD/ACE2 protease domain (PD) cryo-EM structure (PDB code: 6M17) in cartoon mode. ACE2 is colored in green, while SARS-CoV-2-RBD is colored in wheat color. (Top right) SARS-CoV-2-RBD/ACE2 protease domain (PD) cyro-EM structure. Selection of peptide fragments making key contacts colored in red. SARS-CoV-2 RBD is shown in surface mode, colored in wheat color. (Bottom) SARS-CoV-2-RBD/ACE2 key contact close-up. Coloration and presentation identical to top figures.]

The same MD simulations will be used for SARS-CoV-2 M^pro^ in complex with N3 compound (PDB: 6LU7) for the same purpose. Additionally, MD simulations of M^pro^ with screened ligands will be run, using the same research methodology. At the very end of this research project, MD simulations with both small-molecule inhibitors of M^pro^ and peptide binders on the SARS-CoV-2-RBD/ACE2 PPI region will be run to assess the effect of double anti-viral therapy.

### b. Research methodology

#### 1. Design and development of the compound database (Martina Ašenbrener Katić, Sanja Čandrlić, Vedran Miletić)

The potential inhibitors, sourced from vendors of commercially available compounds, will be stored in a compound database. The existing databases were evaluated for the purpose: the freely accessible databases such as ChemSpider^[@ref16](#ref16)^ by Royal Society of Chemistry and PubChem^[@ref17](#ref17)^ by U.S. National Library of Medicine are not open source software that can be extended, interfaced, and changed according to the research requirements of the various projects and in particular the project proposed here. Therefore, the compound database that satisfies the following requirements will be designed and developed:

- Able to hold millions of compounds
- Searchable according to name, structure, and molecular descriptors
- Able to be updated with new data from commercial databases

The compound database will be interfaced with the docking server described in the following to allow the easy screening of selected compounds.

#### 2. Design and development of the docking server (Martina Ašenbrener Katić, Vedran Miletić, Davide Mercadante)

The engine of the screening workflow that will be used is RxDock (formerly RiboDock(r)^[@ref18](#ref18)^ and rDock^[@ref19](#ref19)^), a fast, versatile, and open-source program for docking ligands to proteins and nucleic acids developed by Vernalis R&D, University of York, University of Barcelona, RxTx, and others^[@ref20](#ref20)^. RxDock and its predecessor rDock were specifically designed for high-throughput virtual screening (HTVS) usage and are already used for HTVS of compounds that could bind to SARS-CoV-2 proteins, in particular by Galaxy^[@ref21; @ref22](#ref21)^ and COVID.SI^[@ref23](#ref23)^.

Gorgulla et al. recently had a paper accepted in Nature^[@ref24](#ref24)^ that described VirtualFlow, an approach to running HTVS campaigns on supercomputers managed by a batch system such as SLURM. Two types of tasks are performed by VirtualFlow: ligand preparation (checking the validity of the input files, a transformation of the input files to supported formats, cutting input files into pieces for multi-process execution, etc.) and virtual screening. The approach of VirtualFlow will be used for running HTVS campaigns with RxDock on Bura supercomputer. Instead of running jobs directly via scripts, a custom docking server with a web interface will be used. The interface will allow the user to specify a set of ligands and the target site on the protein and will use a connection to Bura for scheduling docking jobs and gathering results after the jobs have finished executing for presenting them back to the user.

The docking server and the compound database will be deployed at the University of Rijeka Department of Informatics for usage by the researchers working on this project and by the wider research community (in a limited amount so the resource usage does not hinder the progress of this project). These two services will further be maintained for at least two years after the project ends. Additionally, both the docking server and the compound database software developed in the scope of this project will be released under open-source licenses so they can be studied, improved, and deployed elsewhere by other researchers for other purposes after this project ends.

#### 3. Drug repositioning (Davide Mercadante, Željko Svedružić)

Using newly developed infrastructure, a high-throughput virtual screening campaign of all FDA approved drugs will be performed. Crystal structures of SARS-CoV-2 M^pro^ structures (6LU7, 6YB7, 6M17, and any other relevant M^pro^ crystal structure or *in silico* model) be prepared accordingly before HTVS campaign itself. Hydrogens will be added, and structures will subsequently be minimized using the custom force field. The protonation state of His residues and the orientation of hydroxyl groups, Ans residues, and Gln residues will be optimized as well. Compounds from FDA approved drugs database will be protonated and 3D structure will be generated using the MMFF94s force field.

Pharmacophore approaches are one of the crucial tools in drug discovery^[@ref25](#ref25)^. IUPAC defines pharmacophore as "an ensemble of steric and electronic features that is necessary to ensure the optimal supramolecular interactions with a specific biological target and to trigger (or block) its biological response"^[@ref26](#ref26)^. A pharmacophore model can be constructed in two ways:

- ligand-based: by overlapping a set of active molecules and deriving prevailing chemical features that are essential for their bioactivity
- structure-based: by probing possible interaction points between the macromolecular target and ligands

Structure-based pharmacophore modeling will be performed to obtain potential structures and a subset of the database matching the structure will be used for screening.

Once the protein structures are prepared, binding pockets will be defined and calculated using the cavity search engine of RxDock. Molecular docking itself will be carried out using the RxDock docking engine, considering the ligands as flexible but treating the receptor as a rigid structure. A post-docking minimization of the most promising docking results according to the docking score will be performed using the GROMACS software suite. The best candidates will be used in all-atom MD simulations of their disruption effects of SARS-CoV-2/ACE2 enzyme binding. The systems will be explicitly solvated to perform incrementally increasing MD simulation -- starting simulations with initial binders will be evaluated using 20 ns all-atom simulations, with an increase to 50, 100 and 200 ns simulations in each passing run -- using GROMACS on University of Rijeka's Bura supercomputing cluster.

#### 4. HTVS campaign of commercially available small-molecule ligands (Željko Svedružić)

Using the same methodology as in b.3 chapter, commercially available small-molecule ligands from all major and relevant commercial vendors will be evaluated in the HTVS campaign. Three specific subsets of ligands will be defined before HTVS campaign itself:

- Molecules with similar SAR properties of known SARS-CoV-2 ligands (such as N3)
- Molecules that comply with the Lipinski Rule of Five
    - Molecules that comply with the Lipinsky Rule of Three will be specially favored
- Molecules that comply with Ghose filter

Each subset will be evaluated independently. The best candidates from each group will be used in all-atom MD simulations of their disruption effects of SARS-CoV-2/ACE2 enzyme binding using the same MD protocol from b.3 chapter. Parts of this will be done by contractors.

#### 5. Peptide binder design (Davide Mercadante, Željko Svedružić)

Bioactive peptide structures and descriptors will be stored in newly developed database. For simulating these peptides inside a complex with SARS-CoV-2-RBD domain in GROMACS, usage of a99SB-disp and CHARMM36m force fields will be considered for their ability to model coil-to-structure transitions^[@ref27; @ref28](#ref27)^. Synthetic helical peptide sequence of spike-binding peptide 1 derived from the ɑ1 helix of ACE2 peptidase domain (ACE2-PD) and synthetic peptide sequences similar to the bioactive peptides that performed best in MD simulations will also be evaluated (example MD simulation of protein--protein interactions from prior work shown in Fig. 5 and Fig. 6). Parts of this will be done by contractors.

![Figure 5: Molecular dynamics protocols can calculate very subtle protein-protein interactions.]

![Figure 6: During the *molecular dynamics* simulations the complementary surfaces adapt to make tight complementary surfaces. The orange lines illustrate that positive and negative surface match (red-blue surfaces) and that each protrusion on the surfaces binds at the matching cavity on the complementary surface. Drugs that target protein-protein interactions are crucial for blocking of the assembly of the invective COVID-19 particles.]

### c. Time schedule of the main activities

(Assuming the beginning of the project in June 2020, the end of the project in December 2021 and the period of 18 months divided into three half-year periods H1, H2, and H3.)

The project will start with the design and development of the docking server which is expected to take approximately 4 months. The design and development of the compound database will follow and it is expected to take approximately 5 months. Therefore, by mid-H2 both the compound database and the docking server should be up and running in a usable state.

In parallel with the design and development during H1, homology modeling and bioinformatics approaches with SARS-CoV-2 sequences will be used first to compare SARS-CoV-2 proteins and RNA molecules with protein and RNA molecules PDB with similar structures. The special emphasis will be given to human and viral molecules. Molecular dynamics protocols will be used for definition of druggable site, like the active sites, or hydrophobic pockets in the protein surface that can facilitate important physiological interaction for the viral molecules.

The aim of the virtual screening during H2 and the beginning of H3 is to define a group of commercially available molecules that will be sorted based on stability of their interaction with the target molecules, on their agreement with Lipinski rules, and the price of full testing. The aim is to define a group of molecules that share, to different extent, properties of the most desirable drugs. Drug development is risky and expensive process and presented approach is used to maximally minimize the risks and the drug development costs.

During the virtual screening campaigns, the issues in the compound database and the docking server software will be addressed as they are discovered.

Our desire is to include in the screening as many viral molecules as possible. Some of the target molecules: RNA-dependent RNA polymerase form SARS-CoV-2 virus (PDB codes: 6M71, 7BTF), SARS-CoV-2 spike protein (PDB code: 6VXX), protease of SARS-CoV-2 (PDB codes: 6YB7 and 6LU7). The pharmacological investigation of similar molecules have been covered in our earlier computational studies: regulation of human DNA methylation^[@ref29; @ref30](#ref29)^, the forces that regulate protein-protein interactions^[@ref31](#ref31)^, and proteolysis in pathogenic processes^[@ref32; @ref33](#ref32)^.

A workshop is planned for local and international scientific community where, among other presentations, our newly developed infrastructure and drug discovery pipeline will be demonstrated during the H3 period. The attendants of the workshop will be given an option to experiment further with the drug discovery pipeline on their own for a month after the workshop ends in order to be able to try if the developed infrastructure works for their research needs.

The dissemination of results on conferences and workshops will happen during H2 and H3. Due to the pandemic, many changes took place in the schedule of the scientific events world wide and it's hard at the moment to predict which conferences will take place and when. However, it's possible to note which conferences we will target in terms of organizers and topics; we'll aim to present the results of our research at future iterations of events such as EMBO's Advances and Challenges in Biomolecular Simulations^[@ref34](#ref34)^ (still scheduled), CECAM's Current and future challenges in modeling protein interactions^[@ref35](#ref35)^ (postponed indefinitely), SCI-RSC Workshop on Computational Tools for Drug Discovery^[@ref36](#ref36)^ (moved to online format), CCG's European UGM and Conference^[@ref37](#ref37)^ (cancelled), and similar events.

During the H3 most of the papers resulting from the research done in the project will be written and submitted. For publication, the journals from the publishers such as MDPI, PeerJ, PLOS, ACS, and Springer will be the targeted. After that, the project will be wrapped up.

### d. Expected results

We are confident that we can perform full HTVS campaign of the major commercial databases on RNA-dependent RNA polymerase from SARS-CoV-2 virus, SARS-CoV-2 spike protein, protease of SARS-CoV-2 and other emerging targets an also develop the software infrastructure to make future endeavours of the same type easier. In the last several months, the research group of Dr. Svedružić has screened about 500 thousand compounds from Maybridge database as potential inhibitors of the human DNA methyltransferase Dnmt1 that target the enzyme active site. They have also screened more than 100 thousand compounds that bind at phenylalanine rich region as potential inhibitors of ubiquitin binding to OTUB1 protein. Considering these results in the context of the present project, our expected results out of HTVS campaign of FDA approved drugs and commercially available compounds (several million compounds) are couple of dozens of compounds with binding affinity lower of 1 µM.

Every molecule sequence that shows homology with the SARS-CoV-2 genes will be generated and the resulting model will be minimized using coarse-grain and all-atom MD protocols.

The effect of single- and double-anti-viral therapy will be evaluated. The effect of small-molecule ligands and peptide binding to SARS-CoV-2 targets and disruption of SAR-CoV-2/ACE2 complex (single-anti-viral therapy) and combined small-molecule ligand and peptide disruption of SARS-CoV-2/ACE2 complex will be ranked.

### e. Practical applications and/or socioeconomic benefits of research results

Viral epidemics pose an ongoing threat to public health and will continue to pose an unknown risk both to the human well-being and robustness of the world economy. Even with all scientific advancement and improvements of private, national and global healthcare systems during the 20th and early 21st century, humankind still lacks the meaning of dealing with emergent viral strains. Both human immune systems and the global pharmaceutical industry have means to neutralize only the viral strains that previously emerged in the population, and even then, quarantine and various other population isolation systems might be the only solution. One such example is the SARS-CoV-1 virus, the precursor of the SARS-Cov-2 virus responsible for the current epidemic and global shutdown. As of today, there is still no vaccine or drug that affects the SARS-CoV-1 virus, a virus that caused the SARS epidemic that officially ended on June 5th, 2003.

Viral pandemics regularly occurred during the 21st century, the most notable ones being the SARS epidemic (2002-2003), the swine flu epidemic (2009-2010), Middle East respiratory syndrome outbreaks (2012-ongoing), and current COVID-19 (2019-ongoing) pandemic. Viruses will continue to mutate and new epidemics in the future are imminent and unavoidable. Furthermore, novel vaccines cannot be developed prior to specific viral mutation and vaccine development is a process that takes time. During that time, a part of the population will inevitably be affected by whichever viral strain emerges, which is the exact current situation with the COVID-19 pandemic. No matter whether vaccine development is successful or not, and how fast its development is, there will be a large number of patients that will be in dire need of an efficient and specific drug for their ailment. Therefore, viral drug discovery remains one of the most important fields of medicinal chemistry and molecular biology, both in the academic and industrial settings.

These are the reasons the project is proposing to develop a novel computational infrastructure and research model for viral drug discovery and development. The first problem we want to address is the issue of chemical synthesis during the drug discovery phase. Chemical synthesis of novel compounds is a time-consuming process with a variable rate of success and during an ongoing epidemic/pandemic, it is a step that would best be avoided. Our solution for that is the development of a database that queries all major commercial compound databases for compounds that are available for purchase at the starting moment of a HTVS campaign.

The second issue we want to address is a lack of proper and well-documented open-source docking platforms that can easily be scaled on any hardware, from basic workstations to complex supercomputers in any of the major HPC centers, and deployed at any time by any research group without any licensing issues. We will be using RxDock as a novel docking platform specifically designed for HTVS campaigns and open-source any novel features we develop on this project. In that way, any progress we make during this specific project on developing novel drugs for the SAR-Cov-2 virus can be deployed in the future for any project by any research group on whatever (super)computer they have access to. The key component we will develop during this project is a docking server built on top of RxDock, interfaced with compound database for sourcing of ligands and the Bura supercomputer for performing molecular docking. The web interface to the docking server will make the setup of large HTVS campaigns straightforward regardless of the device used by the user for accessing the interface.

The third and final problem we plan to solve is the discovery of novel SARS-CoV-2 drugs using infrastructure we develop in the first part of the project. Since the main targets for viral infections are various proteases, we will utilize our extensive research experience in protease mechanism research^[@ref29; @ref31; @ref32; @ref33; @ref38](#ref29)^ to simulate ligand-protein interactions of the most promising hit candidates from HTVS campaigns. The simulations will be run on local HPC Bura infrastructure. Insights gained during this project will be applied for all subsequent projects with the infrastructure we develop ready to deploy at any time.

Since all parts of the proposed infrastructure are and are going to be open-source, resulting infrastructure will feature the following characteristics:

- *reproducibility* -- all results generated on our software infrastructure will be easily repeated by any research group
- *speed of setup* -- once developed, all software solutions will be easily deployed within a working day on any hardware by any research group or private company
- *collaboration* -- multiple research groups will be able to collaborate using a single server setup

Furthermore, the novel drug discovery infrastructure we develop is target-agnostic and will retain its full usefulness after this project completion, and for any subsequent and inevitable viral epidemic. Some of the ongoing projects that will utilize this infrastructure include the development of DNA methyltransferase inhibitors, ubiquitin iso-peptidase inhibitors, and γ-secretase modulators.

### f. Resources

#### Bura supercomputer (existing equipment)

Bura supercomputer is University of Rijeka's largest computational resource and it is maintained by Center for Advanced Computing and Modeling^[@ref39](#ref39)^. Bura consists of three major parts^[@ref40](#ref40)^:

- Cluster: The multicomputer system consists of 288 compute nodes with two Xeon E5 processors per node (24 physical cores per node). A total of 6912 processor cores are available. Each node has 64 GB of memory and 320 GB of disk space, respectively, while the computer nodes together have 18 TB of memory and 95 TB of disk space.
- GPGPU: Four heterogeneous nodes with two Xeon E5 processors and two NVIDIA Tesla K40 general purpose graphics processing units (GPUs) are available.
- SMP: A multiprocessor system with a large amount of shared memory. SMP is made up of 16 Xeon E7 processors with a total of 256 physical cores, 12 TB of memory and 245 TB of local storage. Two nodes are available.

Cluster part will be used in this project to perform high-throughput virtual screening and some molecular dynamics (MD) simulations. However, most of the MD simulations will be performed on the new GPU compute nodes described below.

An entire node in the cluster costs 384 EUR per month. We will be using 18 nodes for the duration of 6 months after the compound database and the docking server are developed.

**Equipment maintenance costs:**

- H1 (small molecular dynamics runs): 6 nodes x 6 months x 384 EUR per node per month = 13824 EUR
- H2 and H3 (molecular docking): 12 nodes x 6 months = 27648 EUR
- H3 (workshop and post-workshop usage): 6 nodes x 1 month = 2304 EUR
- **Total:** 43 776 EUR

#### GPU compute nodes (new equipment)

Bura's GPGPU part presently has 4 GPU nodes with 8 NVIDIA Tesla K40 GPUs. Since GPUs continue to evolve much faster than CPUs, the GPUs in Bura are very much out of date; high-end consumer GPUs of today offer two to three times more FLOPS than Tesla K40 and even more than that if one takes performance per watt into account.

Kutzner et al. benchmarked GROMACS on GPU systems^[@ref41; @ref42](#ref41)^ and found that these FLOPS translate to MD performance as both non-bonded kernels and particle mesh Ewald kernels running on the GPU gain throughput proportional to or better than the increase in FLOPS. GROMACS doesn't require double precision and therefore runs just as well on relatively cheap consumer NVIDIA GeForce and AMD Radeon GPUs as it does on much more expensive professional NVIDIA Tesla and AMD Radeon Instinct GPUs.

Since the GPUs in Bura do not offer competitive performance for MD simulations, a replacement of K40 GPUs in Bura with recent high-end consumer GPUs was considered in order to perform MD simulations faster and more efficiently. However, due to warranty constraints imposed on Bura by the vendor, the replacement of GPUs is not possible at the moment and, due to physical space constraints of the node casing which is designed for the particular generation of GPUs used at the time, the replacement is likely to be hard to do in the future when the warranty expires.

In particular, the second GROMACS on GPU systems benchmark paper^[@ref42](#ref42)^ from Kutzner et al. explores the option of adding GPUs to existing nodes without GPUs and it was chosen as the approach that will be used here. SuperMicro 2023US-TR4 is a server that can be customized for various usage scenarios and can optionally fit two GPUs. The choice remains between two NVIDIA GeForce and two AMD Radeon GPUs per node; since NVIDIA license for consumer (GeForce) cards^[@ref43](#ref43)^ prohibits data center usage starting from 2018^[@ref44](#ref44)^, the most cost-effective option is AMD Radeon. Finally, high-end consumer Radeon 5900XT GPUs are expected to be released in Fall 2020 at the price of approximately 1500 EUR a piece and they are expected to offer very competitive performance--power and performance--cost ratios, especially when compared to professional GPUs.

Therefore, six SuperMicro 2023US-TR4 servers will be obtained and then independently upgraded with two Radeon 5900XT GPUs each.

**New equipment costs:**

- SuperMicro 2023US-TR4: 6 nodes x 8150 EUR (an offer including just one example of a possible customization for the requested configuration from Senetic is attached; the availability and the pricing of the components is expected to vary in the coming months) = 48900 EUR
- Two Radeon 5900XT per node: 12 GPUs x 1500 EUR = 18000 EUR
- Two L-shaped low profile 8-pin power cables per GPU: 24 x 14.99 USD (a price listing from Moddiy is attached) x 0.9053 EUR/USD (exchange rate on the 30^th^ of March 2020) = 325,69 EUR
- **Total:** 67 225,69 EUR

#### Dedicated servers, rack, and switch (new equipment)

The molecular docking platform supporting the research requires two servers for the compound database and one for the docking server and its web interface. The compound database will be running in a multi-instance configuration supporting both load balancing (distribution of workloads across multiple nodes) and failover (automatic switching to a standby instance upon the failure of one of the instances).

SuperMicro 2023US-TR4 can be customized to fit the requirements of the compound database and docking server (it's the same server used for the GPUs above which simplifies the rack configuration). To house these nodes, we require a rack such as Dell NetShelter SX 42U. To connect these nodes to the network, we require an Ethernet switch such as Cisco WS-C2960X-24PS-L. To connect these nodes to the power source, we require two power strips of German type with surge protection such as Intellinet 714006 19" 1.5U Rackmount 7-Output Power PDU.

**New equipment costs:**

- SuperMicro 2023US-TR4: 3 nodes x 8150 EUR (an offer including just one example of a possible customization for the requested configuration from Senetic is attached; the availability and the pricing of the components is expected to vary in the coming months) = 24450 EUR
- Dell NetShelter SX 42U (an offer from Senetic is attached): 1200 EUR
- Cisco WS-C2960X-24PS-L (an offer from Senetic is attached): 1000 EUR
- Intellinet 714006 19" 1.5U Rackmount 7-Output Power PDU (a price listing from Senetic is attached): 2 x (239,74 kn / 7,606435 kn per EUR) = 63,04 EUR
- **Total:** 26 713,04

#### Recording equipment (new equipment)

The workshop that we will organize will be recorded and streamed to a platform such as YouTube. Basic hardware and software equipment will be provided by the University of Rijeka. The remaining equipment required for recording and streaming is a high-quality consumer web camera such as Logitech Brio and a lavalier microphone for recording the speaker voice such as Rode smartLav+.

**New equipment costs:**

- Logitech Brio (a price listing from Amazon.de is attached): 271,18 EUR
- Rode smartLav+ (a price listing from Amazon.de is attached): 55,00 EUR
- **Total:** 326,18 EUR

### g. Ethical issues

None.

## Section B -- Principal Investigator

### General information

**Principal Investigator's full name:** Martina Ašenbrener Katić

#### Education history

2017: PhD in Informatics

:   Institution: University of Rijeka Department of Informatics, Rijeka, Croatia

2009: Professor of Mathematics and Informatics (equivalent M.Ed. mathematics and informatics)

:   Institution: University of Rijeka Faculty of Arts and Sciences, Rijeka, Croatia

#### Employment history

2019--now: Assistant Professor

:   Institution: University of Rijeka Department of Informatics, Rijeka, Croatia

2017--2019: Senior Research and Teaching Assistant

:   Institution: University of Rijeka Department of Informatics, Rijeka, Croatia

2009--2017: Research and Teaching Assistant

:   Institution: University of Rijeka Department of Informatics, Rijeka, Croatia

2009: Teacher of mathematics and informatics

:   Institution: Ivan Goran Kovačić elementary school, Delnice, Croatia

### Principal Investigator's track record related to the topic of the Call and the project proposal

#### 1. Papers in peer-reviewed scientific journals, book chapters, conference proceedings, monographs etc. (in the last five years)

1. Čandrlić, S., **Ašenbrener Katić, M.** & Jakupović, A. Preliminary Multi-lingual Evaluation of a Question Answering System Based on the Node of Knowledge Method. in (eds. Arai, K. & Bhatia, R.) 998--1009 (Springer, 2020). [doi:10.1007/978-3-030-12388-8_69](https://doi.org/10.1007/978-3-030-12388-8_69).
2. Sinčić, P., **Ašenbrener Katić, M.** & Čandrlić, S. Perception and attitudes on the effects of digital technologies application: a survey. in *Central European Conference on Information and Intelligent Systems* 85--92 (Faculty of Organization and Informatics Varazdin, 2019).
3. Petković, M., Čandrlić, S. & **Ašenbrener Katić, M.** Automatic Testing of Web Applications with the Support of Geb Web Driver. *Zbornik Veleučilišta u Rijeci* **7**, 185--207 (2019).
4. Čandrlić, S., **Ašenbrener Katić, M.** & Pavlić, M. A system for transformation of sentences from the enriched formalized Node of Knowledge record into relational database. *Expert Systems with Applications* **115**, 442--464 (2019).
5. **Asenbrener Katic, M.**, Candrlic, S. & Pavlic, M. Modeling of verbs using the node of knowledge conceptual framework. in *2018 41st International Convention on Information and Communication Technology, Electronics and Microelectronics (MIPRO)* 1022--1027 (2018). [doi:10.23919/MIPRO.2018.8400187](https://doi.org/10.23919/MIPRO.2018.8400187).
6. Rauker Koch, M., **Asenbrener Katic, M.** & Pavlic, M. Fable representation in FNOK and DNOK formalisms using the NOK conceptual framework. in *Annals of DAAAM and Proceedings of the International DAAAM Symposium* 439--445 (2017). [doi:10.2507/28th.daaam.proceedings.061](https://doi.org/10.2507/28th.daaam.proceedings.061).
7. Pavlic, M., Han, Z. D., Jakupovic, A., **Asenbrener Katic, M.** & Candrlic, S. Adjective representation with the method Nodes of Knowledge. in *2017 40th International Convention on Information and Communication Technology, Electronics and Microelectronics (MIPRO)* 1221--1226 (2017). [doi:10.23919/MIPRO.2017.7973610](https://doi.org/10.23919/MIPRO.2017.7973610).
8. **Asenbrener Katic, M.**, Candrlic, S. & Pavlic, M. Comparison of Two Versions of Formalization Method for Text Expressed Knowledge. in *Beyond Databases, Architectures and Structures. Towards Efficient Solutions for Data Analysis and Knowledge Representation* (eds. Kozielski, S., Mrozek, D., Kasprowski, P., Małysiak-Mrozek, B. & Kostrzewa, D.) 55--66 (Springer International Publishing, 2017). [doi:10.1007/978-3-319-58274-0_5](https://doi.org/10.1007/978-3-319-58274-0_5).
9. **Ašenbrener Katić, M.** Sustav za integraciju relacijske baze podataka i jednostavnih rečenica prirodnog jezika primjenom konceptualnog okvira 'Node of knowledge' (A system for integration of relational database and simple sentences in natural language using the 'Node of knowledge' conceptual framework). (2017). \[PhD thesis\]
10. **Ašenbrener Katić, M.**, Čandrlić, S. & Holenko Dlab, M. Introducing collaborative e-learning activities to the e-course "Information systems". in *Proceedings of the 39th International Convention MIPRO 2016* 917--922 (2016).
11. Rauker Koch, M., Pavlić, M. & **Ašenbrener Katić, M.** Homonyms and Synonyms in NOK Method. *Procedia Engineering* **100**, 1055--1061 (2015).
12. Pavlić, M. *et al.* Question answering system in natural language. in *Razvoj poslovnih i informatičkih sustava CASE 27* 5--16 (2015).
13. Holenko Dlab, M., **Asenbrener Katic, M.** & Candrlic, S. Ensuring formative assessment in e-course with online tests. in *2015 10th International Conference on Computer Science Education (ICCSE)* 322--327 (2015). [doi:10.1109/ICCSE.2015.7250264](https://doi.org/10.1109/ICCSE.2015.7250264).
14. **Asenbrener Katic, M.**, Pavlic, M. & Candrlic, S. The Representation of Database Content and Structure Using the NOK Method. *Procedia Engineering* **100**, 1075--1081 (2015).

#### 2. List of previous PI's research projects that are related to the present project proposal, role in the project (PI/team member) and source of funding

- 2019--now: Development of the NOK platform for the transformation of natural language sentences into a relational database, team member, funded by University of Rijeka

- 2018--now: Development of the International Education Program Veleri-OI IoT School, team member, funded by EU European Social Fund

- 2018--2019: Development of a QA system on top of a relational database, team member, funded by University of Rijeka

- 2014--2015: RFID (Internet of Things) based animal individual behaviour intelligent identification technology and application in traceability (REMALLOY), team member, funded by Croatian Ministry of Science, Education, and Sports

- 2013--2018: Extending the information system development methodology with artificial intelligence methods, team member, funded by University of Rijeka

- 2012--2015: Software Engineering -- Computer Science Education and Research Cooperation, team member, funded by German Academic Exchange Service (DAAD) Stability Pact for South Eastern Europe

- 2009--2014: Methodology of information systems analysis and modeling, team member, funded by Croatian Ministry of Science, Education, and Sports

### Other relevant achievements

- Received the University of Rijeka Award for Teaching Excellence in 2018/2019 academic year (awarded 21^st^ of May, 2019)

- Paper review for Computational Intelligence journal (Wiley) (2 papers in 2019--2020)

- Paper review for MIPRO conference (21 papers from 2011--2019)

## Section C -- Research Group

**List all persons involved in the implementation of the proposed research.**

| Full name | Title | Organization | Country | Status in Project (doctoral student/postdoctoral researcher/researcher/technician/consultant) |
| --------- | ----- | ------------ | ------- | --------------------------------------------------------------------------------------------- |
| [Martina Ašenbrener Katić](https://portal.uniri.hr/Portfolio/987) | PhD in Informatics | University of Rijeka Department of Informatics | Croatia | Researcher |
| [Sanja Čandrlić](https://portal.uniri.hr/Portfolio/487) | PhD in Information Sciences | University of Rijeka Department of Informatics | Croatia | Researcher |
| [Vedran](https://vedran.miletic.net/) [Miletić](https://www.miletic.net/) | PhD in Computer Science | University of Rijeka Department of Informatics | Croatia | Researcher |
| [Davide Mercadante](https://lab.mercadante.net/) | PhD in Chemistry | University of Auckland School of Chemical Sciences | New Zealand | Researcher |
| [Željko Svedružić](https://svedruziclab.github.io/) | PhD in Biochemistry | University of Rijeka Department of Biotechnology | Croatia | Researcher |

### Short Description of the Research Group

**Martina Ašenbrener Katić, PhD** is researching in the area of data modeling and teaching software engineering, among other subjects. She will use her expertise in these areas in the process of design of the docking server and the compound database, in particular creating the data model for the database. Last but certainly not the least, Dr. Ašenbrener Katić will be leading the project. University of Rijeka Department of Informatics supports the project application and their support will ensure that Dr. Ašenbrener Katić has enough time and appropriate resources to work on the project.

**Sanja Čandrlić, PhD** received her PhD at the University of Zagreb Faculty of Humanities and Social Sciences. She is researching and teaching in the areas of information system analysis, process and data modeling, software engineering, and team software development, among others. She participated in two scientific projects supported by the Ministry of science, education and sports of the Republic of Croatia and led by Dr. Mile Pavlić: 2002--2006 on "Methodology of Information System Development" (0009026) and 2007--2014 on "Methodology of Information Systems Analysis and Modeling" (318-0161199-1354). She is presently the co-PI of the project "Development of the International Education Program Veleri-OI IoT School" funded by EU European Social Fund. Dr. Čandrlić will be applying her expertise in perform data modeling for the compound database and assist Dr. Ašenbrener Katić with the compound database design.

**Vedran Miletić, PhD** is researching in the interdisciplinary area of computer science, biochemistry, and biophysics. He obtained his PhD from University of Zagreb Faculty of Electrical Engineering and Computing and from 2015. to 2018. he worked as a postdoctoral researcher in Molecular Biomechanics group at Heidelberg Institute for Theoretical Studies in Heidelberg, Germany. He has authored parts of code in a number of free and open source computational chemistry software projects, including CP2K^[@ref45](#ref45)^, GROMACS^[@ref46; @ref47; @ref48](#ref46)^, and RxDock^[@ref49](#ref49)^, and also supercomputing infrastructure software projects such as Mesa^[@ref50](#ref50)^ and Clang/LLVM^[@ref51; @ref52](#ref51)^. Dr. Miletić will be implementing the docking server and compound database and also performing the infrastructure maintenance. He will also be serving as a liaison between the research group and the Centre for Advanced Computing and Modeling as he is a member of the University of Rijeka Supercomputing resources council.

**Davide Mercadante, PhD** is researching in the area of biochemistry. He obtained his PhD at The University of Auckland in 2012. and afterwards worked as a postdoctoral researcher at Heidelberg Institute of Theoretical Studies and Heidelberg University in Heidelberg, Germany (2013--2017) and The University of Zurich in Zurich, Switzerland (2017--2019). In 2015. he was nominated for the Postdoctoral Award from the Biophysical Society, Intrinsically disordered Proteins subgroup and awarded a prize for excellent research at the Heidelberg Institute for Theoretical Studies for the contribution to the understanding of intrinsically disordered protein behaviour. Dr. Mercadante has studied the binding interaction between enzymes and small molecule ligands and also macromolecular organization^[@ref53; @ref54; @ref55](#ref53)^. He will be working on evaluation and optimization of algorithms used in the process of molecular docking, and also prototyping parts of the docking server and compound database implementation.

**Željko Svedružić, PhD** is researching in the area of biochemistry, medicinal chemistry and enzymology. He obtained his PhD from Oklahoma State University Department of Biochemistry and Molecular Biology, and from 1998 to 2000 worked as a postdoctoral researcher at University of California Santa Barbara Department of Chemistry and Epigenx Pharmaceuticals, Inc. on Enzymology and inhibitors of mammalian and bacterial cytosine DNA methyltransferases project, and from 2001 to 2002 as a postdoctoral researcher at Duke University Medical Center Department of Biochemistry working on enzymology of protein phosphatase CDC25B with Cdk2/CycA protein complex as the substrate project. He worked as a senior scientist from 2003 to 2006 at Washington State University School of Molecular Biosciences on a research project of DNA damage induced changes in DNA flexibility and DNA-nucleosome interaction, and DNA repair in nuclear extracts. During his career, Dr. Svedružić specialized in enzymology, and *in vitro*, *in vivo* and *in silico* approaches for studies of structure and function of biomolecules. He will be performing HTVS campaign setup and analysis, subsequent SAR analysis, and MD simulation design, planning and analysis.

Additionally, interested M.Sc. and B.Sc. students will be selected help on the project while working on their diploma and master theses.

## References

### ref1

World Health Organization. Coronavirus disease 2019 (COVID-19) Situation Report - 79.

### ref2

Zhang, G., Pomplun, S., Loftis, A. R., Loas, A. & Pentelute, B. L. *The first-in-class peptide binder to the SARS-CoV-2 spike protein*. <https://biorxiv.org/lookup/doi/10.1101/2020.03.19.999318> (2020) doi:10.1101/2020.03.19.999318.

### ref3

Yang, H. *et al.* Design of Wide-Spectrum Inhibitors Targeting Coronavirus Main Proteases. *PLoS Biol.* **3**, e324 (2005).

### ref4

Xue, X. *et al.* Structures of Two Coronavirus Main Proteases: Implications for Substrate Binding and Antiviral Drug Design. *J. Virol.* **82**, 2515--2527 (2008).

### ref5

Ren, Z. *et al.* The newly emerged SARS-Like coronavirus HCoV-EMC also has an "Achilles' heel": current effective inhibitor targeting a 3C-like protease. *Protein Cell* **4**, 248--250 (2013).

### ref6

Wang, F. *et al.* Michael Acceptor-Based Peptidomimetic Inhibitor of Main Protease from Porcine Epidemic Diarrhea Virus. *J. Med. Chem.* **60**, 3212--3216 (2017).

### ref7

Jin, Z. *et al.* Structure of Mpro from COVID-19 virus and discovery of its inhibitors. *bioRxiv* 2020.02.26.964882 (2020) doi:10.1101/2020.02.26.964882.

### ref8

Bank, R. P. D. RCSB PDB - 6LU7: The crystal structure of COVID-19 main protease in complex with an inhibitor N3. <https://www.rcsb.org/structure/6LU7>.

### ref9

Anand, K., Ziebuhr, J., Wadhwani, P., Mesters, J. R. & Hilgenfeld, R. Coronavirus Main Proteinase (3CLpro) Structure: Basis for Design of Anti-SARS Drugs. *Science* **300**, 1763--1767 (2003).

### ref10

Yang, H. *et al.* The crystal structures of severe acute respiratory syndrome virus main protease and its complex with an inhibitor. *Proc. Natl. Acad. Sci.* **100**, 13190--13195 (2003).

### ref11

Zhao, Q. *et al.* Structure of the Main Protease from a Global Infectious Human Coronavirus, HCoV-HKU1. *J. Virol.* **82**, 8647--8655 (2008).

### ref12

Wrapp, D. *et al.* Cryo-EM structure of the 2019-nCoV spike in the prefusion conformation. *Science* **367**, 1260--1263 (2020).

### ref13

Yan, R. *et al.* Structural basis for the recognition of SARS-CoV-2 by full-length human ACE2. *Science* **367**, 1444--1448 (2020).

### ref14

Smith, M. C. & Gestwicki, J. E. Features of protein-protein interactions that translate into potent inhibitors: topology, surface area and affinity. *Expert Rev. Mol. Med.* **14**, e16 (2012).

### ref15

Josephson, K., Ricardo, A. & Szostak, J. W. mRNA display: from basic principles to macrocycle drug discovery. *Drug Discov. Today* **19**, 388--399 (2014).

### ref16

Pence, H. E. & Williams, A. ChemSpider: An Online Chemical Information Resource. *J. Chem. Educ.* **87**, 1123--1124 (2010).

### ref17

Kim, S. *et al.* PubChem Substance and Compound databases. *Nucleic Acids Res.* **44**, D1202--D1213 (2016).

### ref18

Morley, S. D. & Afshar, M. Validation of an empirical RNA-ligand scoring function for fast flexible docking using RiboDock®. *J. Comput. Aided Mol. Des.* **18**, 189--208 (2004).

### ref19

Ruiz-Carmona, S. *et al.* rDock: A Fast, Versatile and Open Source Program for Docking Ligands to Proteins and Nucleic Acids. *PLOS Comput. Biol.* **10**, e1003571 (2014).

### ref20

RxDock. *rxdock.gitlab.io* <https://rxdock.gitlab.io/>.

### ref21

Anton Nekrutenko *et al.* *galaxyproject/SARS-CoV-2: second biorxiv release*. (Zenodo, 2020). doi:10.5281/zenodo.3685264.

### ref22

Cheminformatics \| COVID-19 analysis on usegalaxy. <https://covid19.galaxyproject.org/cheminformatics/>.

### ref23

Skupnostna znanost in boj proti koronavirusu \| COVID.SI. *COVID.SI* <https://covid.si/>.

### ref24

Gorgulla, C. *et al.* An open-source drug discovery platform enables ultra-large virtual screens. *Nature* 1--8 (2020) doi:10.1038/s41586-020-2117-z.

### ref25

Yang, S.-Y. Pharmacophore modeling and applications in drug discovery: challenges and recent advances. *Drug Discov. Today* **15**, 444--450 (2010).

### ref26

Wermuth, C.-G., Robin Ganellin, C., Lindberg, P. & Mitscher, L. A. Chapter 36 - Glossary of Terms Used in Medicinal Chemistry (IUPAC Recommendations 1997). in *Annual Reports in Medicinal Chemistry* (ed. Bristol, J. A.) vol. 33 385--395 (Academic Press, 1998).

### ref27

Robustelli, P., Piana, S. & Shaw, D. E. Developing a molecular dynamics force field for both folded and disordered protein states. *Proc. Natl. Acad. Sci.* **115**, E4758--E4766 (2018).

### ref28

Huang, J. *et al.* CHARMM36m: an improved force field for folded and intrinsically disordered proteins. *Nat. Methods* **14**, 71--73 (2017).

### ref29

Miletić, V., Odorčić, I., Nikolić, P. & Svedružić, Ž. M. In silico design of the first DNA-independent mechanism-based inhibitor of mammalian DNA methyltransferase Dnmt1. *PLOS ONE* **12**, e0174410 (2017).

### ref30

Svedruzic, Z. M. Mammalian Cytosine DNA Methyltransferase Dnmt1: Enzymatic Mechanism, Novel Mechanism-Based Inhibitors, and RNA-directed DNA Methylation. <https://www.ingentaconnect.com/content/ben/cmc/2008/00000015/00000001/art00007> (2008) doi:info:doi/10.2174/092986708783330700.

### ref31

Svedružić, Ž. M., Odorčić, I., Chang, C. H. & Svedružić, D. Substrate Channeling via a Transient Protein-Protein Complex: The case of D-Glyceraldehyde-3-Phosphate Dehydrogenase and L-Lactate Dehydrogenase. *bioRxiv* 2020.01.22.916023 (2020) doi:10.1101/2020.01.22.916023.

### ref32

Svedružić, Ž. M., Popović, K. & Šendula-Jengić, V. Modulators of γ-Secretase Activity Can Facilitate the Toxic Side-Effects and Pathogenesis of Alzheimer's Disease. *PLoS ONE* **8**, (2013).

### ref33

Svedružić, Ž. M., Popović, K. & Šendula-Jengić, V. Decrease in catalytic capacity of γ-secretase can facilitate pathogenesis in sporadic and Familial Alzheimer's disease. *Mol. Cell. Neurosci.* **67**, 55--65 (2015).

### ref34

Advances and Challenges in Biomolecular Simulations. <https://meetings.embo.org/event/20-biomolecular-simulations>.

### ref35

POSTPONED : Current and future challenges in modeling protein interactions. *CECAM - workshop details* <https://www.cecam.org/workshop-details/32>.

### ref36

SCI-RSC Workshop on Computational Tools for Drug Discovery. <https://www.soci.org/events/scirsc-workshop-on-computational-tools-for-drug-discovery>.

### ref37

CCG \| UGM and Conference 2020 \| Europe. <https://www.chemcomp.com/UGM-2020-Europe.htm>.

### ref38

Nikolić, P., Miletić, V., Odorcić, I. & Svedružić, Ž. M. Chapter 5 - In Silico Optimization of the First DNA-Independent Mechanism-Based Inhibitor of Mammalian DNA Methyltransferase DNMT1. in *Epi-Informatics* (ed. Medina-Franco, J. L.) 113--153 (Academic Press, 2016). doi:10.1016/B978-0-12-802808-7.00005-8.

### ref39

HPC Bura -- Center for Advanced Computing and Modelling. <https://cnrm.uniri.hr/hpc-bura/>.

### ref40

Computing resources -- Center for Advanced Computing and Modelling. <https://cnrm.uniri.hr/bura/>.

### ref41

Kutzner, C. *et al.* Best bang for your buck: GPU nodes for GROMACS biomolecular simulations. *J. Comput. Chem.* **36**, 1990--2008 (2015).

### ref42

Kutzner, C. *et al.* More bang for your buck: Improved use of GPU nodes for GROMACS 2018. *J. Comput. Chem.* **40**, 2418--2431 (2019).

### ref43

License for Customer use of NVIDIA GeForce Software. <https://www.nvidia.com/en-us/drivers/geforce-license/>.

### ref44

Nvidia updates GeForce EULA to prohibit data center use. <https://www.datacenterdynamics.com/en/news/nvidia-updates-geforce-eula-to-prohibit-data-center-use/>.

### ref45

*cp2k/cp2k*. (CP2K, 2020). <https://github.com/cp2k/cp2k>

### ref46

*gromacs/gromacs*. (Gromacs, 2020). <https://gitlab.com/gromacs/gromacs>

### ref47

Franz, F., Aponte-Santamaría, C., Daday, C., Miletić, V. & Gräter, F. Stability of Biological Membranes upon Mechanical Indentation. *J. Phys. Chem. B* **122**, 7073--7079 (2018).

### ref48

Herrera-Rodríguez, A. M., Miletić, V., Aponte-Santamaría, C. & Gräter, F. Molecular Dynamics Simulations of Molecules in Uniform Flow. *Biophys. J.* **116**, 1579--1585 (2019).

### ref49

*rxdock/rxdock*. (RxDock, 2020). <https://gitlab.com/rxdock/rxdock>.

### ref50

Mesa / mesa. *GitLab* <https://gitlab.freedesktop.org/mesa/mesa>.

### ref51

Lattner, C. & Adve, V. LLVM: a compilation framework for lifelong program analysis transformation. in *International Symposium on Code Generation and Optimization, 2004. CGO 2004.* 75--86 (2004). doi:10.1109/CGO.2004.1281665.

### ref52

*llvm/llvm-project*. (LLVM, 2020). <https://github.com/llvm/llvm-project>

### ref53

Mercadante, D. *et al.* Bovine β-lactoglobulin is dimeric under imitative physiological conditions: dissociation equilibrium and rate constants over the pH range of 2.5--7.5. *Biophys. J.* **103**, 303--312 (2012).

### ref54

Mercadante, D., Melton, L. D., Jameson, G. B., Williams, M. A. & De Simone, A. Substrate dynamics in enzyme action: rotations of monosaccharide subunits in the binding groove are essential for pectin methylesterase processivity. *Biophys. J.* **104**, 1731--1739 (2013).

### ref55

Mercadante, D., Melton, L. D., Jameson, G. B. & Williams, M. A. Processive pectin methylesterases: the role of electrostatic potential, breathing motions and bond cleavage in the rectification of Brownian motions. *PLoS One* **9**, (2014).

[^1]: The Application Form should not exceed 15 pages, excluding Section C.

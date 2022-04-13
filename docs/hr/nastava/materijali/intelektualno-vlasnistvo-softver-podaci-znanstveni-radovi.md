---
author: Vedran Miletić
---

# Intelektualno vlasništvo nad softverom, podacima i znanstvenim radovima

## Pojam intelektualnog vlasništva

!!! warning
    Autor ovog teksta nije odvjetnik i tekst koji slijedi ne treba uzeti kao pravni savjet na temu aktualnog ili povijesnog zakonodavstva.

U suvremenom društvu na Zapadu i drugdje omogućeno je vlasništvo nad neopipljivim intelektualnim tvorevinama. [Intelektualno vlasništvo](https://en.wikipedia.org/wiki/Intellectual_property) poznaje dvije vrste prava:

- industrijsko vlasništvo (zaštitni znak, patent),
- autorsko pravo.

Za početak patentnog prava uzima se britanski [Statue of Monopolies](https://en.wikipedia.org/wiki/Statute_of_Monopolies) iz 1624. godine, a za početak autorskog prava također britanski [Statue of Anne](https://en.wikipedia.org/wiki/Statute_of_Anne) iz 1710. godine, poznat i pod nazivom Copyright Act 1710.

Prvi kontakt s pojmom intelektualnog vlasništva najčešće se događa kroz [audiovizualni sadržaj](https://www.mpaa.org/research-policy/), iako se [tu ponekad nailazi na paradokse](https://stonetoss.com/comic/rated-arrr/).

Koncept intelektualnog vlasništva [ima svoje kritičare](https://aeon.co/essays/the-idea-of-intellectual-property-is-nonsensical-and-pernicious) i reformatore ([Electronic Frontier Foundation](https://www.eff.org/) u SAD-u, [EDRi](https://edri.org/) u Europskoj Uniji).

## Intelektualno vlasništvo nad softverom

### Autorsko pravo

Svaki napisan softver [automatski je vlasništvo autora](https://choosealicense.com/no-permission/) i autor ga može [licencirati](https://en.wikipedia.org/wiki/License) drugima pod uvjetima kojima želi. Alternativno, autor se može odreći vlasništva i staviti softver u [javnu domenu](https://creativecommons.org/share-your-work/public-domain/).

#### Vlasnički softver

Vlasnički softveri razvijeni za komercijalne svrhe često imaju nižu cijenu za akademsku upotrebu. Primjerice, kod Gaussiana, [cijena akademske licence je otprilike šest puta niža nego cijena komercijalne licence](https://gaussian.com/pricing/).

S druge strane, kod vlasničkog softvera razvijenog u akademskom okruženju često se sreće shema licenciranja koja definira različite uvjete korištenja u neprofitne (npr. akademska istraživanja) i profitne svrhe (npr. primjena u istraživanja u korporacijama).

Primjerice, [VMD](https://www.ks.uiuc.edu/Research/vmd/) ima sljedeću [licencu](https://www.ks.uiuc.edu/Research/vmd/current/LICENSE.html):

> Upon execution of this Agreement by the party identified below ("Licensee"), The Board of Trustees of the University of Illinois  ("Illinois"), on behalf of The Theoretical and Computational Biophysics Group ("TCBG") in the Beckman Institute, will provide the Visual Molecular Dynamics ("VMD") software in Executable Code and/or Source Code form ("Software") to Licensee, subject to the following terms and conditions. For purposes of this Agreement, Executable Code is the compiled code, which is ready to run on Licensee's computer. Source code consists of a set of files which contain the actual program commands that are compiled to form the Executable Code.
>
> 1. The Software is intellectual property owned by Illinois, and all right, title and interest, including copyright, remain with Illinois.  Illinois grants, and Licensee hereby accepts, a restricted, non-exclusive, non-transferable license to use the Software for academic, research and internal business purposes only, e.g. not for commercial use (see Clause 7 below), without a fee.
>
> (...)

[UCSF Chimera](https://www.cgl.ucsf.edu/chimera/) ima sljedeću [licencu](https://www.cgl.ucsf.edu/chimera/license.html):

> This license agreement ("License"), effective today, is made by and between you ("Licensee") and The Regents of the University of California, a California corporation having its statewide administrative offices at 1111 Franklin Street, Oakland, California 94607-5200 ("The Regents"), acting through its Office of Innovation, Technology & Alliances, University of California San Francisco ("UCSF"), 3333 California Street, Suite S-11, San Francisco, California 94143, and concerns certain software known as "UCSF Chimera," a system of software programs for the visualization and interactive manipulation of molecular models, developed by the Computer Graphics Laboratory at UCSF for research purposes and includes executable code, source code, and documentation ("Software").
>
> 1. General. A non-exclusive, nontransferable, perpetual license is granted to the Licensee to install and use the Software for academic, non-profit, or government-sponsored research purposes. Use of the Software under this License is restricted to non-commercial purposes. Commercial use of the Software requires a separately executed written license agreement.
>
> 2. Permitted Use and Restrictions. Licensee agrees that it will use the Software, and any modifications, improvements, or derivatives to the Software that the Licensee may create (collectively, "Improvements") solely for internal, non-commercial purposes and shall not distribute or transfer the Software or Improvements to any person or third parties without prior written permission from The Regents. The term "non-commercial," as used in this License, means academic or other scholarly research which (a) is not undertaken for profit, or (b) is not intended to produce works, services, or data for commercial use, or (c) is neither conducted, nor funded, by a person or an entity engaged in the commercial use, application or exploitation of works similar to the Software.

[LightDock](https://github.com/brianjimenez/lightdock) je u početku imao [licencu](https://github.com/brianjimenez/lightdock/blob/master/LICENSE) čiji su najvažniji dijelovi:

> The LightDock software ("Software") has been developed by the contributing researchers of the Protein Interactions and Docking group ("Developers"). The Software is made available through the Barcelona Supercomputing Center ("BSC") for your internal, non-profit research use.
>
> (...)
>
> 2\. You agree to make results generated using Software available to other academic researchers for non-profit research purposes. If you wish to obtain Software for any commercial purposes, including fee-based service projects, you will need to execute a separate licensing agreement and pay a fee. (...)

LightDock je danas slobodni softver otvorenog koda licenciran pod licencom GPLv3.

#### Slobodni softver otvorenog koda

[Licenci slobodnog softvera otvorenog koda](https://choosealicense.com/licenses/) razlikujemo prvenstveno po tome jesu li [copyleft](https://en.wikipedia.org/wiki/Copyleft) ili ne. (Copyleft zahtijeva da se promijenjene verzije softvera distribuiraju pod istom licencom.)

Neke od copyleft licenci su:

- [GNU Affero General Public License, version 3 (AGPLv3)](https://www.gnu.org/licenses/agpl-3.0.en.html)
- [GNU General Public License, version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html)
- [GNU Lesser General Public License, version 3 (LGPLv3)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
- [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/)

Neke od ne-copyleft licenci su:

- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [MIT License](https://opensource.org/licenses/MIT)
- [FreeBSD License](https://www.freebsd.org/copyright/freebsd-license/) (poznata i kao [2-Clause BSD License](https://opensource.org/licenses/BSD-2-Clause))
- [The Unlicense](https://unlicense.org/)

Primjerice, [PyMOL](https://pymol.org/) je dostupan pod [licencom](https://github.com/schrodinger/pymol-open-source/blob/master/LICENSE) sličnom licencama MIT i BSD:

> Open-Source PyMOL is Copyright (C) Schrodinger, LLC.
>
> All Rights Reserved
>
> Permission to use, copy, modify, distribute, and distribute modified versions of this software and its built-in documentation for any purpose and without fee is hereby granted, provided that the above copyright notice appears in all copies and that both the copyright notice and this permission notice appear in supporting documentation, and that the name of Schrodinger, LLC not be used in advertising or publicity pertaining to distribution of the software without specific, written prior permission.
>
> (...)

[Avogadro](https://avogadro.cc/) je [dostupan pod licencom GPLv2](https://github.com/cryos/avogadro/blob/master/COPYING):

> 0. This License applies to any program or other work which contains a notice placed by the copyright holder saying it may be distributed under the terms of this General Public License. The "Program", below, refers to any such program or work, and a "work based on the Program" means either the Program or any derivative work under copyright law: that is to say, a work containing the Program or a portion of it, either verbatim or with modifications and/or translated into another language. (Hereinafter, translation is included without limitation in the term "modification".) Each licensee is addressed as "you". Activities other than copying, distribution and modification are not covered by this License; they are outside its scope. The act of running the Program is not restricted, and the output from the Program is covered only if its contents constitute a work based on the Program (independent of having been made by running the Program). Whether that is true depends on what the Program does.
>
> 1. You may copy and distribute verbatim copies of the Program's source code as you receive it, in any medium, provided that you conspicuously and appropriately publish on each copy an appropriate copyright notice and disclaimer of warranty; keep intact all the notices that refer to this License and to the absence of any warranty; and give any other recipients of the Program a copy of this License along with the Program. You may charge a fee for the physical act of transferring a copy, and you may at your option offer warranty protection in exchange for a fee.
>
> 2. You may modify your copy or copies of the Program or any portion of it, thus forming a work based on the Program, and copy and distribute such modifications or work under the terms of Section 1 above, provided that you also meet all of these conditions:
>
>     a) You must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.
>
>     b) You must cause any work that you distribute or publish, that in whole or in part contains or is derived from the Program or any part thereof, to be licensed as a whole at no charge to all third parties under the terms of this License.
>
>     c) If the modified program normally reads commands interactively when run, you must cause it, when started running for such interactive use in the most ordinary way, to print or display an announcement including an appropriate copyright notice and a notice that there is no warranty (or else, saying that you provide a warranty) and that users may redistribute the program under these conditions, and telling the user how to view a copy of this License. (Exception: if the Program itself is interactive but does not normally print such an announcement, your work based on the Program is not required to print an announcement.)

[GROMACS](https://www.gromacs.org/) je [dostupan pod licencom LGPLv2.1 ili novijom (po vašem izboru)](https://github.com/gromacs/gromacs/blob/master/COPYING):

> 0. This License Agreement applies to any software library or other program which contains a notice placed by the copyright holder or other authorized party saying it may be distributed under the terms of this Lesser General Public License (also called "this License"). Each licensee is addressed as "you".
>
>     A "library" means a collection of software functions and/or data prepared so as to be conveniently linked with application programs (which use some of those functions and data) to form executables. The "Library", below, refers to any such software library or work which has been distributed under these terms. A "work based on the Library" means either the Library or any derivative work under copyright law: that is to say, a work containing the Library or a portion of it, either verbatim or with modifications and/or translated straightforwardly into another language. (Hereinafter, translation is included without limitation in the term "modification".)
>
>     "Source code" for a work means the preferred form of the work for making modifications to it. For a library, complete source code means all the source code for all modules it contains, plus any associated interface definition files, plus the scripts used to control compilation and installation of the library.
>
>     Activities other than copying, distribution and modification are not covered by this License; they are outside its scope. The act of running a program using the Library is not restricted, and output from such a program is covered only if its contents constitute a work based on the Library (independent of the use of the Library in a tool for writing it). Whether that is true depends on what the Library does and what the program that uses the Library does.
>
> 1. You may copy and distribute verbatim copies of the Library's complete source code as you receive it, in any medium, provided that you conspicuously and appropriately publish on each copy an appropriate copyright notice and disclaimer of warranty; keep intact all the notices that refer to this License and to the absence of any warranty; and distribute a copy of this License along with the Library. You may charge a fee for the physical act of transferring a copy, and you may at your option offer warranty protection in exchange for a fee.
>
> 2. You may modify your copy or copies of the Library or any portion of it, thus forming a work based on the Library, and copy and distribute such modifications or work under the terms of Section 1 above, provided that you also meet all of these conditions:
>
>     a) The modified work must itself be a software library.
>
>     b) You must cause the files modified to carry prominent notices stating that you changed the files and the date of any change.
>
>     c) You must cause the whole of the work to be licensed at no charge to all third parties under the terms of this License.
>
>     d) If a facility in the modified Library refers to a function or a table of data to be supplied by an application program that uses the facility, other than as an argument passed when the facility is invoked, then you must make a good faith effort to ensure that, in the event an application does not supply such function or table, the facility still operates, and performs whatever part of its purpose remains meaningful.
>
>     (For example, a function in a library to compute square roots has a purpose that is entirely well-defined independent of the application. Therefore, Subsection 2d requires that any application-supplied function or table used by this function must be optional: if the application does not supply it, the square root function must still compute square roots.)
>
> (...)

[RxDock](https://rxdock.gitlab.io/) je, kao i njegov temelj [rDock](http://rdock.sourceforge.net/), [dostupan pod licencom LGPLv3](https://gitlab.com/rxdock/rxdock/blob/master/LICENSE.md).

#### Problemi autorskih prava

- [Copyright trolls](https://www.eff.org/issues/copyright-trolls), kompanije koje podižu masovne tužbe s ciljem postizanja nagodbe s optuženima
- [Digital Rights Management (DRM)](https://www.eff.org/issues/drm), prvenstveno kod [digitalnog videa](https://www.eff.org/issues/digital-video)
- [Digital Millenium Copyright Act (DMCA)](https://www.eff.org/issues/dmca), prvenstveno zbog gubitka [prava na popravak](https://www.eff.org/issues/right-to-repair) (tzv. "anti-circumvention" provision) i davanja vlasnicima autorskog prava brz i jednostavan način za micanje s interneta sadržaja koji navodno krši njihova autorska prava (tzv. "safe harbors")
- [Cenzura sadržaja na internetu putem zakona o autorskom pravu](https://www.eff.org/issues/coica-internet-censorship-and-copyright-bill)

### Patent

[Patent](https://en.wikipedia.org/wiki/Patent) je oblik intelektualnog vlasništa koji vlasniku daje zakonsko pravo da ostalima zabrani stvaranje, korištenje, prodaju i uvoz izuma u zamjenu za javnu objavu izuma. Bazu patenata diljem svijeta moguće je pretraživati pomoću [Google Patents](https://patents.google.com/).

Primjeri patenata:

- [lijek za rak](https://www.eff.org/deeplinks/2014/08/magical-drug-wins-effs-stupid-patent-month),
- [snimanje satova joge](https://www.eff.org/deeplinks/2014/10/octobers-very-bad-no-good-totally-stupid-patent-month-filming-yoga-class),
- [povećanje cijene u slučaju povećane potražnje](https://www.eff.org/deeplinks/2014/12/spirit-holidays-its-not-too-late-uber-avoid-stupid-patent-month),
- [dodavanje novih vrsta trave putem interneta u sportske terene unutar računalne igre](https://www.eff.org/deeplinks/2015/01/january-stupid-patent-month-method-updating-video-game-grass),
- [povezivanje kućanskih uređaja na internet](https://www.eff.org/deeplinks/2015/08/stupid-patent-month-drink-mixer-attacks-internet-things),
- [slider u sučelju računalnog programa](https://www.eff.org/deeplinks/2015/12/stupid-patent-month-microsofts-design-patent-slider).

#### Problemi patenata

- [Patent trollovi](https://www.eff.org/issues/stupid-patent-month), kompanije koje kupe patente koje ne namjeravaju koristiti već samo podižu tužbe za kršenje patenata; [prijedlog reforme zakona](https://www.eff.org/pages/defend-innovation), [prijedlog licence za patente koja rješava problem, a funkcionira unutar postojećeg zakonskog okvira](https://www.defensivepatentlicense.org/)

### Zaštitni znak

[Zaštitni znak](https://en.wikipedia.org/wiki/Trademark) je prepoznatljivi znak, dizajn ili izraz koji razlikuje nečiji proizvod ili uslugu od svih ostalih.

U zajednici slobodnog softvera otvorenog koda postoji nepisano pravilo da se postojeća imena softvera ne dupliciraju, bila registrirana kao zaštitni znak ili ne.

- Mozillin web preglednik, inicijalno nazvan Firebird, postao [Firefox](https://www.mozilla.org/firefox/) kako bi se razlikovao od sustava za upravljanje bazom podataka [Firebird](https://firebirdsql.org/).
- Isto vrijedi i kod odvajanja (tzv. forkanja) projekta, paket uredskih alata [OpenOffice.org](https://www.openoffice.org/) je postao [LibreOffice](https://www.libreoffice.org/), program prevoditelj [GCC](https://gcc.gnu.org/) je neko vrijeme bio poznat kao [EGCS](https://www.gnu.org/software/gcc/egcs-1.0/), a alat za molekularni docking [rDock](http://rdock.sourceforge.net/) forkan je kao [RxDock](https://rxdock.gitlab.io/).

#### Problemi zaštitnih znakova

- [Kompanije koje koriste zaštitni znak da bi utišale kritičare](https://www.eff.org/issues/trademark)

## Autorsko pravo nad podacima

Isto kao kod softvera, autorsko pravo nad podacima postoji automatski.

Neke od licenci:

- [Creative Commons (CC)](https://creativecommons.org/)
- [Open Publication License (OPL)](https://en.wikipedia.org/wiki/Open_Publication_License)
- [Open Database License (ODbL)](https://en.wikipedia.org/wiki/Open_Database_License)
- [Open Game License (OGL)](https://en.wikipedia.org/wiki/Open_Game_License)
- [GNU Free Documentation License (GNU FDL, GFDL)](https://en.wikipedia.org/wiki/GNU_Free_Documentation_License)

## Autorsko pravo nad znanstvenim radovima

Isto kao kod softvera i podataka, autorsko pravo nad znanstvenim radovima postoji automatski, ali se često kod objave prebacuje na izdavača (npr. [Elsevier](https://www.elsevier.com/about/policies/copyright), [Springer](https://www.springer.com/gp/open-access/publication-policies), [American Chemical Society (ACS)](https://pubs.acs.org/page/copyright/journals/jpa_index.html), [Cell Press](https://www.cell.com/trends/editorial-policies), [Nature](https://www.nature.com/nature-research/editorial-policies/self-archiving-and-license-to-publish)).

Kod radova objavljenim u [otvorenom pristupu](https://en.wikipedia.org/wiki/Open_access) (engl. *open access*, kraće OA) prebacivanja autorskih prava na izdavača nema (npr. [MDPI](https://www.mdpi.com/authors/rights), [PLOS](https://plos.org/open-science/why-open-access/#license), detaljnije za časopis [PLOS ONE](https://journals.plos.org/plosone/s/licenses-and-copyright), [PeerJ](https://peerj.com/about/policies-and-procedures/)). Najčešće se koristi licenca [Creative Commons Attribution](https://creativecommons.org/licenses/by/4.0/).

[Aaron Swartz](https://en.wikipedia.org/wiki/Aaron_Swartz) je 2011. godine iskoristio svoj pristup radovima na MIT-u da sistematsi preuzme velik broj radova s ciljem da ih učini dostupnim javnosti zbog čega je bio uhićen od strane MIT-eve policije i optužen od strane Federalne vlade SAD-a. 2013. godine je počinio samoubojstvo, ali je [ostao jedna od značajnih ličnosti u borbi za otvoreni pristup znanstvenim radovima](https://www.eff.org/deeplinks/2019/11/join-eff-aaron-swartz-day-weekend-internet-archive).

Otvoreni pristup dolazi u različitim varijantama koje imaju [nazive po bojama](https://en.wikipedia.org/wiki/Open_access#Colour_naming_system): zlatni (pravi otvoreni pristup), zeleni (pravo na vlastito arhiviranje), hibridni (dio radova u otvorenom pristupu, dio ne), brončani (radovi prelaze u otvoreni pristup nakon određenog vremena), dijamantni/platinasti (autor plaća da bude u otvorenom pristupu), crni (piratstvo, npr. [Sci-Hub](https://en.wikipedia.org/wiki/Sci-Hub) koji je razvila [Alexandra Elbakayan](https://en.wikipedia.org/wiki/Alexandra_Elbakyan) i koji je [vrlo popularan](https://www.science.org/doi/10.1126/science.352.6285.508) unatoč [svom statusu pred zakonom](https://www.nature.com/articles/nature.2015.18876), [Elsevierova tužba](https://www.nature.com/articles/nature.2017.22196), [ACS-ova tužba](https://www.nature.com/articles/nature.2017.22971)).

Uz [izdavače u otvorenom pristupu kao PLOS](https://plos.org/open-science/why-open-access/), [veliku podršku istom daje i Electronic Frontier Foundation](https://www.eff.org/issues/open-access).

### Specifičnosti višeautorskih radova

Većina znanstvenih radova ima veći broj autora i sve je češća praksa da se posebno navodi doprinos pojedinih autora. Jedna od taksonomija doprinosa je [Contributor Roles Taxonomy](https://credit.niso.org/), kraće CRediT, koju koriste [brojni izdavači](https://credit.niso.org/adopters/). CRediT je taksonomija visoke razine koja se koristi za predstavljanje uloga koje obično imaju suradnici u znanstvenom znanstvenom radu. [Uloge](https://credit.niso.org/contributor-roles-defined/) opisuju specifičan doprinos svakog suradnika znanstvenom radu, ima ih 14 i one su:

> 1. [Conceptualization](https://credit.niso.org/contributor-roles/conceptualization/): Ideas; formulation or evolution of overarching research goals and aims.
> 1. [Data curation](https://credit.niso.org/contributor-roles/data-curation/): Management activities to annotate (produce metadata), scrub data and maintain research data (including software code, where it is necessary for interpreting the data itself) for initial use and later re-use.
> 1. [Formal analysis](https://credit.niso.org/contributor-roles/formal-analysis/): Application of statistical, mathematical, computational, or other formal techniques to analyze or synthesize study data.
> 1. [Funding acquisition](https://credit.niso.org/contributor-roles/funding-acquisition/): Acquisition of the financial support for the project leading to this publication.
> 1. [Investigation](https://credit.niso.org/contributor-roles/investigation/): Conducting a research and investigation process, specifically performing the experiments, or data/evidence collection.
> 1. [Methodology](https://credit.niso.org/contributor-roles/methodology/): Development or design of methodology; creation of models.
> 1. [Project administration](https://credit.niso.org/contributor-roles/project-administration/): Management and coordination responsibility for the research activity planning and execution.
> 1. [Resources](https://credit.niso.org/contributor-roles/resources/): Provision of study materials, reagents, materials, patients, laboratory samples, animals, instrumentation, computing resources, or other analysis tools.
> 1. [Software](https://credit.niso.org/contributor-roles/software/): Programming, software development; designing computer programs; implementation of the computer code and supporting algorithms; testing of existing code components.
> 1. [Supervision](https://credit.niso.org/contributor-roles/supervision/): Oversight and leadership responsibility for the research activity planning and execution, including mentorship external to the core team.
> 1. [Validation](https://credit.niso.org/contributor-roles/validation/): Verification, whether as a part of the activity or separate, of the overall replication/reproducibility of results/experiments and other research outputs.
> 1. [Visualization](https://credit.niso.org/contributor-roles/visualization/): Preparation, creation and/or presentation of the published work, specifically visualization/data presentation.
> 1. [Writing - original draft](https://credit.niso.org/contributor-roles/writing-original-draft/): Preparation, creation and/or presentation of the published work, specifically writing the initial draft (including substantive translation).
> 1. [Writing - review & editing](https://credit.niso.org/contributor-roles/writing-review-editing/): Preparation, creation and/or presentation of the published work by those from the original research group, specifically critical review, commentary or revision – including pre- or post-publication stages.

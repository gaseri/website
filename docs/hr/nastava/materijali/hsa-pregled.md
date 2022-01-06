---
author: Vedran Miletić
---

# Pregled heterogene sustavske arhitekture

!!! note
    Složeno prema *HSA Overview*, autor P. Rogers, poglavlje u knjizi *Heterogeneous System Architecture: A New Compute Platform Infrastructure*, urednik Wen-mei W. Hwu, izdavač [Elsevier](https://www.elsevier.com/books/heterogeneous-system-architecture/hwu/978-0-12-800386-2) (dalje u tekstu: HSAANCPI). Autorska prava na izvorni tekst zadržava [HSA Foundation](https://hsafoundation.com/).

Heterogena sustavska arhitektura (engl. *Heterogeneous System Architecture*, HSA) je:

- nova hardverska i softverska platforma koja omogućuje zajednički rad procesora različite vrste u dijeljenoj memoriji
- primjenjiva na pametne telefone, tablete, osobna računala, radne stanice i superračunala
- evolucija [simetričnog multiprocesinga](https://en.wikipedia.org/wiki/Symmetric_multiprocessing) (engl. *symmetric multiprocessing*, SMP): homogeni multiprocesorski sustavi postavili su temelj za današnje heterogene sustave na čipu (engl. *System on Chip*, SoC) i ubrzane procesorske jedinice (engl. *Accelerated Processing Unit*, APU); u zadnjem desetljeću to su čipovi:

    - APU: Intel Core i3, i5, i7; AMD A, E, C, Z, Athloni na socketima AM1 i AM4 te Ryzen serije G i U; NVIDIA Tegra
    - SoC: Samsung Exynos, Qualcomm Snapdragon, ImgTec PowerVR, HiSilicon Kirin, AMD G i R

- APU/SoC: CPU, GPU, DSP, audio-video enkoderi i dekoderi, DMA uređaji, pogoni za kriptografiju
- HSA izvorno fokusirana na GPU-e (koji rade uz CPU-e), ali kasnije proširena i na druge vrste uređaja uglavnom već prisutne u SoC-ima
- GPU-i izvorno vezani na CPU-e kao ulazno-izlazne jedinice

    - odvojeni silicij; integracija CPU-a i GPU-a na isti silicij se pokazala dosta složena
    - korišteni isključivo za grafiku do otkrića [programabilnih shadera](https://en.wikipedia.org/wiki/Shader)
    - HSA Foundation uspostavljena s ciljem razvoja jednostavnijih programskih modela za programiranje softvera koji se izvode na CPU-ima i GPU-ima

## Kratka povijest računanja na GPU-ima

- počinje 2000-ih, GPU-i su tada vezani na CPU-e najčešće putem [PCI-a](https://en.wikipedia.org/wiki/Conventional_PCI) ili [AGP-a](https://en.wikipedia.org/wiki/Accelerated_Graphics_Port) [HSAANCPI, slika 2.1]
- CPU ima sustavsku (glavnu, radnu) memoriju, GPU ima vlastitu memoriju (uglavnom veće širine pojasa od sustavske memorije)

    - programer mora brinuti o kopiranju podataka iz jedne u drugu ovisno o tome koji uređaj želi koristiti za obradu podataka
    - takav sustav se [Non-uniform memory access](https://en.wikipedia.org/wiki/Non-uniform_memory_access) (NUMA)

        - primjer niske neuniformnosti: SMP sustav s 2 procesora i odvojenim bazenima memorije: svaki CPU ima izravan pristup svojem bazenu memorije (niže zadržavanje), a neizravan pristup tuđem (više zadržavanje)
        - primjer visoke neuinformnosti: heterogeni sustav s CPU-om i GPU-om na sabirnici PCI/AGP/PCIe

    - GPU koristi virtualne adrese kod pristupanja memoriji CPU-a pa se pokazivači i reference ne mogu prosljeđivati između CPU-a i GPU-a

        - strukture podataka moraju imati odmake pojedinih podataka umjesto njihovih adresa u memoriji

    - straničenje i virtualna memorija mogu stvoriti probleme kod pristupanja GPU-a memoriji CPU-a

        - GPU-i mogu pristupati samo onim stranicama u memoriji CPU-a koje su zaključane (engl. *page locked*) ili prikačene (engl. *pinned*) pa ih upravljanje virtualnom memorijom neće privremeno preseliti u pohranu podataka
        - postoji ograničenje količine memorije koja može biti zaključana/prikačena i tipično iznosi pola sustavske memorije
        - programer mora brinuti o zaključavanju/kačenju memorije

    - još jedan problem s memorijom je koherencija memorije

        - javlja se kad više niti dijeli podatke u memoriji

!!! todo
    Ovaj dio treba dovršiti.

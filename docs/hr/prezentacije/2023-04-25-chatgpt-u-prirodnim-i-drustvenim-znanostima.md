---
marp: true
theme: default
class: default
author: Vedran Miletić
title: ChatGPT u prirodnim i društvenim znanostima
description: Predavanje na Festivalu znanosti
keywords: large language models, artificial intelligence, chatgpt
curriculum-vitae: |
  Vedran Miletić je docent u polju računarstva na Fakultetu informatike i digitalnih tehnologija, gdje vodi Grupu za aplikacije i usluge na eksaskalarnoj istraživačkoj infrastrukturi. Njegova grupa radi na poboljšanju metoda simulacije molekulske dinamike u slobodnim softverima otvorenog koda namijenjenim za akademsku i industrijsku primjenu na modernim superračunalima.
abstract: |
  ChatGPT je uzdrmao svijet (visokog) obrazovanja i istraživanja kao vrlo malo alata u posljednjem desetljeću. Mnogi komentatori razvoja tehnologije već ga sada smatraju korakom u razvoju tehnologije koji će rezultirati širokim društvenim promjenama usporedivim s uvođenjem Googleovog web pretraživanja ili interneta. Prezentacija će pokriti učinak ovog alata iz perspektive istraživača u prirodnim i društvenim znanostima, ali će se dotaknuti i utjecaja razvoja umjetne inteligencije na obrazovanje i svakodnevni život.
---

# ChatGPT u prirodnim i društvenim znanostima

## [Vedran](https://vedran.miletic.net/) [Miletić](https://www.miletic.net/), Fakultet informatike i digitalnih tehnologija

![FIDIT logo h:350px](https://upload.wikimedia.org/wikipedia/commons/1/14/FIDIT-logo.svg) Izvor slike: [Wikimedia Commons File:FIDIT-logo.svg](https://commons.wikimedia.org/wiki/File:FIDIT-logo.svg)

### Festival znanosti, Rijeka, Otvoreni dan, zgrada sveučilišnih odjela, 25. travnja 2023.

---

<!-- paginate: true -->

## Fakultet informatike i digitalnih tehnologija

- osnovan 2008. godine kao Odjel za informatiku Sveučilišta u Rijeci
- postao fakultet 2022. godine
- [pet laboratorija](https://www.inf.uniri.hr/znanstveni-i-strucni-rad/laboratoriji) sastavljenih od istraživačkih grupa; znanstveni interesi:
    - računalni vid, raspoznavanje uzoraka
    - obrada prirodnog jezika, strojno prevođenje
    - e-učenje, digitalna transformacija
    - paralelno programiranje na superračunalima

![FIDIT logo bg right:35% 90%](https://upload.wikimedia.org/wikipedia/commons/1/14/FIDIT-logo.svg)

Izvor slike: [Wiki. Comm. File:FIDIT-logo.svg](https://commons.wikimedia.org/wiki/File:FIDIT-logo.svg)

---

## Grupa za aplikacije i usluge na eksaskalarnoj istraživačkoj infrastrukturi

- engl. *Group for Applications and Services on Exascale Research Infrastructure*, kraće **GASERI**
- fokus istraživanja: primjena suvremenih eksaskalarnih superračunala za rješavanje problema u računalnoj biokemiji
- glavni cilj: dizajn algoritama visokih performansi za korištenje u akademskim istraživanjima i industrijskom razvoju

![GASERI logo](../../images/gaseri-logo-text.png)

---

## Internet 90-ih

- za ilustraciju: [14 iconic 90s websites](https://webflow.com/blog/90s-website-design), [1990s Internet & World Wide Web](https://nostalgiacafe.proboards.com/thread/133/1990s-internet-world-wide-web)
- možemo ga retroaktivno nazvati "doba prije Googlea"
    - [Archie](https://www.stackscale.com/blog/archie-internet-search-engine/)
    - [Gopher](https://www.howtogeek.com/661871/the-web-before-the-web-a-look-back-at-gopher/)
    - [Ask Jeeves!](https://www.mentalfloss.com/article/94784/why-everyone-stopped-asking-jeeves)
    - Lycos, Excite, AltaVista, Yahoo!, WebCrawler, itd.

---

## Suvremeni internet

- Googleov uspjeh: algoritam za rangiranje stranica u rezultatima [PageRank](https://en.wikipedia.org/wiki/PageRank)
    - [udio na tržištu danas](https://gs.statcounter.com/search-engine-market-share)
- malo ljudi je u doba nastanka Googlea (1998.) vidjelo povijest računarstva/informatike kao doba prije Googlea i nakon Googlea
    - Google Search omogućio Images, News, GMail, Shopping, Scholar, Books, Patents, Maps itd.
        - motivacija za druge autore softvera da svoje glavne aplikacije isporuče kao web aplikacije, a ne desktop
    - YouTube, Android, Street View, Calendar, Flights, Meet, ...

---

## Prijelaz aplikacija i korisnika s desktopa na web

- 1997\. (doba prije Googlea) desktop i Microsoft dominiraju tržištem softvera za osobna računala
- 2007\. (doba Googlea) Paul Graham kaže [Microsoft is Dead](http://www.paulgraham.com/microsoft.html):

    > Microsoft's biggest weakness is that they still don't realize how much they suck. They still think they can write software in house. Maybe they can, by the standards of the desktop world. But that world ended a few years ago.

- slična promjena paradigme događa se i nastavit će se događati uzrokovana širokom primjenom tehnologija koje stoje u pozadini ChatGPT-a

---

## OpenAI

- [osnovan 2015. godine](https://www.businessinsider.com/history-of-openai-company-chatgpt-elon-musk-founded-2022-12), suosnivači Sam Altman i Elon Musk; [charter](https://openai.com/charter/):

    > OpenAI’s mission is to ensure that artificial general intelligence (AGI)—by which we mean highly autonomous systems that outperform humans at most economically valuable work—benefits all of humanity. We will attempt to directly build safe and beneficial AGI, but will also consider our mission fulfilled if our work aids others to achieve this outcome.

- Poznatiji proizvodi: [Gym](https://www.gymlibrary.dev/) ([Retro](https://openai.com/blog/gym-retro/)), [RoboSumo](https://github.com/openai/robosumo), [Debate Game](https://openai.com/research/debate), [Dactyl](https://openai.com/blog/learning-dexterity/), [DALL-E](https://openai.com/dall-e-2/)

![OpenAI logo h:150px](https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg)

Izvor slike: [Wikimedia Commons File:OpenAI Logo.svg](https://commons.wikimedia.org/wiki/File:OpenAI_Logo.svg)

---

## ChatGPT

- izbacuje ga [OpenAI](https://openai.com/) [u studenom 2022. godine](https://openai.com/index/chatgpt/)
- jezični model za razgovor; [nije prvi takav](https://gpt3demo.com/apps/instructgpt)
- *Reinforcement Learning from Human Feedback*
- [treniranje značajnim dijelom ručni rad](https://time.com/6247678/openai-chatgpt-kenya-workers/)
- [rast broja korisnika iznad svih očekivanja](https://www.statista.com/chart/29174/time-to-one-million-users/)
- [primjeri upita](https://platform.openai.com/docs/examples)

---

## Mogućnosti

ChatGPT "zna":

- odgovarati na više (pod)pitanja u nizu i pritom zadržati kontekst
- pojašnjavati koncepte na više načina
- ponašati se kao chat bot u korisničkoj podršci
- pisati opise poslova kod zapošljavanja
- pisati programski kod jednostavnih aplikacija i usluga
- pisati eseje na tipične teme bez činjeničnih pogrešaka
- pisati sažetke članaka
- [i još mnogo toga](https://platform.openai.com/docs/examples) na [95 jezika](https://knowinsiders.com/how-many-languages-chatgpt-supports-updated-36845.html)

---

## Demo

[Službeno web sjedište](https://chatgpt.com/) često kaže `ChatGPT is at capacity right now`; primjeri:

- [List of ChatGPT Examples](https://www.followchain.org/chatgpt-examples/)
- [22 Interesting ChatGPT Examples](https://builtin.com/artificial-intelligence/chatgpt-examples)
- [12 Cool Things You Can Do with ChatGPT](https://beebom.com/cool-things-chatgpt/)
- Rubni slučajevi: [ChatGPT Examples: 22 Interesting Questions Answered By ChatGPT](https://finlightened.com/chatgpt-examples/)

---

## Poslovi na koje ChatGPT najviše utječe

- profesori u školama i na fakultetima: ocjenjivanje studentskih eseja i programa
- novinari: pisanje i sažimanje članaka
- odvjetnici: pisanje prve verzije zakona, pravnih savjeta
- marketingaši: copywriting uz automatsku optimizaciju ključnih riječi za tražilice
    - upitna je dugoročna korist od optimizacije za tražilice ako ih ChatGPT zamijeni
    - optimizacija teksta za ChatGPT?
- tehnička podrška

---

## Sadašnjost znanosti

[Major publishers are banning ChatGPT from being listed as an academic author. What's the big deal?](https://phys.org/news/2023-01-major-publishers-chatgpt-academic-author.html) navodi brojne ~~probleme~~ izazove:

- autorstvo (AI ne može preizeti odgovornost za proizvedeni sadržaj)
    - poredak autora (abecedni ili po doprinosu, koliki je doprinos ChatGPT-a)
    - broj citata i napredovanje u karijeri: što kad ChatGPT stekne uvjete za redovitog profesora u svim granama kroz godinu dana?
    - imalo je [problema](https://www.nature.com/articles/d41586-023-00062-9) i prije ChatGPT-a
- autorsko pravo (engl. *copyright*): [postojeća tužba](https://githubcopilotlitigation.com/), [nezadovoljstvo autora softvera](https://news.ycombinator.com/item?id=33458374)

---

## Nature

[Tools such as ChatGPT threaten transparent science; here are our ground rules for their use](https://www.nature.com/articles/d41586-023-00191-1):

> First, no LLM tool will be accepted as a credited author on a research paper. That is because any attribution of authorship carries with it accountability for the work, and AI tools cannot take such responsibility.  
> Second, researchers using LLM tools should document this use in the methods or acknowledgements sections. If a paper does not include these sections, the introduction or another appropriate section can be used to document the use of the LLM.

---

## Science

[Science Journals: Editorial Policies](https://www.science.org/content/page/science-journals-editorial-policies):

> (...) artificial intelligence tools cannot be authors. (...)  
> Artificial intelligence (AI) policy: Text generated from AI, machine learning, or similar algorithmic tools cannot be used in papers published in Science journals, nor can the accompanying figures, images, or graphics be the products of such tools, without explicit permission from the editors. In addition, an AI program cannot be an author of a Science journal paper. A violation of this policy constitutes scientific misconduct.

---

## Primjena u znanosti

- skiciranje uvoda u rad, pregleda korištenih metoda, zaključka
    - uz nadzor i korecije od strane autora
- pisanje preglednih radova (engl. *state of te art*)
    - vjerojatna budućnost je manja valorizacija preglednih radova
- stvaranje istraživanja? [ChatGPT: Study shows AI can produce academic papers good enough for journals—just as some ban it](https://phys.org/news/2023-01-chatgpt-ai-academic-papers-good.html)

---

## Budućnost obrazovanja

- sporije se kreće od znanosti
- očekivane prve reakcije: otpor, zabrinutost
    - [How ChatGPT could make it easy to cheat on written tests and homework: 'You can NO LONGER give take-home exams or homework'](https://www.dailymail.co.uk/sciencetech/article-11513127/ChatGPT-OpenAI-cheat-tests-homework.html)
    - [ChatGPT appears to pass medical school exams. Educators are now rethinking assessments](https://www.abc.net.au/news/science/2023-01-12/chatgpt-generative-ai-program-passes-us-medical-licensing-exams/101840938)
- jedini način prepoznavanja korištenja ChatGPT-a su velike greške u činjenicama, što će se vremenom ispraviti
    - iduća generacija, GPT-4,  [postiže bolje rezultate na standardiziranim testovima za odvjetnike i biologe](https://openai.com/product/gpt-4)

---

## Primjer predmeta *Programiranje za web* i *Upravljanje računalnim sustavima* s prijediplomskog studija Informatika

- obrnuta didaktika od tipičnih kolegija iz područja programskog inženjerstva
    - omogućena probojem slobodnog softvera otvorenog koda u mainstream tijekom proteklih 30-ak godina
    - fokus na čitanje, razumijevanje i kritiku postojećeg koda
    - pisanje vlastitog koda postoji, ali je sporedno
- napustili smo reprodukcijska pitanja pred nekoliko godina
    - željeli smo pisanje ispita iz teorije "s otvorenom knjigom", ali i bilježnicom, materijalima za nastavu, Wikipedijom, Googleom itd.

---

## Primjer primjene u nastavi na FIDIT-u (1/2)

Upit: `Zaporke se čuvaju u a. Google Docs dokumentima za zaporke b. internoj dokumentaciji, unutar teksta c. službenoj dokumentaciji softvera d. OneNote bilješkama e. upravitelju zaporkama f. internoj dokumentaciji, u posebnom odjeljku za zaporke g. Google Docs tablicama zaporki`

> Najsigurniji i preporučeni način čuvanja zaporki je korištenje upravitelja zaporkama (password manager), stoga bi odgovor bio e. Upravitelji zaporkama su alati koji omogućavaju generiranje jakih zaporki te njihovo sigurno čuvanje i upravljanje. Korisnik treba samo zapamtiti jednu "glavnu" zaporku kojom pristupa upravitelju zaporkama, a sve ostale zaporku upravitelj pamti i automatski upisuje prilikom prijave u odgovarajuće aplikacije ili web stranice.

---

## Primjer primjene u nastavi na FIDIT-u (2/2)

Upit: `cloud-config pod postavkom passwd može imati niz znakova oblika: a. $1$yjn5wBuA$l03L5t7vE.eOe0bcZvzLd/ b. XMmpyK3r$coLOHkrewlLD3/9BVkHG60 c. $5$MFZI/7q6bYbtQG9Z$wG3Rs7qwLCDvHAQbLXpbpkxFgTYy4bp8Uqnx9HUe4HC d. $6$U9UYfuf1DvS0vDsS$Om2BbsD0eoayX5tZ2zzFAbeeT8uflwpxYkVO0NvVhUNSnbR5Qt6lszw4hjWNU2GC9tbeh./.nPoJ9V5hY.YTZ/ e. fidit1234`

> Točni odgovori su a, b, c i d. Ovo su hash-irane zaporke koje se koriste u Unix sustavima za pohranjivanje zaporki korisnika. Format zaporki varira ovisno o korištenoj verziji Unixa, a različiti formati se označavaju različitim brojevima u prvom dijelu hash-a. U tom smislu, ovi nizovi znakova su primjeri zaporki koje se mogu koristiti u cloud-config datotekama. Odgovor e nije ispravan jer nije u formatu Unix zaporki.

---

## Dva pristupa primjene u nastavi

- nastavnici
    - sažimanje sadržaja iz više izvora za skriptu, izrada prezentacija
    - izrada pitanja
- studenti
    - objašnjavanje koncepata u fazi učenja
    - pisanje eseja i programa
    - odgovori na pitanja na ispitu
        - ChatGPT može odgovoriti na pitanja koja je sam izradio
        - gotovo sigurno će većinom studentima biti zabranjeno korištenje

---

## Što još može ML? (1/2)

- integracija ChatGPT-a u postojeće alate: [Blender](https://github.com/gmh5225/chatGPT-BlenderGPT), [PyMOL](https://techserr.com/chatgpt-the-future-of-pymol-automation/)
- sinteza govora: [WaveNet](https://www.deepmind.com/research/highlighted-research/wavenet)
- igranje igre diplomacije: [Cicero](https://ai.meta.com/research/cicero/)
- pomoć u kući: [AI Habitat](https://aihabitat.org/)
- crtati: [Craiyon](https://www.craiyon.com/)/[DALL-E mini](https://dallemini.com/) (similar to OpenAI's [DALL-E 2](https://openai.com/product/dall-e-2))
    - [Playform](https://www.playform.io/) ima drugačiji stil
    - [Bing također to možže](https://www.bing.com/images/create)
    - [NVIDIA Canvas](https://www.nvidia.com/en-us/studio/canvas/) je desktop aplikacija za isto

---

## Što još može ML? (2/2)

- pomoć u uredskom radu: [Notion AI](https://www.notion.so/help/guides/using-notion-ai), [pregled značajki](https://www.notion.so/help/guides/notion-ai-for-docs)
    - sličan [Office 365 Copilotu](https://www.microsoft.com/en-us/microsoft-365/blog/2023/03/16/introducing-microsoft-365-copilot-a-whole-new-way-to-work/), [Generative AI-u](https://workspace.google.com/blog/product-announcements/generative-ai) u Google Workspaceu (Docs/GMail)
- sažimanje teksta: [Eightify](https://eightify.app/)/[summarize.tech](https://www.summarize.tech/)
- pretvorba govora u tekst: [Whisper](https://openai.com/research/whisper)
- predviđanje zamotavanja proteina: [AlphaFold](https://www.deepmind.com/research/highlighted-research/alphafold), [ESMFold](https://www.nature.com/articles/d41586-022-03539-1)
    - integrirani specijalizirani alati: [BioNeMo](https://www.nvidia.com/en-us/gpu-cloud/bionemo/)

---

## GPT-4

- prijelaz [GPT-3 -> GPT-4](https://www.pcmag.com/news/the-new-chatgpt-what-you-get-with-gpt-4-vs-gpt-35)
    - već dostupan kao [usluga uz pretplatu](https://openai.com/product/gpt-4) pod imenom ChatGPT+
    - [koristi ga novi Bing](https://blogs.bing.com/search/march_2023/Confirmed-the-new-Bing-runs-on-OpenAI%E2%80%99s-GPT-4), prešao [100 milijuna aktivnih korisnika u danu](https://www.theverge.com/2023/3/9/23631912/microsoft-bing-100-million-daily-active-users-milestone)
    - i dalje [nije savršen, nekad je GPT-3 bolji](https://mashable.com/article/openai-gpt-4-answers-better-than-gpt-3)
- hrvatski jezični model i prevođenje između engleskog i hrvatskog su već relativno dobri, ali očekuje se da će biti i bolji
    - pitanje mjeseci, eventualno jedne ili dvije godine da hrvatski jezik bude nativno upotrebljiv
    - za ilustraciju: Google Translate nekad i danas

---

## Budućnost

- zasigurno: chatbotovi, korisnička podrška, pisanje eseja (čak i ako bude mimo pravila), odgovori na e-mailove, pomoć kod učenja...
- [You.com](https://you.com/) nadmašuje Google Search?
- Bing nadmašuje Google Search?
    - mainstream IT mediji već špekuliraju  o toj mogućnosti: [How ChatGPT Could Take Microsoft's Search Engine Bing Into the Future](https://www.cnet.com/tech/mobile/how-chatgpt-could-take-microsofts-search-engine-bing-into-the-future/)
- Elon Musk ga koristi kao dodatak za [Neuralink](https://neuralink.com/)?

---

## Idući koraci

Posjetite [chatgpt.com](https://chatgpt.com/), nedajte da vas obeshrabri `ChatGPT is at capacity right now`, registrirajte se i krenite.

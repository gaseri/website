---
author: Vedran Miletić
---

# Upravljanje zaporkama

Za pamćenje većeg broja zaporki potrebno je koristiti upravitelj zaporkama, primjerice [KeePassXC](https://keepassxc.org/), [KeePass](https://keepass.info/) i [Bitwarden](https://bitwarden.com/). Upravitelj zaporkama sprema veći broj zaporki i štiti ih jednom glavnom zaporkom ili ključem, koji može biti izveden softverski ili [hardverski](https://www.theverge.com/2019/2/22/18235173/the-best-hardware-security-keys-yubico-titan-key-u2f).

Brojne preporuke u vezi zaporki postoje, primjerice [upute i najbolje prakse američkog Nacionalnog instituta za standarde i tehnologiju](https://auth0.com/blog/dont-pass-on-the-new-nist-password-guidelines/) i [preporuke politike zaporki za Microsoft 365](https://learn.microsoft.com/microsoft-365/admin/misc/password-policy-recommendations?view=o365-worldwide).

Današnja (super)računala i računalni resursi u oblaku omogućuju [brzo razbijanje jednostavnih zaporki](https://irontechsecurity.com/how-long-does-it-take-a-hacker-to-brute-force-a-password/), a procjenu vremena probijanja pojedinih zaporki moguće je dobiti na [stranici How Secure Is My Password? na Security.org](https://www.security.org/how-secure-is-my-password/).

!!! example "Zadatak"
    U KeePassXC-u stvorite novu bazu zaporki u korištenjem formata [KDBX 4.0](https://keepass.info/help/kb/kdbx_4.html). U bazi stvorite dvije grupe po želji; u prvoj stvorite jednu stavku, a u drugoj dvije. Svakoj stvorenoj stavci dodajte oznake, a na barem jednoj dodajte vrijeme isteka i URL na koji se odnosi.

    Kod stvaranja lozinki iskoristite ugrađeni generator i pronađite potrebne postavke da se provjera zaporke na stranici *How Secure Is My Password?* zeleni. Uvjerite se da:

    - KeePassXC može unijeti korisničko ime i zaporku u web preglednik, odnosno terminal,
    - vrijeme trajanja zaporke može isteći,
    - unose je moguće filtrirati po nazivu i oznakama.

!!! example "Zadatak"
    Proučite u [službenoj dokumentaciji KeeePassXC-a](https://keepassxc.org/docs/) odjeljak o [dodavanju TOTP-a](https://keepassxc.org/docs/KeePassXC_GettingStarted#_adding_totp_to_an_entry). Za aplikaciju po želji uključite dvofaktorsku autentifikaciju.

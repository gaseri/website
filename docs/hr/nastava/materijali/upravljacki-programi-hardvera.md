---
author: Vedran Miletić
---

# Upravljački programi hardvera

## Upravljački programi grafičkih procesora

!!! todo
    Ovaj dio treba napisati u cijelosti.

KMS **Postavljanje rezolucije od strane jezgre** je značajka novijih verzija Linux jezgre (od 2009. godine). Ona ima čitav niz prednosti nad postavljanjem rezolucije u korisničkoj domeni, o čemu više detalja nude [Wikipedija](https://en.wikipedia.org/wiki/Mode_setting) i [Fedorin wiki](https://fedoraproject.org/wiki/Features/KernelModesetting).

KMS pod Linuxom podržavaju sve Intel grafičke kartice, sve AMD/ATi grafičke kartice i sve NVIDIA grafičke kartice. Nažalost, postoje iznimke (najčešće noviji modeli i najčešće na prijenosnim računalima) za koje podrška još nije dovršena; u tom slučaju može biti problema s pokretanjem Plymoutha i kasnije X servera.

TODO opiši user mode setting, atomic mode setting za AMD or so

### Dijelovi upravljačkog programa u korisničkom prostoru

!!! todo
    - OpenGL `glxinfo`
    - OpenCL `clinfo`
    - Vulkan `vulkaninfo`

## Upravljački programi zvučnih uređaja

!!! todo
    Ovaj dio treba napisati u cijelosti.

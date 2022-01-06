---
author: Vedran Miletić
---

# Kontinuirana integracija korištenjem GitLaba

Pored [posluživanja Git repozitorija i povezanih usluga](https://docs.gitlab.com/ee/user/project/), [GitLab nudi i funkcionalnost](https://en.wikipedia.org/wiki/CI/CD) [kontinuirane integracije, isporuke i implementacije](https://en.wikipedia.org/wiki/CI/CD) (engl. *continuous integration, delivery and deployment*, kraće CI/CD).

Kontinuirana integracija omogućuje nam izgradnju, testiranje i pokretanje softvera u čistoj okolini, što nam pomaže pronaći probleme kod integracije promjena koji developeru možda promaknu dok radi u razvojnoj okolini. GitLab omogućuje korištenje [Docker](https://www.docker.com/) kontejnera ili vlastitih (virtualnih) računala u kojima se izvodi [GitLab Runner](https://docs.gitlab.com/runner/).

Korištenje CI/CD u besplatnoj verziji GitLaba [ograničeno je na 400 minuta mjesečno od rujna 2020. godine](https://about.gitlab.com/blog/2020/09/01/ci-minutes-update-free-users/), što je sasvim dovoljno za eksperimentiranje i edukaciju.

## Konfiguracija kontinuirane integracije

Recimo da imamo repozitorij u kojem postoje datoteke:

- `program.py` koja sadrži program
- `biblioteka.py` koja sadrži funkcije
- `test_biblioteka.py` koja sadrži testove funkcija iz datoteke `biblioteka.py` koji se pokreću korištenjem [pytest](https://www.pytest.org/)-a

Kako bismo uključili kontinuiranu integraciju, na GitLabu je potrebno u tom repozitoriju stvoriti datoteku `.gitlab-ci.yml` ([više detalja o formatu YAML](https://yaml.org/)) s postavkama za testiranje (`test:`) i pokretanje (`run:`). Moguće je koristiti predložak za Python koji GitLab nudi. U našem slučaju postavit ćemo da datoteka bude oblika:

``` yaml
# This file is a template, and might need editing before it works on your project.
# Official language image. Look for the different tagged releases at:
# https://hub.docker.com/r/library/python/tags/
image: python:latest

# Change pip's cache directory to be inside the project directory since we can
# only cache local items.
variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

# Pip's cache doesn't store the python packages
# https://pip.pypa.io/en/stable/reference/pip_install/#caching
#
# If you want to also cache the installed packages, you have to install
# them in a virtualenv and cache it as well.
cache:
  paths:
    - .cache/pip
    - venv/

before_script:
  - python -V  # Print out python version for debugging
  - pip install virtualenv
  - virtualenv venv
  - source venv/bin/activate

test:
  script:
    - pip install pytest
    - venv/bin/pytest

run:
  script:
    - python program.py
```

Uočimo da se kod testiranja instalira i pokreće `pytest`, a kod pokretanja sam `program.py`. Odmah nakon commitanja ove datoteke u repozitorij GitLab će pokrenuti navedene poslove, a njihov rezultat možemo pratiti na GitLabu na stranici repozitorija u odjeljku `CI/CD`: stranice `Pipelines` i `Jobs`.

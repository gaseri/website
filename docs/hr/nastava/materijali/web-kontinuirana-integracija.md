---
marp: true
author: Vedran Miletić
title: Kontinuirana integracija i isporuka web aplikacija
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Kontinuirana integracija i kontinuirana isporuka za web aplikacije

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Motivacija

Programer ste u timu i popravili ste *bug* koji vam je bio dodijeljen, odnosno izvršili ste promjene u kodu nakon kojih na vašoj (razvojnoj) verziji aplikacije pokrenutoj lokalno izgleda da sve radi kako treba.

Možete li napraviti iste promjene na (produkcijskoj) verziji aplikacije, odnosno verziji koja je dostupna korisnicima?

---

![Falling is just like flying bg 80% left:60%](https://www.monkeyuser.com/assets/images/2021/231-mixed-feelings.png)

## Pomiješani osjećaji

Izvor: [Mixed Feelings](https://www.monkeyuser.com/2021/mixed-feelings/) (MonkeyUser, 25th November 2021)

---

## Povezane teme

* [dvanaestofaktorska aplikacija](https://12factor.net/)
    - [5. faktor: Izgradite, objavite izdanje, pokrenite](https://12factor.net/build-release-run)
    - [10. faktor: Pariter razvoja/produkcije](https://12factor.net/dev-prod-parity)
* [rječnik pojmova oblaku urođenih aplikacija](https://glossary.cncf.io/)
    - [Continuous integration (CI)](https://glossary.cncf.io/continuous-integration/)
    - [Continuous delivery (CD)](https://glossary.cncf.io/continuous-delivery/)
* [Developer Roadmaps](https://roadmap.sh/): [Frontend](https://roadmap.sh/frontend), [Backend](https://roadmap.sh/backend) i [DevOps](https://roadmap.sh/devops)
* primjena u tvrtkama:
    - Cisco, kompanija temeljena na vlasničkom softveru: [What is CI/CD?](https://www.cisco.com/c/en/us/solutions/data-center/data-center-networking/what-is-ci-cd.html)
    - Red Hat, kompanija temelejna na slobodnom softveru otvorenog koda: [What is CI/CD? (redhat.com)](https://www.redhat.com/en/topics/devops/what-is-ci-cd), [What is CI/CD? (opensource.com)](https://opensource.com/article/18/8/what-cicd)

---

## Beskonačna DevOps petlja

![DevOps Infinity Loop](https://cdn-images-1.medium.com/max/1600/1*TNJ7Rpr5G1OJHtKH-IBEFw.png)

Izvor: [Continuous Delivery Best Practices: Extent and Intent](https://www3.dbmaestro.com/blog/continuous-delivery-best-practices-extent-and-intent)

---

## Kontinuirana integracija

Kontinuirana integracija (engl. *continuous integration*, kraće CI) je prema [Wikipediji](https://en.wikipedia.org/wiki/Continuous_integration):

* praksa spajanja svih radnih kopija programera u zajedničku glavnu razvojnu granu *nekoliko puta dnevno*
* američki programer [Grady Booch](https://en.wikipedia.org/wiki/Grady_Booch) je 1991. godine predložio pojam CI [u svojoj knjizi](https://en.wikipedia.org/wiki/Booch_method) pod naslovom *Object Oriented Design with Applications*
    - nije zagovarao integraciju nekoliko puta dnevno, već *nekoliko puta tjedno ili mjesečno*, ovisno o projektu
* agilna metodologija ekstremno programiranje (engl. *extreme programming*, kraće XP) pretpostavlja da se zahtjevi korisnika/kupca jako brzo mijenjaju
    - usvojila je koncept CI tako da zagovara integriranje više od jednom dnevno, možda i *nekoliko puta na sat*

---

## Tijek rada kontinuirane integracije (1/2)

Najjednostavnije je lokalno pokretanje testova:

* programer pokreće automatizirane jedinične testove *u svojoj lokalnoj okolini*
    - nakon što je završio s razvojem koda
    - prije spajanja svojih promjena na glavnu granu
* **never break the build**: glavna grana uvijek prolazi testove, može se postaviti i pokrenuti

---

![Basic CI workflow](https://cdn-images-1.medium.com/max/1600/1*JMTOcsscPABr3rkd9RjhmA.png)

Izvor: [How to set up an efficient development workflow with Git and CI/CD](https://proandroiddev.com/how-to-set-up-an-efficient-development-workflow-with-git-and-ci-cd-5e8916f6bece)

---

## Tijek rada kontinuirane integracije (2/2)

Uz lokalno izvođenje testova, moguće je i korištenje poslužitelja za CI koji:

* **izgrađuje** softver: povlači zavisnosti, pretvara izvorni kod u izvršni oblik i pretvara popratni sadržaj u oblik u kojem se može iskoristiti
* izgrađuje softver i **izvodi testove**
* izgrađuje softver, izvodi testove i **vrši isporuku, odnosno postavljanje** softvera ako su testovi uspješno prošli
    - kontinuirana isporuka i kontinuirano postavljanje
* primjer: [CI/CD Concepts (GitLab)](https://docs.gitlab.com/ee/ci/introduction/)

---

![GitLab workflow example extended](https://docs.gitlab.com/ee/ci/introduction/img/gitlab_workflow_example_extended_v12_3.png)

---

## Implementacije CI ili CI/CD (1/2)

Platforme u oblaku:

* [GitHub Actions](https://github.com/features/actions) ([prezentacija CI/CD značajki](https://resources.github.com/ci-cd/))
* [GitLab CI/CD](https://docs.gitlab.com/ee/ci/) ([primjeri korištenja](https://docs.gitlab.com/ee/ci/examples/))
* [Atlassian Bitbucket Pipelines](https://www.atlassian.com/software/bitbucket/features/pipelines)
* [CircleCI](https://circleci.com/)
* [JetBrains TeamCity](https://www.jetbrains.com/teamcity/)
* [Azure DevOps Server](https://azure.microsoft.com/en-us/services/devops/server/) ([shematski prikaz](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/apps/devops-dotnet-webapp))
* [AppVeyor](https://www.appveyor.com/)
* [Travis CI](https://travis-ci.org/) (otvoreni kod [dostupan na GitHubu](https://github.com/travis-ci/travis-ci))
* [Cirrus CI](https://cirrus-ci.org/)
* [Semaphore](https://semaphoreci.com/) itd.

---

## Implementacije CI ili CI/CD (2/2)

Softveri otvorenog koda za samostalno postavljanje i održavanje:

* [Jenkins](https://www.jenkins.io/) (usluge postavljanja i održavanja nude [Servana](https://servanamanaged.com/services/managed-jenkins/), [Bitnami](https://bitnami.com/stack/jenkins) i drugi)
* [Buildbot](https://www.buildbot.net/)
* [Apache Gump](https://gump.apache.org/)
* [GoCD](https://www.gocd.org/)

---

## Primjeri korištenja kontinuirane integracije

* Django pokreće testove [kod svake promjene](https://github.com/django/django/commits/main) na dva sustava:
    - [GitHub Actions](https://github.com/django/django/actions)
    - [vlastitoj instanci Jenkinsa](https://djangoci.com/)
* Ruby on Rails [koristi](https://github.com/rails/rails/commits/main) [GitHub Actions](https://github.com/rails/rails/actions) i [Buildkite](https://buildkite.com/rails/rails)
* Discourse [koristi](https://github.com/discourse/discourse/commits/main) [GitHub Actions](https://github.com/discourse/discourse/actions)
* phpBB [koristi](https://github.com/phpbb/phpbb/commits/master) [GitHub Actions](https://github.com/phpbb/phpbb/actions)
* PHP (interpreter) [koristi](https://github.com/php/php-src/commits/master) [Cirrus CI](https://cirrus-ci.com/github/php/php-src), [AppVeyor](https://ci.appveyor.com/project/php/php-src), [Azure Pipelines](https://dev.azure.com/phpazuredevops/PHP/_build) i [Travis CI](https://app.travis-ci.com/github/php/php-src)
* LLVM [koristi](https://github.com/llvm/llvm-project/commits/main) [vlastitu instancu Buildbota](https://lab.llvm.org/buildbot/#/console)
* (primjer van weba) ns-3 [koristi](https://gitlab.com/nsnam/ns-3-dev/-/commits/master) [GitLab CI/CD](https://gitlab.com/nsnam/ns-3-dev/-/pipelines)
    - [konfiguracija](https://docs.gitlab.com/ee/ci/yaml/gitlab_ci_yaml.html) se piše u [obliku YAML](https://yaml.org/)

---

## Dio CI skripte za jednostavan program u Pythonu

``` yaml
image: python:latest

before_script:
  - python -V  # Print out python version for debugging
  - pip install virtualenv
  - virtualenv venv
  - source venv/bin/activate
  - pip install -r requirements.txt

test:
  script:
    - pip install pytest
    - venv/bin/pytest

run:
  script:
    - python program.py
```

---

## Dio CI skripte za Django

``` yaml
services:
  - mysql:latest
cache:
  paths:
    - ~/.cache/pip/

migrations:
  stage: build
  script:
    - python manage.py makemigrations
    # - python3 manage.py makemigrations myapp
    - python manage.py migrate
    - python manage.py check
django-tests:
  stage: test
  script:
    - echo "GRANT ALL on *.* to user;"| mysql -u root -p pass -h mysql
    - python manage.py test
```

---

## Dio CI skripte za jednostavan program u Node.js-u

``` yaml
image: node:latest

services:
  - redis:latest
  - postgres:latest

cache:
  paths:
    - node_modules/

test_async:
  script:
    - npm install
    - node ./specs/start.js ./specs/async.spec.js

test_db:
  script:
    - npm install
    - node ./specs/start.js ./specs/db-postgres.spec.js
```

---

![Won't Work For Us bg 95% left:65%](https://i0.wp.com/www.comicagile.net/wp-content/uploads/2022/04/BA8C83E8-1C86-434C-B8C7-29C878D76232.jpeg?resize=1536%2C1352&ssl=1)

## Nije to za nas

Izvor: [Won't Work For Us](https://www.comicagile.net/comic/wont-work-for-us/) (Comic Agilé #169)

---

## Kontinuirana isporuka (1/2)

Kontinuirana isporuka (engl. *continuous delivery*, kraće CD) je prema [Wikipediji](https://en.wikipedia.org/wiki/Continuous_delivery):

* pristup programskom inženjerstvu u kojem timovi proizvode softver u kratkim ciklusima, osiguravajući da se softver može pouzdano postaviti
    - u bilo kojem trenutku
    - bez dodatnog ručnog rada kod postavljanja
* cilj je izgradnja, testiranje i izdavanje softvera velikom brzinom i učestalošću
* pomaže smanjiti troškove, vrijeme i rizik isporuke promjena dopuštajući *dodatna ažuriranja aplikacija u produkciji*
* izravni i ponovljivi postupak postavljanja je preduvjet kontinuirane isporuke

---

## Kontinuirana isporuka (2/2)

![Continuous Delivery process diagram](https://upload.wikimedia.org/wikipedia/commons/c/c3/Continuous_Delivery_process_diagram.svg)

Izvor: [Wikimedia Commons File:Continuous Delivery process diagram.svg](https://commons.wikimedia.org/wiki/File:Continuous_Delivery_process_diagram.svg)

---

![Falling is just like flying bg 80% left:60%](https://turnoff.us/image/en/deployment-pipeline.png)

## Cjevovod postavljanja

Izvor: [deployment pipeline](https://turnoff.us/geek/deployment-pipeline/) {turnoff.us}

---

## Kontinuirano postavljanje

![CI/CD flow](https://www.redhat.com/cms/managed-files/styles/wysiwyg_full_width/s3/ci-cd-flow-desktop_0.png?itok=QgBYmjA2)

Izvor: [What is a CI/CD pipeline? (Red Hat Topics, Understanding DevOps)](https://www.redhat.com/en/topics/devops/what-cicd-pipeline)

Kontinuirano postavljanje (engl. *continuous deployment*, kraće CD) je prema [Wikipediji](https://en.wikipedia.org/wiki/Continuous_deployment):

* pristup u programskom inženjerstvu u kojem se programska funkcionalnost isporučuje automatiziranim postavljanjem na redovitoj bazi
* slijedi **nakon** s kontinuirane isporuke
    - u fazi kontinuirane isporuke programska se funkcionalnost smatra potencijalno prikladnom za postavljanje, ali se postavljanje *ne* događa
    - u fazi kontinuirane isporuke očekuje se ručno postavljanje

---

## Primjer primjene CI/CD u praksi

1. Dva developera rade promjene svaki na svojoj verziji softvera. Jedan od njih zaokruži cjelinu (npr. implementacija nove značajke, popravak buga) i lokalno izvede testove koji uspješno prođu.
2. Taj developer zatraži od upravitelja projekta da uvrsti njegove promjene u glavnu granu stvaranjem zahtjeva za povlačenjem promjena (engl. *pull request*).
3. Upravitelj projekta spaja upravo dobivene promjene s promjenama koje su izveli drugi developeri i povlači ih u *probnu granu*.
4. Po dolasku promjena u probnu granu, aktivira se CI/CD sustav koji izvodi instalaciju zavisnosti, pokreće testove, postavlja konfiguraciju i pokreće aplikaciju pa javlja rezultate svih navedenih radnji.
5. Ako je CI/CD uspješno prošao, promjene se (uz ručno odobrenje ili bez njega) povlače u *glavnu granu* i od tamo se vrši postavljanje aplikacije.

---

## Zaključak

* dvanaestofaktorska, odnosno oblaku urođena aplikacija može automatizirati:
    - integraciju
    - isporuku
    - postavljanje
* velik broj CI/CD alata
    - usluge u oblaku
    - alati otvorenog koda za samostalno postavljanje
* [DevOps CI/CD Explained in 100 Seconds (Fireship)](https://youtu.be/scEDHsr3APg)

---
author: Mario Ćuro, Vedran Miletić
---

# Automatizacija konfiguracije i održavanja sustava alatom Ansible

[Prema Red Hatu](https://www.redhat.com/en/topics/automation/what-is-infrastructure-automation), pod automatizacijom infrastrukture podrazumijevamo uporabu tehnologije koja obavlja zadatke uz smanjenu ljudsku pomoć kako bi se upravljalo hardverom, softverom, mrežom, operativnim sustavom i uređajima za pohranu podataka koji zajedno pružaju usluge korisnicima.

Nama je zanimljiva automatizacija konfiguracije i održavanja sustava. Time se interaktivna konfiguracija računala, primjerice korištenjem sučelja naredbenog retka, zamjenjuje definicijskim datotekama koje sadrže konfiguraciju jednog ili više računala pa se naziva [infrastrukura u obliku koda](https://en.wikipedia.org/wiki/Infrastructure_as_code) (engl. *infrastructure as code*). Takve definicijske datoteke imaju brojne prednosti, a najvažnije olakšan pregled konfiguracije, mogućnost verzioniranja konfiguracije i mogućnost višestrukog pokretanja procesa konfiguracije infrastrukture po potrebi.

Osim Ansiblea, koji koristimo u nastavku, za sličnu svrhu koriste se [Chef](https://www.chef.io/), [Puppet](https://puppet.com/), [Salt](https://saltproject.io/), [Terraform](https://www.terraform.io/) i drugi alati.

[Službena dokumentacija Ansiblea](https://docs.ansible.com/) podijeljena je u dva dijela:

- dio koji pripada zajednici slobodnog softvera i softvera otvorenog koda: [Community Documentation](https://docs.ansible.com/ansible_community.html), [Core Documentation](https://docs.ansible.com/core.html), [Galaxy Documentation](https://docs.ansible.com/galaxy.html), [Lint Documentation](https://docs.ansible.com/lint.html) i [Community Contributors](https://docs.ansible.com/community.html)
- dio koji je samo za Red Hatove pretplatnike: [Automation Documentation](https://docs.ansible.com/automation.html) i [Collections Documentation](https://docs.ansible.com/automation.html)

Mi ćemo koristiti [Ansible Documentation](https://docs.ansible.com/ansible/latest/index.html) koji pripada zajednici i koji pokriva sve relevantne značajke.

U primjerima u nastavku pretpostavljamo da su računala imenovana `posluzitelj`, `radnastanica1` i `radnastanica2`, da su ta imena navedena u `/etc/hosts` ili u DNS poslužitelju, da je postavljen SSH na svakom od sustava koji se koristi te da je moguća potrebna prijava pomoću ključeva.

## Instalacija Ansiblea

Instalirajmo [Ansible](https://www.ansible.com/) ([službena dokumentacija](https://docs.ansible.com/ansible/)) na poslužitelju i klijentima naredbom:

``` shell
$ sudo pacman -S ansible
```

Uvjerimo se da je uspješno instaliran:

``` shell
$ ansible --version
ansible 2.7.5
```

## Konfiguracija Ansiblea

Izmijenimo datoteku `/etc/ansible/hosts` da bude sadržaja:

``` ini
[local]
posluzitelj

[workstation]
radnastanica1
radnastanica2
```

Provjerimo možemo li doseći sva računala:

``` shell
$ ansible all -m ping
posluzitelj | SUCCESS => {
    "changed": false,
    "ping": "pong"
}
radnastanica1 | SUCCESS => {
    "changed": false,
    "ping": "pong"
}
radnastanica2 | SUCCESS => {
    "changed": false,
    "ping": "pong"
}
```

Provjerimo možemo li pokrenuti naredbu na računalima:

``` shell
$ ansible all -a "/usr/bin/hostname"
posluzitelj | CHANGED | rc=0 >>
posluzitelj

radnastanica1 | CHANGED | rc=0 >>
radnastanica1

radnastanica2 | CHANGED | rc=0 >>
radnastanica2
```

## Stvaranje i pokretanje Ansible playbooka

Stvorimo [YAML](https://yaml.org/) datoteku `/etc/ansible/playbook.yml` koja će tražiti instalaciju paketa koji sadrži MkDocs (paket je nazvan upravo `mkdocs`) i pokrenuti uslugu [Uncomplicated Firewall](https://wiki.archlinux.org/title/Uncomplicated_Firewall) (kraće UFW, ime usluge `ufw`) ako već nije pokrenuta. Datoteka je sadržaja:

``` yaml
- name: Install MkDocs and ensure UFW is running
  hosts: all
  become: yes
  tasks:
    - name: install mkdocs
      package:
        name: mkdocs
        state: present
    - name: ensure ufw is running
      service:
        name: ufw
        state: started
```

Pokrenimo Ansible playbook datoteku `playbook.yml` naredbom:

``` shell
$ ansible-playbook -K /etc/ansible/playbook.yml
SUDO password:

PLAY [Update and Install packages] ********************************************************************************************************************************************************************************

TASK [Gathering Facts] ********************************************************************************************************************************************************************************************
ok: [posluzitelj]
ok: [radnastanica1]
ok: [radnastanica2]

TASK [install mkdocs] *********************************************************************************************************************************************************************************************
changed: [posluzitelj]
changed: [radnastanica1]
changed: [radnastanica2]

TASK [ensure ufw is running] **************************************************************************************************************************************************************************************
ok: [posluzitelj]
ok: [radnastanica1]
ok: [radnastanica2]

PLAY RECAP ********************************************************************************************************************************************************************************************************
posluzitelj                : ok=3    changed=1    unreachable=0    failed=0
radnastanica1              : ok=3    changed=1    unreachable=0    failed=0
radnastanica2              : ok=3    changed=1    unreachable=0    failed=0
```

Paket je instaliran, a usluga je već bila pokrenuta na svim računalima tako da tu nema promjene.

Recimo da u nekom trenutku dodamo još jednu radnu stanicu imena `radnastanica3` koja ima zadovoljene iste preduvjete kao radne stanice koje smo ranije dodali. Provjerimo možemo li je doseći:

``` shell
$ ansible all -m ping
posluzitelj | SUCCESS => {
    "changed": false,
    "ping": "pong"
}
radnastanica1 | SUCCESS => {
    "changed": false,
    "ping": "pong"
}
radnastanica2 | SUCCESS => {
    "changed": false,
    "ping": "pong"
}
radnastanica3 | SUCCESS => {
    "changed": false,
    "ping": "pong"
}
```

Ponovimo pokretanje Ansible playbook `playbook.yml` naredbom:

``` shell
$ ansible-playbook -K /etc/ansible/playbook.yml
SUDO password:

PLAY [Update and Install packages] *********************************************

TASK [Gathering Facts] *********************************************************
ok: [posluzitelj]
ok: [radnastanica1]
ok: [radnastanica2]
ok: [radnastanica3]

TASK [install mkdocs] **********************************************************
ok: [posluzitelj]
ok: [radnastanica1]
ok: [radnastanica2]
changed: [radnastanica3]

TASK [ensure ufw is running] ***************************************************
ok: [posluzitelj]
ok: [radnastanica1]
ok: [radnastanica2]
ok: [radnastanica3]

PLAY RECAP *********************************************************************
radnastanica1              : ok=2    changed=0    unreachable=0    failed=0
radnastanica2              : ok=2    changed=0    unreachable=0    failed=0
radnastanica3              : ok=2    changed=1    unreachable=0    failed=0
posluzitelj                : ok=2    changed=0    unreachable=0    failed=0
```

Uočimo kako se mijenja samo `radnastanica3` koja ranije nije imala instaliran paket `mkdocs`.

Više detalja moguće je pronaći na [stranici Ansible na ArchWikiju](https://wiki.archlinux.org/title/Ansible).

## Pokretanje postojećih Ansible playbooka

Vrlo jednostavan primjer postojećeg playbooka koji instalira HTTP poslužitelj Apache može se pronaći na službenom GitHubu u repozitoriju [ansible/workshop-examples](https://github.com/ansible/workshop-examples). Složeniji primjeri dostupni su u repozitoriju [ansible/ansible-examples](https://github.com/ansible/ansible-examples). Kako su neki od njih pisani specifično za Red Hat Enterprise Linux, potrebne su manje preinake u navođenju sustava za instalaciju paketa i putanjama pojedinih direktorija i datoteka kako bi se mogle pokrenuti na Arch Linuxu.

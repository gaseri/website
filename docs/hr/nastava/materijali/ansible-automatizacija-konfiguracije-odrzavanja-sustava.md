---
author: Mario Ćuro, Vedran Miletić
---

# Automatizacija konfiguracije i održavanja sustava alatom Ansible

[Prema Red Hatu](https://www.redhat.com/en/topics/automation/what-is-infrastructure-automation), pod automatizacijom infrastrukture podrazumijevamo uporabu tehnologije koja obavlja zadatke uz smanjenu ljudsku pomoć kako bi se upravljalo hardverom, softverom, mrežom, operativnim sustavom i uređajima za pohranu podataka koji zajedno pružaju usluge korisnicima.

Nama je zanimljiva automatizacija konfiguracije i održavanja sustava. U nastavku pretpostavljamo da su računala imenovana `posluzitelj`, `radnastanica1` i `radnastanica2`, da su ta imena navedena u `/etc/hosts` ili u DNS poslužitelju, da je postavljen SSH na svakom od sustava koji se koristi te da je moguća potrebna prijava pomoću ključeva.

## Instalacija Ansiblea

Instalirajmo [Ansible](https://www.ansible.com/) ([službena dokumentacija](https://docs.ansible.com/ansible/)) na poslužitelju i klijentima naredbom:

``` shell
$ sudo apt install ansible
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

## Ansible playbook

Stvorimo [YAML](https://yaml.org/) datoteku `/etc/ansible/playbook.yml` koja će tražiti instalaciju paketa koji sadrži MkDocs (paket je nazvan upravo `mkdocs`) i pokrenuti uslugu [Uncomplicated Firewall](https://help.ubuntu.com/community/UFW) (kraće UFW, ime usluge `ufw`) ako već nije pokrenuta. Datoteka je sadržaja:

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

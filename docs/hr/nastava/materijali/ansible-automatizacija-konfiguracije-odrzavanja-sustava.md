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

## Primjeri korištenja Ansiblea za održavanje računalnih praktikuma

### Nadogradnja svih paketa

Za upravljanje paketima na Arch Linuxu koristi se modul zajednice `community.general.pacman` ([dokumentacija](https://docs.ansible.com/ansible/latest/collections/community/general/pacman_module.html)), a playbook `syu.yml` je dostupan u [odjeljku Playbook stranice Ansible na ArchWikiju](https://wiki.archlinux.org/title/Ansible#Playbook).

### Uključivanje pokretanja korištenjem utičnica za Docker i libvirt daemone

Za upravljanje systemd jedinkama koristi se ugrađeni modul `ansible.builtin.systemd` ([dokumentacija](https://docs.ansible.com/ansible/latest/collections/ansible/builtin/systemd_module.html)).

``` yaml
- name: Enable socket activation for Docker and libvirt
  hosts: workstation
  become: yes

  tasks:
    - name: enable systemd socket unit for Docker daemon
      systemd:
        name: docker.socket
        state: started
        enabled: true
    - name: enable systemd socket unit for libvirt daemon
      systemd:
        name: libvirtd.socket
        state: started
        enabled: true
```

### Dodavanje lokalnog korisnika bez administratorskih ovlasti

Za upravljanje korisnicima koristi se ugrađeni modul `ansible.builtin.user` ([dokumentacija](https://docs.ansible.com/ansible/latest/collections/ansible/builtin/user_module.html)).

``` yaml
- name: Add user korisnik and configure groups
  hosts: workstation
  become: yes

  tasks:
    - name: add user korisnik with password korisnik
      user:
        name: korisnik
        comment: FIDIT korisnik
        password: $6$CvycajZY8Z5mk0qC$AdU3seLtogi4XJGXa1MwRuBZ7pHiuLrQaricpAJ2Id5sqV1.UBoUS//yKGPxrK9nzgvmiiIJI2isGamrtGBGD.
        shell: /bin/bash
        groups: docker,libvirt
        append: yes
```

[Hashirani zapis zaporke](https://docs.ansible.com/ansible/latest/collections/ansible/builtin/user_module.html#return-password) naveden kao vrijednost ključa `password` je generiran prema [uputama iz Ansibleove dokumentacije](https://docs.ansible.com/ansible/latest/reference_appendices/faq.html#how-do-i-generate-encrypted-passwords-for-the-user-module) korištenjem [algoritma SHA-512](https://wiki.archlinux.org/title/SHA_password_hashes) i soli `CvycajZY8Z5mk0qC` naredbom

``` shell
$ ansible all -i localhost, -m debug -a "msg={{ 'korisnik' | password_hash('sha512', 'CvycajZY8Z5mk0qC') }}"
[DEPRECATION WARNING]: Encryption using the Python crypt module is deprecated. The Python crypt module is deprecated and will be removed from Python 3.13. Install the passlib library for continued encryption functionality. This feature will be removed in version 2.17. Deprecation warnings can be disabled by setting deprecation_warnings=False in ansible.cfg.
localhost | SUCCESS => {
    "msg": "$6$CvycajZY8Z5mk0qC$AdU3seLtogi4XJGXa1MwRuBZ7pHiuLrQaricpAJ2Id5sqV1.UBoUS//yKGPxrK9nzgvmiiIJI2isGamrtGBGD."
}
```

Korisnik `korisnik` je nakon stvaranja dodan u grupe `docker` i `libvirt`, navedene kao vrijednost ključa `groups`, čime dobiva pravo pokretanja [Docker](https://www.docker.com/) kontejnera i [libvirt](https://libvirt.org/) virtualnih strojeva. Vrijednost `yes` ključa `append` osigurava da će korisnik biti [dodan u navedene grupe](https://docs.ansible.com/ansible/latest/collections/ansible/builtin/user_module.html#return-append) (umjesto da se popis grupa zamijeni navedenim).

### Brisanje datoteka korisnika

Rješenje prezentirano u nastavku pokreće naredbe izravno u ljusci korištenjem ugrađenog modula `ansible.builtin.shell` ([dokumentacija](https://docs.ansible.com/ansible/latest/collections/ansible/builtin/shell_module.html)).

``` yaml
- name: Logout user korisnik
  hosts: workstation

  tasks:
    - name: kill processes owned by user korisnik using loginctl
      shell: loginctl --signal=KILL kill-user korisnik
```

``` yaml
- name: Restore files in home directory of user korisnik to defaults
  hosts: workstation

  tasks:
    - name: remove files in home directory of user korisnik
      shell: rm -r /home/korisnik/{*,.[!.]*}
      args:
        executable: /bin/bash
    - name: copy files from /etc/skel to home directory of user korisnik
      shell: cp -r /etc/skel/{*,.[!.]*} /home/korisnik
      args:
        executable: /bin/bash
```

## Pokretanje postojećih Ansible playbooka

Vrlo jednostavan primjer postojećeg playbooka koji instalira HTTP poslužitelj Apache može se pronaći na službenom GitHubu u repozitoriju [ansible/workshop-examples](https://github.com/ansible/workshop-examples). Složeniji primjeri dostupni su u repozitoriju [ansible/ansible-examples](https://github.com/ansible/ansible-examples). Kako su neki od njih pisani specifično za Red Hat Enterprise Linux, potrebne su manje preinake u navođenju sustava za instalaciju paketa i putanjama pojedinih direktorija i datoteka kako bi se mogle pokrenuti na Arch Linuxu.

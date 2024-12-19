---
author: Vedran Miletić
---

# O računalnim praktikumima za nastavu

[Fakultet informatike i digitalnih tehnologija](https://www.inf.uniri.hr/) održava nastavu u četiri računalna praktikuma:

- O-350/O-351 i O-366/O-367 s po 36 računala (35 studentskih i jedno nastavničko),
- O-359 s 21 računalom (20 studentskih i jedno nastavničko) i
- O-365 s 26 računala (25 studentskih i jedno nastavničko).

## Upute za instalacija i konfiguracija računala u računalnim praktikumima

U nastavku je opisan način instalacije i konfiguracije računala u računalnim praktikumima. Pritom pretpostavljamo da je operacijski sustav Windows, bez obzira radi li se o verziji 10 ili 11, instaliran na računalima u konfiguraciji pokretanja putem UEFI-ja te da su instalirane sve dostupne nadogradnje. Nadalje pretpostavljamo da računala u sebi imaju SSD i HDD, da je sav hardver ispravan, da je UEFI firmware ("BIOS") osvježen na zadnju dostupnu verziju i da je u konfiguraciji firmwarea:

- isključen Secure Boot[^1] i
- uključena podrška za [hardverski potpomognutu virtualizaciju](https://en.wikipedia.org/wiki/Hardware-assisted_virtualization)[^2] ([Intel VT-x](https://www.intel.com/content/www/us/en/virtualization/virtualization-technology/intel-virtualization-technology.html)/[AMD-V](https://www.amd.com/en/solutions/hci-and-virtualization)).

[^1]: Distribucije temeljene na Arch Linuxu moguće je [konfigurirati tako da se pokreću kad je Secure Boot uključen](https://wiki.archlinux.org/title/Unified_Extensible_Firmware_Interface/Secure_Boot), ali to komplicira održavanje i nepotrebno je na računalima koja se koriste samo u nastavne svrhe.

[^2]: Sustav za virtualizaciju [QEMU](https://wiki.archlinux.org/title/QEMU)/[KVM](https://wiki.archlinux.org/title/KVM), kojeg koristi emulator operacijskog sustava [Android](https://wiki.archlinux.org/title/Android), zahtijeva hardverski potpomognutu virtualizaciju za svoj rad.

### Instalacija operacijskog sustava Manjaro uz operacijski sustav Windows

Operacijski sustav [Manjaro](https://manjaro.org/), popularnu distribuciju Linuxa [temeljenu na Arch Linuxu](https://wiki.archlinux.org/title/Arch-based_distributions), instalirat ćemo u konfiguraciji [dualnog pokretanja s operacijskim sustavom Windows](https://wiki.archlinux.org/title/Dual_boot_with_Windows).

Kod instalacije Manjara iskoristit ćemo njegov ugrađeni instalacijski alat [Calamares](https://calamares.io/). Prilikom instalacije ćemo:

- na stranici `Particije`:
    - iskoristiti postojeću EFI particiju stvorenu prilikom instalacije operacijskog sustava Windows
    - particiju koju ćemo montirati na `/` napraviti na SSD-u s datotečnim sustavom `btrfs`
    - particiju koju ćemo montirati na `/home` napraviti na HDD-u s datotečnim sustavom `ext4`
- na stranici `Korisnici`:
    - postaviti ime na `FIDIT Sensei`
    - postaviti korisničko ime na `sensei`
    - postaviti ime domaćina na `odj-oxxx-yyy`
    - postaviti zaporku po želji
    - isključiti automatsku prijavu bez traženja zaporke i uključiti korištenje iste zaporke za administratorski račun

Nakon instalacije ponovno ćemo pokrenuti računalo.

### Konfiguracija operacijskog sustava Manjaro nakon instalacije

Konfigurirat ćemo [GRUB](https://wiki.archlinux.org/title/GRUB). U datoteci `/etc/default/grub` ćemo:

- promijeniti liniju `GRUB_DEFAULT=0` u `GRUB_DEFAULT=2` tako da se kao zadani pokreće operacijski sustav Windows
- promijeniti liniju `GRUB_TIMEOUT=5` u `GRUB_TIMEOUT=90` tako da onaj tko pokreće sva računala u računalnom praktikumu za potrebe održavanja ima vremena odabrati pokretanje operacijskog sustava Manjaro prije nego pokrene zadani operacijski sustav
- odkomentirati liniju `#GRUB_DISABLE_LINUX_UUID=true`, odnosno maknuti `#` i ostaviti `GRUB_DISABLE_LINUX_UUID=true`, tako da se koriste nazivi uređaja umjesto UUID-a diskova što olakšava kloniranje particije operacijskog sustava među računalima

Dodatno ćemo instalirati [Avahi](https://wiki.archlinux.org/title/Avahi) i uključiti pokretanje njegovog daemona:

``` shell
$ sudo pamac install avahi
(...)
$ sudo systemctl enable --now avahi-daemon.service
```

Naposlijetku ćemo uključiti pokretanje [OpenSSH](https://wiki.archlinux.org/title/Secure_Shell) daemona:

``` shell
$ sudo systemctl enable --now sshd.service
```

i generirati novi SSH ključ sa zadanim postavkama i bez zaporke:

``` shell
$ ssh-keygen
Generating public/private rsa key pair.
Enter file in which to save the key (/home/sensei/.ssh/id_rsa):
Created directory '/home/sensei/.ssh'.
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/sensei/.ssh/id_rsa
Your public key has been saved in /home/sensei/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:x8BAgC63dlAV/QJclVtUHuB7w9XWdcQFVNyJ6z8vjUU sensei@odj-oxxx-yyy
The key's randomart image is:
+---[RSA 3072]----+
|   ..==+...oo+**X|
|  . . oo. ...o oO|
| . .   .o. o. o =|
|. +     .oo  + oE|
| o o    S.o o +. |
|  o .    .   o ..|
| . .          .+ |
|              oo.|
|               .+|
+----[SHA256]-----+
```

Dodat ćemo novostvoreni ključ u `.ssh/authorized_keys` ručno ili naredbom:

``` shell
$ ssh-copy-id localhost
/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: "/home/sensei/.ssh/id_rsa.pub"
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys

Number of key(s) added: 1

Now try logging into the machine, with:   "ssh 'localhost'"
and check to make sure that only the key(s) you wanted were added
```

### Kloniranje slike diska instaliranog operacijskog sustava na računala

Iskoristit ćemo [Clonezillu](https://clonezilla.org/) za stvaranje slike diska koja se može prebaciti na ostala računala i pohraniti je na neki vanjski medij. Zatim ćemo prebaciti tu sliku na ostala računala.

!!! note
    U nastavku ćemo pretpostaviti da su upute pisane za računalni praktikum O-359. Ostali računalni praktikumi se konfiguriraju analogno.

Nakon kloniranja i uspješnog pokretanja operacijskog sustava Manjaro postavit ćemo ime domaćina naredbom:

``` shell
$ sudo hostnamectl hostname odj-o359-101
```

te u datoteci `/etc/hosts` promijeniti liniju `127.0.1.1  odj-oxxx-yyy` u `127.0.1.1  odj-o359-101`. Varirat ćemo broj koji se odnosi na računalni praktikum i broj koji se odnosi na pojedino računalo po potrebi.

### Konfiguracija nastavničkog računala

Na nastavničkom računalu ćemo instalirati [Ansible](https://wiki.archlinux.org/title/Ansible) naredbom:

``` shell
$ sudo pamac install ansible
```

Stvorit ćemo datoteku `/etc/ansible/hosts` sadržaja:

``` ini
[o359]
odj-o359-101.local
odj-o359-102.local
odj-o359-103.local
odj-o359-104.local
odj-o359-105.local
odj-o359-106.local
odj-o359-107.local
odj-o359-108.local
odj-o359-109.local
odj-o359-110.local
odj-o359-111.local
odj-o359-112.local
odj-o359-113.local
odj-o359-114.local
odj-o359-115.local
odj-o359-116.local
odj-o359-117.local
odj-o359-118.local
odj-o359-119.local
odj-o359-120.local
odj-o359-121.local

[o359stud]
odj-o359-102.local
odj-o359-103.local
odj-o359-104.local
odj-o359-105.local
odj-o359-106.local
odj-o359-107.local
odj-o359-108.local
odj-o359-109.local
odj-o359-110.local
odj-o359-111.local
odj-o359-112.local
odj-o359-113.local
odj-o359-114.local
odj-o359-115.local
odj-o359-116.local
odj-o359-117.local
odj-o359-118.local
odj-o359-119.local
odj-o359-120.local
odj-o359-121.local
```

## Ansible playbook datoteke za instalaciju, konfiguraciju i održavanje računalnog praktikuma te pomoćne radnje prilikom održavanja nastave i ispita

!!! note
    [Ansible](https://www.ansible.com/) [playbook datoteke](https://docs.ansible.com/ansible/latest/playbook_guide/playbooks_intro.html) u nastavku provjerene su korištenjem [Ansible Linta](https://ansible.readthedocs.io/projects/lint/) i [profila](https://ansible.readthedocs.io/projects/lint/profiles/) `production`.

### Administratorske operacije u računalnom praktikumu

#### Nadogradnja svih paketa operacijskog sustava

``` yaml
- name: All hosts up-to-date
  hosts: o359
  become: true

  tasks:
    - name: Full system upgrade
      community.general.pacman:
        update_cache: true
        upgrade: true

    - name: Run Pamac update including AUR packages
      ansible.builtin.command: /usr/bin/pamac update --no-confirm --aur
      register: pamac_update_aur
      changed_when: pamac_update_aur.rc == 0

    - name: Clean up afterwards
      ansible.builtin.file:
        path: /var/tmp/pamac-build-sensei
        state: absent
```

#### Ponovno pokretanje računala

``` yaml
- name: Reboot student hosts
  hosts: o359stud
  become: true

  tasks:
    - name: Unconditionally reboot the machine
      ansible.builtin.reboot:
```

#### Isključivanje računala

``` yaml
- name: Shut down student hosts
  hosts: o359stud
  become: true

  tasks:
    - name: Unconditionally shut down the machine
      community.general.shutdown:
```

#### Dohvaćanje informacije o verziji UEFI firmwarea, odnosno BIOS-a

``` yaml
- name: Print UEFI firmware or BIOS version to standard output
  hosts: o359
  become: true

  tasks:
    - name: Print to stdout and filter
      ansible.builtin.shell: set -o pipefail && dmidecode | grep Version
      register: dmidecode_version
      changed_when: false

    - name: Prin information retrieved in previous task
      ansible.builtin.debug:
        var: dmidecode_version.stdout_lines
```

### Instalacija i konfiguracija Veyona

!!! quote "ToDo"
    Ovdje treba dodatno opisati proces instalacije i konfiguracije [Veyona](https://veyon.io/).

#### Instalacija softvera za studije na FIDIT-u

``` yaml
- name: Install software for courses at FIDIT
  hosts: o359
  become: true

  tasks:
    - name: Install npm and solidity
      community.general.pacman:
        name:
          - emacs
          - tree
          - php
          - xdebug
          - jq
          - siege
          - npm
          - python-cryptography
          - docker
          - docker-buildx
          - docker-compose
          - virt-manager
          - qemu-full
          - ansible
          - ansible-lint
          - python-django
          - python-pylint
          - mypy
          - autopep8
          - python-isort
          - python-black
          - flake8
          - python-flake8-black
          - python-flake8-docstrings
          - python-flake8-isort
          - python-pytest
          - python-pytest-isort
          - python-pytest-flake8
          - python-mpi4py
          - shadow
          - jdk11-openjdk
          - intellij-idea-community-edition
        state: present

    - name: Run Pamac update including AUR packages
      ansible.builtin.command: /usr/bin/pamac build --no-confirm visual-studio-code-bin mozart2-bin emacs-oz-mode gns3-gui gns3-server ns3 simgrid solidity-bin
      register: pamac_build
      changed_when: pamac_build.rc == 0

    - name: Clean up afterwards
      ansible.builtin.file:
        path: /var/tmp/pamac-build-sensei
        state: absent
```

#### Instalacija softvera za kolegij INF-BioTech

``` yaml
- name: Install software for INF-BioTech course
  hosts: o359
  become: true

  tasks:
    - name: Install Open Babel with GUI, PyMOL, and Python packages
      community.general.pacman:
        name:
          - openbabel
          - wxwidgets-gtk3
          - pymol
          - python-matplotlib
          - python-pandas
          - python-scipy
          - python-sympy
        state: present

    - name: Build and install Avogadro and UGENE
      ansible.builtin.command: /usr/bin/pamac build --no-confirm avogadroapp ugene
      register: pamac_build
      changed_when: pamac_build.rc == 0

    - name: Clean up afterwards
      ansible.builtin.file:
        path: /var/tmp/pamac-build-sensei
        state: absent
```

#### Konfiguracija zadanog OS-a za pokretanje u GRUB-u

``` yaml
- name: Configure GRUB default
  hosts: o359
  become: true

  tasks:
    - name: Ensure GRUB boots Manjaro by default
      ansible.builtin.lineinfile:
        path: /etc/default/grub
        regexp: '^GRUB_DEFAULT='
        line: 'GRUB_DEFAULT=0'

    - name: Make GRUB config
      ansible.builtin.command: grub-mkconfig -o /boot/grub/grub.cfg
      register: grub_mkconfig
      changed_when: grub_mkconfig.rc == 0
```

#### Konfiguracija bilježenja vremena pristupa u datotečnom sustavu

``` yaml
- name: Enable atime in /etc/fstab
  hosts: o359
  become: true

  tasks:
    - name: Ensure SELinux is set to enforcing mode
      ansible.builtin.lineinfile:
        path: /etc/fstab
        regexp: '^UUID=6797c252-79f2-4278-9563-1bb6a7bbe222'
        line: 'UUID=6797c252-79f2-4278-9563-1bb6a7bbe222 /home ext4 defaults,strictatime 0 0 '
```

#### Konfiguracija Dockera

``` yaml
- name: Configure Docker
  hosts: o359
  become: true

  tasks:
    - name: Create data directory
      ansible.builtin.file:
        path: /home/data
        state: directory
        mode: '0755'

    - name: Create directory for Docker data
      ansible.builtin.file:
        path: /home/data/docker
        state: directory
        mode: '0710'

    - name: Create directory for Docker configuration
      ansible.builtin.file:
        path: /etc/docker
        state: directory
        mode: '0755'

    - name: Configure Docker data path
      ansible.builtin.copy:
        content: |
          {
            "data-root": "/home/data/docker"
          }
        dest: /etc/docker/daemon.json
        mode: '0755'

    - name: Restart Docker daemon
      ansible.builtin.systemd:
        name: docker.service
        state: restarted

    - name: Remove existing docker data root
      ansible.builtin.file:
        path: /var/lib/docker
        state: absent

    - name: Enable socket for Docker daemon
      ansible.builtin.systemd:
        name: docker.socket
        state: started
        enabled: true
```

#### Konfiguracija libvirta

``` yaml
- name: Configure libvirt
  hosts: o359
  become: true

  tasks:
    - name: Create data directory
      ansible.builtin.file:
        path: /home/data
        state: directory
        mode: '0755'

    - name: Destroy default libvirt pool
      ansible.builtin.command: /usr/bin/virsh pool-destroy default
      ignore_errors: true
      register: virsh_pool_destroy
      changed_when: virsh_pool_destroy.rc == 0

    - name: Remove existing libvirt images in default pool directory
      ansible.builtin.file:
        path: /var/lib/libvirt/images
        state: absent

    - name: Delete default libvirt pool
      ansible.builtin.command: /usr/bin/virsh pool-delete default
      ignore_errors: true
      register: virsh_pool_delete
      changed_when: virsh_pool_delete.rc == 0

    - name: Undefine default libvirt pool
      ansible.builtin.command: /usr/bin/virsh pool-undefine default
      ignore_errors: true
      register: virsh_pool_undefine
      changed_when: virsh_pool_undefine.rc == 0

    - name: Create directory for libvirt images
      ansible.builtin.file:
        path: /home/data/libvirt/images
        state: directory
        mode: '0755'

    - name: Define new default libvirt pool
      ansible.builtin.command: /usr/bin/virsh pool-define-as default dir - - - - /home/data/libvirt/images
      register: virsh_pool_define_as
      changed_when: virsh_pool_define_as.rc == 0

    - name: Build new default libvirt pool
      ansible.builtin.command: /usr/bin/virsh pool-build default
      register: virsh_pool_build
      changed_when: virsh_pool_build.rc == 0

    - name: Start new default libvirt pool
      ansible.builtin.command: /usr/bin/virsh pool-start default
      register: virsh_pool_start
      changed_when: virsh_pool_start.rc == 0

    - name: Autostart new default libvirt pool
      ansible.builtin.command: /usr/bin/virsh pool-autostart default
      register: virsh_pool_autostart
      changed_when: virsh_pool_autostart.rc == 0

    - name: Restart libvirt daemon
      ansible.builtin.systemd:
        name: libvirtd.service
        state: restarted

    - name: Enable socket for libvirt daemon
      ansible.builtin.systemd:
        name: libvirtd.socket
        state: started
        enabled: true
```

#### Konfiguracija zaporke korisnika s administratorskim ovlastima

``` yaml
- name: Change password for user sensei
  hosts: o359
  become: true

  tasks:
    - name: Change password for user sensei
      ansible.builtin.user:
        name: sensei
        password: $6$o9vnobuw1GRKliS1$V1IzPOorSQFzdtc/FSlGhZzqLEtAPoH2sw8eJyYRJphrdCa8AS1UXuvgSgWdFzUu3Y9UQp11y7VkXP4sLUW9V1
```

#### Konfiguracija (običnog) korisnika

``` yaml
- name: Add user korisnik, amend user directory and skeleton with useful files
  hosts: o359
  become: true

  tasks:
    - name: Add user korisnik
      ansible.builtin.user:
        name: korisnik
        comment: FIDIT korisnik
        password: $6$CvycajZY8Z5mk0qC$AdU3seLtogi4XJGXa1MwRuBZ7pHiuLrQaricpAJ2Id5sqV1.UBoUS//yKGPxrK9nzgvmiiIJI2isGamrtGBGD.
        shell: /bin/bash
        groups: docker,libvirt
        append: true

    - name: Create configuration directory for SSH
      ansible.builtin.file:
        path: /home/korisnik/.ssh
        state: directory
        owner: korisnik
        group: korisnik
        mode: '0700'

    - name: Add FIDIT authorized key
      ansible.builtin.copy:
        content: |
          ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCxmpMTCt+Tc6T31WXou0RIjzrJwtvtR\
          0eAcu1NMdACQv0SaOrcSPaXwT7mQJzaI1yHJuU0jGvA5mKEETPBIyZb63drWuqpsC0oCF\
          YRUZ/drXv8EPyy+J/iPDAFGywetewuqMSzMG7BsePXVjRWgJUN3ZU2wDEeCndUjn5TMLg\
          99UFueAUI3/o5G0HyeCR3PbzbSP4iVK+dr0yEs1q/2RfSwm2qABA7X1cY5x/Hf2MZx9rx\
          rvzXMvkeYYnChsVjOtKTEuTKNOA+ylQ12o6FVNZrNNWD+gZrc5eI0yPwX3WPRkNKFG8h4\
          pwz7zGSYQAR+SzcvNr8G+vqRW1eYvTPw/KwzreX4FhxDEYqYK8KouhQypZss8REPGP61j\
          8gicHoSpdYq52TJhIEK0Z9tYV4zV1HdCoTKY81OVYew3V0P2ycgNJEfPVbg1nF2v3w3wK\
          y9pvP0xELyQZDNjU8XtuPI6PLBoD3XPssJQY7Z7T4ZJ0OXw6dCzodlRhgXji0Og7IWCk=
          fidit-korisnik
        dest: /home/korisnik/.ssh/authorized_keys
        owner: korisnik
        group: korisnik
        mode: '0600'

    - name: Create configuration directory for SSH
      ansible.builtin.file:
        path: /etc/skel/.ssh
        state: directory
        mode: '0755'

    - name: Add FIDIT authorized key
      ansible.builtin.copy:
        content: |
          ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCxmpMTCt+Tc6T31WXou0RIjzrJwtvtR\
          0eAcu1NMdACQv0SaOrcSPaXwT7mQJzaI1yHJuU0jGvA5mKEETPBIyZb63drWuqpsC0oCF\
          YRUZ/drXv8EPyy+J/iPDAFGywetewuqMSzMG7BsePXVjRWgJUN3ZU2wDEeCndUjn5TMLg\
          99UFueAUI3/o5G0HyeCR3PbzbSP4iVK+dr0yEs1q/2RfSwm2qABA7X1cY5x/Hf2MZx9rx\
          rvzXMvkeYYnChsVjOtKTEuTKNOA+ylQ12o6FVNZrNNWD+gZrc5eI0yPwX3WPRkNKFG8h4\
          pwz7zGSYQAR+SzcvNr8G+vqRW1eYvTPw/KwzreX4FhxDEYqYK8KouhQypZss8REPGP61j\
          8gicHoSpdYq52TJhIEK0Z9tYV4zV1HdCoTKY81OVYew3V0P2ycgNJEfPVbg1nF2v3w3wK\
          y9pvP0xELyQZDNjU8XtuPI6PLBoD3XPssJQY7Z7T4ZJ0OXw6dCzodlRhgXji0Og7IWCk=
          fidit-korisnik
        dest: /etc/skel/.ssh/authorized_keys
        mode: '0644'

    - name: Create directory for Konsole profiles
      ansible.builtin.file:
        path: /home/korisnik/.local/share/konsole
        state: directory
        owner: korisnik
        group: korisnik
        mode: '0755'

    - name: Add FIDIT profile to Konsole
      ansible.builtin.copy:
        content: |
          [General]
          Command=/usr/bin/bash
          Name=FIDIT
        dest: /home/korisnik/.local/share/konsole/fidit.profile
        owner: korisnik
        group: korisnik
        mode: '0644'

    - name: Configure Konsole to have FIDIT profile as default
      ansible.builtin.copy:
        content: |
          [Desktop Entry]
          DefaultProfile=fidit.profile
        dest: /home/korisnik/.config/konsolerc
        owner: korisnik
        group: korisnik
        mode: '0600'

    - name: Create directory for Konsole profiles
      ansible.builtin.file:
        path: /etc/skel/.local/share/konsole
        state: directory
        mode: '0755'

    - name: Add FIDIT profile to Konsole
      ansible.builtin.copy:
        content: |
          [General]
          Command=/usr/bin/bash
          Name=FIDIT
        dest: /etc/skel/.local/share/konsole/fidit.profile
        mode: '0644'

    - name: Configure Konsole to have FIDIT profile as default
      ansible.builtin.copy:
        content: |
          [Desktop Entry]
          DefaultProfile=fidit.profile
        dest: /etc/skel/.config/konsolerc
        mode: '0644'
```

### Nastavničke operacije u računalnom praktikumu

#### Odjava korisnika (`logout.yml`)

``` yaml
- name: Logout user korisnik
  hosts: o359stud

  tasks:
    - name: Kill processes owned by user korisnik using loginctl
      ansible.builtin.command: /usr/bin/loginctl --signal=KILL kill-user korisnik
      ignore_unreachable: true
      changed_when: true
```

#### Brisanje datoteka u kućnom direktoriju korisnika (`delete.yml`)

``` yaml
- name: Restore files in home directory of user korisnik to defaults
  hosts: o359stud

  tasks:
    - name: Remove files in home directory of user korisnik
      ansible.builtin.shell: rm -r /home/korisnik/{*,.[!.]*}
      args:
        executable: /bin/bash
      ignore_errors: true
      register: rm_home_korisnik
      changed_when: true

    - name: Copy files from /etc/skel to home directory of user korisnik
      ansible.builtin.shell: cp -r /etc/skel/.[!.]* /home/korisnik
      args:
        executable: /bin/bash
      changed_when: true

    - name: Change permisions for SSH configuration directory
      ansible.builtin.file:
        path: /home/korisnik/.ssh
        state: directory
        mode: '0700'

    - name: Change permisions for SSH authorized keys file
      ansible.builtin.file:
        path: /home/korisnik/.ssh/authorized_keys
        state: file
        mode: '0600'

    - name: Change permisions for Konsole resource configuration
      ansible.builtin.file:
        path: /home/korisnik/.config/konsolerc
        state: file
        mode: '0600'
```

### Instalacija Confluent Platform korištenjem Dockera

``` yaml
- name: Install and run Confluent Platform via Docker
  hosts: o359

  tasks:
    - name: Git checkout
      ansible.builtin.git:
        repo: 'https://github.com/confluentinc/cp-all-in-one.git'
        dest: /home/korisnik
        version: 7.3.0-post

    - name: Docker compose
      ansible.builtin.command: /usr/bin/docker-compose up -d -- build
      args:
        chdir: /home/korisnik/cp-all-in-one/cp-all-in-one/
      changed_when: true
```

## Ograničavanje pristupa Moodle testu na temelju IP adrese

Moodleova aktivnost `Test` u pripadnim `Postavkama` sadrži odjeljak `Dodatna ograničenja tijekom rješavanja`. Među dodatnim ograničenjima postoji i mogućnost `Ograničavanja pristupa samo ovim IP adresama` ([dokumentacija](https://docs.moodle.org/401/en/Quiz_settings#Extra_restrictions_on_attempts)). Za IP adresu se navodi javna adresa odgovarajućeg računalnog praktikuma iz popisa u nastavku; u slučaju da postoji potreba za provjerom adrese na licu mjesta, može se iskoristiti [Detektor adrese IP](https://apps.group.miletic.net/ip/).

### O-350/351

``` ip
193.198.209.231
```

### O-359

``` ip
193.198.209.234
```

### O-365

``` ip
193.198.209.232
```

### O-366/367

``` ip
193.198.209.233
```

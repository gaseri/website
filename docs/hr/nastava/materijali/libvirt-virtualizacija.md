---
author: Vedran Miletić
---

# Virtualizacija korištenjem libvirta

[libvirt](https://libvirt.org/) je skup alata otvorenog koda za upravljanje [različitim virtualizacijskim platformama](https://libvirt.org/drivers.html), uključujući [QEMU/KVM](https://libvirt.org/drvqemu.html). Nudi [aplikacijsko programsko sučelje](https://libvirt.org/html/index.html) u programskim jezicima C, Python i [brojnim drugim](https://libvirt.org/bindings.html), kao i [sučelje naredbenog retka](https://libvirt.org/manpages/index.html#tools), a pokreće se na [operacijskim sustavima sličnim Unixu](https://libvirt.org/platforms.html) i [Windowsima](https://libvirt.org/windows.html). [Mnoge aplikacije](https://libvirt.org/apps.html) koriste libvirt, uključujući [virt-manager](https://libvirt.org/apps.html#desktop-applications).

libvirt koristi [XML dokumente za konfiguraciju](https://libvirt.org/format.html). Specifično, struktura dokumenta koja opisuje virtualne strojeve prikazana je u [dijelu Domains u dokumentaciji](https://libvirt.org/formatdomain.html). Primjerice, za tipični QEMU/KVM virtualni stroj kojemu su dodijeljene 2 virtualne procesorske jezgre i 4 gigabajta RAM-a, domena zapisana u XML-u je oblika:

``` xml
<domain type="kvm">
  <name>ubuntu20.04</name>
  <uuid>3f74a815-6eb0-4e1c-abae-372afd171049</uuid>
  <metadata>
    <libosinfo:libosinfo xmlns:libosinfo="http://libosinfo.org/xmlns/libvirt/domain/1.0">
      <libosinfo:os id="http://ubuntu.com/ubuntu/20.04"/>
    </libosinfo:libosinfo>
  </metadata>
  <memory unit="KiB">4194304</memory>
  <currentMemory unit="KiB">4194304</currentMemory>
  <vcpu placement="static">2</vcpu>
  <os>
    <type arch="x86_64" machine="pc-q35-6.1">hvm</type>
    <boot dev="hd"/>
  </os>
  <features>
    <acpi/>
    <apic/>
    <vmport state="off"/>
  </features>
  <cpu mode="host-model" check="partial"/>
  <clock offset="utc">
    <timer name="rtc" tickpolicy="catchup"/>
    <timer name="pit" tickpolicy="delay"/>
    <timer name="hpet" present="no"/>
  </clock>
  <on_poweroff>destroy</on_poweroff>
  <on_reboot>restart</on_reboot>
  <on_crash>destroy</on_crash>
  <pm>
    <suspend-to-mem enabled="no"/>
    <suspend-to-disk enabled="no"/>
  </pm>
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <disk type="file" device="disk">
      <driver name="qemu" type="qcow2"/>
      <source file="/var/lib/libvirt/images/ubuntu20.04.qcow2"/>
      <target dev="vda" bus="virtio"/>
      <address type="pci" domain="0x0000" bus="0x04" slot="0x00" function="0x0"/>
    </disk>
    <disk type="file" device="cdrom">
      <driver name="qemu" type="raw"/>
      <target dev="sda" bus="sata"/>
      <readonly/>
      <address type="drive" controller="0" bus="0" target="0" unit="0"/>
    </disk>
    <controller type="usb" index="0" model="qemu-xhci" ports="15">
      <address type="pci" domain="0x0000" bus="0x02" slot="0x00" function="0x0"/>
    </controller>
    <controller type="sata" index="0">
      <address type="pci" domain="0x0000" bus="0x00" slot="0x1f" function="0x2"/>
    </controller>
    <controller type="pci" index="0" model="pcie-root"/>
    <controller type="pci" index="1" model="pcie-root-port">
      <model name="pcie-root-port"/>
      <target chassis="1" port="0x10"/>
      <address type="pci" domain="0x0000" bus="0x00" slot="0x02" function="0x0" multifunction="on"/>
    </controller>
    <controller type="pci" index="2" model="pcie-root-port">
      <model name="pcie-root-port"/>
      <target chassis="2" port="0x11"/>
      <address type="pci" domain="0x0000" bus="0x00" slot="0x02" function="0x1"/>
    </controller>
    <controller type="pci" index="3" model="pcie-root-port">
      <model name="pcie-root-port"/>
      <target chassis="3" port="0x12"/>
      <address type="pci" domain="0x0000" bus="0x00" slot="0x02" function="0x2"/>
    </controller>
    <controller type="pci" index="4" model="pcie-root-port">
      <model name="pcie-root-port"/>
      <target chassis="4" port="0x13"/>
      <address type="pci" domain="0x0000" bus="0x00" slot="0x02" function="0x3"/>
    </controller>
    <controller type="pci" index="5" model="pcie-root-port">
      <model name="pcie-root-port"/>
      <target chassis="5" port="0x14"/>
      <address type="pci" domain="0x0000" bus="0x00" slot="0x02" function="0x4"/>
    </controller>
    <controller type="pci" index="6" model="pcie-root-port">
      <model name="pcie-root-port"/>
      <target chassis="6" port="0x15"/>
      <address type="pci" domain="0x0000" bus="0x00" slot="0x02" function="0x5"/>
    </controller>
    <controller type="pci" index="7" model="pcie-root-port">
      <model name="pcie-root-port"/>
      <target chassis="7" port="0x16"/>
      <address type="pci" domain="0x0000" bus="0x00" slot="0x02" function="0x6"/>
    </controller>
    <controller type="virtio-serial" index="0">
      <address type="pci" domain="0x0000" bus="0x03" slot="0x00" function="0x0"/>
    </controller>
    <interface type="network">
      <mac address="52:54:00:67:31:2e"/>
      <source network="default"/>
      <model type="virtio"/>
      <address type="pci" domain="0x0000" bus="0x01" slot="0x00" function="0x0"/>
    </interface>
    <serial type="pty">
      <target type="isa-serial" port="0">
        <model name="isa-serial"/>
      </target>
    </serial>
    <console type="pty">
      <target type="serial" port="0"/>
    </console>
    <channel type="unix">
      <target type="virtio" name="org.qemu.guest_agent.0"/>
      <address type="virtio-serial" controller="0" bus="0" port="1"/>
    </channel>
    <channel type="spicevmc">
      <target type="virtio" name="com.redhat.spice.0"/>
      <address type="virtio-serial" controller="0" bus="0" port="2"/>
    </channel>
    <input type="tablet" bus="usb">
      <address type="usb" bus="0" port="1"/>
    </input>
    <input type="mouse" bus="ps2"/>
    <input type="keyboard" bus="ps2"/>
    <graphics type="spice" autoport="yes">
      <listen type="address"/>
      <image compression="off"/>
    </graphics>
    <sound model="ich9">
      <address type="pci" domain="0x0000" bus="0x00" slot="0x1b" function="0x0"/>
    </sound>
    <audio id="1" type="spice"/>
    <video>
      <model type="qxl" ram="65536" vram="65536" vgamem="16384" heads="1" primary="yes"/>
      <address type="pci" domain="0x0000" bus="0x00" slot="0x01" function="0x0"/>
    </video>
    <redirdev bus="usb" type="spicevmc">
      <address type="usb" bus="0" port="2"/>
    </redirdev>
    <redirdev bus="usb" type="spicevmc">
      <address type="usb" bus="0" port="3"/>
    </redirdev>
    <memballoon model="virtio">
      <address type="pci" domain="0x0000" bus="0x05" slot="0x00" function="0x0"/>
    </memballoon>
    <rng model="virtio">
      <backend model="random">/dev/urandom</backend>
      <address type="pci" domain="0x0000" bus="0x06" slot="0x00" function="0x0"/>
    </rng>
  </devices>
</domain>
```

Uočimo:

- da se koristi KVM (`<domain type="kvm">`)
- da se virtualni stroj zove `ubuntu20.04` (`<name>ubuntu20.04</name>`)
- da virtualni stroj posjeduje [univerzalno jedinstveni identifikator](https://en.wikipedia.org/wiki/Universally_unique_identifier) (engl. *universally unique identifier*, kraće UUID) (`<uuid>3f74a815-6eb0-4e1c-abae-372afd171049</uuid>`)
- da su virtualnom stroju zaista dodijeljeni 4 gigabajta RAM-a (`<memory unit="KiB">4194304</memory>`, `<currentMemory unit="KiB">4194304</currentMemory>`) i 2 virtualne procesorske jezgre (`<vcpu placement="static">2</vcpu>`)
- da je arhitektura virtualnog stroja x86-64, koristi emulirani skup čipova [Intel Q35](https://ark.intel.com/content/www/us/en/ark/products/31918/intel-82q35-graphics-and-memory-controller.html) i pokreće se s čvrstog diska:

    ``` xml
    <os>
      <type arch="x86_64" machine="pc-q35-6.1">hvm</type>
      <boot dev="hd"/>
    </os>
    ```

- da CPU virtualnog stroja podržava iste značajke kao CPU domaćina (`<cpu mode="host-model" check="partial"/>`)
- da virtualni stroj ima velik broj uređaja, specifično:
    - čvrsti disk koji je slika u formatu qcow2 na određenoj putanji u datotečnom sustavu

        ``` xml
        <disk type="file" device="disk">
            <driver name="qemu" type="qcow2"/>
            <source file="/var/lib/libvirt/images/ubuntu20.04.qcow2"/>
            <target dev="vda" bus="virtio"/>
            <address type="pci" domain="0x0000" bus="0x04" slot="0x00" function="0x0"/>
        </disk>
        ```

    - mrežno sučelje kod kojeg je navedena MAC (hardverska, fizička) adresa:

        ``` xml
        <interface type="network">
            <mac address="52:54:00:67:31:2e"/>
            <source network="default"/>
            <model type="virtio"/>
            <address type="pci" domain="0x0000" bus="0x01" slot="0x00" function="0x0"/>
        </interface>
        ```

U Virtual Machine Manageru moguće je za pojedini virtualni stroj pronaći ovaj zapis. Klikom na gumb `Open`, odnosno `Show the virtual machine console and details` otvara se konzola virtualnog stroja. Zatim klikom na gumb `Show virtual hardware details` zapis postaje dostupan pod karticom `XML`.

## virsh

!!! hint
    Za dodatne primjere naredbi proučite [stranicu libvirt na ArchWikiju](https://wiki.archlinux.org/title/libvirt).

[virsh](https://libvirt.org/manpages/virsh.html) je korisničko sučelje naredbenog retka za upravljanje virtualnim strojevima.

Uvjerimo se da je instaliran pokretanjem naredbe `virsh` s parametrom `--version`, odnosno `-v`:

``` shell
$ virsh -v
7.10.0
```

Korištenjem argumenta `nodeinfo` moguće je dobiti informacije o domaćinu na kojem je libvirt pokrenut, a korištenjem naredbe `nodememstats` o zauzeću memorije na domaćinu:

``` shell
$ virsh nodeinfo
CPU model:           x86_64
CPU(s):              32
CPU frequency:       3400 MHz
CPU socket(s):       1
Core(s) per socket:  8
Thread(s) per core:  2
NUMA cell(s):        2
Memory size:         32780200 KiB

$ virsh nodememstats
total  :             32780200 KiB
free   :              1481684 KiB
buffers:               115852 KiB
cached :             22103936 KiB
```

Baratanje pojedinim virtualnim strojevima možemo izvesti argumentima `start` i `shutdown`, koji pokreću virtualni, odnosno šalju mu signal za isključivanje:

``` shell
$ virsh start ubuntu20.04
Domain 'ubuntu20.04' started

$ shutdown ubuntu20.04
Domain 'ubuntu20.04' is being shutdown
```

Nama je virsh zanimljiv za brzo stvaranje velikog broja virtualnih strojeva sa sličnim postavkama. Iskoristit ćemo XML zapis domene od ranije, koji smo mogli dobiti i argumentom `dumpxml` na način:

``` shell
$ virsh dumpxml ubuntu20.04
<domain type='kvm' id='1'>
  <name>ubuntu20.04</name>
...
```

Taj XML dokument je zapis postojećeg virtualnog stroja koji je jedinstven pa ćemo ga kod stvaranja svakog novog virtualnog stroja morati donekle izmijeniti. XML dokument koji možemo iskoristiti za stvaranje novog virtualnog stroja treba imati:

- ima drugačije ime od svih već stvorenih virtualnih strojeva
- ima novi univerzalno jedinstveni identifikator (iskoristit ćemo naredbu ljuske `uuidgen` za generiranje)
- stvorit ćemo novu sliku diska u formatu qcow2 (bilo kopiranjem `focal-server-cloudimg-amd64.img` i širenjem na željenu veličinu, bilo stvaranjem nove prazne slike željene veličine)
- promijenit ćemo posljednjih 3 okteta u MAC adresi mrežnog sučelja `52:54:00:67:31:2e` na proizvoljnju vrijednost; podsjetimo se da su prvih tri okteta oznaka proizvođača (u našem slučaju `52:54:00`), a posljednjih tri okteta oznaka konkretnog mrežnog adaptera (u našem slučaju `67:31:2e`), npr. možemo iskoristiti `bb:bb:bb` ([prigodna tema](https://youtu.be/vy1w3j_mUfs))

Nakon uređivanja XML dokumenta virtualni stroj stvaramo naredbom:

``` shell
$ virsh create moja-domena.xml
```

!!! adomonition "Zadatak"
    Stvorite novi virtualni stroj s 1 virtualnom procesorskom jezgrom i 2 gigabajta RAM-a, a zatim mu dodajte još jedan (prazan) tvrdi disk veličine 50 gigabajta.

virsh podržava velik broj argumenata, a njihov popis i kratak opis možemo dobiti korištenjem argumenta `help`:

``` shell
$ virsh help
Grouped commands:

    Domain Management (help keyword 'domain'):
        attach-device                  attach device from an XML file
        attach-disk                    attach disk device
        attach-interface               attach network interface
        autostart                      autostart a domain
        blkdeviotune                   Set or query a block device I/O tuning parameters.
        blkiotune                      Get or set blkio parameters
        blockcommit                    Start a block commit operation.
        blockcopy                      Start a block copy operation.
        blockjob                       Manage active block operations
        blockpull                      Populate a disk from its backing image.
        blockresize                    Resize block device of domain.
        change-media                   Change media of CD or floppy drive
        console                        connect to the guest console
        ...
```

Detalje o pojedinim naredbama moguće je pronaći u man stranici `virsh(1)` (naredba `man 1 virsh`).

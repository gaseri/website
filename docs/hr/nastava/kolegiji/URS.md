---
tags:
  - (kolegij) urs
---

# Upravljanje računalnim sustavima

## Vježbe

Za rad na vježbama potrebno je u postavkama računala uključiti podršku za [hardverski potpomognutu virtualizaciju](https://en.wikipedia.org/wiki/X86_virtualization#Hardware-assisted_virtualization).

Nakon postavljanja virtualizacijskog hardvera računala, potrebno je na operacijskom sustavu:

- [GNU](https://www.debian.org/)/[Linux](https://fedoraproject.org/): instalirati [Virtual Machine Manager](https://virt-manager.org/) i [QEMU](https://www.qemu.org/)/[KVM](https://linux-kvm.org/)
- [FreeBSD](https://www.freebsd.org/): koristiti [bhyve](https://bhyve.org/)
- [macOS](https://www.apple.com/macos/) i [Windows](https://www.microsoft.com/windows/): instalirati [Oracle VirtualBox](https://www.virtualbox.org/)

Nakon instalacije virtualizacijskog softvera na operacijskom sustavu, potrebno je preuzeti ISO sliku za instalaciju distribucije kompatibilne s [Red Hat Enterprise Linuxom](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux) [9](https://www.redhat.com/en/blog/hot-presses-red-hat-enterprise-linux-9) ili [10](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux-10):

- [najpopularniji](https://linuxiac.com/rocky-linux-is-the-most-preferred-enterprise-linux-distribution/) izbor: [Rocky Linux](https://rockylinux.org/)
- druge mogućnosti: [AlmaLinux OS](https://almalinux.org/) i [CentOS Stream](https://www.centos.org/)

Sve navedene distribucije omogućuju korištenje [Red Hat Ansible Automation Platforme](https://www.redhat.com/en/technologies/management/ansible), koja je zastupljena u dijelu vježbi.

U radu na vježbama koristit će se [službena Red Hatova dokumentacija](https://docs.redhat.com/en).

### Uvod i motivacija

- [Command Line Heroes - An original podcast from Red Hat](https://www.redhat.com/en/command-line-heroes)

### Osnove upravljanja računalnim sustavom

!!! info
    Za ovaj dio vježbi koristi se [dokumentacija Red Hat Enterprise Linuxa](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/) [9](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9) ili [10](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10).

#### Instalacija sustava

=== "Enterprise Linux 9"

    - [Interactively installing RHEL from installation media](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/interactively_installing_rhel_from_installation_media/index)

=== "Enterprise Linux 10"

    - [Interactively installing RHEL from installation media](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/interactively_installing_rhel_from_installation_media/index)

#### Upravljanje sustavom

=== "Enterprise Linux 9"

    - [Interacting with the command-line assistant powered by RHEL Lightspeed](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/interacting_with_the_command-line_assistant_powered_by_rhel_lightspeed/index)
    - [Configuring basic system settings](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/configuring_basic_system_settings/index)
    - [Managing software with the DNF tool](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/managing_software_with_the_dnf_tool/index)
    - [Deploying web servers and reverse proxies](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/deploying_web_servers_and_reverse_proxies)
    - [Installing and using dynamic programming languages](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/installing_and_using_dynamic_programming_languages/index)
    - [Configuring and using database servers](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/configuring_and_using_database_servers/index)
    - [Managing, monitoring, and updating the kernel](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/managing_monitoring_and_updating_the_kernel/index)
    - [Using systemd unit files to customize and optimize your system](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/using_systemd_unit_files_to_customize_and_optimize_your_system/index)

=== "Enterprise Linux 10"

    - [Interacting with the command-line assistant powered by RHEL Lightspeed](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/interacting_with_the_command-line_assistant_powered_by_rhel_lightspeed/index)
    - [Managing software with the DNF tool](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/managing_software_with_the_dnf_tool/index)
    - [Deploying web servers and reverse proxies](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/deploying_web_servers_and_reverse_proxies)
    - [Installing and using dynamic programming languages](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/installing_and_using_dynamic_programming_languages/index)
    - [Configuring and using database servers](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/configuring_and_using_database_servers/index)
    - [Managing, monitoring, and updating the kernel](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/managing_monitoring_and_updating_the_kernel/index)
    - [Using systemd unit files to customize and optimize your system](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/using_systemd_unit_files_to_customize_and_optimize_your_system/index)

#### Razvoj aplikacija

=== "Enterprise Linux 9"

    - [Developing C and C++ applications in RHEL 9](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/developing_c_and_cpp_applications_in_rhel_9)
    - [Packaging and distributing software](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/packaging_and_distributing_software/index)

=== "Enterprise Linux 10"

    - [Developing C and C++ applications in RHEL 10](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/developing_c_and_cpp_applications_in_rhel_10)
    - [Packaging and distributing software](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/packaging_and_distributing_software/index)

#### Sigurnost sustava

=== "Enterprise Linux 9"

    - [Managing and monitoring security updates](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/managing_and_monitoring_security_updates/index)
    - [Security hardening](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/security_hardening/index)
    - [Using SELinux](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/using_selinux/index)
    - [Securing networks](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/securing_networks/index)
    - [Configuring firewalls and packet filters](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/configuring_firewalls_and_packet_filters/index)

=== "Enterprise Linux 10"

    - [Managing and monitoring security updates](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/managing_and_monitoring_security_updates/index)
    - [Security hardening](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/security_hardening/index)
    - [Using SELinux](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/using_selinux/index)
    - [Securing networks](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/securing_networks/index)
    - [Configuring firewalls and packet filters](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/configuring_firewalls_and_packet_filters/index)

#### Umrežavanje sustava

=== "Enterprise Linux 9"

    - [Configuring and managing networking](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/configuring_and_managing_networking)

=== "Enterprise Linux 10"

    - [Configuring and managing networking](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/configuring_and_managing_networking)

#### Upravljanje pohranom

=== "Enterprise Linux 9"

    - [Managing file systems](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/managing_file_systems/index)
    - [Configuring and using network file services](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/configuring_and_using_network_file_services/index)
    - [Managing storage devices](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/managing_storage_devices/index)

=== "Enterprise Linux 10"

    - [Managing file systems](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/managing_file_systems/index)
    - [Configuring and using network file services](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/configuring_and_using_network_file_services/index)
    - [Managing storage devices](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/managing_storage_devices/index)

#### Upravljanje kontejnerima i virtualnim strojevima

=== "Enterprise Linux 9"

    - [Building, running, and managing containers](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/building_running_and_managing_containers)
    - [Configuring and managing virtualization](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/configuring_and_managing_virtualization/index)

=== "Enterprise Linux 10"

    - [Building, running, and managing containers](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/building_running_and_managing_containers)
    - [Configuring and managing Linux virtual machines](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/configuring_and_managing_linux_virtual_machines/index)
    - [Configuring and managing Windows virtual machines](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/configuring_and_managing_windows_virtual_machines/index)

#### Upravljanje oblakom

=== "Enterprise Linux 9"

    - [Configuring and managing cloud-init for RHEL 9](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/configuring_and_managing_cloud-init_for_rhel_9/index)

=== "Enterprise Linux 10"

    - [Configuring and managing cloud-init for RHEL](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/configuring_and_managing_cloud-init_for_rhel/index)

#### Upravljanje identitetima

=== "Enterprise Linux 9"

    - [Planning Identity Management](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/planning_identity_management/index)
    - [Installing Identity Management](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/installing_identity_management/index)
    - [Accessing Identity Management services](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/accessing_identity_management_services/index)
    - [Configuring authentication and authorization in RHEL](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/configuring_authentication_and_authorization_in_rhel/index)
    - [Managing IdM users, groups, hosts, and access control rules](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/managing_idm_users_groups_hosts_and_access_control_rules/index)
    - [Managing certificates in IdM](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/managing_certificates_in_idm/index)
    - [Working with vaults in Identity Management](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/working_with_vaults_in_identity_management)
    - [Working with DNS in Identity Management](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/working_with_dns_in_identity_management)

=== "Enterprise Linux 10"

    - [Planning Identity Management](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/planning_identity_management/index)
    - [Installing Identity Management](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/installing_identity_management/index)
    - [Accessing Identity Management services](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/accessing_identity_management_services/index)
    - [Configuring authentication and authorization in RHEL](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/configuring_authentication_and_authorization_in_rhel/index)
    - [Managing IdM users, groups, hosts, and access control rules](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/managing_idm_users_groups_hosts_and_access_control_rules/index)
    - [Managing certificates in IdM](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/managing_certificates_in_idm/index)
    - [Working with vaults in Identity Management](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/working_with_vaults_in_identity_management)
    - [Working with DNS in Identity Management](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/working_with_dns_in_identity_management)

### Automatizacija upravljanja računalnim sustavima

!!! info
    Za ovaj dio vježbi koristi se [dokumentacija Red Hat Ansible Automation Platforme 2.6](https://docs.redhat.com/en/documentation/red_hat_ansible_automation_platform/2.6).

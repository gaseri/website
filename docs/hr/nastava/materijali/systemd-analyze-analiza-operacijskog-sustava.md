---
author: Vedran Miletić
---

# Analiza upravitelja operacijskog sustava

Korištenjem naredbe `systemd-analyze` ([službena dokumentacija](https://www.freedesktop.org/software/systemd/man/systemd-analyze.html)) možemo analizirati stanje operacijskog sustava.

## Analiza vremena potrebnog za pokretanje

Pokretanje operacijskog sustava sastoji se od:

- [pokretanja jezgre](https://wiki.archlinux.org/title/Arch_boot_process#Kernel)
- [pokretanja korisničkog](https://wiki.archlinux.org/title/Arch_boot_process#Early_userspace) [prostora](https://wiki.archlinux.org/title/Arch_boot_process#Late_userspace)

Bez argumenata `systemd-analyze` daje podatke o vremenu pokretanja računala:

``` shell
$ systemd-analyze
Startup finished in 1.544s (kernel) + 1min 45.693s (userspace) = 1min 47.238s
graphical.target reached after 1min 45.692s in userspace
```

Naredbe za detaljniju analizu i vizualizaciju vremena pokretanja su:

- `systemd-analyze blame`
- `systemd-analyze critical-chain`
- `systemd-analyze plot`
- `systemd-analyze dot | dot -Tsvg > systemd.svg` (zahtijeva instaliran paket [graphviz](https://archlinux.org/packages/?name=graphviz), [više detalja](https://wiki.archlinux.org/title/Graphviz))

## Analiza sigurnosnih ograničenja usluga

Naredbom `systemd-analyze` uz korištenje argumenta `security` i imena usluge možemo saznati više informacija o sigurnosnim ograničenjima koja se postavljaju kod njenog pokretanja:

``` shell
$ systemd-analyze security httpd.service
  NAME                                                        DESCRIPTION                                                             EXPOSURE
✗ RootDirectory=/RootImage=                                   Service runs within the host's root directory                                0.1
  SupplementaryGroups=                                        Service runs as root, option does not matter
  RemoveIPC=                                                  Service runs as root, option does not apply
✗ User=/DynamicUser=                                          Service runs as root user                                                    0.4
✗ CapabilityBoundingSet=~CAP_SYS_TIME                         Service processes may change the system clock                                0.2
✗ NoNewPrivileges=                                            Service processes may acquire new privileges                                 0.2
✓ AmbientCapabilities=                                        Service process does not receive ambient capabilities
✗ PrivateDevices=                                             Service potentially has access to hardware devices                           0.2
✗ ProtectClock=                                               Service may write to the hardware clock or system clock                      0.2
✗ CapabilityBoundingSet=~CAP_SYS_PACCT                        Service may use acct()                                                       0.1
✗ CapabilityBoundingSet=~CAP_KILL                             Service may send UNIX signals to arbitrary processes                         0.1
✗ ProtectKernelLogs=                                          Service may read from or write to the kernel log ring buffer                 0.2
✗ CapabilityBoundingSet=~CAP_WAKE_ALARM                       Service may program timers that wake up the system                           0.1
✗ CapabilityBoundingSet=~CAP_(DAC_*|FOWNER|IPC_OWNER)         Service may override UNIX file/IPC permission checks                         0.2
✗ ProtectControlGroups=                                       Service may modify the control group file system                             0.2
✗ CapabilityBoundingSet=~CAP_LINUX_IMMUTABLE                  Service may mark files immutable                                             0.1
✗ CapabilityBoundingSet=~CAP_IPC_LOCK                         Service may lock memory into RAM                                             0.1
✗ ProtectKernelModules=                                       Service may load or read kernel modules                                      0.2
✗ CapabilityBoundingSet=~CAP_SYS_MODULE                       Service may load kernel modules                                              0.2
✗ CapabilityBoundingSet=~CAP_SYS_TTY_CONFIG                   Service may issue vhangup()                                                  0.1
✗ CapabilityBoundingSet=~CAP_SYS_BOOT                         Service may issue reboot()                                                   0.1
✗ CapabilityBoundingSet=~CAP_SYS_CHROOT                       Service may issue chroot()                                                   0.1
✗ SystemCallArchitectures=                                    Service may execute system calls with all ABIs                               0.2
✗ CapabilityBoundingSet=~CAP_BLOCK_SUSPEND                    Service may establish wake locks                                             0.1
✗ MemoryDenyWriteExecute=                                     Service may create writable executable memory mappings                       0.1
✗ RestrictNamespaces=~user                                    Service may create user namespaces                                           0.3
✗ RestrictNamespaces=~pid                                     Service may create process namespaces                                        0.1
✗ RestrictNamespaces=~net                                     Service may create network namespaces                                        0.1
✗ RestrictNamespaces=~uts                                     Service may create hostname namespaces                                       0.1
✗ RestrictNamespaces=~mnt                                     Service may create file system namespaces                                    0.1
✗ CapabilityBoundingSet=~CAP_LEASE                            Service may create file leases                                               0.1
✗ CapabilityBoundingSet=~CAP_MKNOD                            Service may create device nodes                                              0.1
✗ RestrictNamespaces=~cgroup                                  Service may create cgroup namespaces                                         0.1
✗ RestrictSUIDSGID=                                           Service may create SUID/SGID files                                           0.2
✗ RestrictNamespaces=~ipc                                     Service may create IPC namespaces                                            0.1
✗ ProtectHostname=                                            Service may change system host/domainname                                    0.1
✗ CapabilityBoundingSet=~CAP_(CHOWN|FSETID|SETFCAP)           Service may change file ownership/access mode/capabilities unrestricted      0.2
✗ CapabilityBoundingSet=~CAP_SET(UID|GID|PCAP)                Service may change UID/GID identities/capabilities                           0.3
✗ LockPersonality=                                            Service may change ABI personality                                           0.1
✗ ProtectKernelTunables=                                      Service may alter kernel tunables                                            0.2
✗ RestrictAddressFamilies=~AF_PACKET                          Service may allocate packet sockets                                          0.2
✗ RestrictAddressFamilies=~AF_NETLINK                         Service may allocate netlink sockets                                         0.1
✗ RestrictAddressFamilies=~AF_UNIX                            Service may allocate local sockets                                           0.1
✗ RestrictAddressFamilies=~…                                  Service may allocate exotic sockets                                          0.3
✗ RestrictAddressFamilies=~AF_(INET|INET6)                    Service may allocate Internet sockets                                        0.3
✗ CapabilityBoundingSet=~CAP_MAC_*                            Service may adjust SMACK MAC                                                 0.1
✗ RestrictRealtime=                                           Service may acquire realtime scheduling                                      0.1
✗ CapabilityBoundingSet=~CAP_SYS_RAWIO                        Service has raw I/O access                                                   0.2
✗ CapabilityBoundingSet=~CAP_SYS_PTRACE                       Service has ptrace() debugging abilities                                     0.3
✗ CapabilityBoundingSet=~CAP_SYS_(NICE|RESOURCE)              Service has privileges to change resource use parameters                     0.1
✗ DeviceAllow=                                                Service has no device ACL                                                    0.2
✓ PrivateTmp=                                                 Service has no access to other software's temporary files
✗ CapabilityBoundingSet=~CAP_NET_ADMIN                        Service has network configuration privileges                                 0.2
✗ ProtectSystem=                                              Service has full access to the OS file hierarchy                             0.2
✗ ProtectProc=                                                Service has full access to process tree (/proc hidepid=)                     0.2
✗ ProcSubset=                                                 Service has full access to non-process /proc files (/proc subset=)           0.1
✗ ProtectHome=                                                Service has full access to home directories                                  0.2
✗ CapabilityBoundingSet=~CAP_NET_(BIND_SERVICE|BROADCAST|RAW) Service has elevated networking privileges                                   0.1
✗ CapabilityBoundingSet=~CAP_AUDIT_*                          Service has audit subsystem access                                           0.1
✗ CapabilityBoundingSet=~CAP_SYS_ADMIN                        Service has administrator privileges                                         0.3
✗ PrivateNetwork=                                             Service has access to the host's network                                     0.5
✗ PrivateUsers=                                               Service has access to other users                                            0.2
✗ CapabilityBoundingSet=~CAP_SYSLOG                           Service has access to kernel logging                                         0.1
✓ KeyringMode=                                                Service doesn't share key material with other services
✓ Delegate=                                                   Service does not maintain its own delegated control group subtree
✗ SystemCallFilter=~@clock                                    Service does not filter system calls                                         0.2
✗ SystemCallFilter=~@cpu-emulation                            Service does not filter system calls                                         0.1
✗ SystemCallFilter=~@debug                                    Service does not filter system calls                                         0.2
✗ SystemCallFilter=~@module                                   Service does not filter system calls                                         0.2
✗ SystemCallFilter=~@mount                                    Service does not filter system calls                                         0.2
✗ SystemCallFilter=~@obsolete                                 Service does not filter system calls                                         0.1
✗ SystemCallFilter=~@privileged                               Service does not filter system calls                                         0.2
✗ SystemCallFilter=~@raw-io                                   Service does not filter system calls                                         0.2
✗ SystemCallFilter=~@reboot                                   Service does not filter system calls                                         0.2
✗ SystemCallFilter=~@resources                                Service does not filter system calls                                         0.2
✗ SystemCallFilter=~@swap                                     Service does not filter system calls                                         0.2
✗ IPAddressDeny=                                              Service does not define an IP address allow list                             0.2
✓ NotifyAccess=                                               Service child processes cannot alter service state
✓ PrivateMounts=                                              Service cannot install system mounts
✗ UMask=                                                      Files created by service are world-readable by default                       0.1

→ Overall exposure level for httpd.service: 9.2 UNSAFE 😨
```

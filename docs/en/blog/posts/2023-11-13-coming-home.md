---
author: Vedran Miletić
authors:
  - vedranmiletic
date: 2023-11-13
tags:
  - freebsd
  - zfs
  - linux
  - unix
---

# Coming `/home`

---

![brown wooden chair beside white wall](https://unsplash.com/photos/eJ6HREYjLr8/download?w=1920)

Photo source: [Julian Hochgesang (@julianhochgesang) | Unsplash](https://unsplash.com/photos/brown-wooden-chair-beside-white-wall-eJ6HREYjLr8)

---

FreeBSD [14.0-RELEASE](https://www.freebsd.org/releases/14.0R/) annoucement is [immiment](https://www.freebsd.org/releases/14.0R/schedule/). Due to faster (re)boot and related improvements by [Colin Percival](https://www.tarsnap.com/about.html), this version [made headlines](https://www.theregister.com/2023/08/29/freebsd_boots_in_25ms/) in tech media even before it got released, which got me interested in trying it out on some of our machines. I installed [the first beta](https://lists.freebsd.org/archives/freebsd-current/2023-September/004614.html) on one of [our servers](2023-06-23-what-hardware-software-and-cloud-services-do-we-use.md#servers) and shortly afterward [reported](https://lists.freebsd.org/archives/freebsd-current/2023-September/004635.html) [an upgrade bug](https://bugs.freebsd.org/bugzilla/show_bug.cgi?id=273661), which got fixed during the beta cycle and was [shipped as an errata](https://www.freebsd.org/security/advisories/FreeBSD-EN-23:12.freebsd-update.asc) in [13.2-RELEASE-p4](https://www.freebsd.org/releases/13.2R/errata/) and [12.4-RELEASE-p6](https://www.freebsd.org/releases/12.4R/errata/).

I was following the subsequent pre-releases with great interest as well. The final FreeBSD 14.0-RELEASE brings [Clang](https://clang.llvm.org/)/[LLVM](https://llvm.org/) 16.0 (which we use in [scientific software development](../../software.md) and [course teaching](../../teaching/courses/CO.md)), [OpenSSL](https://www.openssl.org/) 3.0, [OpenZFS](https://openzfs.org/) 2.2, [Lua configuration](https://cgit.freebsd.org/src/commit/?id=3cb2f5f369ec) support in [the boot loader](https://man.freebsd.org/cgi/man.cgi?query=loader&sektion=8&format=html), [upgraded WireGuard](https://cgit.freebsd.org/src/commit/?id=744bfb213144) in [the kernel wg driver](https://man.freebsd.org/cgi/man.cgi?query=wg&sektion=4&format=html), and plenty of [other changes](https://www.freebsd.org/releases/14.0R/relnotes/) that are relevant to our usage. I found it well worth the time it took to go through these changes and learn what to expect from the release.

<!-- more -->

## FreeBSD 14 is moving `/usr/home` to `/home`

As [Skylar Grey](https://skylargreymusic.com/) would put it in her song [Coming Home, Pt. II](https://youtu.be/k84QxVJd0tI) (typewriter styling and slashes added by the author of the blog post):

> See you can doubt, and you can hate  
> But I know, no matter what it takes
>
> I'm coming `/home`  
> I'm coming `/home`  
> Tell the `world` I'm coming `/home`

On a less artistic and a more technical note, in [the Userland Application Changes section](https://www.freebsd.org/releases/14.0R/relnotes/#userland-programs) of [the release notes](https://www.freebsd.org/releases/14.0R/relnotes/) for FreeBSD 14.0-RELEASE, a block of text caught my eye (typewriter styling added in the quoted text):

> The [pw(8)](https://man.freebsd.org/cgi/man.cgi?query=pw&sektion=8&format=html) and [bsdinstall(8)](https://man.freebsd.org/cgi/man.cgi?query=bsdinstall&sektion=8&format=html) programs now create home directories for users in `/home` by default rather than `/usr/home`. The default symbolic link for `/home`, referencing `/usr/home`, is no longer created. [bbb2d2ce4220](https://cgit.freebsd.org/src/commit/?id=bbb2d2ce4220)

This is further explained in the commit message of [bbb2d2ce4220](https://cgit.freebsd.org/src/commit/?id=bbb2d2ce4220) (typewriter styling added):

> When adding a user, `pw` will create the path to the home directory
> if needed.  However, if creating a path with just one component,
> i.e. that appears to be in the root directory, `pw` would create the
> directory in `/usr`, and create a symlink from the root directory.
> Most commonly, this meant that the default of `/home/$user` would turn
> into `/usr/home/$user`.  This was added in a self-described kludge 26
> years ago.  It made (some) sense when root was generally a small
> partition, with most of the space in `/usr`.  However, the default is
> now one large partition.  `/home` really doesn't belong under `/usr`,
> and anyone who wants to use `/usr/home` can specify it explicitly.
> Remove the kludge to move `/home` under `/usr` and create the symlink,
> and just use the specified path.  Note that this operation was
> done only on the first invocation for a path, and this happened most
> commonly when adding a user during the install.

Interesting bit of history and a cool anecdote to remember for [operating](../../../hr/nastava/kolegiji/OS1.md) [systems](../../../hr/nastava/kolegiji/OS2.md) and [sysadmin](../../../hr/nastava/kolegiji/URS.md) courses. As far as I know, ([GNU/](https://stallman-copypasta.github.io/))Linux had `home` directory set up like this since the beginning, so this change is also helpful to new users who these days mostly come with prior experience with Unix-like operating systems obtained only on (GNU/)Linux.

## "Upgrading" `/usr/home` to `/home` on an existing installation

I started to wonder if this change was also made while performing an `upgrade` via [freebsd-update(8)](https://man.freebsd.org/cgi/man.cgi?query=freebsd-update&sektion=8&format=html), but it expectedly turned out that it was not.

``` shell
% ls -ld /home
lrwxr-xr-x  1 root wheel 11  7 tra   2023 /home -> usr/home
```

Since I made sure to keep the pre-upgrade ZFS snapshot (that's fairly easy when you normally run [zfs-destroy(8)](https://openzfs.github.io/openzfs-docs/man/master/8/zfs-destroy.8.html) on old snapshots only when disk usage becomes a problem), there was little worry that the machine would be put in a nonrecoverable state. Let's see if `/usr/home` can simply be moved to `/home`.

``` shell
% doas zfs rename zroot/usr/home zroot/home
Password:
cannot unmount '/usr/home': pool or dataset is busy
```

Of course not. That was a bit too optimistic, wasn't it? Even for ZFS.

### Preparing for the move

It is impossible to perform this operation without unmounting the filesystem first, and unmounting can't be done on a busy filesystem. A busy filesystem in this case means a busy home directory, and a busy home directory in general means user processes running, be it session processes or per-user daemons. As a consequence, all regular users have to be logged out and all of their processes should be killed before the move can happen. Logging out is easy, killing session processes too, so let's check which daemons are running for my user.

``` shell
% grep vedran /etc/rc.conf
syncthing_home="/home/vedran/.config/syncthing"
syncthing_user="vedran"
syncthing_group="vedran"
```

No wonder [Syncthing](https://syncthing.net/) is running, it is a very simple and useful file synchronization tool. Let's make sure it is stopped so it does not try to access its configuration directory via the `/home -> /usr/home` symlink while we are moving the home directory.

``` shell
% doas service syncthing stop
Password:
Stopping syncthing.
Waiting for PIDS: 34480.
```

One can now log out as a regular user and log in as `root` on the display and the keyboard attached to the system, if any. As this particular machine is headless and without an input device, I opted to log in as `root` via SSH. To do this, one has to replace the line in the OpenSSH daemon configuration file `/etc/ssh/sshd_config`. Instead of

``` aconf
#PermitRootLogin no
```

there should be

``` aconf
PermitRootLogin yes
```

Of course, after changing the configuration, one should also activate the changes using the [service(8)](https://man.freebsd.org/cgi/man.cgi?query=service&sektion=8&format=html) command:

``` shell
% doas service sshd reload
Password:
Performing sanity check on sshd configuration.
```

### Moving the home directory

It looks like we can finally log in as `root` and perform the move. Let's first get the existing symlink out of the way.

``` shell
# rm /home
```

ZFS is smart enough to unmount and mount on filesystem rename so running just [zfs-rename(8)](https://openzfs.github.io/openzfs-docs/man/master/8/zfs-rename.8.html) should do the whole job for us.

``` shell
# zfs rename zroot/usr/home zroot/home
```

That went with issues, good! Let's use [zfs-get(8)](https://openzfs.github.io/openzfs-docs/man/master/8/zfs-get.8.html) to see what we got.

``` shell
# zfs get mountpoint zroot/home
NAME        PROPERTY    VALUE        SOURCE
zroot/home  mountpoint  /zroot/home  inherited from zroot
# ls /zroot
home
```

Honestly, `/zroot` is not exactly the place where we want the `home` directory to be. Let's fix that using [zfs-set(8)](https://openzfs.github.io/openzfs-docs/man/master/8/zfs-set.8.html).

``` shell
# zfs set mountpoint=/home zroot/home
# zfs get mountpoint zroot/home
NAME        PROPERTY    VALUE       SOURCE
zroot/home  mountpoint  /home       local
# ls /zroot
```

It looks like that worked. Is the home directory finally where it should be?

``` shell
# ls -ld /home
drwxr-xr-x  3 root wheel 3 15 kol  00:25 /home
```

The home directory is indeed in its proper place, great!

!!! tip
    When my user was created during installation, [pw(8)](https://man.freebsd.org/cgi/man.cgi?query=pw&sektion=8&format=html) set its home directory to `/home/vedran`, as can be easily seen from the relevant part of the `/etc/passwd` file:

    ``` shell
    # grep vedran /etc/passwd
    vedran:*:1001:1001:Vedran Miletic:/home/vedran:/bin/tcsh
    ```

    Some software might choose to use the `/usr/home/vedran` path to access the user's home directory regardless of this setting. Of course, the configuration of such software should be updated with the correct path. However, until this configuration is updated, it could be useful to have a "reverse" symlink:

    ``` shell
    # ln -s /home /usr/home
    ```

### Cleaning up after the move

So, the move is done, but we still have to undo some of the preparatory work. Let's not forget to change that line in the OpenSSH daemon configuration file `/etc/ssh/sshd_config` back to

``` aconf
#PermitRootLogin no
```

and reload the service

``` shell
# service sshd reload
```

Of course, it also would be useful to get Syncthing running again.

``` shell
# service syncthing start
```

And that's it! That's what I like about FreeBSD (and ZFS), there is absolutely no need to reinstall the operating system just to get all the latest and greatest [directory hierarchy](https://docs.freebsd.org/en/books/handbook/basics/#dirstructure) conventions from 14.0-RELEASE applied. To the contrary, the required changes can be done fairly easily in an existing installation that was upgraded from 13.2-RELEASE or 12.4-RELEASE.

---
author: Vedran Miletić
authors:
  - vedranmiletic
date: 2025-03-16
tags:
  - scientific software
  - gromacs
  - linux
  - debian
  - red hat
  - free and open-source software
  - c++
---

# Cross-building and cross-testing GROMACS for Windows on Linux with MinGW and Wine

---

![opened brown wooden window](https://unsplash.com/photos/FQYCJSqER_0/download?w=1920)

Photo source: [Katerina (@kat_katerina) | Unsplash](https://unsplash.com/photos/opened-brown-wooden-window-FQYCJSqER_0)

---

[GROMACS](https://www.gromacs.org/) [2025.1](https://manual.gromacs.org/2025.1/index.html) ([doi:10.5281/zenodo.15006630](https://doi.org/10.5281/zenodo.15006630)) [bugfix](https://gitlab.com/gromacs/gromacs/-/milestones/163) [release](https://gitlab.com/gromacs/gromacs/-/merge_requests/5026) came out on Tuesday with [a few bugfixes](https://manual.gromacs.org/2025.1/release-notes/2025/2025.1.html). Among them is [a fix for energy correction map (CMAP) parsing code](https://manual.gromacs.org/2025.1/release-notes/2025/2025.1.html#fix-parsing-of-cmap-types-with-arbitrary-number-of-spaces-between-elements), a feature which I [introduced in 2025 release](https://manual.gromacs.org/2025.1/release-notes/2025/major/features.html#support-for-amino-acid-specific-energy-correction-maps-cmaps) in preparation for adding a converted version of [Amber](https://ambermd.org/) [ff19SB](https://github.com/csimmerling/ff19SB_201907) [force](https://ambermd.org/AmberModels.php) [field](https://ambermd.org/AmberModels_proteins.php) ([doi:10.1021/acs.jctc.9b00591](https://doi.org/10.1021/acs.jctc.9b00591)) to GROMACS.

[The fix](https://gitlab.com/gromacs/gromacs/-/merge_requests/5006) for [the issue](https://gitlab.com/gromacs/gromacs/-/issues/5312) also added several tests to avoid future regressions. I like developing free and open-source software in general and GROMACS in particular; it feels a lot like [the postdoc years in Heidelberg](2023-07-28-alumni-meeting-2023-at-hits-and-the-reminiscence-of-the-postdoc-years.md) again. However, that was not the only change I proposed that got merged for the bugfix release.

<!-- more -->

## Compiling and running a Windows version of GROMACS

First, a bit of background. Back in 2019, the late [Dr. Željko Svedružić](https://svedruziclab.github.io/principal-investigator.html) wanted to expand [the Computational design of biologically active molecules course](https://svedruziclab.github.io/teaching.html#computational-design-of-biologically-active-molecules) with the topic of running molecular dynamics simulation to analyze potential drugs. This, of course, included the usage of GROMACS, the go-to free and open-source molecular dynamics simulator.

Dr. Svedružić asked me to provide a Windows build that the students of biotechnology can run on their laptops. This way, the students would get a feeling of working with GROMACS in a familiar environment before trying to run it on [the Bura supercomputer](https://cnrm.uniri.hr/bura/). By then, he and I [collaborated successfully for years](2015-07-28-joys-and-pains-of-interdisciplinary-research.md), both on research and teaching, so I was eager to help him improve the course. While we strongly preferred when students used Linux (or macOS) due to ease of running computational chemistry software, we [supported the Windows users to the largest possible extent](../../../hr/nastava/materijali/inf-biotech-instalacija-softvera-windows-ubuntu.md) starting from [the introductory Informatics course](../../../hr/nastava/kolegiji/INF-BioTech.md).

Back then I was using [Fedora](https://fedoraproject.org/), which [packaged](https://docs.fedoraproject.org/en-US/packaging-guidelines/MinGW/) [MinGW](https://fedoraproject.org/wiki/MinGW) for cross-compiling programs to run on Windows since [Fedora 11 release (June 2009)](https://fedoraproject.org/wiki/Features/Windows_cross_compiler). I installed MinGW on my Fedora and gave cross-compiling GROMACS a shot. I didn't get very far, attributed that to my (lack of) cross-compiling skills, and opted to use [the Microsoft Visual C++ compiler](https://visualstudio.microsoft.com/vs/features/cplusplus/) on Windows. Students attending the course were quite happy either way as long as it ran fine on their laptops, which it did.

## Fixing MinGW GCC build

Since 2025.1, GROMACS can finally be [cross-built for Windows on Linux with MinGW](https://manual.gromacs.org/2025.1/release-notes/2025/2025.1.html#fixed-cross-compile-for-windows-with-mingw-gcc-on-linux).

It turns out that there was [a missing `#include <cstdint>` directive and wrong file name casing](https://gitlab.com/gromacs/gromacs/-/issues/5088) of `Windows.h` and `Winsock.h`; the latter was [present around 2019 as well](https://gitlab.com/gromacs/gromacs/-/issues/2341#note_310442710). Namely, when cross-compiling for Windows from Linux with MinGW, they [have to be lowercase](https://gitlab.com/gromacs/gromacs/-/merge_requests/5022), i.e. `windows.h` and `winsock.h`, while Windows [doesn't care either way](https://learn.microsoft.com/en-us/windows/wsl/case-sensitivity). This also explains why [the GROMACS package](https://packages.msys2.org/base/mingw-w64-gromacs) in [the MSYS2 distribution](https://www.msys2.org/), which uses [mingw-w64](https://www.mingw-w64.org/) on Windows, [required adding the missing include](https://github.com/msys2/MINGW-packages/blob/master/mingw-w64-gromacs/002-include-missing-header.patch), but didn't require the change in casing.

## Cross-building GROMACS with MinGW GCC

I have a [Debian](https://www.debian.org/devel/testing.html) [testing](https://www.debian.org/devel/testing.html) (*[trixie](https://www.debian.org/releases/trixie/)*) installation at hand this time, but I would expect Fedora to be similar. [MinGW GCC](https://tracker.debian.org/pkg/gcc-mingw-w64) C++ compiler can be installed using the command:

``` shell
apt install g++-mingw-w64
```

Of course, building GROMACS as instructed below also requires [CMake](https://tracker.debian.org/pkg/cmake) and [Ninja](https://tracker.debian.org/pkg/ninja-build). In summary, something along the lines of:

``` shell
cmake -S . -B crossbuild -D CMAKE_SYSTEM_NAME=Windows -D CMAKE_C_COMPILER=x86_64-w64-mingw32-gcc -D CMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ -D GMX_FFT_LIBRARY=fftpack -G Ninja
cmake --build crossbuild
```

configures and builds GROMACS for Windows on Linux. I would also add:

- `-D GMX_DEVELOPER_BUILD=ON` to enable building of tests and
- `-D GMX_ENABLE_CCACHE=ON` to save some time on rebuilds.

After some minutes, the build process produces a bunch of executable files for Windows, like:

``` shell
file crossbuild/bin/gmx.exe
```

``` shell-session
crossbuild/bin/gmx.exe: PE32+ executable (console) x86-64, for MS Windows, 20 sections
```

which made me wonder how far can one get without actually having to use Windows.

## Cross-running GROMACS with Wine

[Wine](https://tracker.debian.org/pkg/wine), a software package which [is not an emulator of Windows](https://www.winehq.org/), can be installed using the command:

``` shell
apt install wine
```

The [latest stable version](https://www.winehq.org/) is [10.0](https://gitlab.winehq.org/wine/wine/-/releases/wine-10.0), [released in January this year](https://www.winehq.org/news/2025012101), while Debian provides [version 9.0](https://gitlab.winehq.org/wine/wine/-/releases/wine-9.0) from the last year. This is new enough for now.

Initial configuration via [winecfg](https://gitlab.winehq.org/wine/wine/-/wikis/Commands/winecfg) utility requires GUI, but `gmx.exe` is a CLI application. Assuming that Wine, when no GUI is available, will

- use the default settings and
- run the console application in the current terminal,

it might be possible to avoid having to use the GUI. Let's see:

``` shell
cd crossbuild
wine bin/gmx.exe
```

``` shell-session
it looks like wine32 is missing, you should install it.
multiarch needs to be enabled first.  as root, please
execute "dpkg --add-architecture i386 && apt-get update &&
apt-get install wine32:i386"
wine: created the configuration directory '/home/vedranmiletic/.wine'
004c:err:winediag:nodrv_CreateWindow Application tried to create a window, but no driver could be loaded.
004c:err:winediag:nodrv_CreateWindow L"The explorer process failed to start."
004c:err:systray:initialize_systray Could not create tray window
004c:err:ole:StdMarshalImpl_MarshalInterface Failed to create ifstub, hr 0x80004002
004c:err:ole:CoMarshalInterface Failed to marshal the interface {6d5140c1-7436-11ce-8034-00aa006009fa}, hr 0x80004002
004c:err:ole:apartment_get_local_server_stream Failed: 0x80004002
0044:err:winediag:nodrv_CreateWindow Application tried to create a window, but no driver could be loaded.
0044:err:winediag:nodrv_CreateWindow L"Make sure that your X server is running and that $DISPLAY is set correctly."
0054:err:winediag:nodrv_CreateWindow Application tried to create a window, but no driver could be loaded.
0054:err:winediag:nodrv_CreateWindow L"Make sure that your X server is running and that $DISPLAY is set correctly."
0054:err:ole:apartment_createwindowifneeded CreateWindow failed with error 3
0054:err:ole:apartment_createwindowifneeded CreateWindow failed with error 0
0054:err:ole:apartment_createwindowifneeded CreateWindow failed with error 14007
0054:err:ole:StdMarshalImpl_MarshalInterface Failed to create ifstub, hr 0x800736b7
0054:err:ole:CoMarshalInterface Failed to marshal the interface {6d5140c1-7436-11ce-8034-00aa006009fa}, hr 0x800736b7
0054:err:ole:apartment_get_local_server_stream Failed: 0x800736b7
0054:err:ole:start_rpcss Failed to open RpcSs service
002c:err:winediag:nodrv_CreateWindow Application tried to create a window, but no driver could be loaded.
002c:err:winediag:nodrv_CreateWindow L"Make sure that your X server is running and that $DISPLAY is set correctly."
0090:err:winediag:nodrv_CreateWindow Application tried to create a window, but no driver could be loaded.
0090:err:winediag:nodrv_CreateWindow L"Make sure that your X server is running and that $DISPLAY is set correctly."
wine: failed to open L"C:\\windows\\syswow64\\rundll32.exe": c0000135
wine: configuration in L"/home/vedranmiletic/.wine" has been updated.
0024:err:module:import_dll Library libgcc_s_seh-1.dll (which is needed by L"Z:\\home\\vedranmiletic\\workspace\\gromacs\\crossbuild\\bin\\gmx.exe") not found
0024:err:module:import_dll Library libgomp-1.dll (which is needed by L"Z:\\home\\vedranmiletic\\workspace\\gromacs\\crossbuild\\bin\\gmx.exe") not found
0024:err:module:import_dll Library libstdc++-6.dll (which is needed by L"Z:\\home\\vedranmiletic\\workspace\\gromacs\\crossbuild\\bin\\gmx.exe") not found
0024:err:module:loader_init Importing dlls for L"Z:\\home\\vedranmiletic\\workspace\\gromacs\\crossbuild\\bin\\gmx.exe" failed, status c0000135
```

Oh, that looks better than I expected. (It is safe to ignore the warning about `wine32` and `i386`, there are only `x86-64` executables here and the `wine` package, which installs `wine64`, fully supports them.)

I remembered tweaking DLL paths from pre-Proton Linux gaming (that is, Steam on Wine) and [the list of useful registry keys](https://gitlab.winehq.org/wine/wine/-/wikis/Useful-Registry-Keys) is helful in finding exactly which keys one has to modify. While Wine's [regedit](https://gitlab.winehq.org/wine/wine/-/wikis/Commands/regedit) is a GUI application (just like winecfg), the `~/.wine/system.reg` file is a plain text file which can be easily edited by hand without any special-purpose tools. In the section:

``` ini
[System\\CurrentControlSet\\Control\\Session Manager\\Environment]
```

the `PATH` setting (which is in Windows used for both executable and library paths) can be changed from

``` ini
"PATH"=str(2):"%SystemRoot%\\system32;%SystemRoot%;%SystemRoot%\\system32\\wbem;%SystemRoot%\\system32\\WindowsPowershell\\v1.0"
```

to

``` ini
"PATH"=str(2):"%SystemRoot%\\system32;%SystemRoot%;%SystemRoot%\\system32\\wbem;%SystemRoot%\\system32\\WindowsPowershell\\v1.0;Z:\\usr\\lib\\gcc\\x86_64-w64-mingw32\\14-win32;Z:\\usr\\x86_64-w64-mingw32\\lib"
```

That adds MinGW's compiler libraries under `/usr/lib/gcc/x86_64-w64-mingw32/14-win32` and system libraries under `/usr/x86_64-w64-mingw32/lib` to Wine's library path. Now `gmx.exe` runs happily:

``` shell
wine bin/gmx.exe
```

``` shell-session
it looks like wine32 is missing, you should install it.
multiarch needs to be enabled first.  as root, please
execute "dpkg --add-architecture i386 && apt-get update &&
apt-get install wine32:i386"
0050:err:winediag:nodrv_CreateWindow Application tried to create a window, but no driver could be loaded.
0050:err:winediag:nodrv_CreateWindow L"The explorer process failed to start."
0050:err:systray:initialize_systray Could not create tray window
 :-) GROMACS - gmx, 2025.2-dev-20250312-65914d71fb-dirty (-:

Executable:   Z:\home\vedranmiletic\workspace\gromacs\crossbuild\bin\gmx.exe
Data prefix:  \home\vedranmiletic\workspace\gromacs (source tree)
Working dir:  Z:\home\vedranmiletic\workspace\gromacs\crossbuild
Command line:
 gmx

SYNOPSIS

gmx [-[no]h] [-[no]quiet] [-[no]version] [-[no]copyright] [-nice <int>]
 [-[no]backup]

OPTIONS

Other options:

 -[no]h                     (no)
 Print help and quit
 -[no]quiet                 (no)
 Do not print common startup info or quotes
 -[no]version               (no)
 Print extended version information and quit
 -[no]copyright             (no)
 Print copyright information on startup
 -nice   <int>              (19)
 Set the nicelevel (default depends on command)
 -[no]backup                (yes)
 Write backups if output files exist

Additional help is available on the following topics:
 commands    List of available commands
 selections  Selection syntax and usage
To access the help, use 'gmx help <topic>'.
For help on a command, use 'gmx help <command>'.

GROMACS reminds you: "I didn't know what MD was. I think I've managed to catch up." (Berk Hess)
```

Great, this looks promising!

## Cross-testing GROMACS with Wine and binfmt_misc

While it is now possible to also run tests one by one, it would be nicer if I could tell CMake to prefix CTest commands with Wine and wouldn't have to worry about which executables are tests and how to run them.

Of course, CMake supports [almost everything you can think of](../../tutorials/cmake-cross-supercomputer-make.md) and this particular thing can be achieved by setting the [CMAKE_CROSSCOMPILING_EMULATOR](https://cmake.org/cmake/help/latest/variable/CMAKE_CROSSCOMPILING_EMULATOR.html) variable. However, there is a nicer approach using [binfmt_misc](https://en.wikipedia.org/wiki/Binfmt_misc) (suggested by my colleague [Henri Menke](https://www.henrimenke.de/) from [MPCDF](https://www.mpcdf.mpg.de/)), which allows Linux to recognize Windows executables and run them with Wine. Debian provides [a Wine binfmt_misc package](https://packages.debian.org/testing/wine-binfmt) that configures this feature, which can be installed with:

``` shell
apt install wine-binfmt
```

The new misc binary format is disabled by default:

``` shell
update-binfmts --display
```

``` shell-session
wine (disabled):
 package = wine
 type = magic
 offset = 0
 magic = MZ
 mask =
 interpreter = /usr/bin/wine
 detector =
```

It can be enabled with:

``` shell
update-binfmts --enable wine
```

Afterwards, it is shown as enabled:

``` shell
update-binfmts --display
```

``` shell-session
wine (enabled):
 package = wine
 type = magic
 offset = 0
 magic = MZ
 mask =
 interpreter = /usr/bin/wine
 detector =
```

I would also restart the related systemd service just in case:

``` shell
systemctl restart systemd-binfmt.service
```

Running `gmx.exe` like one would run a Linux ELF file now works:

``` shell
bin/gmx.exe
```

``` shell-session
it looks like wine32 is missing, you should install it.
multiarch needs to be enabled first.  as root, please
execute "dpkg --add-architecture i386 && apt-get update &&
apt-get install wine32:i386"
 :-) GROMACS - gmx, 2025.2-dev-20250312-65914d71fb-dirty (-:

Executable:   Z:\home\vedranmiletic\workspace\gromacs\crossbuild\bin\gmx.exe
Data prefix:  \home\vedranmiletic\workspace\gromacs (source tree)
Working dir:  Z:\home\vedranmiletic\workspace\gromacs\crossbuild
Command line:
 gmx

SYNOPSIS

gmx [-[no]h] [-[no]quiet] [-[no]version] [-[no]copyright] [-nice <int>]
 [-[no]backup]

OPTIONS

Other options:

 -[no]h                     (no)
 Print help and quit
 -[no]quiet                 (no)
 Do not print common startup info or quotes
 -[no]version               (no)
 Print extended version information and quit
 -[no]copyright             (no)
 Print copyright information on startup
 -nice   <int>              (19)
 Set the nicelevel (default depends on command)
 -[no]backup                (yes)
 Write backups if output files exist

Additional help is available on the following topics:
 commands    List of available commands
 selections  Selection syntax and usage
To access the help, use 'gmx help <topic>'.
For help on a command, use 'gmx help <command>'.

GROMACS reminds you: "The absence of real intelligence doesn't prove you're using AI" (Magnus Lundborg)
```

Let's try running the tests:

``` shell
cd ..
cmake --build crossbuild --target check
```

``` shell-session
[1/80] Generating git version information
[1/3] Running all tests except physical validation
Test project /home/vedranmiletic/workspace/gromacs/crossbuild
      Start  1: TestUtilsUnitTests
 1/80 Test  #1: TestUtilsUnitTests ........................   Passed    4.52 sec
      Start  2: TestUtilsMpiUnitTests
 2/80 Test  #2: TestUtilsMpiUnitTests .....................   Passed    2.52 sec
      Start  3: UtilityUnitTests
 3/80 Test  #3: UtilityUnitTests ..........................   Passed    2.56 sec
      Start  4: UtilityMpiUnitTests
 4/80 Test  #4: UtilityMpiUnitTests .......................   Passed    2.92 sec
      Start  5: GmxlibTests
 5/80 Test  #5: GmxlibTests ...............................   Passed    2.73 sec
      Start  6: MdlibUnitTest
 6/80 Test  #6: MdlibUnitTest .............................   Passed    5.08 sec
      Start  7: AwhTest
 7/80 Test  #7: AwhTest ...................................   Passed    4.24 sec
(...)
      Start 76: MdrunFEPTests
76/80 Test #76: MdrunFEPTests .............................   Passed   14.55 sec
      Start 77: MdrunPullTests
77/80 Test #77: MdrunPullTests ............................   Passed    6.06 sec
      Start 78: MdrunRotationTests
78/80 Test #78: MdrunRotationTests ........................   Passed    6.46 sec
      Start 79: MdrunSimulatorComparison
79/80 Test #79: MdrunSimulatorComparison ..................   Passed    2.74 sec
      Start 80: MdrunVirtualSiteTests
80/80 Test #80: MdrunVirtualSiteTests .....................   Passed   16.86 sec

91% tests passed, 7 tests failed out of 80

Label Time Summary:
GTest              = 1522.73 sec*proc (80 tests)
IntegrationTest    = 444.77 sec*proc (22 tests)
MpiTest            = 1092.22 sec*proc (21 tests)
QuickGpuTest       = 218.64 sec*proc (20 tests)
SlowGpuTest        = 1098.19 sec*proc (14 tests)
SlowTest           = 926.38 sec*proc (14 tests)
UnitTest           = 151.59 sec*proc (44 tests)

Total Test time (real) = 899.11 sec

The following tests FAILED:
 40 - GmxAnaTest (Failed)                               GTest IntegrationTest
 48 - TrajectoryAnalysisUnitTests (Failed)              GTest SlowTest
 50 - ToolUnitTests (Failed)                            GTest SlowTest
 55 - MdrunModulesTests (Failed)                        GTest IntegrationTest QuickGpuTest
 57 - MdrunTestsOneRank (Failed)                        GTest IntegrationTest MpiTest SlowGpuTest
 58 - MdrunTestsTwoRanks (Failed)                       GTest IntegrationTest MpiTest SlowGpuTest
 64 - MdrunMpiTests (Failed)                            GTest IntegrationTest MpiTest QuickGpuTest
FAILED: CMakeFiles/run-ctest-nophys /home/vedranmiletic/workspace/gromacs/crossbuild/CMakeFiles/run-ctest-nophys
cd /home/vedranmiletic/workspace/gromacs/crossbuild && /usr/bin/ctest --output-on-failure -E physicalvalidationtests
ninja: build stopped: subcommand failed.
```

The test failures could also be due to Wine bugs, but I [reported it](https://gitlab.com/gromacs/gromacs/-/issues/5321) to keep track of the issue. Regardless of the cause of these failures, I would say I got quite far without actually having to use Windows.

Furthermore, while MinGW GCC is not exactly compatible with Microsoft Visual C++ (MSVC), it is nice to know that [a recent pull request fixing MSVC build](https://gitlab.com/gromacs/gromacs/-/merge_requests/5038) on the `main` branch also fixed the previously broken MinGW GCC build. Therefore, cross-building and cross-testing with MinGW GCC can be considered an easily accessible, but only somewhat accurate check of Windows compatibility.

## What if... GUI was required?

The Debian machine is headless, but it is possible to configure X11 over SSH and, from my experience in teaching [other](../../../hr/nastava/kolegiji/UMS.md) [courses](../../../hr/nastava/kolegiji/PPHS.md), it works reasonably well. (A good source from the top of my mind is [documentation for Red Hat Enterprise Linux](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/getting_started_with_the_gnome_desktop_environment/remotely-accessing-an-individual-application-x11_getting-started-with-the-gnome-desktop-environment) and clones; the intersection of familiarity with Fedora and [requirements of vendor-packaged software such as ROCm](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) gets you to appreciate Red Hat's way of doing things, but I digress.)

More interestingly, what if GUI was required, but not really? That is, what if Wine required an X11 display to exist to put windows (pun intended) on it, but the user never actually needed to click any buttons? There is a way to do it with [X Window Virtual Framebuffer (Xvfb)](https://www.x.org/releases/current/doc/man/man1/Xvfb.1.xhtml):

``` shell
apt install xvfb
```

Now, run Xvfb in the background:

``` shell
Xvfb :0 &
```

This will create server number 0 with default settings, where only screen 0 exists and has the dimensions 1280x1024x24. This is enough for the purpose.

GROMACS can be run with:

``` shell
DISPLAY=:0.0 wine bin/gmx.exe
```

``` shell-session
it looks like wine32 is missing, you should install it.
multiarch needs to be enabled first.  as root, please
execute "dpkg --add-architecture i386 && apt-get update &&
apt-get install wine32:i386"
 :-) GROMACS - gmx, 2025.2-dev-20250312-65914d71fb-dirty (-:

Executable:   Z:\home\vedranmiletic\workspace\gromacs\crossbuild\bin\gmx.exe
Data prefix:  \home\vedranmiletic\workspace\gromacs (source tree)
Working dir:  Z:\home\vedranmiletic\workspace\gromacs\crossbuild
Command line:
 gmx

SYNOPSIS

gmx [-[no]h] [-[no]quiet] [-[no]version] [-[no]copyright] [-nice <int>]
 [-[no]backup]

OPTIONS

Other options:

 -[no]h                     (no)
 Print help and quit
 -[no]quiet                 (no)
 Do not print common startup info or quotes
 -[no]version               (no)
 Print extended version information and quit
 -[no]copyright             (no)
 Print copyright information on startup
 -nice   <int>              (19)
 Set the nicelevel (default depends on command)
 -[no]backup                (yes)
 Write backups if output files exist

Additional help is available on the following topics:
 commands    List of available commands
 selections  Selection syntax and usage
To access the help, use 'gmx help <topic>'.
For help on a command, use 'gmx help <command>'.

GROMACS reminds you: "Or (horrors!) use Berendsen!" (Justin Lemkul)
```

Notice the absence of lines regarding the failure to create a window:

``` shell-session
0050:err:winediag:nodrv_CreateWindow Application tried to create a window, but no driver could be loaded.
0050:err:winediag:nodrv_CreateWindow L"The explorer process failed to start."
0050:err:systray:initialize_systray Could not create tray window
```

After enabling Wine binfmt_misc, the command looks like:

``` shell
DISPLAY=:0.0 bin/gmx.exe
```

``` shell-session
it looks like wine32 is missing, you should install it.
multiarch needs to be enabled first.  as root, please
execute "dpkg --add-architecture i386 && apt-get update &&
apt-get install wine32:i386"
 :-) GROMACS - gmx, 2025.2-dev-20250312-65914d71fb-dirty (-:

Executable:   Z:\home\vedranmiletic\workspace\gromacs\crossbuild\bin\gmx.exe
Data prefix:  \home\vedranmiletic\workspace\gromacs (source tree)
Working dir:  Z:\home\vedranmiletic\workspace\gromacs\crossbuild
Command line:
 gmx

SYNOPSIS

gmx [-[no]h] [-[no]quiet] [-[no]version] [-[no]copyright] [-nice <int>]
 [-[no]backup]

OPTIONS

Other options:

 -[no]h                     (no)
 Print help and quit
 -[no]quiet                 (no)
 Do not print common startup info or quotes
 -[no]version               (no)
 Print extended version information and quit
 -[no]copyright             (no)
 Print copyright information on startup
 -nice   <int>              (19)
 Set the nicelevel (default depends on command)
 -[no]backup                (yes)
 Write backups if output files exist

Additional help is available on the following topics:
 commands    List of available commands
 selections  Selection syntax and usage
To access the help, use 'gmx help <topic>'.
For help on a command, use 'gmx help <command>'.

GROMACS reminds you: "The soul? There's nothing but chemistry here" (Breaking Bad)
```

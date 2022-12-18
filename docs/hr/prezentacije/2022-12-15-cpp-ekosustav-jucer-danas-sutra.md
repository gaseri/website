---
marp: true
author: Vedran Miletić, Marin Jurjević
title: C++ ekosustav jučer/danas/sutra
description: Predavanje na 16. unConferenceu IT zajendnice Rijeka
keywords: c++, isocpp, qt
theme: default
class: _invert
paginate: true
abstract: |
  Programski jezik C++, uz C, za mnoge je onaj koji su naučili na ranim godinama srednjoškolskog ili fakultetskog obrazovanja i potom rijetko kasnije koristili. Prevladavajuća je percepcija da je C++ težak za korištenje i nedovoljno bogat ugrađenom funkcionalnošću. Ipak, posljednje desetljeće donijelo je velike pomake naprijed kroz revizije standarda C++11/14/17/20 i njima potaknuta poboljšanja program-prevoditelja i razvojnih okolina. Predavanje će dati pregled stanja C++ ekosustava: gdje se primjenjuje, koji su softveri u njemu razvijeni, kojim se alatima programski inženjeri koriste, što donose novi standardi i koja je budućnost jezika u doba rasta programskih jezika Rust i Go.
curriculum-vitae: |
  Vedran Miletić radi kao docent na Fakultetu informatike i digitalnih tehnologija, na kojem je voditelj Grupe za aplikacije i usluge na eksaskalarnoj istraživačkoj infrastrukturi. Bavi se znanstvenim istraživanjem u području računalne biokemije, razvija softver za superračunala te predaje kolegije iz područja IT infrastrukture i web aplikacija.

  Marin Jurjević radi kao razvojni inženjer u GlobalLogicu, gdje razvija rješenja u programskom jeziku C++. Zanimaju ga sve sfere razvoja, no iskustvo dosadašnje radno iskustvo je stjecao u sferi računalnih ugradbenih sustava, okvira Qt i C++ programskog jezika.
host: |
  Nekadašnji Odjel za informatiku, a danas Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci bavi se znanstvenim istraživanjima u područjima informatike i računarstva te održavanjem nastave informatičkih kolegija na svim razinama studija, za studente informatike i drugih studijskih programa. Nalazi se na adresi Radmile Matejčić 2 na Kampusu na Trsatu.
---

# C++ ekosustav jučer/danas/sutra

## Vedran Miletić i Marin Jurjević

### 16. unConference IT zajednice Rijeka

#### Predavaona prof. dr. sc. Pavle Dragojlovića O-028, Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, 15. prosinca 2022.

---

## Predstavljanje predavača

[Vedran Miletić](https://vedran.miletic.net/):

- Docent, Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci
- Znanstvenik, računarstvo/biokemija
    - Razvoj istraživačkog softvera za superračunala i oblake
    - Trivia: [Research Software Engineers](https://software.ac.uk/research-software-engineers) ([Software Sustainability Institute](https://software.ac.uk/))

Marin Jurjević:

- Softverski inženjer, GlobalLogic
- Auto industrija, embedded

---

## Zanimljivi podaci o stanju ekosustava u 2021. godini

- Velik broj developera ne koristi alate za statičku analizu / oblikovanje koda
- Stara verzija standarda C++98/03 je i dalje u upotrebi (12% korisnika)
- Nova verzija standarda C++20 se brzo prihvaća (18%), iako nije potpuno podržana u kompajlerima

Izvor: [C++ Ecosystem in 2021 (JetBrains)](https://blog.jetbrains.com/clion/2021/07/cpp-ecosystem-in-2021/); [sirovi podaci](https://www.jetbrains.com/lp/devecosystem-2021/cpp/)

---

## Kronologija razvoja programskih jezika: FORTRAN II

``` fortran
C AREA OF A TRIANGLE WITH A STANDARD SQUARE ROOT FUNCTION
C INPUT - TAPE READER UNIT 5, INTEGER INPUT
C OUTPUT - LINE PRINTER UNIT 6, REAL OUTPUT
C INPUT ERROR DISPLAY ERROR OUTPUT CODE 1 IN JOB CONTROL LISTING
      READ INPUT TAPE 5, 501, IA, IB, IC
  501 FORMAT (3I5)
C IA, IB, AND IC MAY NOT BE NEGATIVE OR ZERO
C FURTHERMORE, THE SUM OF TWO SIDES OF A TRIANGLE
C MUST BE GREATER THAN THE THIRD SIDE, SO WE CHECK FOR THAT, TOO
      IF (IA) 777, 777, 701
  701 IF (IB) 777, 777, 702
  702 IF (IC) 777, 777, 703
  703 IF (IA+IB-IC) 777, 777, 704
  704 IF (IA+IC-IB) 777, 777, 705
  705 IF (IB+IC-IA) 777, 777, 799
  777 STOP 1
C USING HERON'S FORMULA WE CALCULATE THE
C AREA OF THE TRIANGLE
  799 S = FLOATF (IA + IB + IC) / 2.0
      AREA = SQRTF( S * (S - FLOATF(IA)) * (S - FLOATF(IB)) *
     +     (S - FLOATF(IC)))
      WRITE OUTPUT TAPE 6, 601, IA, IB, IC, AREA
  601 FORMAT (4H A= ,I5,5H  B= ,I5,5H  C= ,I5,8H  AREA= ,F10.2,
     +        13H SQUARE UNITS)
      STOP
      END
```

---

## Kronologija razvoja programskih jezika: Fortran 90+

``` fortran
TYPE point
   REAL x, y
END TYPE point
TYPE triangle
   TYPE(point) a, b, c
END TYPE triangle

! keyword capitalization is no longer required
program helloworld
     print *, "Hello, World!"
end program helloworld
```

---

## Kronologija razvoja programskih jezika: C-asti C++ kakav se podučava na osnovama programiranja

``` c++
int main(void) {
    enum { RED, GREEN, BLUE };
    const char *nm[] = {
        [RED] = "red",
        [GREEN] = "green",
        [BLUE] = "blue",
    };
}
```

---

## Kronologija C++ standarda: C++98

Primjer koda koji izvodi zbrajanje dva broja:

``` c++
int sum(const int &a, const int &b) {
    return a + b;
}

int main() {
    int a = 5;
    int b = 7;

    int c = sum(a, b);
    std::cout << c;
    return 0;
}

```

## Kronologija C++ standarda: C++11/14

Primjer koda koji izvodi zbrajanje dva broja korištenjem lambda funkcije koja vraća i prima auto argumente i rezultat:

``` c++
// auto sum(auto a, auto b) -> int {
//     return a + b;
// }

int main(){
    // this returns std::function
    auto sum = [](auto a, auto b) -> int {
        return a+b;
    };
    auto a = 5;
    auto b = 7;

    auto c = sum(std::move(a), std::move(b));
    std::cout << c;
    return 0;
}
```

## Memory management i smart pointeri: C++98

Primjer koda s `new` i `delete`:

``` c++
// c++98
class Foo {

public:
    Foo() {
        name = "Foo";
    }

    Foo(const std::string &newName) {
        name = newName;
    }

    void printName() {
        std::cout << name;
    }
private:
    std::string name;
};

int main() {
    Foo *foo = new Foo("IT Zajednica");
    foo->printName();
    delete foo;
    return 0;
}
```

---

## Memory management i smart pointeri: C++11/14

Primjer koda s `unique_ptr`:

``` c++
// c++11/14
class Foo {
public:
    Foo() = default;
    explicit Foo(const std::string &newName) : name(newName){}

    void printName() {
        std::cout << name;
    }

private:
    std::string name;
};

int main() {
    auto foo = std::make_unique<Foo>("IT Zajednica");
    foo->printName();
    return 0;
}

```

---

## Novosti u C++17 (dio)

- Standardna biblioteka za rad s datotečnim sustavom
- `std::any`, `std::variant` -- mogućnost korištenja vise tipova unutar iste varijable
- `std::optional` -- opcionalna vrijednost koja ne mora biti prisutna
- `std::tuple` -- *n*-torka
- `std::byte` -- implementacija koncepta *bytea* -- objekt koji sadrži definiciju 8 bitovnog polja, slično kao `char` i `unsigned char`
- `std::size`/`std::empty`/`std::data` -- kao ne-članovi

Izvor: [C++17 (cppreference.com)](https://en.cppreference.com/w/cpp/17)

---

## Novosti u C++20 (dio)

- Moduli
- `std::range` -- ekstenzija iteratora koja ih čini moćnijima i manje podložnima greškama
- Koncepti -- validacija template argumenata tokom vremena kompajliranja
- Korutine -- Funkcije koje mogu prekinuti izvođenje i biti nastavljene kasnije
- `std::format` -- formatiranje rečenica (konačno!)
- Nema 100%-tnu [podršku u program-prevoditeljima](https://en.cppreference.com/w/cpp/compiler_support); [stanje kompajlera za verziju C++20](https://en.cppreference.com/w/cpp/compiler_support#C.2B.2B20_features)

Izvor: [C++20 (cppreference.com)](https://en.cppreference.com/w/cpp/20)

---

## Novosti u C++23 (dio)

- `std::exception`
- `auto(x)`
- `std::out_ptr` i `std::inout_ptr`
- Veća podrska za Atribute
- Moduli za `std`

Izvor: [C++23 (cppreference.com)](https://en.cppreference.com/w/cpp/23)

---

## Stanje kompajlera otvorenog koda

- GNU Compiler Collection (GCC)
    - brz razvoj tijekom 2000-ih omogućio standardizaciju C++0x (C++11)
    - [poboljšanja dijagnostike](https://gcc.gnu.org/wiki/ClangDiagnosticsComparison) iz verzije u verziju; [horori dijagnostike](https://codegolf.stackexchange.com/questions/1956/generate-the-longest-error-message-in-c)
- Clang/LLVM
    - započeo 2000. godine na UIUC kao istraživački projekt za tehnike optimizacije statičkih i dinamičkih programskih jezika
    - virtualni stroj, podržava [dinamičko prevođenje programa](https://en.wikipedia.org/wiki/Dynamic_compilation), slično kao JVM
    - [Clang-Tidy](https://clang.llvm.org/extra/clang-tidy/) -- statička analiza koda
    - [clangd](https://clangd.llvm.org/) -- language server

---

## Ostali kompajleri

- [MSVC](https://en.wikipedia.org/wiki/Microsoft_Visual_C%2B%2B)
    - ima svojih specifičnosti, npr. [visibility](https://gcc.gnu.org/wiki/Visibility)
- Intel Studio (icc), IBM XL, NVIDIA HPC SDK (ex. PGI)
- online sučelje za kompajlere: [Coliru](https://coliru.stacked-crooked.com/), [Godbolt](https://godbolt.org/)

---

## Razvojna okruženja

Relevantna:

- Visual Studio (VS) Code
- CLion
- Visual Studio
- Qt Creator
- Vim/Emacs

Legacy (za C++):

- Atom
- NetBeans
- Eclipse
- Code::Blocks

---

## Softveri napisani u C++-u

- [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/)
- [GROMACS](https://www.gromacs.org/), [Open](https://openfoam.org/)[FOAM](https://www.openfoam.com/)
- [KDE Plasma](https://kde.org/plasma-desktop/)
- NVIDIA CUDA, AMD ROCm
- Chromium
- Android, Windows, ROS (Robot Operating System)
- MariaDB, MongoDB, PostgreSQL
- Java Virtual Machine
- Unreal Engine, Unity
- Adobe Photoshop
- [Popis softvera napisanih u C++-u (Bjarne Stroustrup)](https://stroustrup.com/applications.html)
- [Popis knjižnica funkcija napisanih u C++-u (cppreference.com)](https://en.cppreference.com/w/cpp/links/libs)

---

## Industrijska primjena: najveće firme IT industrije

- Google
- Meta (ex. Facebook)
- Amazon
- Microsoft
- Apple
- ...

---

## Industrijska primjena: autoindustrija

- Rimac
- Tesla
- BMW
- Mercedes
- Hyundai
- Kia
- Citroen
- ...

---

## Industrijska primjena: ostale

- Aeronautika
    - NASA
    - SpaceX
- Pomorska industrija
- Financije
- Ugradbeni uređaji
- Robotika
- ...

---

## Qt

- programski okvir za razvijanje platformski nezavisnih programskih rješenja -*de facto* standard za razvijanje multiplatformskih aplikacija
- značajke:
    - odraz (eng. *reflection*) -- introspekcija, pretraživanje i pretraga struktura (objekata); još uvijek nije efikasno implementiran u C++
    - implementacija dizajna promatrača (eng. *observer pattern*) -- sustav signala (engl. *signals*) i utora (engl. *slots*)
    - vlastita implementacija/omotač (eng. *wrapper*) više manje svih `std` struktura podataka i algoritama
- 2 standardna načina razvoja Qt aplikacije:
    - Qt widgets (zastario, iako se još koristi)
    - QML (Qt Modeling Language)

---

### Razvoj tehnologije Qt

[Qt History (Qt Wiki)](https://wiki.qt.io/Qt_History)

---

### Primjer sučelja u QML-u

``` qml
import QtQuick 2.0
import "content" as Content

Rectangle {
    id: root
    width: 640; height: 320
    color: "#646464"

    ListView {
        id: clockview
        anchors.fill: parent
        orientation: ListView.Horizontal
        cacheBuffer: 2000
        snapMode: ListView.SnapOneItem
        highlightRangeMode: ListView.ApplyRange

        delegate: Content.Clock { city: cityName; shift: timeShift }
        model: ListModel {
            ListElement { cityName: "New York"; timeShift: -4 }
            ListElement { cityName: "London"; timeShift: 0 }
            ListElement { cityName: "Oslo"; timeShift: 1 }
            ListElement { cityName: "Mumbai"; timeShift: 5.5 }
            ListElement { cityName: "Tokyo"; timeShift: 9 }
            ListElement { cityName: "Brisbane"; timeShift: 10 }
            ListElement { cityName: "Los Angeles"; timeShift: -8 }
        }
    }

    Image {
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.margins: 10
        source: "content/arrow.png"
        rotation: -90
        opacity: clockview.atXBeginning ? 0 : 0.5
        Behavior on opacity { NumberAnimation { duration: 500 } }
    }

    Image {
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 10
        source: "content/arrow.png"
        rotation: 90
        opacity: clockview.atXEnd ? 0 : 0.5
        Behavior on opacity { NumberAnimation { duration: 500 } }
    }
}
```

Izvor: [Qt Quick Demo - Clocks](https://doc.qt.io/qt-6/qtdoc-demos-clocks-example.html)

---

### Primjena Qt-a u autoindustriji

- [Peugeuot 308, Citröen CX5, DS DS4 i Opel Astra](https://www.qt.io/blog/case-stellantis-one-project-4-unique-vehicle-brands)
- [Rimac Nevera](https://www.qt.io/rimac-automobili-built-with-qt)

---

### Primjena Qt-a u pomorstvu

- [Navico](https://www.qt.io/navico-built-with-qt)
- [BEP Marine](https://www.qt.io/bep-marine-built-with-qt)

---

### Qt - medicina

- [Embedded Software Development for Medical Devices](https://www.qt.io/industry/qt-in-medical/)
    - [Ekyona](https://www.qt.io/eykona-built-with-qt)
    - [Medec](https://www.qt.io/medec-built-with-qt)

---

## Infrastruktura

- upravitelj paketa: [Conan](https://conan.io/)
- sustavi za izgradnju softvera: [CMake](https://cmake.org/), [Meson](https://mesonbuild.com/), [Bazel](https://bazel.build/); [Ninja](https://ninja-build.org/), [MSBuild](https://learn.microsoft.com/en-us/visualstudio/msbuild/msbuild)

---

## Budućnost

- Rust, Go
- [Carbon](https://github.com/carbon-language/carbon-lang), [cppfront](https://github.com/hsutter/cppfront)
- [Boost](https://www.boost.org/) -> `std` [library](https://en.cppreference.com/w/cpp/standard_library)

---

## Hoće li C++ umrijeti?

C++ neće umrijeti jer:

- Studija [Ranking Programming Languages by Energy Efficiency](https://haslab.github.io/SAFER/scp21.pdf) postavlja ga na (stranica 16):
    - treće mjesto u kategoriji potrošnje energije i vremena
    - peto mjesto u kategoriji potrošnje memorije
- Uostalom, [nije ni Fortran umro u svojoj niši](https://en.wikipedia.org/wiki/List_of_quantum_chemistry_and_solid-state_physics_software)

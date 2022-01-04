---
author: Mia Doričić, Vedran Miletić
---

# rocALUTION: ROCm SPARSE Linear Algebra PACkage

U nastavku koristimo kod iz repozitorija [rocALUTION](https://github.com/ROCmSoftwarePlatform/rocALUTION) ([službena dokumentacija](https://rocalution.readthedocs.io/)).

rocALUTION je biblioteka rijetke linearne algebre koja ima istu ulogu za rijetke matrice koju rocSOLVER ima za obične matrice.

Glavni načini vršenja računanja sa rocALUTION su (prema [službenoj dokumentaciji](https://rocalution.readthedocs.io/en/master/usermanual.html#user-manual)):

- Single-node Computation (jednočvorno računanje)
- Multi-node Computation (višečvorno računanje)
- Solvers

    - Fixed-Point iteracija -- Jacobi, Gauss-Seidel, ...
    - Krylove potprostorne metode -- CR, CG, BiCGStab, GMRES, IDR, ...
    - Mixed-precision sheme korekcije defekata
    - Chebysheve iteracije
    - MultiGrid sheme -- geometrijske i algebarske

- Preconditioners

    - Dijeljenje matrice -- Jacobi, (Višeobojiv) Gauss-Seidel, ...
    - Faktorizacija -- ILU(0), ILU(p), Multi-Eliminacija ILU (rekurzivno), ILUT, ...
    - Aproksimativna inverzija -- Chevyshev matrični polinom, SPAI, FSAI, ...
    - Blok-dijagonalni preconditioner za rješavanje problema Minimax točke
    - Blok tip podpreconditionera/solvera
    - Zbrajajuća Schwarz i restriktirana zbrajajuća Schwarz metoda
    - Preconditioners tipova varijabli

- Backends

    - Host (Domaćin) -- rezervni backend za CPU
    - GPU/HIP -- ubrzani backend za AMD GPU-ove koji podržavaju HIP
    - OpenMP -- dizajnirano za višejezgrene procesore
    - MPI -- dizajnirano za višečvorne i multi-GPU konfiguracije

## Primjeri

### Primjer `cg.cpp`

Službeni primjer `clients/samples/cg.cpp` ([poveznica na kod](https://github.com/ROCmSoftwarePlatform/rocALUTION/blob/develop/clients/samples/cg.cpp)) koristi jednu od Krylovih potprostornih metoda CR (Conjugate Residual Method). To je iterativna metoda za riješavanje rijetkih simetričnih polu-pozitivnih određenih linearnih sustava Ax=b. (Za više informacija o ovoj metodi pogledajte [dio službene dokumentacije koji govori o solverima](https://rocalution.readthedocs.io/en/master/solvers.html#cr).)

Kod je oblika:

``` c++
int main(int argc, char* argv[])
{
  // Check command line parameters
  if(argc == 1)
  {
      std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
      exit(1);
  }

  // Initialize rocALUTION
  init_rocalution();

  // Check command line parameters for number of OMP threads
  if(argc > 2)
  {
      set_omp_threads_rocalution(atoi(argv[2]));
  }

  // Print rocALUTION info
  info_rocalution();

  // rocALUTION objects
  LocalVector<double> x;
  LocalVector<double> rhs;
  LocalVector<double> e;
  LocalMatrix<double> mat;

  // Read matrix from MTX file
  mat.ReadFileMTX(std::string(argv[1]));

  // Move objects to accelerator
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  e.MoveToAccelerator();

  // Allocate vectors
  x.Allocate("x", mat.GetN());
  rhs.Allocate("rhs", mat.GetM());
  e.Allocate("e", mat.GetN());

  // Linear Solver
  CG<LocalMatrix<double>, LocalVector<double>, double> ls;

  // Preconditioner
  Jacobi<LocalMatrix<double>, LocalVector<double>, double> p;

  // Initialize rhs such that A 1 = rhs
  e.Ones();
  mat.Apply(e, &rhs);

  // Initial zero guess
  x.Zeros();

  // Set solver operator
  ls.SetOperator(mat);
  // Set solver preconditioner
  ls.SetPreconditioner(p);

  // Build solver
  ls.Build();

  // Verbosity output
  ls.Verbose(1);

  // Print matrix info
  mat.Info();

  // Start time measurement
  double tick, tack;
  tick = rocalution_time();

  // Solve A x = rhs
  ls.Solve(rhs, &x);

  // Stop time measurement
  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack - tick) / 1e6 << " sec" << std::endl;

  // Clear solver
  ls.Clear();

  // Compute error L2 norm
  e.ScaleAdd(-1.0, x);
  double error = e.Norm();
  std::cout << "||e - x||_2 = " << error << std::endl;

  // Stop rocALUTION platform
  stop_rocalution();

  return 0;
}
```

Krenimo od funkcije `main()`.

Prvo se pojavljuje uvjet `if` sa postavljenim ispitivanjem je li broj argumenata `argc` jednak 1. U slučaju da jest, provodi se ispis operatorom `<<` na na standardni izlaz za greške `std::cerr` ([za više infromacija proučite std::cerr na cppreference.com](https://en.cppreference.com/w/cpp/io/cerr)).
Dakle, u slučaju bez pojavljivanja greške, vrši se output za `argv[0]`, amtricu i broj niti:

``` c++
int main(int argc, char* argv[])
{
  if(argc == 1)
  {
      std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
      exit(1);
  }
```

Poziva se funkcija `init_rocalution()`. Ova funkcija definira backend deskriptor sa informacijama o hardveru i njegovim specifikacijama. (Za više detalja pogledajte [službenu dokumentaciju](https://rocalution.readthedocs.io/en/master/basics.html#initialization-of-rocalution).)

Slijedi uvjetovanje `if` gdje u slučaju da je `argc` veći od 2, poziva se funkcija `set_omp_threads_rocalution()`, kojom se postavlja broj niti koji će rocALUTION koristiti. To će se također postići pomoću funkcije `atoi()`, obzirom da je `argv` tipa char.

Iza toga stoji poziv funkciji `info_rocalution()` koja ispisuje informacije o rocALUTION platformi.

Za više informacija o funkciji za postavljanje broja niti pogledajte [API dokumentaciju](https://rocalution.readthedocs.io/en/master/api.html#_CPPv4N10rocalution26set_omp_threads_rocalutionEi), a a više informacija o funkciji `atoi()` pogledajte [cppreference](https://en.cppreference.com/w/c/string/byte/atoi).

``` c++
init_rocalution();

if(argc > 2)
{
    set_omp_threads_rocalution(atoi(argv[2]));
}

info_rocalution();
```

Kreće inicijalizacija objekata/vektora i matrice koji će se koristiti u nastavku.
Iza inicijalizacije stoji funkcija za čitanje MTX (Matrix Market Format) datoteke, te se pomoću novog objekta za matricu otvara ta datoteka.

``` c++
LocalVector<double> x;
LocalVector<double> rhs;
LocalVector<double> e;
LocalMatrix<double> mat;

mat.ReadFileMTX(std::string(argv[1]));
```

Sve objekte koje smo u prijašnjem koraku stvorili sada pomoću funkcije `MoveToAccelerator()` mičemo u akcelerator.
Nakon toga slijedi alokacija novih imena i veličine za vektore. Funkcija `GetN()` vraća broj stupaca u matrici, a `GetM()` broj redaka u matrici.

``` c++
mat.MoveToAccelerator();
x.MoveToAccelerator();
rhs.MoveToAccelerator();
e.MoveToAccelerator();

x.Allocate("x", mat.GetN());
rhs.Allocate("rhs", mat.GetM());
e.Allocate("e", mat.GetN());
```

U CG slover učitavamo matricu na kojoj će biti vršene operacije, vektor i njegov tip i tip nove varijable. Jednako tako radimo za Jacobi preconditioner. Jacobi riješava linearni sustav u kojem dominiraju dijagonale (Za više informacija o ovoj metodi pogledajte [službenu dokumentaciju](https://rocalution.readthedocs.io/en/master/precond.html#jacobi-method).

``` c++
CG<LocalMatrix<double>, LocalVector<double>, double> ls;

Jacobi<LocalMatrix<double>, LocalVector<double>, double> p;
```

Na vektoru `e` putem funkcije `Ones()` postavljamo sve jedinice, te putem `Apply()` funkcije primjenjujemo te promjene na matricu.
Na sličan način na vektor `x` postavljaju se nule funkcijom `Zeros()`.

``` c++
e.Ones();
mat.Apply(e, &rhs);

x.Zeros();
```

Pomoću varijable prije stvorene putem CG solvera, postavljamo operator i prekondicioner. Zatim se solver izgrađuje putem funkcije `Build()`. U to ulaze alokacija podataka, struktura i numerička komputacija.
`Verbose()` pruža opširniji ispis slovera. Ako je postavljen na 0, neće ispisati ništa. Ako je postavljen na 1, ispisati će informacije o solveru, početak i kraj. Ako je postavljen na 2, ispisati će informacije o iteratoru.
Zatim, ispisuju se informacije o određenom objektu, u ovom slučaju matrici `mat`.

``` c++
ls.SetOperator(mat);
ls.SetPreconditioner(p);

ls.Build();

ls.Verbose(1);

mat.Info();
```

Kreće mjerenje vremena izvođenja operacija.
Iza toga se pokreće operacija riješavanja `Solve()` koju riješava solver. Nakon toga završava mjerenje vremena.

``` c++
double tick, tack;
tick = rocalution_time();

ls.Solve(rhs, &x);

tack = rocalution_time();
```

Slijedi ispis vremena potrebnog kroz operaciju, te se oslobađa memorija zauzeta sa strane solvera.

``` c++
std::cout << "Solver execution:" << (tack - tick) / 1e6 << " sec" << std::endl;

ls.Clear();
```

Ažurira se vektor `e` sa navedenim vrijednostima unutar funkcije `ScaleAdd()`. Također, računa se i norma vektora u obliku novog objekta `error`.

``` c++
e.ScaleAdd(-1.0, x);
double error = e.Norm();
std::cout << "||e - x||_2 = " << error << std::endl;
```

Zaustavlja se rocALUTION, i program završava.

``` c++
  stop_rocalution();

  return 0;
}
```

Kraj programa.

### Primjer `async.cpp`

S `using namespace` uvodi se `rocalution`. Kreće main(). Ponovno se definiraju `argc` i `argv`. Kreće `if` kojemu je uvjet ako je argc jednake vrijednosti kao 1, putem std::cerr (character error stream) funkcije se ispisuje vrijednost `argv[0]` i `'<matrix> [Num threads]'`. 'Izlaz' je true.

Inicijalizira se rocAlution. 'If'-om se kontrolira broj OMP threadova; ako je argc veći od 2, funkcijom `set_omp_threads_rocalution()` se postavlja broj threadova u `atoi(argv[2])`.

Onda se inicijalizira funkcija za ispis infa rocalution-a.

Kreiraju se objekti klase LocalVector. (LocalVector se zove local jer uvijek ostaje na jednom sistemu, a sistem može sadržavati više CPU-a putem UMA ili NUMA sistema memorije). Kreiraju se `x` i `y` tipa double, i `LocalMatrix<double>` objekt `mat`. (Za LocalMatrix vrijedi sve isto kao i za LocalVector).

Još se stvaraju objekti tipa double nazvani `tick, tack, tickg, tackg`.

Sada apparently čita matricu iz MTX file-a koristeći objekt koji predstavlja matricu `mat`, i pohranjuje ju u `argv[1]` u tipu `string`.

Koristeći objekte `x` i `y`, alociraju se vektori, koristeći i objekt `mat` za funkcije `GetN()` i `GetM()` (što su brojevi redova (M) i stupaca (N)).

`X`-eve postavlja na 1, `y`-one na 0. Objekt `tickg` će nositi rezultat funkcije `rocalution_time()` koja vraća trenutačno vrijeme u mikrosekundama. (S obzirom na ovaj `g`, zaključila bih da se ovo odnosi na GPU, jer se sljedeći odnosi na CPU).

Ispisuje se `info` o sva tri objekta (x, y i mat).

Ponovno se računa trenutačno vrijeme i sprema u objekt `tick`, ali ovaj put se to odnosi na CPU.

Ide `fr` petlja koja kaže za svaki i manji od 100, pokreće se funkcija za matricu `mat` imena `ApplyAdd()` koji objektu `y` pridodaje umnožak objekata `mat` i `x`.

Objekt `tack` ponovno računa vrijeme, i ispod toga se ispisuje koliko je CPU-u trebalo vremena da executa, na način da se objekti tack i tick oduzmu i podijele sa brojem 1e6. Isto tako se ispisuje `Dot product` odnosno Skalarni umnožak `x` i `y`.

Objekt `tick` opet mjeri vrijeme izvođenja.

Funkcijom `MoveToAccelerator()` vrijednosti/memorija objekata `mat`, `x` i `y` se pomiče na Accelerator Backend.

Ponovno, ispisuje se info o sva tri objekta.

`Tack` ponovno mjeri vrijeme, i ispisuje se vrijeme koje je bilo potrebno za Sync transfer prije, na isti način kao i za Execute CPU-a.

`Y`-one se postavlja na 0, i ponovno `tick` računa vrijeme, ali ovaj put za Accelerator.

Opet se ponavlja ista `for` petlja kao i prije, i `tack` računa vrijeme. Ispisuje se vrijeme koje je bilo potrebno Akceleratoru da executa na isti način kao i prijašnje računice, i ponovno se računa i Skalarni produkt `x` i `y`.

Iza toga se računa vrijeme za `tackg` koje se koristi za ispis vremena cijelog executa + transfera bez funkcije `Async` tako da se `tackg` i `tickg` oduzmu i opet podijele sa 1e6.

Za sva tri objekta `mat`, `x` i `y` se pomiče data na host-a. `Y`-oni se opet ponstavljaju na 0.

Kroz komentar kaže da se sada ovo odnosi na Async, i vidi se razlika jer se sada u isto vrijeme računaju vremena za i `tickg` i `tick`.

Sada se vrši pomicanje vrijednosti/memorije objekata `mat` i `x` na Accelerator, ali sada sa opcijom Async.

Ispisuje se info o sva tri objekta. Računa s vrijeme za `tack` i ispisuje koliko je vremena Async transfer trebao, i to na isti način kao i prije.

Računa se vrijeme `tick` za CPU. Opet ide ista `for` petlja kao i prije.

Računa se vrijeme za `tack` i ispisuje se opet koliko je CPU execute trajao i Skalarni produkt.

Sada se objekti `mat` i `x` sinkroniziraju. I `y` se pomiče na Accelerator.

Ispisuje se opet info sva tri objekta.

`Y`-oni se postavljaju na 0, i računa se vrijeme `tick` za Accelerator.

Ponovno ista `for` petlja. Računa se vrijeme za `tack`, i ispisuje Accelerator execution vrijeme, i Skalarni produkt.

Iza toga se računa ponovno vrijeme `tackg` i ispisuje vrijeme kompletnog executa + transfera (async).

Zaustavlja se rocAlution, i kraj programa.

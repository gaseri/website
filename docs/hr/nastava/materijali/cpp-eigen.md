---
author: Vedran Miletić
---

# C++ biblioteka predložaka za linearnu algebru Eigen

[Eigen](https://eigen.tuxfamily.org/) je popularna C++ biblioteka predložaka koja implementira:

- podatkovne strukture za linearnu algebru kao što su matrice i vektori te
- s njima povezane algoritme kao što su rješavanje linearnih jednažbi, računanje karakterističnih vrijednosti i drugi

Funkcionalnost:

- vektori i matrice svih veličina, od malih matrica fiksnih veličina do velikih matrica dinamički promjenjive veličine
- svi standardni numerički tipovi podataka za elemente matrice, uključujući `std::complex` ([dokumentacija na cppreference.com](https://en.cppreference.com/w/cpp/numeric/complex))
- gusti i rijetki vektori i matrice, [dekompozicija matrica](https://eigen.tuxfamily.org/dox/group__TopicLinearAlgebraDecompositions.html), [računanje linearnih najmanjih kvadrata](https://eigen.tuxfamily.org/dox/group__LeastSquares.html) i [geometrijske transformacije](https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html)
- u [nepodržanim modulima](https://eigen.tuxfamily.org/dox/unsupported/): tenzori, nelinearna optimizacija, rješavanje polinomskih jednadžbi i brojne druge

Performanse:

- omogućuje primjenu "lijenog izračuna" (engl. *lazy evaluation*) gdje god to ima smisla
- provodi eksplicitnu vektorizaciju za SSE2/SSE3/SSE4, AVX/AVX2/AVX512, FMA i srodne ekstenzije na drugim arhitekturama
- izbjegava se dinamička alokacija memorije kod matrica fiksne veličine
- odpetljavaju se petlje kad to ima smisla

Osim toga, Eigen je dizajniran da bude jednostavan za korištenje programerima naviklim na C++.

## Guste matrice i polja

[Kratki pregled načina korištenja](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html).

Moduli:

- `Core` implementira vektore, matrice i polja te linearnu algebru i osnovne funkcije za rad s poljima (`#include <Eigen/Core>`)
- `Eigenvalues` implementira računanje karakterističnih vrijednosti i vektora te pripadne dekompozicije (`#include <Eigen/Eigenvalues>`)

### Polja, matrice i vektori

- `Eigen::Matrix` je tip kojim se zapisuju matrice
- `Eigen::Vector` je specijalizacije tipa `Eigen::Matrix` koja ima jedan stupac
- `Eigen::Array` je polje (produkt dva polja računa se element po element, a ne kao produkt matrica)

Inicijalizacija:

- `Eigen::Matrix<float, 4, 3>` matrica od 4 retka i 3 stupca čiji su elementi brojevi s pomičnim zarezom jednostruke preciznosti
- `Eigen::Matrix<double, Dynamic, Dynamic>` matrica dinamički promjenjivog broja redaka i stupaca čiji su elementi brojevi s pomičnim zarezom dvostruke preciznosti
- pomoćni nazivi tipova: `Eigen::Matrix3f` 3x3 matrica s `float` vrijednostima, `Eigen::Matrix2d` 2x2 matrica s `double` vrijednostima

Aritmetički operatori (neka su `mat1`, `mat2` i `mat3` matrice, `vec1` i `vec2` bilo kakvi vektori iste vrste, `row1` retčani vektor, `col1` stupčani vektor te `s1` skalar):

- zbrajanje: `mat1 + mat2`
- množenje sklalarom: `s1 * mat1`, `mat1 * s1`, `mat1 / s1`
- množenje matrice i vektora: `mat1 * col1`, `row1 * mat1`
- množenje matrica: `mat1 * mat2`
- transponiranje: `mat2.transpose()`
- skalarni (unutarnji) produkt: `vec1.dot(vec2)`
- vektorski produkt: `vec1.cross(vec2)`
- vanjski produkt: `col1 * row1`
- norma: `vec1.norm()`

Primjer korištenja:

``` c++
#include <Eigen/Core>
#include <iostream>

int main() {
   Eigen::Matrix2d A;
   A << 2., 0., -0.5, 1.;

   std::cout << A << std::endl;
   std::cout << "trag: " << A.trace() << std::endl;
   std::cout << "norma: " << A.norma() << std::endl;
   std::cout << "minimalni koeficijent: " << A.minCoeff() << std::endl;
   std::cout << "maksimalni koeficijent: " << A.maxCoeff() << std::endl;

   return 0;
}
```

!!! admonition "Zadatak"
    - Dopunite primjer tako da inicijalizirate još jednu matricu `B` od 4 retka i 2 stupca pa izračunajte produkt te matrice i matrice `A`. Transponirajte matricu `B` pa izračunajte produkt matrice `A` i matrice dobivene transponiranjem `B`.
    - Dohvatite prvi stupac matrice `B` u vektor i izračunajte njegovu normu, a zatim dohvatite oba stupca matrice `B` u dva vektora i izračunajte njihov skalarni produkt.

Operatori na poljima (neka su `array1` i `array2` polja istog oblika, a `s1` skalar):

- aritmetički operatori element po element: `array1 + array2`, `array1 - array2`, `array1 * array2`, `array1 / array2`
- usporedbe: `array1 < array2`, `array1 => array2`, `array2 > s1` itd. (rezultat usporedbe je polje elemenata tipa `bool`)
- funkcije: `array1.abs()`, `array1.log()`, `array1.pow(array2)`, `array1.pow(s1)` itd.

!!! admonition "Zadatak"
    Inicijalizirajte dva polja i dvije matrice s elementima iste vrijednosti pa usporedite rezultate korištenja aritmetičkih operatora `+` i `*`.

## Rijetke matrice

[Kratki pregled načina korištenja](https://eigen.tuxfamily.org/dox/group__SparseQuickRefPage.html).

- `Sparse` implementira rijetke matrice i pripadnu linearnu algebru (`#include <Eigen/Sparse>`)

Inicijalizacija:

- `SparseMatrix<double> spmat(1000, 100)` rijetka matrica od 1000 stupaca i 100 redaka čiji su elementi brojevi s pomičnim zarezom dvostruke preciznosti
- `spmat.insert(i, j) = v` dodaje element vrijednosti `v` na mjestu `i, j` ako element ne postoji, u protivnom:

    - `spmat.coeffRef(i, j) = v` postavlja element na mjestu `i, j` na vrijednost `v`
    - `spmat.coeffRef(i, j) += v` uvećava element na mjestu `i, j` za vrijednost `v`
    - `spmat.coeffRef(i, j) -= v` umanjuje element na mjestu `i, j` za vrijednost `v`

Dohvaćanje svojstava:

- `spmat.nonZeros()` broj elemenata različitih od nule
- `spmat.norm()` euklidska norma matrice
- `spmat.isCompressed()` provjera je li matrica zapisana u komprimiranom obliku

Aritmetički operatori (`spmat1` i `spmat2` rijetke matrice, `s1` skalar, `dm1` gusta matrica, `dv1` gusti vektor):

- zbrajanje: `spmat1 + spmat2`, oduzimanje: `spmat1 - spmat2`
- množenje skalarom: `spmat1 * s1`, `spmat1 / s1`
- množenje rijetkih matrica: `spmat1 * spmat2`
- množenje rijetkih matrica gustim matricama i vektorima: `spmat1 * dm1`, `spmat1 * dv1`
- transponiranje: `spmat1.transpose()`

!!! admonition "Zadatak"
    - Napravite program koji inicijalizira gustu matricu s 2000 redaka i 2000 stupaca i elementima tipa `double`. Za vrijeme dok se program izvodi izmjerite zauzeće memorije, korištenjem `top`-a (stupac RES) ili `ps`-a (stupac RSS).
    - Napravite program koji inicijalizira rijetku matricu s 2000 redaka i 2000 stupaca i elementima tipa `double`. Postavite vrijednosti svih elemenata na dijagonali. Za vrijeme dok se program izvodi izmjerite zauzeće memorije, a zatim usporedite s prethodnim slučajem.

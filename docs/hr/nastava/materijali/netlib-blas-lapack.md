---
author: Vedran Miletić
---

# BLAS i LAPACK

U nastavku se bavimo linearnom algebrom na računalu. Razmotrit ćemo dvije biblioteke:

- [Basic Linear Algebra Subprograms (BLAS)](https://www.netlib.org/blas/)
- [Linear Algebra PACKage (LAPACK)](https://www.netlib.org/lapack/)

Obje biblioteke pisane su za programske jezike C i Fortran. Za C++ postoje ekvivalentne biblioteke [BLAS++](https://bitbucket.org/icl/blaspp) i [LAPACK++](https://bitbucket.org/icl/lapackpp) koje nećemo posebno proučavati.

Dodatno, projekti [PLASMA](https://icl.cs.utk.edu/plasma/) i [MAGMA](https://icl.utk.edu/magma/) implementiraju paralelnu linearnu algebru na višejzegrenim procesorima te masivno paralelnim i grafičkim procesorima.

## BLAS

BLAS je skup funkcija (rutina u terminologiji Fortrana) koje izvode osnovne operacije na vektorima i matricama. Implementiran je u [tri nivoa](https://www.netlib.org/blas/#_blas_routines):

- Nivo 1: operacije s vektorima ([službena dokumentacija](https://www.netlib.org/blas/#_level_1))
- Nivo 2: operacije s matricama i vektorima ([službena dokumentacija](https://www.netlib.org/blas/#_level_2))
- Nivo 3: operacije s matricama ([službena dokumentacija](https://www.netlib.org/blas/#_level_3))

Za svaki od nivoa podržana su četiri tipa podataka:

- realni brojevi jednostruke preciznosti (`SINGLE`, kraće `S`)
- realni brojevi dvostruke preciznosti (`DOUBLE`, kraće `D`)
- kompleksni brojevi jednostruke preciznosti (`COMPLEX`, kraće `C`)
- kompleksni brojevi dvostruke preciznosti (`DOUBLE COMPLEX`, kraće `Z`)

Objašnjenja svake od funkcija moguće je pronaći u [dijelu Reference BLAS službene dokumentacije](https://www.netlib.org/lapack/explore-html/modules.html). Za ilustraciju, spomenimo neke od njih:

- Na nivou 1:

    - `SSCAL()` vrši množenje vektora čiji su elementi realni brojevi jednostruke preciznosti skalarom (koji je istog tipa kao elementi vektora)
    - `SAXPY()` množi vektor čiji su elementi realni brojevi jednostruke preciznosti skalarom (koji je istog tipa kao elementi vektora) i pribraja mu drugi vektor čiji su elementi istog tipa (računa izraz oblika `a * x + y`)
    - `SDOT()` izvodi skalarni produkt dva vektora čiji su elementi realni brojevi jednostruke preciznosti
    - `DNRM2()` računa euklidsku normu vektora čiji su elementi realni brojevi dvostruke preciznosti

- Na nivou 2:

    - `SGEMV()` vrši množenje matrice čiji su elementi realni brojevi jednostruke preciznosti i vektora čiji su elementi istog tipa
    - `STRMV()` vrši množenje trokutaste matrice čiji su elementi realni brojevi jednostruke preciznosti i vektora čiji su elementi istog tipa
    - `DSYMV()` vrši množenje simetrične matrice čiji su elementi realni brojevi dvostruke preciznosti i vektora čiji su elementi istog tipa

- Na nivou 3:

    - `SGEMM()` vrši množenje matrica čiji su elementi realni brojevi jednostruke preciznosti
    - `DTRMM()` vrši množenje trokutastih matrica čiji su elementi realni brojevi dvostruke preciznosti

## LAPACK

LAPACK je skup funkcija (rutina u terminologiji Fortrana) za rješavanje sustava linearnih jednadžbi, računanje linearnih najmanjih kvadrata, rješavanje problema karakterističnih vrijednosti i dekompozicija na singularne vrijednosti matrice. Također omogućuje korištenje funkcija z afaktorizaciju matrice metodama LU, Choleskog, QR, Schur i generalizirani Schur. Kao i BLAS na kojem se temelji, podržava realne i kompleksne brojeve jednostruke i dvostruke preciznosti.

Funkcije u LAPACK-u se dijele u tri vrste:

- upravljačke (engl. *driver*) koje rješavaju čitav problem, npr. rješavanje sustava linearnih jednadžbi ili računanje karakterističnih vrijednosti simetrične matrice
- računske (engl. *computational*) koje rješavaju određen izdvojeni zadatak, npr. LU faktorizacija ili redukcija simetrične matrice na tridijagonalni oblik.
- pomoćne (engl. *auxiliary*) koje se dodatno dijele na:

    - funkcije koje obavljaju podzadatke algoritama na blokovima matrice
    - funkcije koje obavljaju računanja niskog nivoa slična onima koja već postoje u BLAS-u, ali koja zasad nisu prisutna u BLAS-u i mogu eventualno biti razmatrana za dodavanje u buduće verzije BLAS-a
    - proširenja BLAS-a, npr. rotacije u kompleksnoj ravnini

Preporučeno je koristiti upravljačke funkcije ako postoje odgovarajuće. U protivnom, moguće je kombinirati sve vrste funkcija.

Primjeri upravljačkih funkcija su:

- za rješavanje sustava linearnih jednadžbi

    - `SGESV()` rješava opći sustav linearnih jednadžbi čiji su koeficijenti realni brojevi jednostruke preciznosti
    - `SPOSV()` rješava simetrični pozitivno definitan sustav linearnih jednadžbi čiji su koeficijenti realni brojevi jednostruke preciznosti
    - `DGESV()` rješava opći sustav linearnih jednadžbi čiji su koeficijenti realni brojevi dvoostruke preciznosti

- za računanje linearnih najmanjih kvadrata

    - `SGELS()` računa linearne najmanje kvadrate korištenjem QR ili LQ faktorizacije (elementi matrica i vektora su realni brojevi jednostruke preciznosti)
    - `DGELSS()` računa linearne najmanje kvadrate korištenjem dekompozicije singularne vrijednosti matrice (elementi matrica i vektora su realni brojevi dvostruke preciznosti)

- za računanje karakterističnih i singularnih vrijednosti

    - `SGEES()` računa Schurovu faktorizaciju matrice čiji su elementi realni brojevi jednostruke preciznosti
    - `DGEEV()` računa karakteristične vrijednosti te lijeve i desne karakteristične vektore matrice čiji su elementi realni brojevi dvostruke preciznosti

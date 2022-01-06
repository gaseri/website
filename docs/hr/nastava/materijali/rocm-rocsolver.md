---
author: Mia Doričić, Vedran Miletić
---

# rocSOLVER: ROCm-ov linear algebra SOLVER

U nastavku koristimo kod iz repozitorija [rocSOLVER](https://github.com/ROCmSoftwarePlatform/rocSOLVER) ([službena dokumentacija](https://rocsolver.readthedocs.io/)).

rocSOLVER je implementacija podskupa funkcija iz LAPACK (Linear Algebra PACKage) biblioteke, te je i dalje u stanju usavršavanja.
Potrebno je instalirati i rocBLAS da bi funkcije bile uspješno odrađene.

Kao primjer, koristiti ćemo kod koji pokazuje kako izračunati QR faktorizaciju od matrice m*n u double-precision.

## Primjer korištenja

Razmotrimo službeni primjer `samples/example_basic.cpp` ([poveznica na kod](https://github.com/ROCmSoftwarePlatform/rocSOLVER/blob/develop/clients/samples/example_basic.cpp)). Kod je oblika:

``` c++
void get_example_matrix(std::vector<double>& hA,
                        rocblas_int& M,
                        rocblas_int& N,
                        rocblas_int& lda) {

  const double A[3][3] = { {  12, -51,   4},
                        {   6, 167, -68},
                        {  -4,  24, -41} };
  M = 3;
  N = 3;
  lda = 3;

  hA.resize(size_t(lda) * N);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      hA[i + j*lda] = A[i][j];
    }
  }
}

int main() {
  rocblas_int M;
  rocblas_int N;
  rocblas_int lda;
  std::vector<double> hA;
  get_example_matrix(hA, M, N, lda);

  printf("A = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", hA[i + j*lda]);
    }
    printf(";\n");
  }
  printf("]\n");

  rocblas_handle handle;
  rocblas_create_handle(&handle);

  size_t size_A = size_t(lda) * N;
  size_t size_piv = size_t(std::min(M, N));

  double *dA, *dIpiv;
  hipMalloc(&dA, sizeof(double)*size_A);
  hipMalloc(&dIpiv, sizeof(double)*size_piv);

  hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);

  rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);

  std::vector<double> hIpiv(size_piv);
  hipMemcpy(hA.data(), dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
  hipMemcpy(hIpiv.data(), dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);

  printf("R = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", (i <= j) ? hA[i + j*lda] : 0);
    }
    printf(";\n");
  }
  printf("]\n");

  hipFree(dA);
  hipFree(dIpiv);
  rocblas_destroy_handle(handle);
}
```

Započnimo od funkcije `main()`, koja glasi:

``` c++
int main() {
  rocblas_int M;
  rocblas_int N;
  rocblas_int lda;
  std::vector<double> hA;
  get_example_matrix(hA, M, N, lda);

  printf("A = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", hA[i + j*lda]);
    }
    printf(";\n");
  }
  printf("]\n");

  rocblas_handle handle;
  rocblas_create_handle(&handle);

  size_t size_A = size_t(lda) * N;
  size_t size_piv = size_t(std::min(M, N));

  double *dA, *dIpiv;
  hipMalloc(&dA, sizeof(double)*size_A);
  hipMalloc(&dIpiv, sizeof(double)*size_piv);

  hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);

  rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);

  std::vector<double> hIpiv(size_piv);
  hipMemcpy(hA.data(), dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
  hipMemcpy(hIpiv.data(), dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);

  printf("R = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", (i <= j) ? hA[i + j*lda] : 0);
    }
    printf(";\n");
  }
  printf("]\n");

  hipFree(dA);
  hipFree(dIpiv);
  rocblas_destroy_handle(handle);
}
```

Kod počinje inicijalizacijom varijabli koje označavaju retke, stupce, vodeću dimenziju i matricu, koju definiramo u funkciji `get_example_matrix()`.

``` c++
rocblas_int M;
rocblas_int N;
rocblas_int lda;
std::vector<double> hA;
get_example_matrix(hA, M, N, lda);
```

Slijedi ispis matrice koristeći `printf` i dvije `for` petlje, koje vrte petlju do broja redaka i broja stupaca.

``` c++
printf("A = [\n");
for (size_t i = 0; i < M; ++i) {
  printf("  ");
  for (size_t j = 0; j < N; ++j) {
    printf("% .3f ", hA[i + j*lda]);
  }
  printf(";\n");
}
printf("]\n");
```

Stvara se rocBLAS `handle`, odnosno poveznica do rocBLAS biblioteke.

``` c++
rocblas_handle handle;
rocblas_create_handle(&handle);
```

Kao sljedeći korak, postavljaju se vrijednosti cijelih brojeva `size_A` i `size_piv`:

- `size_A` će nositi rezultat umnoška veličine vodeće dimenzije i broja stupaca, što će biti broj elemenata u matrici A
- `size_piv` će nositi rezultat algoritma `std::min(M, N)` ([više informacija o funkciji std::min na cppreference.com](https://en.cppreference.com/w/cpp/algorithm/min))

``` c++
size_t size_A = size_t(lda) * N;
size_t size_piv = size_t(std::min(M, N));
```

Vrši se alokacija memorije, a zatim kopiraju podatci na GPU.

```
double *dA, *dIpiv;
hipMalloc(&dA, sizeof(double)*size_A);
hipMalloc(&dIpiv, sizeof(double)*size_piv);

hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);
```

Sljedeći korak je korištenje rocSOLVER funkcije `rocsolver_dgeqrf()` koja računa QR faktorizaciju na GPU.

```
rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);
```

Vraćamo podatke i rezultate natrag na procesor nakon što inicijaliziramo polje u koje ćemo spremiti podatke.

```
std::vector<double> hIpiv(size_piv);
hipMemcpy(hA.data(), dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
hipMemcpy(hIpiv.data(), dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);
```

Nakon svih ovih koraka rezultati su pohranjeni u `hA` i `hIpiv`. Ispis ovih rezultata izvršiti ćemo na ovaj način:

```
printf("R = [\n");
for (size_t i = 0; i < M; ++i) {
  printf("  ");
  for (size_t j = 0; j < N; ++j) {
    printf("% .3f ", (i <= j) ? hA[i + j*lda] : 0);
  }
  printf(";\n");
}
printf("]\n");
```

Kao završni korak u funkciji `main()` moramo počistiti kod, te oslobađamo memoriju koju smo prethodno alocirali, i uništavamo `handle`.

Ako pogledamo funkciju koju smo na početku koda spomenuli, ona izgleda ovako:

```
void get_example_matrix(std::vector<double>& hA,
                        rocblas_int& M,
                        rocblas_int& N,
                        rocblas_int& lda) {

  const double A[3][3] = { {  12, -51,   4},
                           {   6, 167, -68},
                           {  -4,  24, -41} };
  M = 3;
  N = 3;
  lda = 3;

  hA.resize(size_t(lda) * N);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      hA[i + j*lda] = A[i][j];
    }
  }
}
```

Kao početak, postavljamo konstantne vrijednosti matrice, te broj redaka, stupaca i vodeću dimenziju.

```
const double A[3][3] = { {  12, -51,   4},
                         {   6, 167, -68},
                         {  -4,  24, -41} };

M = 3;
N = 3;
lda = 3;
```

Po pravilu, rocSOLVER matrice moraju biti pohranjene u formatu stupac po stupac (engl. *column-major*), o kojemu možete više pročitati u [odjeljku Eigenove dokumentacije Column-major and row-major storage](https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html).

Taj format ćemo, u ovom slučaju, postići tako da nećemo koristiti klasičnu metodu za dohvaćanje vrijednosti matrice na mjestu `(i, j)`, već ćemo vodeću dimenziju pomnožiti sa brojem stupaca, i zbrojiti taj iznos sa brojem redaka. Time kopiramo 2D polje (matricu `A`) u `hA`, 1D polje, postavljeno u formatu stupac po stupac.

```
hA.resize(size_t(lda) * N);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      hA[i + j*lda] = A[i][j];
    }
  }
}
```

Kraj programa.

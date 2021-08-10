---
author: Mia Doričić, Vedran Miletić
---

# rocFFT: ROCm Fast Fourier Transforms

U nastavku koristimo kod iz repozitorija [rocFFT](https://github.com/ROCmSoftwarePlatform/rocFFT) ([službena dokumentacija](https://rocfft.readthedocs.io/)).

rocFFT je softverska biblioteka za računanje brzihi Fourierovih transformacija, te:

- pruža brzu i preciznu platformu za računanje diskretnih FFT-a
- podržava single i double precision floating point formate
- podržava 1D, 2D i 3D transformacije
- podržava računanje transformacija u hrpama
- podržava realne i kompleksne FFT
- podržava dužine koje sadrže bilo koju kombinaciju umnožaka 2, 3 i 5

## Tipovi podataka

Postoji par struktura podataka koje su implementirane u ovu biblioteku. Dvije od najvažnijih su:

### Plan

U plan računanja ulaze svi parametri potrebni za izvedbu transformacije. Parametri su:

- [out] `plan` -- plan handle
- [in] `placement` -- pohrana rezultata
- [in] `transform_type` -- tip transformacije
- [in] `precision` -- koji tip (single ili double) precision-a će koristiti
- [in] `dimensions` -- dimenzije
- [in] `lengths` -- dimenzije veličine polja duljina transformacije
- [in] `number_of_transforms` -- broj transformacija
- [in] `description` -- opis plana

### Izvođenje

Nakon što je plan napravljen, izvođenje može biti korišteno da pokrene plan da računa transformaciju na zadanim podatcima. Njegovi parametri su:

- [in] `plan` -- plan handle
- [in] `in_buffer` -- polje (veličine 1 kod isprepletenih podataka, veličine 2 kod planarnih podataka) input buffer-a
- [inout] `out_buffer` -- polje (veličine 1 kod isprepletenih podataka, veličine 2 kod planarnih podataka) output buffer-a, može biti i `nullptr` kod `inplace` postavljanja rezultata
- [in] `info` -- pokretanje info handle-a stvorenog `rocfft_execution_info_create(...)` funkcijom

### Primjer

Službeni primjer `clients/samples/rocfft/rocfft_example_set_stream.cpp` ([poveznica na kod](https://github.com/ROCmSoftwarePlatform/rocFFT/blob/develop/clients/samples/rocfft/rocfft_example_set_stream.cpp)) pokazuje primjer kako provesti dvije `inplace` transformacije sa dva toka.

Program izgleda ovako:

```
#define CHECK_HIP_ERR(err)                                    \
 if(err != hipSuccess)                                     \
 {                                                         \
     std::cerr << "hip error code : " << err << std::endl; \
     exit(-1);                                             \
 }

#define CHECK_ROCFFT_ERR(err)                                    \
 if(err != rocfft_status_success)                             \
 {                                                            \
     std::cerr << "rocFFT error code : " << err << std::endl; \
     exit(-1);                                                \
 }

struct fft_fixture_t
{
 double2*              cpu_buf;
 double2*              gpu_buf;
 hipStream_t           stream;
 rocfft_execution_info info;
 rocfft_plan           plan;
};

int main(int argc, char* argv[])
{
 std::cout << "rocfft example of 2 inplace transforms with 2 streams.\n" << std::endl;

 rocfft_status rc = rocfft_status_success;

 size_t length      = 8;
 size_t total_bytes = length * sizeof(double2);

 fft_fixture_t ffts[2];

 /// preparation
 for(auto& it : ffts)
 {
     // create cpu buffer
     it.cpu_buf = new double2[length];

     // init cpu buffer...

     // create gpu buffer
     CHECK_HIP_ERR(hipMalloc(&(it.gpu_buf), total_bytes));

     // copy host to device
     CHECK_HIP_ERR(hipMemcpy(it.gpu_buf, it.cpu_buf, total_bytes, hipMemcpyHostToDevice));

     // create stream
     CHECK_HIP_ERR(hipStreamCreate(&(it.stream)));

     // create execution info
     CHECK_ROCFFT_ERR(rocfft_execution_info_create(&(it.info)));

     // set stream
     // NOTE: The stream must be of type hipStream_t.
     // It is an error to pass the address of a hipStream_t object.
     CHECK_ROCFFT_ERR(rocfft_execution_info_set_stream(it.info, it.stream));

     // create plan
     CHECK_ROCFFT_ERR(rocfft_plan_create(&it.plan,
                                         rocfft_placement_inplace,
                                         rocfft_transform_type_complex_forward,
                                         rocfft_precision_double,
                                         1,
                                         &length,
                                         1,
                                         nullptr));
     size_t work_buf_size = 0;
     CHECK_ROCFFT_ERR(rocfft_plan_get_work_buffer_size(it.plan, &work_buf_size));
     assert(work_buf_size == 0); // simple 1D inplace fft doesn't need extra working buffer
 }

 /// execution
 for(auto& it : ffts)
 {
     CHECK_ROCFFT_ERR(
         rocfft_execute(it.plan, (void**)&(it.gpu_buf), (void**)&(it.gpu_buf), nullptr));
 }

 /// wait and copy back
 for(auto& it : ffts)
 {
     CHECK_HIP_ERR(hipStreamSynchronize(it.stream));
     CHECK_HIP_ERR(hipMemcpy(it.cpu_buf, it.gpu_buf, total_bytes, hipMemcpyDeviceToHost));
 }

 /// clean up
 for(auto& it : ffts)
 {
     CHECK_ROCFFT_ERR(rocfft_plan_destroy(it.plan));
     CHECK_ROCFFT_ERR(rocfft_execution_info_destroy(it.info));
     CHECK_HIP_ERR(hipStreamDestroy(it.stream));
     CHECK_HIP_ERR(hipFree(it.gpu_buf));
     delete[] it.cpu_buf;
 }

 return 0;
}
```

Krenimo od funkcije `main()`. Prvo stvaramo novi objekt tipa `rocfft_status` kojim program može javiti postoji li greška, i gdje se nalazi. U ovom slučaju postavljen je da javlja uspjeh.

Varijabla `length` postavljena je na 8, te se definira način računanja za konačan broj bajtova `total_bytes`. Nakon toga, inicijalizira se i polje tipa strukture `fft_fixture_t`.

```
std::cout << "rocfft example of 2 inplace transforms with 2 streams.\n" << std::endl;

rocfft_status rc = rocfft_status_success;

size_t length      = 8;
size_t total_bytes = length * sizeof(double2);

fft_fixture_t ffts[2];
```

Slijedi priprema; kreće petlja `for` u kojoj, kao raspon, stoji izraz `(auto& it : ffts)`, pri čemu `auto& it` služi da se ne stvaraju kopije elemenata unutar `ffts`, a operator `:` govori da će petlja vrtjeti svaki element dok ne dođe do kraja `ffts`.

Stvara se CPU buffer, a iza toga GPU buffer putem nama već poznatih funkcija, `CHECK_HIP_ERR(hipMalloc(...))`.

```
for(auto& it : ffts)
{

    it.cpu_buf = new double2[length];

    CHECK_HIP_ERR(hipMalloc(&(it.gpu_buf), total_bytes));
```

Prebacujemo/kopiramo podatke sa domaćina na uređaj, iza čega stvaramo tok.

```
CHECK_HIP_ERR(hipMemcpy(it.gpu_buf, it.cpu_buf, total_bytes, hipMemcpyHostToDevice));

CHECK_HIP_ERR(hipStreamCreate(&(it.stream)));
```

Također, stvara se `execution info` (za više informacija o ovom pojmu proučite [službenu dokumentaciju](https://rocfft.readthedocs.io/en/rocm-4.3.0/api.html#execution-info)).

Postavlja se tok, koji mora biti tipa hipStream_t (u suprotnom bi došlo do greške).

```
CHECK_ROCFFT_ERR(rocfft_execution_info_create(&(it.info)));

CHECK_ROCFFT_ERR(rocfft_execution_info_set_stream(it.info, it.stream));
```

Napslijetku, stvaramo plan računanja; u planu stoje parametri: objekt za plan, placement varijabla, tip transformacije, tip precision-a koji će se koristiti, dimenzije, duljina, broj transformacija i opis plana.

```
CHECK_ROCFFT_ERR(rocfft_plan_create(&it.plan,
                                    rocfft_placement_inplace,
                                    rocfft_transform_type_complex_forward,
                                    rocfft_precision_double,
                                    1,
                                    &length,
                                    1,
                                    nullptr));
```

Veličina buffera koji radi se postavlja na 0, te se pokreće funkcija kojom se doseže njegova veličina.

```
    size_t work_buf_size = 0;
    CHECK_ROCFFT_ERR(rocfft_plan_get_work_buffer_size(it.plan, &work_buf_size));
    assert(work_buf_size == 0);
}
```

Nakon postavljanja plana, kreće `execution` koji koristi iste parametre za petlju `for`, te provodi funkciju `rocfft_execute()` (za više infromacija o ovoj funkciji pogledajte [službenu dokumentaciju ROCm biblioteka](https://rocmdocs.amd.com/en/latest/ROCm_Libraries/ROCm_Libraries.html#execution)):

```
for(auto& it : ffts)
{
    CHECK_ROCFFT_ERR(
        rocfft_execute(it.plan, (void**)&(it.gpu_buf), (void**)&(it.gpu_buf), nullptr));
}
```

Program čeka da se prijašnja funkcija izvrši, te zatim kopira podatke natrag na domaćina.

```
for(auto& it : ffts)
{
    CHECK_HIP_ERR(hipStreamSynchronize(it.stream));
    CHECK_HIP_ERR(hipMemcpy(it.cpu_buf, it.gpu_buf, total_bytes, hipMemcpyDeviceToHost));
}
```

Za kraj programa, slijedi pročišćivanje i oslobađanje memorije:

```
 for(auto& it : ffts)
 {
     CHECK_ROCFFT_ERR(rocfft_plan_destroy(it.plan));
     CHECK_ROCFFT_ERR(rocfft_execution_info_destroy(it.info));
     CHECK_HIP_ERR(hipStreamDestroy(it.stream));
     CHECK_HIP_ERR(hipFree(it.gpu_buf));
     delete[] it.cpu_buf;
 }

 return 0;
}
```

Ako se vratimo na početak programa, definirani su `CHECK_HIP_ERR` i `CHECK_ROCFFT_ERR` sa vlastitim ispisima u slučaju da dođe do greške:

```
#define CHECK_HIP_ERR(err)                                    \
 if(err != hipSuccess)                                     \
 {                                                         \
     std::cerr << "hip error code : " << err << std::endl; \
     exit(-1);                                             \
 }

#define CHECK_ROCFFT_ERR(err)                                    \
 if(err != rocfft_status_success)                             \
 {                                                            \
     std::cerr << "rocFFT error code : " << err << std::endl; \
     exit(-1);                                                \
 }
```

Isto tako, definirana je struktura `fft_fixture_t` u kojoj su stvoreni elementi potrebni za transformacije.

```
struct fft_fixture_t
{
 double2*              cpu_buf;
 double2*              gpu_buf;
 hipStream_t           stream;
 rocfft_execution_info info;
 rocfft_plan           plan;
};
```

Kraj programa.

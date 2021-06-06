# Vježbe 3: Relacije među modelima. Upiti

###### tags: `DWA2 Vjezbe` `Django` `Django Modeli` `Pyhton`


## Modeli: Relacije među modelima

[Relacije: Django dokumnetacija](https://docs.djangoproject.com/en/3.1/topics/db/examples/)
- [Many to many](https://docs.djangoproject.com/en/3.1/topics/db/examples/many_to_many/)
- [One to one](https://docs.djangoproject.com/en/3.1/topics/db/examples/one_to_one/)
- [Many to one](https://docs.djangoproject.com/en/3.1/topics/db/examples/many_to_one/)



### Many to many

Na prethodnim vježbama stvorili smo model koji sadržava klasu Predmet, dodajmo sada klasu Student koja će biti povezana s Predmetom relacijom ManyToMany.


:::success
**Zadatak**

Model iz Vježbi2 `main/models.py` nadopunite tako da stvorite novu klasu `Student`. Klasa student sadrži stupce naziva `student_ime`, `student_prezime`, `student_broj_xice` i `student_prvi_upis_predmeta`.
Tipovi podataka neka budu `CharField()` za `student_ime` sa `max_length` na `25`, `student_prezime` sa `max_length` na `50`. Vrijednost `student_broj_xice` postavite na `CharField()` za `student_ime` sa `max_length` na `10`. 

Dodajte vrijednost `student_predmeti` koja će biti povezan s klasom `Predmet`, tip veze neka bude `ManyToMany`. 

Unutar klase `Predmet` izmijenite vrijednost `predmet_vrijeme_objave` tako da joj postavite zadanu vrijednost na `timezone.now`.

Nakon kreirane klase pokrenite naredbe `makemigrations` i `migrate`.

:::spoiler Rješenje
```python
from django.db import models
from django.utils import timezone

class Predmet(models.Model):
    predmet_naslov = models.CharField(max_length=100)
    predmet_sadrzaj = models.TextField()
    predmet_vrijeme_objave = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.predmet_naslov

class Student(models.Model):
    student_ime = models.CharField(max_length=25)
    student_prezime = models.CharField(max_length=50)
    student_broj_xice = models.CharField(max_length=10)
    student_predmeti = models.ManyToManyField(Predmet)
```
:::


:::success
**Zadatak**

definirajte funkciju `__str__()` unutar klase Student koja vraća `student_broj_xice`. Zatim dodajte klasu `Student` unutar `main/admin.py` tako da ona postane vidljiva u admin panelu. 

:::spoiler Rješenje
```python
#Unutar klase
def __str__(self):
    return self.student_broj_xice
    
#ovaj kod ide u main/admin.py  
from django.contrib import admin
from .models import *

model_list = [Predmet, Student]
admin.site.register(model_list)
```
:::


Nakon što ste nadopunili `main/models.py` primjenite pripremu i nakon toga migraciju s naredbama `makemigrations` i `migrate`. 


:::info
Provjerite radi li vam sve tako da posjetite `http://127.0.0.1:8000/admin/`.
:::

### Many to one

:::success
**Zadatak**
Definirajte klasu `Profesor` koja sadrži vrijednosti, `prof_ime` i `prof_prezime` koji su `CharField` duljine 30, zatim definirajte `prof_email` koji je tipa `EmailField`.

Unutar klase zadajte `__str__` funkciju koja vraća email adresu od profesora.

Nakon kreirane klase pokrenite naredbe `makemigrations` i `migrate`.

:::spoiler Rješenje
```python
class Profesor(models.Model):
    prof_ime = models.CharField(max_length=30)
    prof_prezime = models.CharField(max_length=30)
    prof_email = models.EmailField()

    def __str__(self):
        return self.prof_email   
```
:::


:::success
**Zadatak**
Izmijenite klasu `Predmet` tako da joj dodate nositelja, vrijednost `nositelj` neka bude tip veze One to many. Za definiranje veze koristite `ForeignKey`.

Nakon kreirane klase pokrenite naredbe `makemigrations` i `migrate`.

:::spoiler Rješenje
```python

class Predmet(models.Model):
    predmet_naslov = models.CharField(max_length=100)
    predmet_sadrzaj = models.TextField()
    predmet_vrijeme_objave = models.DateTimeField(default=timezone.now)
    predmet_nositelj = models.ForeignKey(Profesor, on_delete=models.CASCADE)

    def __str__(self):
        return self.predmet_naslov 
```
:::


### One to one 
Student i Profesor povezat ćemo u klasi Zavrsni_rad, na kojem radi student, dok mu je profesor mentor. 
Svaki završni rad ima Studenta koji ga piše i profesora koji mu je mentor. Ovo ćemo ostvariti *One to one* vezama.

:::success
**Zadatak**

Definirajte klasu `Završni_rad`. 
`Zavrsni_rad` neka ima zadanog nositelja `mentor` koji je povezan sa `Profesor` pomoću `OneToOne` veze, dodatni parametar koje zadajete u definiciji veze je `primary_key=True`. 
Klasu `Završni_rad` zatim povežite sa `Student`, tip veze neka bude `One-to-one`, dodatni parametri koje zadajete su `on_delete=models.CASCADE` i `primary_key=True`. 

Zatim dodajte vrijednosti `rad_naslov` i `rad_zadatak` koji su CharField duljine 25 i 50 i bool vrijednost `rad_prvi_upis` koja po defaultu ima vrijednost `True`. 

Nakon kreirane klase pokrenite naredbe `makemigrations` i `migrate`.

:::spoiler Rješenje
```python
class Zavrsni_rad(models.Model):
    mentor = models.OneToOneField(
        Profesor,
        on_delete=models.CASCADE,
    )


    student = models.OneToOneField(
        Student,
        on_delete=models.CASCADE,
        primary_key=True,
    )
    
    rad_naslov = models.CharField(max_length=50)
    rad_zadatak = models.CharField(max_length=50)
    rad_prvi_upis = models.BooleanField(default=True)
```
:::


:::success
**Zadatak**
Definirajte funkciju `__str__()` unutar klase `Završni_rad` koja vraća `student_broj_xice`. Zatim dodajte klasu `Završni rad` unutar `main/admin.py` tako da ona postane vidljiva u admin panelu. 
:::spoiler Rješenje
```python
#Unutar klase Zavrsni_rad ovo dopunjavamo
def __str__(self):
        return "Zavrsni rad od: %s " % self.student.student_broj_xice
                
#u admin.py samo dopunite model_list
model_list = [Predmet, Student, Zavrsni_rad]
```
:::

:::info
**manage.py** 
- Nakon svake novo stvorene klase u modelu pokrenite naredbe `makemigrations` i `migrate`.
- `./manage.py flush` koristite za očistiti bazu padatak od prethodno unešenih vrijednosti.
- ./manage.py dbshell, ALTER TABLE main_predmet ADD COLUMN "predmet_nositelj_id" integer;
:::


### Upiti

Naredba za pokretanje Python shell-a:
```
./manage.py shell 
```

Za definiranje instance klase:
```
>>> profesor=Profesor()
>>> predmet=Predmet()

```

Povezivanje instanci Profesor i Predmet pomoću vanjskog ključa, odnosno dodavanje nositelja na predmet:

```
>>> predmet.predmet_nositelj=profesor
```

Instanci klase student možemo dodati *n* predmeta:

```
>>> student.student_predmeti.add(predmet1, predmet2)
```

Kreiranje instance Zavrsni_rad:
```
>>> zr = Zavrsni_rad()
```

Dodavanje veze One to one, na instancu klase Završni rad:
```
>>> zr.mentor=profesor
>>> zr.student=student
```

Upittza koji vraća sve instance tražene klase:
```
profesors = Profesor.objects.all()
```

Pretraga po zadanoj vrijednosti:
```
email = Profesor.objects.get(prof_email="prof_mail@uniri.hr")
```

Pretraga svih instanci koji imaju traženo ime:
```
prof_peros = Profesor.objects.filter(prof_ime__contains="Pero")
``````

Uzimanje prvih 5 zapisa:
```
Profesor.objects.all()[:5]
```

Sortiranje i dohvaćanje prvog u listi:
```
Profesor.objects.order_by('prof_ime')[0]
```


## Cjelovit kod današnjih vježbi 

```python
from django.db import models
from django.utils import timezone

# Create your models here.

class Profesor(models.Model):
    prof_ime = models.CharField(max_length=30)
    prof_prezime = models.CharField(max_length=30)
    prof_email = models.EmailField()

    def __str__(self):
        return self.prof_email

class Predmet(models.Model):
    predmet_naslov = models.CharField(max_length=100)
    predmet_sadrzaj = models.TextField()
    predmet_vrijeme_objave = models.DateTimeField(default=timezone.now)
    predmet_nositelj = models.ForeignKey(Profesor, default=1, on_delete=models.CASCADE)

    def __str__(self):
        return self.predmet_naslov


class Student(models.Model):
    student_ime = models.CharField(max_length=25)
    student_prezime = models.CharField(max_length=50)
    student_broj_xice = models.CharField(max_length=10)
    student_predmeti = models.ManyToManyField(Predmet)

    def __str__(self):
        return self.student_broj_xice


class Zavrsni_rad(models.Model):
    mentor = models.OneToOneField(
        Profesor,
        on_delete=models.CASCADE,
    )
    student = models.OneToOneField(
        Student,
        on_delete=models.CASCADE,
        primary_key=True
    )
    rad_naslov = models.CharField(max_length=50)
    rad_zadatak = models.CharField(max_length=150)
    rad_prvi_upis = models.BooleanField(default=True)

    def __str__(self):
        return "Zavrsni rad od: %s " % self.student.student_broj_xice
```

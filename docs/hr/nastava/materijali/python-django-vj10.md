# Vježba 10: Autentifikacija. Autorizacija


## Stvaranje projekta

:::info
**Priprema za rad**
Stvorite projekt naziva `vj10`, unutar njega aplikaciju naziva `main`.
Provedite migraciju.
Zatim kreirajte administratora, za stvaranje korisnika sa administratorskim ovlastima koristite naredbu `./manage.py createsuperuser`.
:::

Povezivanje projekta i aplikacije

:::info
vj10/urls.py

```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main.urls')),
    path('accounts/', include('django.contrib.auth.urls'))
]

```
:::

### Homepage
:::info
main/urls.py

```python
from django.urls import path
from . import views 

urlpatterns = [
    path('', views.index, name='index'),
]

```
:::


Stvaranje pogleda za index
```python
def index(request):
    return render(request, 'main/index.html')
```

Unutar aplikacije `main` stvorite si direktorij `templates`, unutar kojeg kreirate `index.html`.
Html template:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title></title>
</head>
<body>

</body>
</html>
```

## Kreiranje korisnika
:::info
Posjetite `/accounts` i `/accounts/login`.

:::
Prilikom posjete `/accounts/login/` javila se greška `TemplateDoesNotExist at /accounts/login/`. Gdje možemo vidjeti iz poruke `Exception Value: registration/login.html` da Django ne može pronaći traženi predložak.

:::success
**Zadatak**
Unutar `templates/registration` stvorite `login.html`.
:::

### Login
```html
{% if form.errors %}
    <h3>Unos nije ispravan.</h3>
{% endif %}
```

```html
{% if next %}
    {% if user.is_authenticated %}
        <p>Your account doesn't have access to this page. To proceed,
        please login with an account that has access.</p>
    {% else %}
        <p>Please login to see this page.</p>
    {% endif %}
{% endif %}
```

```html
<form method="post" action="{% url 'login' %}">
    {% csrf_token %}
    <table>
      <tr>
        <td>{{ form.username.label_tag }}</td>
        <td>{{ form.username }}</td>
      </tr>
      <tr>
        <td>{{ form.password.label_tag }}</td>
        <td>{{ form.password }}</td>
      </tr>
    </table>
    <input type="submit" value="login" />
    <input type="hidden" name="next" value="{{ next }}" />
  </form>
```


Detaljnije o [csrf_tokenu](https://docs.djangoproject.com/en/3.2/ref/csrf/)

Postavljanje lokacije gdje želimo da korisnik bude usmjeren nakon uspješnog logina radimo unutar `settings.py`, tako da dodamo npr. `LOGIN_REDIRECT_URL = '/'` za usmjeravanje na `index.html`.

### Registracija 
Za registraciju koristimo gotovu formu sa:
```python
from django.contrib.auth.forms import UserCreationForm
```
I kreiramo funkciju `register`:

```python
def register(request):
    form = UserCreationForm()
    context = {'form': form}
    
    return render(request, 'registration/register.html', context)
```
Kreirajmo `register.html`:
```html    

<form method="post" action="{% url 'register' %}">
    {% csrf_token %}

    {% if form.errors %}
        <h3>Greska.</h3>
    {% endif %}

    {{ form }}

    <input type="submit" value="Register" />

  </form>
```


Izmjenimo `register` funkciju:

```python
from django.contrib.auth import authenticate, login

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)

        if form.is_valid():
            form.save()
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']

            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('index')

    else:
        form = UserCreationForm()

    context = {'form': form}

    return render(request, 'registration/register.html', context)
```

Izmjene na `index.html` ako je korisnik ulogiran.

```
<h1>This is our homepage</h1>

{% if user.is_authenticated %}
<h3> Vase ime:  {{ user.username}} </h3>   

{% else %}
<h3>Niste ulogirani</h3>

{% endif %}
```

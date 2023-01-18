---
marp: true
author: Vedran Miletić
title: "Objektno modeliranje i programiranje web aplikacija. Objektno-relacijsko preslikavanje"
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Objektno orijentirano modeliranje i programiranje web aplikacija. Objektno-relacijsko preslikavanje

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Objektno orijentirano modeliranje i programiranje

Za modeliranje koristit ćemo po potrebi [Allen Holub's UML Quick Reference](https://holub.com/uml/):

- Use-Case (Story) Diagram
- Static-Model (Class) Diagram
- Interaction (Dynamic-Model) Diagrams

Objektno orijentirano programiranje (prema [Wikipediji](https://en.wikipedia.org/wiki/Object-oriented_programming)):

- programska paradigma temeljena na objektima koji sadrže podatke (polja, atribute ili svojstva) i kod (procedure ili metode)
- četiri temelja: enkapsulacija, apstrakcija, nasljeđivanje i polimorfizam

---

## Enkapsulacija (1/2)

🙋 **Pitanje:** Je li ovo primjer enkapsulacije?

``` ruby
class Account < ApplicationRecord
  def initialize(iban, balance)
    @iban = iban
    @balance = balance
  end

  attr_accessor :iban, :balance
end

account = Account.find_by(iban: 'HR1234567890')
account.balance += 1000
account.save!
```

---

## Enkapsulacija (2/2)

🙋 **Pitanje:** Je li ovo primjer enkapsulacije?

``` php
<?php

class Account extends Model {
  private string $iban; private int $balance;
  public function __construct(string $iban, int $balance) { /* ... */ };

  public function getBalance(): int
  {
    return $this->balance;
  }

  public function alterBalance(int $amount): void
  {
    $this->iban += $amount;
  }
}
```

---

## Apstrakcija

``` php
<?php

interface EmailProvider
{
  public function send();
}

class SmtpEmailProvider implements EmailProvider
{
  public function send() {
    // ...
    $smtp = Mail::factory('smtp', ['host' => $host, 'port' => $port,
      'auth' => true, 'username' => $username, 'password' => $password]);
    $mail = $smtp->send($to, $headers, $email_body);
    // ...
  }
}
```

---

![Ready to use bg 95% left:55%](https://www.monkeyuser.com/assets/images/2021/225-reusable-components.png)

## Ponovno iskoristive komponente

Izvor: [Reusable Components](https://www.monkeyuser.com/2021/reusable-components/) (MonkeyUser, 7th September 2021)

---

## Nasljeđivanje

``` javascript
class PrintJob {
  constructor(printer, numberOfPages) {
    this.printer = printer;
    this.numberOfPages = numberOfPages;
  }
}

class PaidPrintJob extends PrintJob {
  constructor(printer, paymentAccount) {
    this.paymentAccount = paymentAccount;
    super(printer);
  }

  chargeAccount() {
    let pricePerPage = this.printer->getPricePerPage();
    this.paymentAccount->pay(this.numberOfPages * pricePerPage);
  }
}
```

---

## Polimorfizam

``` python
from django.db import models

class Question(models.Model):
  question_text = models.CharField(max_length=200)
  pub_date = models.DateTimeField('date published')

class Article(models.Model):
  article_text = models.TextField()
  pub_date = models.DateTimeField('date published')

q = Question(question_text="What's new?", pub_date=timezone.now())
a = Article(article_text="Lorem ipsum dolor sit amet, consectetur...",
            pub_date=timezone.now())

objs = [q, a]
for obj in objs:
  print(obj.pub_date)
```

---

![There is a time consuming difference between running a feature and having a runnable feature bg 95% left:55%](https://www.monkeyuser.com/assets/images/2019/153-edge-cases.png)

## Rubni slučajevi

Izvor: [Web App - Edge Cases](https://www.monkeyuser.com/2019/edge-cases/) (MonkeyUser, 15th October 2019)

---

## SOLID (1/2)

Prema [Wikipediji](https://en.wikipedia.org/wiki/SOLID), SOLID je skup načela koji je 2000. godine predložio američki softverski inženjer [Robert C. Martin](https://en.wikipedia.org/wiki/Robert_C._Martin) (kolokvijalno *Uncle Bob*) u svome radu [Design Principles and Design Patterns](https://wnmurphy.com/assets/pdf/Robert_C._Martin_-_2000_-_Principles_and_Patterns.pdf):

> What goes wrong with software? The design of many software applications begins as a vital image in the minds of its designers. At this stage it is clean, elegant, and compelling. (...)
>
> But then something begins to happen. **The software starts to rot.** At first it isn't so bad. An ugly wart here, a clumsy hack there, but the beauty of the design still shows through. Yet, over time as **the rotting continues, the ugly festering sores and boils accumulate until they dominate the design** of the application. The program becomes a festering mass of code that the developers find increasingly hard to maintain. (...)

---

## SOLID (2/2)

[Michael Feathers](https://michaelfeathers.silvrback.com/) je uveo mnemoničku kraticu za pet načela dizajna čiji je cilj softver učiniti razumljivijim, fleksibilnijim i lakšim za održavanje:

- (**S**) Načelo pojedinačne odgovornosti (engl. *single-responsibility principle*)
- (**O**) Načelo otvoreno-zatvoreno (engl. *open-closed principle*)
- (**L**) Liskovino načelo zamjene (engl. *Liskov substitution principle*)
- (**I**) Načelo razdvajanja sučelja (engl. *interface segregation principle*)
- (**D**) Načelo inverzije zavisnosti (engl. *dependency inversion principle*)

---

## Načelo pojedinačne odgovornosti (1/4)

(**S**) Načelo pojedinačne odgovornosti (engl. *single-responsibility principle*): klasa bi trebala imati samo jednu odgovornost, odnosno samo promjene na jednom dijelu specifikacije softvera trebale bi utjecati na specifikaciju klase (prema [Wikipediji](https://en.wikipedia.org/wiki/Single-responsibility_principle)).

---

## Načelo pojedinačne odgovornosti (2/4)

🙋 **Pitanje:** Zadovoljava li ova klasa načelo pojedinačne odgovornosti?

``` php
<?php

class Article {
  private Author $author;
  private $title;

  public function getAuthorAsJson() {
    return json_encode(["name" => $this->author->getName(),
                        "surname" => $this->author->getName()]);
  }

  public function getTitle() {
    return $this->title;
  }
}
```

---

## Načelo pojedinačne odgovornosti (3/4)

🙋 **Pitanje:** Zadovoljava li ova klasa načelo pojedinačne odgovornosti?

``` php
<?php

class Article {
  private Author $author;
  private $title;

  public function getAuthor() {
    return $this->author;
  }

  public function getTitle() {
    return $this->title;
  }
}
```

Kako ćemo dobiti podatke o autoru u formatu JSON?

---

## Načelo pojedinačne odgovornosti (4/4)

Jedno moguće rješenje:

``` php
<?php

class Author {
  private $name;
  private $surname;

  public function asJson() {
    return json_encode(["name" => $this->name,
                        "surname" => $this->surname])
  }
}
```

---

## Načelo otvoreno-zatvoreno (1/4)

(**O**) Načelo otvoreno-zatvoreno (engl. *open-closed principle*): dijelovi softvera trebali bi biti otvoreni za proširenje, ali zatvoreni za izmjene (prema [Wikipediji](https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle)).

---

## Načelo otvoreno-zatvoreno (2/4)

🙋 **Pitanje:** Zadovoljava li ova klasa načelo otvoreno-zatvoreno?

``` php
<?php

class Auth {
  public function login($user) {
    if ($user instanceof Student) {
      $this->loginStudent($user);
    } else if ($user instanceof Professor) {
      $this->loginProfessor($user);
    }
  }

  // ...
}
```

---

## Načelo otvoreno-zatvoreno (3/4)

🙋 **Pitanje:** Zadovoljava li ova klasa načelo otvoreno-zatvoreno?

``` php
<?php

class Auth {
  public function login($user) {
    $user->autheticate();
  }

  // ...
}
```

---

## Načelo otvoreno-zatvoreno (4/4)

``` php
<?php

interface AuthInterface {
  public function authenticate();
}

class Student implements AuthInterface {
  public function authenticate() {
    // ...
  };
}

class Professor implements AuthInterface {
  public function authenticate() {
    // ...
  };
}
```

---

## Liskovino načelo zamjene (1/3)

(**L**) Liskovino načelo zamjene (engl. *Liskov substitution principle*): objekti u programu trebali bi biti zamjenjivi primjercima svojih podtipova bez promjene ispravnosti tog programa (prema [Wikipediji](https://en.wikipedia.org/wiki/Liskov_substitution_principle)).

---

## Liskovino načelo zamjene (2/3)

🙋 **Pitanje:** Zadovoljavaju li ove klase Liskovino načelo zamjene?

``` php
<?php

interface FileRepository {
  public function getFiles();
}

class UserUploads implements FileRepository {
  public function getFiles() {
    return File::select('name')->where('user_id', $this->user->id)->get();
  }
}
```

---

## Liskovino načelo zamjene (3/3)

``` php
<?php

class PublicUploads implements FileRepository {
  public function getFiles() {
    $files = array();
    $handle = opendir('/var/www/html/uploads');

    if ($handle) {
      while (($entry = readdir($handle)) !== FALSE) {
        $files[] = $entry;
      }
    }

    closedir($handle);
  }
}
```

---

## Načelo razdvajanja sučelja (1/4)

(**I**) Načelo razdvajanja sučelja (engl. *interface segregation principle*): mnoga sučelja specifična za pojedinog klijenta bolja su od jednog sučelja opće namjene (prema [Wikipediji](https://en.wikipedia.org/wiki/Interface_segregation_principle)).

---

## Načelo razdvajanja sučelja (2/4)

🙋 **Pitanje:** Zadovoljavaju li ove klase načelo razdvajanja sučelja?

``` php
<?php

interface User {
  public function authenticate();
  public function getUserEmail();
}
```

---

## Načelo razdvajanja sučelja (3/4)

``` php
<?php

class Student implements User {
  public function authenticate() {
    // ...
  }

  public function getUserEmail() {
    // ...
  }
}
```

---

## Načelo razdvajanja sučelja (4/4)

``` php
<?php

class Guest implements User {
  public function authenticate() {
    return null;
  };

  public function getUserEmail() {
    // ...
  };
}
```

---

## Načelo inverzije zavisnosti (1/3)

(**D**) Načelo inverzije zavisnosti (engl. *dependency inversion principle*): ovisnost o apstrakcijama, \[ne\] o konkretizacijama (prema [Wikipediji](https://en.wikipedia.org/wiki/Dependency_inversion_principle)).

---

## Načelo inverzije zavisnosti (2/3)

🙋 **Pitanje:** Zadovoljavaju li ove klase načelo inverzije zavisnosti?

``` php
<?php

class MySqlConnection {
  public function connect() { /* ... */  }
}

class PageLoader {
  private $dbConnection;
  public function __construct(MySqlConnection $dbConnection) {
      $this->dbConnection = $dbConnection;
  }

  // ...
}
```

---

## Načelo inverzije zavisnosti (3/3)

``` php
<?php

interface DbConnectionInterface {
  public function connect();
}

class MySqlConnection implements DbConnectionInterface {
  public function connect() { /* ... */  }
}

class PageLoader {
  private $dbConnection;
  public function __construct(DbConnectionInterface $dbConnection) {
      $this->dbConnection = $dbConnection;
  }

  // ...
}
```

---

## Primjena SOLID načela u suvremenim aplikacijama

[Katerina Trajchevska](https://trajchevska.com/) iz [Adeve](https://www.adeva.com/) u objavi na blogu pod naslovom [SOLID Design Principles: The Guide to Becoming Better Developers](https://adevait.com/software/solid-design-principles-the-guide-to-becoming-better-developers) piše:

> Product owners don't always understand the implications of bad software design principles. It's on us, as engineers, to consider the best design practices when estimating and make sure we write code that's easy to maintain and extend. SOLID design principles can help us achieve that.

Snimka predavanja na temu: [Becoming a better developer by using the SOLID design principles](https://youtu.be/rtmFCcjEgEw) by [Katerina Trajchevska](https://trajchevska.com/) ([Laracon EU](https://laracon.eu/), 28th January 2019)

---

## Relacijski model

Recimo da je [neka teretana](https://youtu.be/7d9HZ9G2iuQ) predstavljena relacijskim modelom (stil označavanja: ***primarni ključ***, *vanjski ključ*):

- Član (***ID člana***, Ime, Adresa, Grad, Poštanski broj, Županija, Telefonski broj, E-mail) (npr. Hrvoje, Dora)
- Članarina (***Kod članarine***, Opis članarine) (npr. mjesečna, godišnja)
- Uplata članarina (***Broj uplate članarine***, *ID člana*, Datum) (npr. 1 godišnja + 2 mjesečne na 14. listopada 2021.)
- Stavka uplate članarine (***Broj uplate članarine***, ***Broj stavke uplate članarine***, *Kod članarine*, Količina) (npr. godišnja članarina na specifičnoj uplati)

Za vizualizaciju možete iskoristiti npr. [PonyORM Entity Relationship Diagram Creator](https://editor.ponyorm.com/) koji može i generirati kod za PonyORM (u Pythonu, donekle sličan Djangu).

---

## Pretvorba relacijskog modela u Django modele

``` python
from django.db import models

class Member(models.Model):
  name = models.CharField(max_length=50)
  address = models.CharField(max_length=50)
  city = models.CharField(max_length=30)
  postal_code = models.IntegerField()
  county = models.CharField(max_length=30)
  phone_number = models.CharField(max_length=20)

# class Membership

class MembershipPayment(models.Model):
  member = models.ForeignKey(Member, on_delete=models.CASCADE)
  date = models.DateField()

# class MembershipPaymentItem
```

---

## Pretvorba Django modela u SQL (1/2)

Django će na temelju definiranih modela generirati SQL. Oblik (tzv. dijalekt) SQL-a ovisit će o korištenom sustavu za upravljanju bazom podataka, a [podržani sustavi](https://docs.djangoproject.com/en/3.2/ref/databases/) su:

- MariaDB/MySQL (prvi primjer u nastavku)
- PostgreSQL (ostali primjeri u nastavku)
- SQLite
- Oracle

---

## Pretvorba Django modela u SQL (2/2)

``` sql
CREATE TABLE members (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  address VARCHAR(50),
  city VARCHAR(30),
  postal_code INT,
  county VARCHAR(30),
  phone_number VARCHAR(20)
);

CREATE TABLE membership_payment (
  id INT PRIMARY KEY,
  member_id INT UNSIGNED NOT NULL,
  date DATE,
  CONSTRAINT `fk_membership_payment_member`
    FOREIGN KEY (member_id) REFERENCES member (id)
    ON DELETE CASCADE
    ON UPDATE RESTRICT
);
```

---

## Upiti u Djangu i relacijskoj bazi (1/2)

Koristimo ranije prikazane klase i podatke spremamo korištenjem sustava za upravljanje bazom podataka [PostgreSQL](https://www.postgresql.org/). Na temelju koda:

``` python
members = Member.objects.all()
```

Django generira upit:

``` sql
SELECT * FROM members;
```

Na temelju koda:

``` python
ivan_horvat = Member.objects.get(name="Ivan Horvat")
```

Django generira upit:

``` sql
SELECT * FROM members WHERE name = "Ivan Horvat";
```

---

## Upiti u Djangu i relacijskoj bazi (2/2)

Na temelju koda:

``` python
ivans = Member.objects.filter(name__contains="Ivan")
```

Django generira upit:

``` sql
SELECT * FROM members WHERE name LIKE "%Ivan%";
```

Na temelju koda:

``` python
payments_ivan_horvat = MembershipPayment.objects.get(member=ivan_horvat)
```

Django generira upit:

``` sql
SELECT * FROM membership_payments WHERE member_id IN
    (SELECT id FROM members WHERE name = "Ivan Horvat");
```

---

## Objektno-relacijsko preslikavanje

Prema [Wikipediji](https://en.wikipedia.org/wiki/Object%E2%80%93relational_mapping):

- programska tehnika za pretvorbu između nekompatibilnih sustava tipova korištenjem objektno orijentiranih programskih jezika
- unos u tablici u relacijskoj bazi podataka postaje objekt u programu i obrnuto
- npr. [Hibernate](https://hibernate.org/) (Java), [PonyORM](https://ponyorm.org/) (Python), [Sequelize](https://sequelize.org/) (JavaScript) i [Doctrine](https://www.doctrine-project.org/projects/orm.html) (PHP)
- čitavi okviri uglavnom integriraju ORM, primjerice:
    - [Django Model](https://opensource.com/article/17/11/django-orm)
    - [Laravel Eloquent](https://laravel.com/docs/8.x/eloquent)
    - [Ruby on Rails ActiveRecord](https://guides.rubyonrails.org/active_record_basics.html)
- generira upite za dohvaćanje podataka i migracije (promjene sheme)

---

## Migracije u Djangu

``` shell
$ ./manage.py makemigrations
Migrations for 'members':
  members/migrations/0002_auto.py:
    - Add field e-mail on member
```

``` python
from django.db import migrations, models

class Migration(migrations.Migration):
  dependencies = [('migrations', '0001_initial')]
  operations = [
    migrations.AddField('Member', 'e-mail',
                        models.CharField(max_length=30)),
  ]
```

Na temelju ove migracije Django generira upit:

``` sql
ALTER TABLE members ADD COLUMN e_mail VARCHAR(30);
```

---

## Migracije u web aplikacijama otvorenog koda

- [Phabricator](https://secure.phabricator.com/): migracije se nalaze [u repozitoriju izvornog koda, u direktoriju `resources/sql`](https://github.com/phacility/phabricator/tree/master/resources/sql) u poddirektorijima `patches` i `autopatches`; napisan u obliku SQL upita za MariaDB/MySQL (drugi sustavi nisu podržani)
- [Flarum](https://flarum.org/): migracije se nalaze [u repozitoriju izvornog koda, u direktoriju `migrations`](https://github.com/flarum/core/tree/master/migrations); koristi Laravel Eloquent
- [Canvas LMS](https://www.instructure.com/canvas): migracije se nalaze [u repozitoriju izvornog koda, u direktoriju `db/migrate`](https://github.com/instructure/canvas-lms/tree/master/db/migrate); koristi Ruby on Rails ActiveRecord

---

## Nerelacijske baze podataka

Prema [Wikipediji](https://en.wikipedia.org/wiki/NoSQL), dio [Yenove taksonomije](https://www.christof-strauch.de/nosqldbs.pdf) NoSQL baza podataka:

- međuspremnik ključeva i vrijednosti (engl. *key–value cache*)
    - npr. [Memcached](https://www.memcached.org/), [Redis](https://redis.io/)
    - često se koriste kao međuspremnik podataka dohvaćenih iz relacijskih baza, npr. [Memcached u Djangu](https://docs.djangoproject.com/en/3.2/topics/cache/), [Memcached ili Redis u Ruby on Railsu](https://guides.rubyonrails.org/caching_with_rails.html)
- pohrana ključeva i vrijednosti (engl. *key–value store*)
    - npr. [ArangoDB](https://www.arangodb.com/), [Couchbase](https://www.couchbase.com/), [Redis](https://redis.io/)
- pohrana dokumenata, npr. [ArangoDB](https://www.arangodb.com/), [Couchbase](https://www.couchbase.com/), [Firebase](https://firebase.google.com/), [MongoDB](https://www.mongodb.com/)
- pohrana grafova, npr. [ArangoDB](https://www.arangodb.com/), [Neo4j](https://neo4j.com/), [OrientDB](https://orientdb.org/)

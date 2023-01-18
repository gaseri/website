---
marp: true
author: Vedran Miletić
title: "Paradigme programiranja u web aplikacijama: proceduralna, objektno orijentirana, funkcijska"
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Paradigme programiranja u web aplikacijama: proceduralna, objektno orijentirana, funkcijska

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Softverska kriza 1960-ih

Prema [Wikipediji](https://en.wikipedia.org/wiki/Software_crisis):

- Projekti razvoja softvera koji su prelazili dana financijska sredstva za izradu
- Projekti razvoja softvera koji su prelazili dane vremenske okvire izrade
- Softver je bio vrlo neučinkovit
- Softver je bio loše kvalitete
- Softver često nije udovoljavao zahtjevima
- Projektima se nije moglo upravljati i bilo ih je teško održavati
- Softver nikada nije bio isporučen (engl. *vaporware*; najpoznatiji primjer softvera pod tom etiketom je dugo vremena bila [igra Duke Nukem Forever](https://www.wired.com/2011/06/duke-nukem-vaporware/))

---

## Rješenje softverske krize 1960-ih

Razvijeno više metodologija razvoja softvera; najvažnije još aktualne su [strukturirano programiranje](https://en.wikipedia.org/wiki/Structured_programming) i [objektno orijentirano programiranje](https://en.wikipedia.org/wiki/Object-oriented_programming).

Američki računalnih arhitekt i programski inženjer [Fred Brooks](https://en.wikipedia.org/wiki/Fred_Brooks) objavljuje 1987. godine [članak u časopisu IEEE Computer](https://ieeexplore.ieee.org/document/1663532) pod naslovom [No Silver Bullet – Essence and Accident in Software Engineering](https://en.wikipedia.org/wiki/No_Silver_Bullet) u kojem piše:

- > "there is no single development, in either technology or management technique, which by itself promises even **one order of magnitude \[tenfold\] improvement within a decade in productivity, in reliability, in simplicity**"
- > "we **cannot expect ever to see two-fold gains every two years [in software development**, as there is in hardware development (Moore's law)]"

---

## Proceduralna paradigma (1/4)

``` json
{
  "rm": "http://example.group.miletic.net/nastava/RM/",
  "pw": "http://example.group.miletic.net/nastava/PW/"
}
```

``` php
<?php

$links_json = file_get_contents('links.json');
$links = json_decode($links_json, true);
$link_key = $_SERVER['REQUEST_URI'];

if (array_key_exists($link_key, $links)) {
  http_response_code(301);
  header('Location: ' . $links[$link_key]);
} else {
  http_response_code(404);
  echo '<p>Unknown key <strong>' . $link_key . '</strong>.</p>\n';
}
```

---

## Proceduralna paradigma (2/4)

🙋 **Pitanje:** Je li i ovo proceduralni kod?

``` php
<?php

$mysqli = new mysqli("example.com", "user", "password", "database");
if ($mysqli->connect_errno) {
  echo "Failed to connect to MySQL: " . $mysqli->connect_error;
}

$res = $mysqli->query("SELECT given_name AS _name FROM profiles");
$row = $res->fetch_assoc();
echo $row['_name'];
```

---

## Proceduralna paradigma (3/4)

🙋 **Pitanje:** Je li i ovo proceduralni kod?

``` php
<?php

class FetchName {
  public function fetchName() {
    $mysqli = new mysqli("example.com", "user", "password", "database");
    if ($mysqli->connect_errno) {
      echo "Failed to connect to MySQL: " . $mysqli->connect_error;
    }

    $res = $mysqli->query("SELECT given_name AS _name FROM profiles");
    $row = $res->fetch_assoc();
    return $row['_name'];
  }
}
```

---

## Proceduralna paradigma (4/4)

🙋 **Pitanje:** Je li i ovo proceduralni kod?

``` php
<?php

class ProfileDataFetcher {
  public function __construct($mysqli) {
    // ...
  }
  public function getAll(): array|null {
    // ...
  }
}

$fetcher = new ProfileDataFetcher(new mysqli("example.com", "user",
                                             "password", "database"));
$profiles = $fetcher->getAll();
foreach ($profiles as $profile) {
  echo $profile->getName();
}
```

---

## Objektno orijentirana paradigma (1/2)

``` python
from django.db import models
from django.utils import timezone

class Question(models.Model):
  question_text = models.CharField(max_length=200)
  pub_date = models.DateTimeField('date published')
  def was_published_recently(self):
    return self.pub_date >= timezone.now() - datetime.timedelta(days=1)

class Choice(models.Model):
  question = models.ForeignKey(Question, on_delete=models.CASCADE)
  choice_text = models.CharField(max_length=200)
  votes = models.IntegerField(default=0)

q = Question(question_text="What's new?", pub_date=timezone.now())
q.save()

q.choice_set.create(choice_text='Not much', votes=0)
q.choice_set.create(choice_text='The sky', votes=0)
```

---

## Objektno orijentirana paradigma (2/2)

``` ruby
def Student < ApplicationRecord
  has_one :address
  has_many :exams
end

def Address < ApplicationRecord
  belongs_to :student
end

def Exam < ApplicationRecord
  belongs_to :student
end

student = Student.create(name: "Ivan Horvat", jmbag: "0123456789")

ivan_horvat = Student.find_by(name: "Ivan Horvat")
```

---

## Funkcijska paradigma (1/4)

Koriste se funkcije kao parametri drugih funkcija:

``` javascript
router.post('/', function (req, res) {
  res.send('Got a POST request');
});
```

``` javascript
var myLogger = function (req, res, next) {
  console.log('LOGGED');
  next();
}

app.use(myLogger)
```

---

## Funkcijska paradigma (2/4)

Pretjerano korištenje povratnih poziva funkcija kao parametara drugih funkcija vodi u tzv. *callback hell*:

``` javascript
call_endpoint('api/getidbyusername/hotcakes', result => {
  call_endpoint(`api/getfollowersbyid/${result.userID}`, result => {
    call_endpoint('api/someothercall/' + result.followers, result => {
      // ...
    });
  });
});
```

Suvremeni JavaScript ovo izbjegava korištenjem `async` i `await`, za one koji žele znati više: [Escape from Callback Hell](https://jst.hashnode.dev/callback-hell) (Incognito, 1st November 2020)

---

## Funkcijska paradigma (3/4)

Prema [Wikipediji](https://en.wikipedia.org/wiki/Functional_programming):

- deklarativna paradigma temeljena na korištenju funkcija i kompozicija funkcija
- funkcije se mogu dodjeljivati kao vrijednost, prosljeđivati kao argumenti drugih funkcija i vratiti kao rezultat izvođenja drugih funkcija
- [čisto funkcijsko programiranje](https://en.wikipedia.org/wiki/Purely_functional_programming): funkcija vraća uvijek isti rezultat za iste ulazne parametre, što olakšava testiranje i otklanjanje grešaka

---

## Funkcijska paradigma (4/4)

> (...) many functional languages are seeing use today in industry and education, including Common Lisp, Scheme, Clojure, Wolfram Language, Racket, Erlang, **Elixir**, OCaml, Haskell, and F#.
>
> Functional programming is also key to some languages that have found success in specific domains, like **JavaScript in the Web**, R in statistics, (...)
>
> In addition, many other programming languages support programming in a functional style or have implemented features from functional programming, such as **C++11**, Kotlin, Perl, **PHP**, **Python**, Go, Rust, Raku, Scala, and Java (...)

Primjeri korištenja funkcijskih jezika na webu:

- [Phoenix Framework](https://www.phoenixframework.org/) za [Elixir](https://elixir-lang.org/) ima [17k+ GitHub zvjezdica](https://github.com/phoenixframework/phoenix) ([za usporedbu](https://gitstar-ranking.com/repositories))
- [Hacker News](https://news.ycombinator.com/) ([izvorni kod](https://github.com/wting/hackernews) napisan u [jeziku Arc](http://www.arclanguage.org/)) ima [10M+ pogleda mjesečno](https://www.similarweb.com/website/news.ycombinator.com/)

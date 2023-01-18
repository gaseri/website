---
marp: true
author: Vedran Miletić
title: Uzorci dizajna u web aplikacijama. Uzorci model-pogled-*
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Uzorci dizajna u web aplikacijama. Uzorci model-pogled-*

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Kompleksnost u razvoju softvera

> Managing complexity is the most important technical topic in software development. In my view, it's so important that **Software's Primary Technical Imperative has to be managing complexity**.
>
> Dijkstra pointed out \[in 1972\] that no one's skull is really big enough to contain a modern computer program, which means that we as software developers shouldn't try to cram whole programs into our skulls at once; we should try to organize our programs in such a way that we can safely focus on one part of it at a time. The goal is to **minimize the amount of a program you have to think about at any one time**.

Izvor: [Steve McConnel](https://stevemcconnell.com/), [Code Complete](https://stevemcconnell.com/books/), [2nd Edition](https://www.microsoftpressstore.com/store/code-complete-9780735619678), 2004.

---

## FizzBuzz

Napišite program koji ispisuje prvih 100 prirodnih brojeva tako da ispiše:

- `Fizz` ako je broj djeljiv s 3,
- `Buzz` ako je broj djeljiv s 5,
- `FizzBuzz` ako je broj djeljiv i s 3 i s 5,
- inače ispiše samo broj.

---

## Dizajn programa FizzBuzz (1/3)

``` python
for n in range(1, 101):
  if n % 3 == 0 and n % 5 == 0:
    print('FizzBuzz')
  elif n % 3 == 0:
    print('Fizz')
  elif n % 5 == 0:
    print('Buzz')
  else:
    print(n)
```

🙋 **Pitanje:** Je li ovo jedini pristup?

---

## Dizajn programa FizzBuzz (2/3)

``` python
for n in range(1, 101):
  s = ''
  if n % 3 == 0:
    s = s + 'Fizz'
  if n % 5 == 0:
    s = s + 'Buzz'
  if n % 5 != 0 and n % 3 != 0:
    s = s + str(n)
  print(s)
```

🙋 **Pitanje:** Postoji li još mogućnosti?

---

## Dizajn programa FizzBuzz (3/3)

``` java
public final class Main {
  /**
   * @param args
   */
  public static void main(final String[] args) {
    final ApplicationContext context = new
      ClassPathXmlApplicationContext(Constants.SPRING_XML);
    final FizzBuzz myFizzBuzz = (FizzBuzz)
      context.getBean(Constants.STANDARD_FIZZ_BUZZ);
    final FizzBuzzUpperLimitParameter fizzBuzzUpperLimit = new
      DefaultFizzBuzzUpperLimitParameter();
    myFizzBuzz.fizzBuzz(fizzBuzzUpperLimit.obtainUpperLimitValue());

    ((ConfigurableApplicationContext) context).close();
  }
}
```

Čitav projekt na GitHubu: [EnterpriseQualityCoding/FizzBuzzEnterpriseEdition](https://github.com/EnterpriseQualityCoding/FizzBuzzEnterpriseEdition)

---

## Dizajn softvera

Prema [Wikipediji](https://en.wikipedia.org/wiki/Software_design): proces kojim agent (ovdje u značenju: programer s puno iskustva) stvara specifikaciju softverskog artefakta (npr. programa, okvira ili biblioteke) namijenjenog postizanju ciljeva, koristeći skup primitivnih komponenata i podložan ograničenjima; dvije mogućnosti:

- > sve aktivnosti uključene u konceptualiziranje, uokvirivanje, implementaciju, puštanje u rad i konačno modificiranje složenih sustava
- > aktivnost koja slijedi nakon specifikacije zahtjeva i prije programiranja, kao ... \[u\] stiliziranom procesu programskog inženjerstva

---

## Mjerila kvalitete dizajna softvera

Dobar dizajn softvera:

- **maksimizira koherentnost**: dijelovi softvera zajedno rade na logičan, razuman, lako uočljiv način
- **minimizira sprezanje**: dijelovi softvera se mogu koristiti odvojeno jedni od drugih, što specijalno olakšava njihovo ponovno korištenje

---

## Primjer ponovnog iskorištenje koda (1/2)

Želimo Pythonov objekt:

``` python
['foo', {'bar': ('baz', None, 1.0, 2)}]
```

pretvoriti u [oblik JSON](https://www.json.org/) ([Wikipedia](https://en.wikipedia.org/wiki/JSON)):

``` json
["foo", {"bar": ["baz", null, 1.0, 2]}]
```

Moramo li pisati svoj kod za tu svrhu?

📝 **Napomena:** uočimo da Pythonovi i JSON-ovi tipovi podataka nisu isti.

---

## Primjer ponovnog iskorištenje koda (2/2)

Pythonov modul `json` omogućuje ponovnu iskoristivost funkcija za rad s JSON-om (npr. pretvorba u, pretvorba iz):

``` python
import json

o = ['foo', {'bar': ('baz', None, 1.0, 2)}]
j = json.dumps(o)
# j će imati vrijednost '["foo", {"bar": ["baz", null, 1.0, 2]}]'
```

🙋 **Pitanje:** Možemo li dizajn softvera ponovno iskoristiti kao što ponovno iskorištavamo kod?

---

## Ponovno iskorištenje dizajna i uzorci dizajna

Prema [Wikipediji](https://en.wikipedia.org/wiki/Software_design_pattern):

- opće, ponovno iskoristivo rješenje za problem koji se često javlja kod dizajna softvera; *nije gotov dizajn* koji se može odmah prevesti u izvorni kod
- opis postupka ili predložak za rješavanje problema koji se može koristiti u različitim situacijama
- formalizirane najbolje prakse koje programer može koristiti za rješavanje uobičajenih problema prilikom dizajniranja aplikacije ili sustava

---

## Početak uzoraka dizajna

- Gang of Four (GoF), [Design Patterns book](https://www.pearson.com/us/higher-education/program/Gamma-Design-Patterns-Elements-of-Reusable-Object-Oriented-Software/PGM14333.html), 1994.
- [kritika](https://en.wikipedia.org/wiki/Design_Patterns#Criticism) ([Paul Graham](http://www.paulgraham.com/) i [Peter Norvig](https://www.norvig.com/)): uzorci dizajna služe za zaobilaženje nedostataka C++-a
    - mogu se za istu svrhu iskoristiti makroi (predprocesorska naredba `#define`)
    - funkcijski jezici kao Lisp nemaju potrebu za većinom uzoraka

![Design Patterns book bg right 80%](https://learningactors.com/wp-content/uploads/2017/10/Gang_of_four.jpg)

---

## Suvremena literatura za uzorke dizajna

- [Design Patterns (Refactoring.Guru)](https://refactoring.guru/design-patterns); stranica kaže o sebi:
    - > makes it easy for you to discover everything you need to know about refactoring, design patterns, SOLID principles, and other smart programming topics
- [Design Patterns (Coursera)](https://www.coursera.org/learn/design-patterns) kojeg nudi Sveučilište u Alberti, dio serijala [Software Design and Architecture Specialization](https://www.coursera.org/specializations/software-design-architecture)
- brojni drugi, npr. popis na [Design Patterns Tutorials and Courses (Hackr.io)](https://hackr.io/tutorials/learn-software-design-patterns)

---

![Code by the rules bg 85% left](https://www.monkeyuser.com/assets/images/2017/62-design-patterns-bureaucracy.png)

## Uzorci dizajna - birokracija

Izvor: [Design Patterns - Bureaucracy](https://www.monkeyuser.com/2017/design-patterns-bureaucracy/) (MonkeyUser, 26th September 2017)

---

## Kreacijski uzorci (engl. *creational patterns*)

- (**C**) Apstraktna tvornica (engl. *abstract factory*)
- (**C**) Graditelj (engl. *builder*)
- (**C**) Tvornica (engl. *factory*)
- (**C**) Prototip (engl. *prototype*)
- (**C**) Singleton

---

## Strukturni uzorci (engl. *structural patterns*)

- (**S**) Adapter
- (**S**) Most (engl. *bridge*)
- (**S**) Smjesa (engl. *composite*)
- (**S**) Dekorator (engl. *decorator*)
- (**S**) Fasada (engl. *facade*)
- (**S**) Muhavac (engl. *flyweight*)
- (**S**) Opunomoćenik (engl. *proxy*)

---

## Uzorci ponašanja (engl. *behavioral patterns*)

- (**B**) Lanac odgovornosti (engl. *chain of responsibility*)
- (**B**) Naredba (engl. *command*)
- (**B**) Interpreter
- (**B**) Iterator
- (**B**) Posrednik (engl. *mediator*)
- (**B**) Uspomena (engl. *memento*)
- (**B**) Promatrač (engl. *observer*)
- (**B**) Stanje (engl. *state*)
- (**B**) Strategija (engl. *strategy*)
- (**B**) Predložak (engl. *template*)
- (**B**) Posjetitelj (engl. *visitor*)

---

## Pregled svih 23 uzoraka dizajna prema GoF

![The 23 Gang of Four Design Patterns](https://live.staticflickr.com/7197/6804689564_8a6ff3efff_c.jpg)

Izvor: [Design Patterns Quick Reference](http://www.mcdonaldland.info/files/designpatterns/designpatternscard.pdf), autor [Jason McDonald](http://www.mcdonaldland.info/) (2007.)

---

## Tvornica (1/5)

``` php
<?php

class Motor {
  // ...
}

class MotorFactory {
  public function create() : Motor {
    $motor = new Motor();
    return $motor;
  }
}

$motorFactory = MotorFactory();
$motor = $motorFactory->create();
```

---

## Tvornica (2/5)

``` php
<?php

class Motor {}

class GasolineMotor extends Motor {}
class DieselMotor extends Motor {}
class ElectricMotor extends Motor {}

class Car {
  public Motor $motor;

  public function __construct() {
    $motor = new Motor();
  }
}
```

🙋 **Pitanje:** Što ćemo s različitim vrstama motora?

---

## Tvornica (3/5)

``` php
<?php

class Car {}

class DieselCar extends Car {
  public Motor $motor;

  public function __construct() {
    $motor = new DieselMotor();
  }
}

class GasolineCar extends Car {
  // analogno
}
```

🙋 **Pitanje:** Možemo li napraviti bolji dizajn od ovog?

---

## Tvornica (4/5)

``` php
<?php

class MotorFactory {
  string $type;

  public function create() : Motor {
    $motor = new $type();
    return $motor;
  }
}

$dieselMotorFactory = MotorFactory();
$dieselMotorFactory->type = "DieselMotor";
```

---

## Tvornica (5/5)

``` php
<?php

class Car {
  public Motor $motor;

  public function __construct(MotorFactory $factory) {
    $motor = $factory->create();
  }
}

$car = new Car($dieselMotorFactory);

$electricMotorFactory = MotorFactory();
$electricMotorFactory->type = "ElectricMotor";
$bmwConceptI4 = new Car($electricMotorFactory);
```

---

## Apstraktna tvornica (1/2)

``` ruby
class Game
  attr_accessor :title
  def initialize(title)
    @title = title
  end
end

class Rpg < Game
  def description
    puts "I am a RPG named #{@title}"
  end
end

class Arcade < Game
  def description
    puts "I am an Arcade named #{@title}"
  end
end
```

---

## Apstraktna tvornica (2/2)

``` ruby
class GameFactory
  def create(title)
    raise NotImplementedError, "You should implement this method"
  end
end

class RpgFactory < GameFactory
  def create(title)
    Rpg.new title
  end
end

class ArcadeFactory < GameFactory
  def create(title)
    Arcade.new title
  end
end
```

---

## Graditelj

``` python
# class Builder

class UserBuilder(Builder):

  def __init__(self):
    self._user_ = User()

  def user(self):
    user = self._user_
    self._user_ = User()
    return user

  def facebook_connection(self):
    self._user_.add_connection("Facebook")

  def google_connection(self):
    self._user_.add_connection("Google")

  def github_connection(self):
    self._user_.add_connection("Github")
```

---

## Prototip (1/2)

``` php
<?php

class BlogArticle
{
  private $title;
  private $body;
  private $author;
  private $date;
  private $comments = [];

  public function __construct(string $title, string $body,
                              Author $author)
  {
    $this->title = $title;
    $this->body = $body;
    $this->author = $author;
    $this->date = new \DateTime();
  }
}
```

---

## Prototip (2/2)

``` php
<?php

class BlogArticle
{
  // ...

  public function __clone()
  {
    $this->title = "Copy of " . $this->title;
    $this->date = new \DateTime();
    $this->comments = [];
  }
}
```

---

## Singleton

``` python
class Singleton:
  __instance = None

  @staticmethod
  def getInstance():
    if Singleton.__instance == None:
      Singleton()
    return Singleton.__instance

  def __init__(self):
    if Singleton.__instance != None:
      raise Exception("This class is a singleton!")
    else:
      Singleton.__instance = self

s1 = Singleton()
s2 = Singleton.getInstance()

# s1 == s2
```

---

## Adapter

``` python
class WeatherForecast:
  def getTemperature(self, location, date_time):
    # returns F

  def getWindSpeed(self, location, date_time):
    # returns mph

class WeatherForecastAdapter:
  __forecast = None

  def __init__(self, forecast):
    self.__forecast = forecast

  def getTermperature(self, location, date_time):
    return (forecast.getTemperature(location, date_time) - 32) / 1.8

  def getWindSpeed(self, location, date_time):
    return forecast.getWindSpeed(location, date_time) * 1.609344
```

---

## Most (1/5)

``` php
<?php

abstract class Article
{
  protected $renderer;

  public function __construct(Renderer $renderer)
  {
    $this->renderer = $renderer;
  }

  public function changeRenderer(Renderer $renderer): void
  {
    $this->renderer = $renderer;
  }

  abstract public function view(): string;
}
```

---

## Most (2/5)

``` php
<?php

class Letter extends Article
{
  // ...
  public function view(): string
  {
    return $this->renderer->renderParts( /* ... */ );
  }
}

class JournalArticle extends Article
{
  // ...
  public function view(): string
  {
    return $this->renderer->renderParts( /* ... */ );
  }
}
```

---

## Most (3/5)

``` php
<?php

interface Renderer
{
    public function renderTitle(string $title): string;

    public function renderTextBlock(string $text): string;

    public function renderImage(string $url): string;

    public function renderLink(string $url, string $title): string;

    public function renderHeader(): string;

    public function renderFooter(): string;

    public function renderParts(array $parts): string;
}
```

---

## Most (4/5)

``` php
<?php

class HTMLRenderer implements Renderer
{
  public function renderTitle(string $title): string
  {
    return "<h1>$title</h1>";
  }

  public function renderTextBlock(string $text): string
  {
    return "<p>$text</p";
  }

  // ...
}

class PDFRenderer implements Renderer { /* ... */ }
```

---

## Most (5/5)

``` php
<?php

class JSONRenderer implements Renderer
{
  // ...
}

class XMLRenderer implements Renderer
{
  // ...
}
```

---

## Dekorator (1/3)

``` php
<?php

interface OpenerInterface {
  public function open() : void;
}

class Door implements OpenerInterface {
  public function open() : void {
    // opens the door
  }
}

class Window implements OpenerInterface {
  public function open() : void {
    // opens the window
  }
}
```

---

## Dekorator (2/3)

``` php
<?php

class SmartDoor extends Door {
  public function open() : void {
    parent::open();
    $this->temperature();
  }
}

class SmartWindow extends Window {
  public function open() : void {
    parent::open();
    $this->temperature();
  }
}
```

🙋 **Pitanje:**  Moramo li ponavljati nasljeđivanje za svaki pametni uređaj?

---

## Dekorator (3/3)

``` php
<?php

class SmartOpener implements OpenerInterface  {
  private $opener;

  public function __construct(OpenerInterface $opener) {
    $this->opener = $opener;
  }
  public function open() : void {
    $this->opener->open();
    $this->temperature();
  }
}

$door = new Door();
$window = new Window();

$smartDoor = new SmartOpener($door);
$smartWindow = new SmartOpener($window);
```

---

## Fasada (1/2)

``` python
class DatabaseConnection:
  def query(self, data) -> str:
    # ...

class CacheConnection:
  def is_available(self, data) -> str:
    # ...

  def get(self, data) -> str:
    # ...
```

---

### Fasada (2/2)

``` python
class Facade:
  def __init__(self, databaseConnection: DatabaseConnection,
               cacheConnection: CacheConnection) -> None:
    self._databaseConnection = databaseConnection or DatabaseConnection()
    self._cacheConnection = cacheConnection or CacheConnection()

  def operation(self, data) -> str:
    if self._cacheConnection.is_available(data):
      return self._cacheConnection.get(data)
    else
      return self._databaseConnection.query(data)

if __name__ == "__main__":
  databaseConnection = DatabaseConnection()
  cacheConnection = CacheConnection()
  facade = Facade(databaseConnection, cacheConnection)
  # data ...
  facade.get(data)
```

---

## Opunomoćenik (1/3)

``` php
<?php

interface DataRetriever
{
  public function retrieve(string $data): string;
}

class DatabaseRetriever implements DataRetriever
{
  private $dbConnection;

  public function retrieve(string $data): string
  {
    return $dbConnection->query($data);
  }
}
```

---

## Opunomoćenik (2/3)

``` php
<?php

class CachingRetriever implements DataRetriever
{
  private $databaseRetriever;
  private $cache = [];

  public function __construct(DatabaseRetriever $databaseRetriever)
  {
      $this->databaseRetriever = $databaseRetriever;
  }
}
```

---

## Opunomoćenik (3/3)

``` php
<?php

class CachingRetriever implements DataRetriever
{
  // ...

  public function retrieve(string $data): string
  {
    if (isset($this->cache[$data])) {
      return $this->cache[$url];
    }

    $result = $this->databaseRetriever->retrieve($url);
    $this->cache[$url] = $result;
    return $result;
  }
}
```

---

## Lanac odgovornosti (1/2)

``` python
class Handler:
  _next_handler = None

  def set_next(self, handler):
    self._next_handler = handler
    return handler

  def handle(self, request):
    if self._next_handler:
      return self._next_handler.handle(request)

    return None
```

---

## Lanac odgovornosti (2/2)

``` python
class FTPDownloader(Handler):
  def handle(self, request):
    if request.startswith("ftp://"):
      # ...
    else:
      return super().handle(request)


class HTTPDownloader(Handler):
  def handle(self, request) -> str:
    if request.startswith("http://") or request.startswith("https://"):
      # ...
    else:
      return super().handle(request)
```

---

## Naredba

``` javascript
import { exec } from 'child_process';

interface Command {
  execute(): void;
}

class ImageMagickCommand implements Command {
  private parameters;
  constructor(parameters) {
    this.parameters = parameters;
  }

  execute() {
    exec('imagemagick ' + parameters, (err, stdout, stderr) => {
      /* ... */
    });
  }
}

myCommand = ImageMagickCommand('-crop 32x32+16+16 image.png');
```

---

## Interpreter

``` python
exec("""for i in range(10):\n    print('Hello world!')""")
result = eval("5 + 2 * a")
```

---

## Iterator (1/2)

``` ruby
class Iterator
  attr_accessor :reverse
  private :reverse

  attr_accessor :collection
  private :collection

  def initialize(collection, reverse = false)
    @collection = collection
    @reverse = reverse
  end

  def each(&block)
    return @collection.reverse.each(&block) if reverse

    @collection.each(&block)
  end
end
```

---

## Iterator (2/2)

``` ruby
class ArticlesCollection
  attr_accessor :collection
  private :collection

  def initialize(collection = [])
    @collection = collection
  end

  def iterator
    Iterator.new(@collection)
  end

  def reverse_iterator
    Iterator.new(@collection, true)
  end

  def add_item(item)
    @collection << item
  end
end
```

---

## Posrednik (1/2)

``` javascript
interface ArticleRetriever {
  getArticle(id: number): void;
}

class DatabaseArticleRetriever implements ArticleRetriever {
  public getArticle(id: number): void {
    /* ... */
  }
}
```

---

## Posrednik (2/2)

``` javascript
class Proxy implements ArticleRetriever {
  private articleRetriever: ArticleRetriever;

  constructor(articleRetriever: ArticleRetriever) {
    this.articleRetriever = articleRetriever;
  }

  public getArticle(id: number): void {
    if (this.isValid(id)) {
      this.realSubject.request(id);
    }
  }

  private isValid(id: number): boolean {
    /* ... */
  }
}
```

---

## Promatrač

``` javascript
function Order() {
  this.observers = [];
}
Order.prototype = {
  subscribe: function(fn) {
    this.observers.push(fn); },
  unsubscribe: function(fn) {
    this.observers = this.observers.filter(
      function(item) {
        if (item !== fn) {
          return item;
    }}); },
  fire: function() {
    this.observers.forEach(
      function(fn) {
        fn.call();
    }); }
}
```

---

## Stanje

``` php
<?php

class Ad {
  private string $state;

  public function publish() {
    switch ($state) {
      case "draft":
        $state = "moderation";
        break;
      case "moderation":
        if ($currentUser.isModerator()) {
          $state = "published";
        }
        break;
      case "published":
        break;
  }
}
```

---

## Strategija

``` ruby
class MapRouter
  def do_routing(start, end)
    raise NotImplementedError,
          "#{self.class} has not implemented method '#{__method__}'"
  end
end

class OSRMMapRouter < MapRouter
  def do_routing(start, end)
    # ...
  end
end

class GraphHopperMapRouter < MapRouter
  def do_routing(start, end)
    # ...
  end
end
```

---

## Primjeri primjene uzoraka dizajna

🙋 **Pitanje:** Koje uzorke dizajna koriste navedeni programi, okviri i biblioteke?

- [Eloquent Collections](https://laravel.com/docs/8.x/eloquent-collections)
- [Laravel Mail](https://laravel.com/docs/8.x/mail#sending-mail)
- [Django Generic Views Test](https://github.com/django/django/blob/main/tests/generic_views/test_edit.py)
- Django Basic Tests: [models](https://github.com/django/django/blob/main/tests/basic/models.py), [tests](https://github.com/django/django/blob/main/tests/basic/tests.py)
- [Magento `ListAction`](https://github.com/magento/magento2/blob/2.4-develop/app/code/Magento/Review/Controller/Product/ListAction.php)

---

## Model, pogled i upravitelj (1/3)

Engl. *model-view-controller*, kraće MVC. Prema [Wikipediji](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller):

- uzorak softverskog dizajna koji se obično koristi za razvoj grafičkih korisničkih sučelja
- dijeli povezanu programsku logiku na tri međusobno povezana elementa
- odvaja unutarnji prikaz informacija od načina na koji se informacije prezentiraju korisniku i prihvaćaju korisnički unosi

![Model, view, controller, and user bg 95% right:45%](https://upload.wikimedia.org/wikipedia/commons/a/a0/MVC-Process.svg)

---

## Model, pogled i upravitelj (2/3)

Koristi se kod razvoja:

- web aplikacija, primjerice:
    - [Djangov model-pogled-predložak](https://docs.djangoproject.com/en/3.2/faq/general/#django-appears-to-be-a-mvc-framework-but-you-call-the-controller-the-view-and-the-view-the-template-how-come-you-don-t-use-the-standard-names)
    - [ASP.NET-ov MVC](https://dotnet.microsoft.com/apps/aspnet/mvc)
    - [Laravelov MVC](https://laracasts.com/series/laravel-8-from-scratch/episodes/1)
- mobilnih i stolnih aplikacija
    - [Android](https://upday.github.io/blog/model-view-controller/) (nije [jedini](https://upday.github.io/blog/model-view-presenter/) [pristup](https://upday.github.io/blog/model-view-viewmodel/))
    - [Qt Model/View](https://doc.qt.io/qt-5/model-view-programming.html)

![Model, view, controller, and user bg 95% right:55%](https://krify.co/wp-content/uploads/2019/06/Django-Work-flow.jpg)

---

## Model, pogled i upravitelj (2/2)

Prema [Wikipediji](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller):

- **Model** je središnja komponenta uzorka, neovisna o korisničkom sučelju
- **Pogled** je bilo koji prikaz informacija kao što su grafikon, dijagram ili tablica; mogući su višestruki prikazi istih podataka
- **Kontroler** prihvaća ulaz i pretvara ga u naredbe za model ili pogled

Interakcija komponenata:

- *Model* upravlja podacima aplikacije. Prima korisnički unos od *kontrolera*.
- *Pogled* prikazuje prezentaciju *modela* u određenom obliku.
- *Kontroler* reagira na korisnički unos i interagira s objektima podatkovnog *modela*. *Kontroler* prima ulaz, po želji ga potvrđuje i zatim prosljeđuje ulaz *modelu*.

---

## Model, pogled i adapter

Engl. *model-view-adapter*, kraće MVA. Prema [Wikipediji](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93adapter):

- za razliku od MVC-a, spaja linearno model i pogled putem adaptera
- model ne komunicira s pogledom osim preko adaptera
- model ne zna koji pogledi postoje
- pogled ne zna koji modeli postoje

![Model, view, and adapter bg 95% right:45%](https://khanlou.com/images/MVA.png)

---

## Model, pogled i prezenter

Engl. *model-view-presenter*, kraće MVP. Prema [Wikipediji](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93presenter):

- MVP je derivat MVC-a, prezenter zamjenuje kontroler
- pogled je pasivan element
- prezenter sadrži svu prezentacijsku logiku

![Model, view, and presenter bg 70% right:40%](https://upload.wikimedia.org/wikipedia/commons/d/dc/Model_View_Presenter_GUI_Design_Pattern.png)

---

## Model, pogled i pogled modela

Engl. *model-view-viewmodel*, kraće MVVM. Prema [Wikipediji](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93viewmodel):

- pogled modela je sličan kao prezenter, ali ne koristi pogled izravno
- pogled koristi osvježava svojstva pogleda modela korištenjem bindera

![Model, view, and viewmodel](https://upload.wikimedia.org/wikipedia/commons/8/87/MVVMPattern.png)

---

## MVC, MVA, MVP i MVVM

- u suštini vrlo slični, razlike među njima nisu velike
    - lako je prijeći iz jednog od pristupa u drugi
- odabir pristupa uglavnom vršimo odabirom okvira u kojem radimo
    - vještina rada u jednom MVC okviru je prenosiva u drugi MVC okvir, npr. iz ASP.NET MVC u Ruby on Rails ili Laravel

---

## Primjena uzoraka dizajna u razvoju softvera

Uzorci dizajna koriste se po potrebi. Primjerice:

- okvir u kojem je vaša aplikacija napravljena može biti MVC,
- možete iskoristiti adapter za pozivanje neke biblioteke čiji API ne paše,
- za stvaranje objekata na temelju modela možete koristiti apstraktnu tvornicu
- u kontroleru možete iskoristiti promatrača u radu s modelima
- itd.

---
marp: true
author: Vedran Miletić
title: Statička analiza programskog koda web aplikacija
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Statička analiza programskog koda web aplikacija

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Statička analiza programa

Prema [Wikipediji](https://en.wikipedia.org/wiki/Static_program_analysis):

Statička analiza programa (engl. *static program analysis*) je analiza programa koja se izvodi bez njegovog pokretanja. Kod skriptnih jezika uglavnom se izvodi na izvornom kodu, a kod jezika s program-prevoditeljem na objektnom kodu.

Statičku analizu provode programski alati. U suprotnom, kad programer izučava kod, to nazivamo [razumijevanjem programa](https://en.wikipedia.org/wiki/Program_comprehension) (engl. *program comprehensions*), [recenzijom koda](https://en.wikipedia.org/wiki/Code_review) (engl. *code review*), [inspekcijom softvera](https://en.wikipedia.org/wiki/Software_inspection) (engl. *software inspection*) ili [šetnjom kroz softver](https://en.wikipedia.org/wiki/Software_walkthrough) (engl. *software walkthrough*).

---

## Alati za statičku analizu

- [Pylint](https://pylint.org/) (Python)
    - korišten od strane Visual Studio Codea kad je instalirana [Microsoftova ekstenzija za Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) ([detaljan opis konfiguracije i načina korištenja](https://code.visualstudio.com/docs/python/linting))
- [ESLint](https://eslint.org/) (JavaScript)
    - može se integrirati u Visual Studio Code korištenjem [ekstenzije ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint)
- [RuboCop](https://rubocop.org/) (Ruby)
- [Phan](https://github.com/phan/phan) i [Psalm](https://psalm.dev/) (PHP)
- [Clippy](https://github.com/rust-lang/rust-clippy) (Rust)
- mnogi drugi alati za mnoge druge jezike, uključujući i markup jezike, npr. [html-lint](https://www.npmjs.com/package/html-lint) za HTML, [stylelint](https://www.npmjs.com/package/stylelint) za CSS te [markdownlint](https://github.com/DavidAnson/markdownlint) za Markdown, postoji i [ekstenzija za Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint) koja je intenzivno korištena kod izrade ovih prezentacija

---

## Komercijalna rješenja za statičku analizu

- uglavnom podržavaju više jezika, fokusirana na raspoznavanje i prijavu složenijih problema neovisnih o jeziku
- često su [open core](https://opensource.com/article/21/11/open-core-vs-open-source) softveri: osnovna funkcionalnost je otvorenog koda, a dodatna funkcionalnost je jedino dio softvera koji se isporučuje kao usluga
- primjeri: [Codacy](https://www.codacy.com/) i [SonarQube](https://www.sonarqube.org/)

---

## Statički i dinamički tipovi podataka

TypeScript vs JavaScript

PHP `strict_types` TODO

---

## ESLint: globalni objekti ne mogu biti pozvani ❌

``` javascript
var math = Math();

var newMath = new Math();

var json = JSON();

var newJSON = new JSON();

var reflect = Reflect();

var newReflect = new Reflect();

var atomics = Atomics();

var newAtomics = new Atomics();
```

---

## ESLint: globalni objekti ne mogu biti pozvani ✅

``` javascript
function area(r) {
    return Math.PI * r * r;
}

var object = JSON.parse("{}");

var value = Reflect.get({ x: 1, y: 2 }, "x");

var first = Atomics.load(foo, 0);
```

---

## Argumenti funkcije mogu biti navedeni samo jednom ❌ i ✅

``` javascript
function foo(a, b, a) {
    console.log("value of the second a:", a);
}

var bar = function (a, b, a) {
    console.log("value of the second a:", a);
};
```

``` javascript
function foo(a, b, c) {
    console.log(a, b, c);
}

var bar = function (a, b, c) {
    console.log(a, b, c);
};
```

---

## Uvjeti u if-u mogu biti navedeni samo jednom ❌

``` javascript
if (isSomething(x)) {
    foo();
} else if (isSomething(x)) {
    bar();
}

if (n === 1) {
    foo();
} else if (n === 2) {
    bar();
} else if (n === 3) {
    baz();
} else if (n === 2) {
    quux();
} else if (n === 5) {
    quuux();
}
```

---

## Uvjeti u if-u mogu biti navedeni samo jednom ✅

``` javascript
if (isSomething(x)) {
    foo();
} else if (isSomethingElse(x)) {
    bar();
}

if (n === 1) {
    foo();
} else if (n === 2) {
    bar();
} else if (n === 3) {
    baz();
} else if (n === 4) {
    quux();
} else if (n === 5) {
    quuux();
}
```

---

## Što je sve pogrešno u kodu ispod? (1/4)

``` php
<?php

function takesAnInt(int $i) : array<string> {
  return [$i, "hello"];
}

$data = ["some text", 5];
takesAnInt($data[0]);

$condition = rand(0, 5);
if ($condition) {
} elseif ($condition) {
}
```

---

## Što je sve pogrešno u kodu ispod? (2/4)

``` php
<?php

abstract class Road {
  abstract static function fix() : void;
}

class TarmacRoad extends Road {}

new Road();

Road::fix();
```

---

## Što je sve pogrešno u kodu ispod? (3/4)

``` php
<?php

class Audi extends Car {}
class Audi {}

final class VW {}
class VWID extends VW {}

$dobroNjemackoAuto = new Audi();
```

---

## Što je sve pogrešno u kodu ispod? (4/4)

``` php
<?php

function sum_ints(int $a, int $b): int {
    return $a + $b;
}

function do_exit() : void {
    exit();
}

sum_ints(x: 0, y: 1);

$exit_code = do_exit();
```

---
marp: true
author: Vedran Miletić
title: Faktori razvoja web aplikacija
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Faktori razvoja web aplikacija

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

![Finding a good reason for not doing a good job can be a full time job bg 95% left](https://www.monkeyuser.com/assets/images/2019/154-everyday-excuses.png)

## Svakodnevne isprike

Izvor: [Everyday Excuses](https://www.monkeyuser.com/2019/everyday-excuses/) (MonkeyUser, 22nd October 2019)

---

## Web sjedište [The Twelve-Factor App](https://12factor.net/)

- skup pravila za razvoj web aplikacija namijenjenih za izvođenje u oblaku
    - knjiga na engleskom jeziku namijenjena za čitanje online
    - autor izvorne verzije [Adam Wiggins](https://adamwiggins.com/)
    - sadržaj danas održava [Heroku](https://www.heroku.com/) (pružatelj platformi kao usluga u oblaku)
- sadržaj preveden na brojne jezike, ali ne i na hrvatski
    - postoji [radna verzija hrvatskog prijevoda](../../arhiva/web-sjedista/12factor-net/intro.md) u izvedbi [Grupe za aplikacije i usluge na eksaskalarnoj istraživačkoj infrastrukturi](../index.md)
- ima [180k pogleda mjesečno](https://www.similarweb.com/website/12factor.net/)

---

## 1. faktor: Baza izvornog koda ([Codebase](https://12factor.net/codebase))

🙋 **Pitanje:** Slijede li ovaj faktor:

- [Moodle](https://moodle.org/) ([MoodleNet](https://moodle.net/), [Merlin](https://moodle.srce.hr/), [loomen](https://loomen.carnet.hr) itd.)
- [WordPress](https://wordpress.org/) ([WordPress.com](https://wordpress.com/), [Sveučilište u Rijeci](https://uniri.hr/), [Njuškalo blog](https://blog.njuskalo.hr/), [Novi list](https://www.novilist.hr/) itd.)
- [SofaScore](https://www.sofascore.com/)
- [IonicaBizau/react-todo-app](https://github.com/IonicaBizau/react-todo-app)
- [Turris OS](https://www.turris.com/en/turris-os/) ([How to try future releases of Turris OS?](https://docs.turris.cz/geek/testing/))
- [Facebook](https://developers.facebook.com/products/) ([Facebook](https://www.facebook.com/), [Facebook Beta](https://www.beta.facebook.com/))
- studentski projekti

---

## 2. faktor: Zavisnosti ([Dependencies](https://12factor.net/dependencies), 1/2)

Sustavi upravljanja zavisnostima:

- [cpanminus](https://metacpan.org/pod/App::cpanminus) za Perl
- [RubyGems](https://rubygems.org/) za Ruby
- [pip](https://pip.pypa.io/) za Python
- [Composer](https://getcomposer.org/) za PHP
- [npm](https://www.npmjs.com/) i [Yarn](https://yarnpkg.com/) za Node.js
- [Maven](https://maven.apache.org/) za Javu
- [NuGet](https://www.nuget.org/) za C#
- [Hex](https://hex.pm/) za Elixir
- [Cargo](https://doc.rust-lang.org/cargo/) za Rust
- itd.

---

## 2. faktor: Zavisnosti (2/2)

🙋 **Pitanje:** Slijede li ovaj faktor:

- [Flarum](https://flarum.org/) ([composer.json](https://github.com/flarum/core/blob/master/composer.json))
- [D3](https://github.com/d3) ([package.json](https://github.com/d3/d3/blob/master/package.json))
- [Electron](https://www.electronjs.org/) ([package.json](https://github.com/electron/electron/blob/master/package.json))
- [Elixir Companies](https://github.com/beam-community/elixir-companies) ([mix.exs](https://github.com/beam-community/elixir-companies/blob/master/mix.exs))
- [alexurquhart/flask-webapp](https://github.com/alexurquhart/flask-webapp) ([requirements.txt](https://github.com/alexurquhart/flask-webapp/blob/master/requirements.txt))
- studentski projekti

---

## 3. faktor: Konfiguracija ([Configuration](https://12factor.net/config))

🙋 **Pitanje:** Slijede li ovaj faktor:

- [Moodle](https://moodle.org/) ([konfiguracija](https://docs.moodle.org/38/en/Configuration_file) i [primjer konfiguracijske datoteke](https://github.com/moodle/moodle/blob/master/config-dist.php))
- [Odoo](https://www.odoo.com/) ([konfiguracija](https://www.odoo.com/documentation/14.0/reference/cmdline.html#configuration-file))
- [Trac](https://trac.edgewall.org/) ([konfiguracija](https://trac.edgewall.org/wiki/TracIni))
- [Django](https://www.djangoproject.com/) ([konfiguracija](https://docs.djangoproject.com/en/3.2/topics/settings/))
- [Laravel](https://laravel.com/) ([primjer konfiguracijske datoteke](https://github.com/laravel/laravel/blob/master/.env.example))
- [Ruby on Rails](https://rubyonrails.org/) ([konfiguracija](https://guides.rubyonrails.org/configuring.html))
- studentski projekti

---

## 4. faktor: Prateće usluge ([Backing services](https://12factor.net/backing-services))

🙋 **Pitanje:** Slijede li ovaj faktor:

- [Moodle](https://moodle.org/) ([konfiguracija](https://docs.moodle.org/38/en/Configuration_file) i [primjer konfiguracijske datoteke](https://github.com/moodle/moodle/blob/master/config-dist.php))
- [Odoo](https://www.odoo.com/) ([konfiguracija](https://www.odoo.com/documentation/14.0/reference/cmdline.html#configuration-file))
- [Trac](https://trac.edgewall.org/) ([konfiguracija](https://trac.edgewall.org/wiki/TracIni))
- [Django](https://www.djangoproject.com/) ([konfiguracija](https://docs.djangoproject.com/en/3.2/topics/settings/))
- [Laravel](https://laravel.com/) ([primjer konfiguracijske datoteke](https://github.com/laravel/laravel/blob/master/.env.example))
- [Ruby on Rails](https://rubyonrails.org/) ([konfiguracija](https://guides.rubyonrails.org/configuring.html))
- studentski projekti

---

## 5. faktor: Izgradnja, objava, pokretanje ([Build, release, run](https://12factor.net/build-release-run))

🙋 **Pitanje:** Slijede li ovaj faktor:

- [Canvas LMS](https://www.instructure.com/canvas) ([tags](https://github.com/instructure/canvas-lms/tags))
- [Discourse](https://www.discourse.org/) ([tags](https://github.com/discourse/discourse/tags))
- [Ghost](https://ghost.org/) ([releases](https://github.com/TryGhost/Ghost/releases))
- studentski projekti

---

## 6. faktor: Procesi ([Processes](https://12factor.net/processes))

🙋 **Pitanje:** Slijede li ovaj faktor:

- neke od ranije spomenutih web aplikacija (što biste gledali?)
- studentski projekti

---

## 7. faktor: Povezivanje na vrata ([Port binding](https://12factor.net/port-binding))

🙋 **Pitanje:** Slijede li ovaj faktor:

- [Canvas LMS](https://www.instructure.com/canvas) ([Production Start](https://github.com/instructure/canvas-lms/wiki/Production-Start))
- [Joomla!](https://www.joomla.org/) ([Installing Joomla](https://docs.joomla.org/Installing_Joomla))
- [Ghost](https://ghost.org/) ([How to install Ghost](https://ghost.org/docs/install/))
- studentski projekti

---

## 8. faktor: Konkurentnost ([Concurrency](https://12factor.net/concurrency))

🙋 **Pitanje:** Slijede li ovaj faktor:

- [Jitsi Meet](https://jitsi.org/) ([Using Jitsi Meet for self-hosted video conferencing](https://puppet.com/blog/using-jitsi-meet-for-self-hosted-video-conferencing/))
- [MediaWiki](https://www.mediawiki.org/) ([Wikipedia webrequest flow](https://commons.wikimedia.org/wiki/File:Wikipedia_webrequest_flow_2020.png))
- [OpenStreetMap](https://www.openstreetmap.org/) ([Component overview](https://wiki.openstreetmap.org/wiki/Component_overview))
- studentski projekti

---

## 9. faktor: Jednokratna upotreba ([Disposability](https://12factor.net/disposability))

🙋 **Pitanje:** Slijede li ovaj faktor:

- [WordPress](https://wordpress.org/) ([Where Does WordPress Store Images on Your Site?](https://www.wpbeginner.com/beginners-guide/where-does-wordpress-store-images-on-your-site/))
- [Moodle](https://moodle.org/) ([File Storage Plugintype](https://docs.moodle.org/dev/File_Storage_Plugintype))
- studentski projekti

---

![test bg 68% left](https://www.monkeyuser.com/assets/images/2021/220-measure-twice-cut-once.png)

## Mjeri dvaput, reži jednom

Izvor: [Measure Twice, Cut Once](https://www.monkeyuser.com/2021/measure-twice-cut-once/) (MonkeyUser, 15th June 2021)

---

## 10. faktor: Paritet razvoja/produkcije ([Dev/prod parity](https://12factor.net/dev-prod-parity))

🙋 **Pitanje:** Slijede li ovaj faktor:

- [phpBB](https://www.phpbb.com/) ([phpBB Demo](https://www.phpbb.com/demo/))
- [Discourse](https://www.discourse.org/) ([Demo](https://try.discourse.org/))
- [Facebook](https://www.facebook.com/) ([Facebook Beta](https://www.beta.facebook.com/))
- studentski projekti

---

## 11. faktor: Zapisnici ([Logs](https://12factor.net/logs))

🙋 **Pitanje:** Slijede li ovaj faktor:

- [Moodle](https://moodle.org/) ([Moodle](https://docs.moodle.org/311/en/Logging))
- [Odoo](https://www.odoo.com/) ([CLI: odoo-bin](https://www.odoo.com/documentation/master/developer/misc/other/cmdline.html))
- studentski projekti

---

## 12. faktor: Administrativni procesi ([Administrativni procesi](https://12factor.net/admin-processes))

🙋 **Pitanje:** Slijede li ovaj faktor:

- [Phabricator](https://github.com/phacility/phabricator/tree/master/scripts)
- [Canvas LMS](https://github.com/instructure/canvas-lms/tree/master/script)
- studentski projekti

---

## Umjesto zaključka

- za samostalno istraživanje: Kevin Hoffman, [Beyond the Twelve-Factor App: Exploring the DNA of Highly Scalable, Resilient Cloud Applications](https://tanzu.vmware.com/content/blog/beyond-the-twelve-factor-app), [O'Reilly](https://www.oreilly.com/library/view/beyond-the-twelve-factor/9781492042631/), 2016.
- dvanaestofaktorske aplikacije => oblaku urođene aplikacije ([Cloud Native Computing Foundation (CNCF)](https://www.cncf.io/))
    - [članovi](https://www.cncf.io/about/members/)
    - [projekti](https://www.cncf.io/projects/)
    - [rječnik](https://glossary.cncf.io/)

---
author: Vedran Miletić
---

# Preporuke za pisanje završnih i diplomskih radova

## Akademsko pisanje

Završni i diplomski radovi su akademski radovi, a prilikom pisanja akademskih radova slijedimo uvriježena stilska i strukturna pravila kako bi ti radovi bili čitljivi i potencijalno korisni i drugim članovima akademske zajednice. O pisanju radova [dr. sc. Vanja Pupovac](https://portal.uniri.hr/Portfelj/1712) na [mrežnim stranicama Akademsko pisanje](https://www.akademsko-pisanje.uniri.hr/o-nama/) i u istoimenoj knjižici kaže sljedeće:

> Akademsko pisanje je oblik komunikacije u akademskoj zajednici. Postoje različiti oblici akademskog pisma, a neki od najpoznatijih su esej, seminarski rad, završni, diplomski ili doktorski rad, te stručni ili znanstveni rad. Svrha akademskog pisanje je savladavanje i sažimanje znanja o pojedinoj temi, proširivanje rasprava (novim informacijama ili mišljenjima), prenošenje informacija (znanstvenicima, stručnjacima ili javnosti) i ovjeravanje znanja.
>
> Osnovna karakteristika akademskog rada je logički način razmišljanja; ideje moraju biti sustavno i jednostavno prezentirane, a mišljenja moraju biti potkrijepljena relevantnim i uvjerljivim razlozima i dokazima. No, to nije uvijek lako ostvarivo, stoga proces pisanja akademskog rada dijelimo u četiri faza. Na mrežnim stranicama [Proces pisanja](https://www.akademsko-pisanje.uniri.hr/proces-pisanja/) donosimo pregled osnovnih pravila i aktivnosti za svaku fazu u procesu pisanja.
>
> Pisanjem akademskog rada uključujemo se u aktualnu raspravu o odabranoj temi. Za uspješno uključivanje u raspravu potrebno je upoznati se s već postojećim činjenicama, idejama i argumentima čitajući relevantnu stručnu ili znanstvenu literaturu. Na taj način osigurat ćemo da vlastite ideje razvijemo na čvrstim znanstvenim temeljima i iskazat ćemo poštovanje prema iskusnijim znanstvenicima. Povjerenje zajednice u vlastiti rad zadobit ćemo preciznim citiranjem i referiranjem izvora što je pojašnjeno na stranici [Rad s izvorima](https://www.akademsko-pisanje.uniri.hr/rad-s-izvorima/). Kredibilitet i vjerodostojnost autora se sporo gradi u akademskoj zajednici, ali lako narušava, stoga savjetujemo da proučite materijale prezentirane na stranici [Akademska čestitost](https://www.akademsko-pisanje.uniri.hr/akademska-cestitost/).

Za više informacija slijedite poveznice u tekstu iznad i proučite [mrežne stranice Akademsko pisanje](https://www.akademsko-pisanje.uniri.hr/) ili [preuzmite knjižicu Akademsko pisanje u formatu PDF](../../../downloads/knjizica-akademsko-pisanje.pdf).

## Softverski alati

Za pisanje radova koristite [LibreOffice](https://www.libreoffice.org/) i format [OpenDocument](https://opendocumentformat.org/). Ako preferirate pisanje markupa umjesto korištenja alata tipa [WYSIYG](https://en.wikipedia.org/wiki/WYSIWYG), umjesto LibreOfficea i OpenDocumenta možete koristiti neki od sljedećih alata i formata:

- [Markdown](https://commonmark.org/), [Pandoc](https://pandoc.org/) i proširenja navedena u [objavi Jaana Tollandera de Balscha](https://jaantollander.com/post/scientific-writing-with-markdown/) (preporučeni uređivači: [HackMD](https://hackmd.io/) i [Visual Studio Code](https://code.visualstudio.com/)),
- [LaTeX](https://www.latex-project.org/) i [Edvin Močibobov](https://edvin.me/) [predložak za završni i diplomski rad](https://bitbucket.org/emocibob/latex-predlozak-za-diplomski-i-zavrsni-rad) (preporučeni uređivači: [TeXstudio](https://texstudio.org/) i [Texmaker](https://www.xm1math.net/texmaker/)),
- [reStructuredText](https://docutils.sourceforge.io/rst.html), [Sphinx](https://www.sphinx-doc.org/en/master/) i [Jeff Terraceov](https://jeffterrace.com/) [snop ekstenzija za akademske radove](https://jterrace.github.io/sphinxtr/) (preporučeni uređivač: [Visual Studio Code](https://code.visualstudio.com/) s proširenjem [reStructuredText](https://marketplace.visualstudio.com/items?itemName=lextudio.restructuredtext)) ili
- [AsciiDoc](https://asciidoc.org/) i [Asciidoctor](https://asciidoctor.org/) (preporučeni uređivač: [Visual Studio Code](https://code.visualstudio.com/) s proširenjem [AsciiDoc](https://marketplace.visualstudio.com/items?itemName=asciidoctor.asciidoctor-vscode)).

LibreOffice i LaTeX rade izvrsne PDF-ove (Sphinx također stvara PDF korištenjem LaTeX-a). Dodatno mogu, ako je potrebno, napraviti [PDF/A](https://en.wikipedia.org/wiki/PDF/A), varijantu PDF-a prikladnu za arhiviranje i dugoročno čuvanje digitalnih dokumenata.

Za upravljanje navodima možete koristiti [Zotero](https://www.zotero.org/) (koji ima i plug-in za LibreOffice) i/ili [BibTeX](http://www.bibtex.org/). Koristite [stil citiranja IEEE](https://ieee-dataport.org/sites/default/files/analysis/27/IEEE%20Citation%20Guidelines.pdf).

Kojom god temom da se bavite, svakako koristite slobodne softvere otvorenog koda kad god možete. Na će taj način onaj koji koristi vaš rad za vlastito učenje moći lako doći do softvera s kojim u radu radite.

## Korištenje Pandocovog dijalekta Markdowna

### Instalacija potrebnih paketa

Na [Arch Linuxu](https://archlinux.org/) i [distribucijama temeljenim na Archu](https://wiki.archlinux.org/title/Arch-based_distributions) (kao što su [Manjaro](https://manjaro.org/), [Garuda Linux](https://garudalinux.org/) i [EndeavourOS](https://endeavouros.com/)) potrebno je instalirati grupe paketa [texlive](https://archlinux.org/groups/x86_64/texlive/) i [texlive-lang](https://archlinux.org/groups/x86_64/texlive-lang/) te pakete [pandoc-cli](https://archlinux.org/packages/extra/x86_64/pandoc-cli/) i [pandoc-crossref](https://archlinux.org/packages/extra/x86_64/pandoc-crossref/). To možemo učiniti naredbom:

``` shell
$ sudo pacman -S pandoc-cli pandoc-crossref texlive texlive-lang
(...)
```

### Stvaranje potrebnih datoteka

#### Tekst rada (Markdown)

Uzmimo da je tekst rada je pohranjen u datoteci `tekst.md`. On je napisan u [Pandocovom dijalektu Markdowna](https://pandoc.org/MANUAL.html#pandocs-markdown) i dodatno sadrži zaglavlje s metapodacima u formatu YAML koji se koriste u predlošku:

``` markdown
---
sveučilište: Sveučilište u Rijeci
fakultet: Fakultet informatike i digitalnih tehnologija
studij: Sveučilišni prijediplomski studij Informatika
autor: Faust Vrančić
naslov: Simulacija rada padobrana korištenjem superračunala Bura
vrsta: Završni rad
mentor: prof. dr. sc. Antun Vrančić
mjesto: Rijeka
datum: 1. srpnja 1571.
ključne-riječi:
  - padobran
  - superračunalo
  - simulacija
  - optimizacija
  - mehanika fluida
sažetak: |
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
---

## Uvod

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo [@Hajn01]. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt [@Samp05; @Sim03]. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?

## Glavni

## Dio

## Ima

## Poglavlja

## Prema

## Potrebi

## Zaključak

At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga [@Rsoft]. Et harum quidem rerum facilis est et expedita distinctio. Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit quo minus id quod maxime placeat facere possimus, omnis voluptas assumenda est, omnis dolor repellendus. Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint et molestiae non recusandae. Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus asperiores repellat [@Wirt99; @Will93; @Jone12].

## Literatura
```

Poglavlje *Literatura* bez sadržaja mora biti posljednje jer služi kao mjesto gdje će biti umetnut generirani popis literature.

#### Bibliografija (BibTeX)

U Pandocovom dijalektu Markdowna sintaksa oblika `[@AutorGodina]` se koristi za citiranje izvora. Bibliografija se pohranjuje u datoteci `bibliografija.bib` u formatu BibTeX:

``` bibtex
@book{Hajn01,
    title     = {Medical Image Registration},
    editor    = {J. V. Hajnal and D. Hill and D. J. Hawkes},
    publisher = {CRC Press LLC},
    address   = {Boca Raton, USA},
    year      = {2001},
}

@incollection{Samp05,
    author    = {M. P. Sampat and M. K. Markey and A. C. Bovik},
    editor    = {A.C. Bovik},
    booktitle = {Handbook of Image and Video Processing},
    title     = {{Computer-Aided Detection and Diagnosis in Mammography}},
    pages     = {1195-1217},
    publisher = {Elsevier Academic Press},
    year      = {2005},
    address   = {Amsterdam},
}

@article{Sim03,
    author  = {T. Sim and S. Baker and M. Bsat},
    title   = {{The CMU Pose, Illumination, and Expression Database}},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    volume  = {25},
    number  = {12},
    pages   = {1615-1618},
    year    = {2003},
    month   = {December},
}

@conference{Wirt99,
    author    = {M. A. Wirth and C. Choi and A. Jennings},
    title     = {{A Nonrigid-Body Approach to Matching Mammograms}},
    booktitle = {{Seventh International Conference on Image Processing and Its Applications}},
    address   = {Manchester, UK},
    pages     = {484-488},
    year      = {1999},
    month     = {July},
}

@phdthesis{Will93,
    author  = {J. Williams},
    title   = {{Narrow-band Analyzer}},
    year    = {1993},
    school  = {Harvard University},
    address = {Cambridge, MA, SAD},
}

@misc{Jone12,
    author = {J. Jones},
    title  = {Networks},
    url    = {http://www.atm.com},
    year   = {(28. srpnja 2012.)},
}

@manual{Rsoft,
    title        = {{R: A Language and Environment for Statistical Computing}},
    author       = {{R Core Team}},
    organization = {R Foundation for Statistical Computing},
    address      = {Vienna, Austria},
    year         = 2012,
    url          = {http://www.R-project.org}
}
```

Osim ručne pripreme ove datoteke, moguće je [skup](https://www.zotero.org/support/collections_and_tags) prikupljenih referenci u [Zoteru](https://www.zotero.org/support/) [izvesti](https://www.zotero.org/support/kb/exporting) kao datoteku u formatu BibTeX.

#### Predložak (LaTeX)

Datoteka `predlozak.latex` sadrži LaTeX predložak u kojem Pandoc zamjenjuje varijablu `$body$` tekstom rada, a sve ostale varijable istoimenim metapodacima iz zaglavlja:

``` latex
% This is the main file for the template for doctoral thesis at
% University of Rijeka. It is based on the template for doctoral thesis at
% University of Zagreb, Faculty of Electrical Engineering and Computing
% in Zagreb, Croatia.
% Initial version was created in May 2023.

% Author: Jelena Bozek, jelena.bozek@fer.hr
% Modified for University of Rijeka: Vedran Miletic, vmiletic@inf.uniri.hr


%%%%%%%%%%%%%%%%%%%%%%%%% POSTAVKE / SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\documentclass[12pt, oneside, a4paper]{book}
\usepackage{etex}
\usepackage{xcolor}
$if(graphics)$
\usepackage[pdftex]{graphicx}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
\makeatother
% Scale images if necessary, so that they will not overflow the page
% margins by default, and it is still possible to overwrite the defaults
% using explicit options in \includegraphics[width, height, ...]{}
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
% Set default figure placement to htbp
\makeatletter
\def\fps@figure{htbp}
\makeatother
$endif$
\usepackage{rotating}
\usepackage{epsfig}
\usepackage{epstopdf}
% required for printing index
% use \index{name} in text
%\usepackage{makeidx}
%\makeindex
% required for printing nomenclature
% use \nomenclature{symbol}{description} in text
%\usepackage{nomencl}
%\makenomenclature
%\renewcommand{\nomname}{Popis oznaka}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{cmap}
\usepackage[croatian]{babel}
\usepackage{ae}
\usepackage[unicode]{hyperref}
\usepackage{mathptmx}
\usepackage{amscd}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{amsfonts}

\usepackage[left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm]{geometry}
\usepackage{setspace}
\onehalfspacing
\usepackage[skip=3pt]{parskip}
\usepackage{fancyhdr} % setting up header and position of page numbers
\pagestyle{fancyplain}
\fancyhf{}
\lhead{\nouppercase{\fancyplain{}{\leftmark}}}
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\rfoot{\thepage}

\usepackage{hhline}
\usepackage{enumerate}
\usepackage{delarray}
$if(tables)$
\usepackage{longtable,booktabs,array} % packages for some table properties
$if(multirow)$
\usepackage{multirow}
$endif$
\usepackage{calc} % for calculating minipage widths
$if(beamer)$
\usepackage{caption}
% Make caption package work with longtable
\makeatletter
\def\fnum@table{\tablename~\thetable}
\makeatother
$else$
% Correct order of tables after \paragraph or \subparagraph
\usepackage{etoolbox}
\makeatletter
\patchcmd\longtable{\par}{\if@noskipsec\mbox{}\fi\par}{}{}
\makeatother
% Allow footnotes in longtable head/foot
\IfFileExists{footnotehyper.sty}{\usepackage{footnotehyper}}{\usepackage{footnote}}
\makesavenoteenv{longtable}
$endif$
$endif$
\usepackage{tabularx} % package that allows dynamical changing table cell width
\usepackage{multirow} % package that enables multiple rows in a table
\usepackage[bf, font=small]{caption}
\usepackage[labelfont=small, font=small]{subcaption}
\usepackage{wasysym}
\usepackage{subeqnarray}
\usepackage{aeguill}
\usepackage{pdflscape} % setting page into landscape view
\usepackage{enumitem} % for itemize lists
\setlist{nolistsep} % setting for itemize lists

\renewcommand{\thefootnote}{\fnsymbol{footnote}} % to get unnumbered footnotes
\renewcommand{\arraystretch}{1.5} % stretching row height

\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
$if(highlighting-macros)$
$highlighting-macros$
$endif$
$if(csl-refs)$
\newlength{\cslhangindent}
\setlength{\cslhangindent}{1.5em}
\newlength{\csllabelwidth}
\setlength{\csllabelwidth}{3em}
\newlength{\cslentryspacingunit} % times entry-spacing
\setlength{\cslentryspacingunit}{\parskip}
\newenvironment{CSLReferences}[2] % #1 hanging-ident, #2 entry spacing
 {% don't indent paragraphs
  \setlength{\parindent}{0pt}
  % turn on hanging indent if param 1 is 1
  \ifodd #1
  \let\oldpar\par
  \def\par{\hangindent=\cslhangindent\oldpar}
  \fi
  % set entry spacing
  \setlength{\parskip}{#2\cslentryspacingunit}
 }%
 {}
\usepackage{calc}
\newcommand{\CSLBlock}[1]{#1\hfill\break}
\newcommand{\CSLLeftMargin}[1]{\parbox[t]{\csllabelwidth}{#1}}
\newcommand{\CSLRightInline}[1]{\parbox[t]{\linewidth - \csllabelwidth}{#1}\break}
\newcommand{\CSLIndent}[1]{\hspace{\cslhangindent}#1}
$endif$
\usepackage[square, numbers, sort]{natbib}
% change the name of Bibliography heading into "Literatura"
\addto\captionscroatian{%
  \renewcommand{\bibname}{Literatura}
}

% Adding a dot after chapter number in TOC
\let\savenumberline\numberline
\def\numberline#1{\savenumberline{#1.}}

% Adding dots after chapter titles to page number in TOC
\makeatletter
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
  \addpenalty{-\@highpenalty}%
  \vskip 1.0em \@plus\p@
  \setlength\@tempdima{1.5em}%
  \begingroup
  \parindent \z@ \rightskip \@pnumwidth
  \parfillskip -\@pnumwidth
  \leavevmode \bfseries
  \advance\leftskip\@tempdima
  \hskip -\leftskip
  #1\nobreak\normalfont\leaders\hbox{\(\m@th
    \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
    mu\)}\hfill\nobreak\hb@xt@\@pnumwidth{\hss #2}\par
  \penalty\@highpenalty
  \endgroup
  \fi}
\makeatother

% adjust the line spacing in a matrix
\makeatletter
\renewcommand*\env@matrix[1][\arraystretch]{%
  \edef\arraystretch{#1}%
  \hskip -\arraycolsep
  \let\@ifnextchar\new@ifnextchar
  \array{*\c@MaxMatrixCols c}}
\makeatother

% remove footer (page number) from TOC, list of figures and list of tables
\AtBeginDocument{\addtocontents{toc}{\protect\thispagestyle{empty}}}
\AtBeginDocument{\addtocontents{lof}{\protect\thispagestyle{empty}}}
\AtBeginDocument{\addtocontents{lot}{\protect\thispagestyle{empty}}}


\begin{document}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\frontmatter

%%%%%%%%%%%%%%%%%%%% NASLOVNICA / FRONT COVER PAGE %%%%%%%%%%%%%%%%%%%%%%%%
% \begin{titlepage}
%     \fontsize{16pt}{20pt}\selectfont
%     \fontfamily{phv}\fontseries{mc}\selectfont
%     \newgeometry{left=3cm,right=3cm,top=3cm,bottom=3cm}
%     \setlength{\intextsep}{0pt plus 0pt minus 0pt}

%     \begin{center}
%         {SVEUČILIŠTE U RIJECI} \\
%         {NOSITELJ/NOSITELJI STUDIJA} \\
%         \vspace{3cm}
%         Ime i prezime autora \\
%         \vspace{2cm}
%         {\fontsize{22pt}{22pt}\selectfont\textbf{NASLOV DOKTORSKOGA RADA}} \\
%         \vspace{2cm}
%         DOKTORSKI RAD \\
%         \vfill{Rijeka, godina.}
%     \end{center}
%     \restoregeometry
% \end{titlepage}

%%%%%%%%%%%%%%% PRVA UNUTARNJA STRANICA / FIRST INNER PAGE %%%%%%%%%%%%%%%%
\begin{titlepage}
    \fontsize{16pt}{20pt}\selectfont
    \fontfamily{phv}\fontseries{mc}\selectfont
    \newgeometry{left=3cm,right=3cm,top=3cm,bottom=3cm}
    \setlength{\intextsep}{0pt plus 0pt minus 0pt}

    \begin{center}
        {$sveučilište$} \\
        {$fakultet$} \\
    {$studij$} \\
        \vspace{3cm}
        $autor$ \\
        \vspace{2cm}
        {\fontsize{22pt}{22pt}\selectfont\textbf{$naslov$}} \\
        \vspace{2cm}
        $vrsta$ \\
        \vspace{5cm} % adjust this spacing if necessary
        Mentor: $mentor$ \\ % Mentor/mentori
        % Komentor: $komentor$ \\ % Komentor/komentori
        \vfill{$mjesto$, $datum$}
    \end{center}
    \restoregeometry
\end{titlepage}

%%%%%%%%%%%%%% DRUGA UNUTARNJA STRANICA / SECOND INNER PAGE %%%%%%%%%%%%%%%
% \begin{titlepage}
%   \fontsize{16pt}{20pt}\selectfont
%   \fontfamily{phv}\fontseries{mc}\selectfont
%   \newgeometry{left=3cm,right=3cm,top=3cm,bottom=3cm}
%   \setlength{\intextsep}{0pt plus 0pt minus 0pt}

%   \begin{center}
%     {UNIVERSITY OF RIJEKA} \\
%     {HOLDER/HOLDERS OF STUDY} \\
%     \vspace{3cm}
%     Author's name and surname \\
%     \vspace{2cm}
%     {\fontsize{22pt}{22pt}\selectfont\textbf{TITLE OF THE DOCTORAL THESIS}} \\
%     \vspace{2cm}
%     DOCTORAL THESIS \\
%     \vspace{5cm} % adjust this spacing if necessary
%     Supervisor/supervisors: title, name, surname, and institution of employment \\
%     Co-supervisor/co-supervisors: title, name, surname, and institution of employment \\
%     \vfill{Rijeka, year}
%   \end{center}
%   \restoregeometry
% \end{titlepage}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \begin{titlepage}
%   \newgeometry{left=3cm,right=3cm,top=3cm,bottom=3cm}
%   \setlength{\intextsep}{0pt plus 0pt minus 0pt}
%   \vspace*{8cm}
%   Mentor/mentori: titula, ime, prezime i institucija zaposlenja

%   Komentor/komentori: titula, ime, prezime i institucija zaposlenja

%   \vspace{8cm}
%   Doktorski rad obranjen je dana \line(1,0){252}\ u/na \line(1,0){424}, pred povjerenstvom u sastavu:

%   \begin{enumerate}
%     \item \line(1,0){397}
%     \item \line(1,0){397}
%     \item \line(1,0){397}
%     \item \line(1,0){397}
%     \item \line(1,0){397}
%   \end{enumerate}
%   \restoregeometry
% \end{titlepage}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% insert optional page with thanks (to funding source) or dedication
%\include{eg_thanks_dedication}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% insert page with abstract
%\include{eg_abstract}

\thispagestyle{empty}


\section*{Sažetak}


$sažetak$

\vspace{1cm}
\textbf{Ključne riječi}:
$for(ključne-riječi)$
$ključne-riječi$;
$endfor$

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% insert page with extended abstract
% prošireni sažetak na hrvatskom, ako rad nije pisan na tom jeziku
%\include{eg_prosireni_sazetak}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\clearpage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TOC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pagestyle{empty} % remove header/footer
\tableofcontents
\cleardoublepage % start new page

\pagestyle{fancyplain} % puts headers/footers back on


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\mainmatter
%%%%%%%%%%%%%%%%%%%%%%%% POGLAVLJA / CHAPTERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \include{eg_introduction}
% \include{eg_making}
% \include{eg_binding}
% \include{eg_sources}
% \include{eg_examples}

$body$

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\backmatter

%%%%%%%%%%%%%%%%%%%%%%% LITERATURA / BIBLIOGRAPHY %%%%%%%%%%%%%%%%%%%%%%%%%
%\addcontentsline{toc}{chapter}{Literatura}
%\bibliographystyle{IEEEtran}
%\bibliography{eg_biblio}
%\cleardoublepage % start new page

%%%%%%%%%%%%%%%%%%%%%%% POPIS OZNAKA / NOMENCLATURE %%%%%%%%%%%%%%%%%%%%%%%
% notation and list of symbols if needed
%\printnomenclature
%\cleardoublepage % start new page

%%%%%%%%%%%%%%%%%%%%%%%%%%% KAZALO POJMOVA / INDEX %%%%%%%%%%%%%%%%%%%%%%%%
% optional index
%\printindex
%\cleardoublepage % start new page

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% insert list of figures
%\listoffigures
%\cleardoublepage % start new page

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% insert list of tables
%\listoftables
%\cleardoublepage % start new page

\end{document}
```

Ovu datoteku nije potrebno mijenjati.

#### Stil citiranja (CSL)

Stil citiranja [IEEE with URL](https://citationsy.com/styles/ieee-with-url) koji ćemo koristiti može se preuzeti iz repozitorija [citation-style-language/styles](https://github.com/citation-style-language/styles) na GitHubu pomoću cURL-a:

``` shell
$ curl -O https://raw.githubusercontent.com/citation-style-language/styles/master/ieee-with-url.csl
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 14576  100 14576    0     0  71280      0 --:--:-- --:--:-- --:--:-- 71450
```

Ovu datoteku također nije potrebno mijenjati.

### Prevođenje u format PDF

Nakon što su stvorene sve četiri datoteke s odgovarajućim imenima i sadržajem u trenutnom direktoriju, prevođenje rada u format PDF korištenjem Pandoca (i LaTeX-a, kojeg će Pandoc pozvati za nas) izvodimo naredbom:

``` shell
$ pandoc --template=predlozak.latex --top-level-division=part -F pandoc-crossref -C --bibliography=bibliografija.bib --csl=ieee-with-url.csl -o rad.pdf tekst.md
(...)
```

---
author: Alen Hamzić, Darian Žeko, Matea Turalija
---

# Aritmetičke operacije

Aritmetičke operacije su neizostavan dio našeg svakodnevnog života, a upoznajemo se s njima još od rane dobi. Osnovne matematičke operacije koje nam omogućuju manipuliranje brojevima su zbrajanje, oduzimanje, množenje i dijeljenje. Ove operacije se često kombiniraju, što može dovesti do složenijih izraza koji se rješavaju prema određenim pravilima. S druge strane, računalo je stroj koji uglavnom izvodi računske operacije, a svi podaci u računalu su prikazani u binarnom obliku. Stoga, kako bismo razumjeli kako računalo radi, važno je poznavati osnovne računske operacije u binarnom sustavu.

## Operacija zbrajanja

Zbrajanje u binarnom zapisu osnovna je aritmetička operacija koja se koristi u digitalnim sustavima. Ova operacija se izvodi na dva binarna broja, gdje se svaki bit zbraja s odgovarajućim bitom drugog broja, uz eventualni prijenos iz prethodnog stupca. Prilikom zbrajanja koristit ćemo sljedeću tablicu zbrajanja binarnih brojeva:

|    |   |
| -: | - |
| $0 + 0$ | $0$ |
| $0 + 1$ | $1$ |
| $1 + 1$ | $\mathbf{1}$ $0$ ($0$ pišemo, $1$ prenosimo dalje) |
| $1 + 1 + 1$ | $\mathbf{1}$ $1$ ($1$ pišemo, $1$ prenosimo dalje) |

!!! admonition "Zadatak"
    Izračunaj u binarnom sustavu zbroj brojeva $1010001_{(2)}$ i $10101_{(2)}$.

**Rješenje:**

U prvom koraku ćemo nadopuniti broj nulama tako da imaju jednak broj znamenaka:

|   |    |
| - | -: |
| | $1010001$ |
| $+$ | $0010101$ |

Zatim zbrojimo binarne brojeve imajući na umu tablicu zbrajanja jednoznamenkastih binarnih brojeva:

|   |   |
| - | - |
| | $010001$ - prijenos |
| | $1010001$ |
| $+$ | $0010101$ |
| | $1100110$ - rezultat |

Rezultat možemo provjeriti pretvaranjem brojeva i rezultata u dekadski sustav:

$$1010001_{(2)} = 1 \cdot 2^6 + 1 \cdot 2^4 + 1 \cdot 2^0 = 64 + 16 + 1 = 81_{(2)},$$

$$10101_{(2)} = 1 \cdot 2^4 + 1 \cdot 2^2 + 1 \cdot 2^0 = 16 + 4 + 1 = 21_{(2)},$$

$$1100110_{(2)} = 1 \cdot 2^6 + 1 \cdot 2^5 + 1 \cdot 2^2 + 1 \cdot 2^1 = 64 + 32 + 4 + 2 = 102_{(2)}.$$

Budući da je $81 + 21 = 102$, rezultat je očito točan.

!!! admonition "Zadatak"
    1. Zbrojite sljedeće binarne brojeve i provjerite dobivene rezultate: $010101_{(2)} + 101_{(2)}, 1110_{(2)} + 10101110_{(2)}$.

## Operacija oduzimanja

Budući da s pomoću dvojnog komplementa često prikazujemo negativnu vrijednost broja, oduzimanje se svodi na zbrajanje s dvojnim komplementom umanjitelja.

!!! admonition "Zadatak"
    Oduzmi brojeve $6_{(10)}$ i $3_{(10)}$.

$$6-3 = 6 + (-3) = 110_{(2)} + 101_{(2)} = 𝟏011.$$

Prvi bit $1$ je preljev (engl. *Overflow*) i zanemaruje se. Dakle, za $6-3$ dobili smo $011$ što je binarno $3$.

!!! admonition "Zadatak"
    2. Oduzmite sljedeće dekadske brojeve u binarnom obliku pomoću dvojnog komplementa i provjerite dobivene rezultate: $5 - 1$, $10 - 6$, $7 - 2$.

## Operacija množenja

Binarno množenje slično je množenju decimalnih brojeva. Budući da su samo binarne znamenke uključene u binarno množenje, možemo množiti samo 0 i 1. Pravila za binarno množenje su sljedeća:

| Množenik | Množitelj | Rezultat |
| -------: | --------: | -------: |
| $0$ | $0$ | $0 \cdot 0 = 0$ |
| $0$ | $1$ | $0 \cdot 1 = 0$ |
| $1$ | $0$ | $1 \cdot 0 = 0$ |
| $1$ | $1$ | $1 \cdot 1 = 1$ |

!!! admonition "Zadatak"
    Pomnožite $100_{(2)}$ i $011_{(2)}$ (množenje brojeva $4$ i $3$).

**Rješenje:**

Pomnožimo krajnje lijevu znamenku množitelja $011$ sa svim znamenkama množenika $100$. Ponavljamo isti postupak za sve sljedeće znamenke množitelja pri tome pazeći da dobiveni rezultat pišemo u novi red s jednim pomaknutim mjestom u desno. Nakon toga, redove zbrojimo koristeći pravila binarnog zbrajanja kako bismo dobili konačni rezultat, odnosno umnožak.

$$
\begin{eqnarray*}
\underline{100 \cdot 011}\\
\hspace{-0.9cm}000\\
\hspace{-0.6cm} 100\\
\underline{\hspace{-0.45cm}+100}\\
\hspace{-0.3cm}1100
\end{eqnarray*}
$$

Provjerom dobivenog rezultata ($4 \cdot 3 = 12$) možemo se uvjeriti da je rezultat točan. Ista pravila množenja vrijede i za binarne brojeve s decimalnom točkom.

!!! admonition "Zadatak"
    1. Pomnožite brojeve $11011_{(2)}$ i $101_{(2)}$
    2. Pomnožite brojeve $1011.1_{(2)}$ i $110_{(2)}$

### Operacija dijeljenja

Algoritam za binarno dijeljenje također je sličan decimalnom dijeljenju, ali koristi pravila koja se primjenjuju na znamenke $0$ i $1$. Jednostavnost binarnog dijeljenja proizlazi iz činjenice da se koriste samo dvije znamenke. Operacije koje se koriste u postupku binarnog dijeljenja su binarno množenje i oduzimanje. Pravila binarnog dijeljenja koja koristimo su sljedeća:

| Dijeljenik | Dijelitelj | Rezultat |
| ---------: | ---------: | -------: |
| $0$ | $0$ | $0 : 0 = \text{nedjeljivo}$ |
| $1$ | $0$ | $1 : 0 = \infty$ |
| $0$ | $1$ | $0 : 1 = 0$ |
| $1$ | $1$ | $1 : 1 = 1$ |

!!! admonition "Zadatak"
    Podjelite broj $100111_{(2)}$ s $11_{(2)}$.

**Rješenje:**

Usporedimo dijeljenik $100111$ s dijeliteljem $11$. Kako je dijeljenik veći od dijelitelja, možemo uzeti prva tri lijeva bita dijeljenika i podijeliti ga s $11$ te zapisati rezultat na vrhu kao kvocjent, baš kao i kod decimalnog dijeljenja. Zatim oduzimamo rezultat od znamenke, zapišemo razliku ispod i spustimo sljedeću znamenku dijeljenika te ponavljamo postupak do kraja. Dakle, dijelimo na sljedeći način:

$$
\begin{eqnarray*}
\underline{100111 : 11 = 1101}\\
\hspace{-2.1cm}\underline{- 11}\\
\hspace{-1.45cm}11\\
\hspace{-1.8cm}\underline{- 11}\\
\hspace{-1.15cm}01\\
\hspace{-1.3cm}\underline{- 0}\\
\hspace{-1.1cm}011\\
\hspace{-1.2cm}\underline{- 11}\\
\hspace{-0.6cm}0
\end{eqnarray*}
$$

!!! admonition "Zadatak"
    1. Podijelite broj $101011_{(2)}$ s brojem $10_{(2)}$.
    2. Podijelite broj $10010_{(2)}$ s brojem $11_{(2)}$.
    3. Podijelite broj $101011_{(2)}$ s brojem $101_{(2)}$.

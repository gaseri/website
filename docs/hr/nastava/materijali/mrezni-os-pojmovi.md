# Osnovni pojmovi mrežnih operacijskih sustava

Operacijski sustav (engl. *operating system*, kraće OS) sistemski je softver koji upravlja sklopovljem računala i pruža usluge korisničkim programima. Jednostavnije rečeno, to je sučelje između hardvera računala i korisnika. Postoje mnoge vrste operacijskih sustava, ovisno o njihovim značajkama i funkcionalnostima. Mogu biti *batch OS*, *multitasking OS*, *multiprocessing OS*, *network OS*, *hybrid OS* itd.

Ovdje ćemo se usredotočit na mrežni operacijski sustav. Naučit ćemo dvije vrste mrežnih operacijskih sustava, njihove prednosti i nedostatke te izraditi i implementirati svoje distribuirane aplikacije u oblaku korištenjem usluga mrežnih operacijskih sustava.

## Mrežni operacijski sustav

Za razliku od operacijskih sustava kako ih svakodnevno koristimo, koji su dizajnirani da jedan korisnik kontrolira jedno računalo ili veći broj korisnika dijele resurse serijski umjesto istovremeno, mrežni operacijski sustavi (engl. *network operating systems*, kraće NOS) distribuiraju svoje funkcije među većim brojem umreženih računala.

NOS je moćan sustav koje sadrži softver i mrežne protokole za komunikaciju s drugim računalima putem mreže. Olakšava sigurnost i sposobnost upravljanja podacima, korisnicima, grupama, aplikacijama, hardverskim uređajima i drugim mrežnim funkcijama. Drugim riječima, mrežni operativni sustav omogućuje različitim računalima da se povežu i komuniciraju preko mreže.

NOS nije prisutan na svakom računalu, klijent ima samo dovoljno softvera za pokretanje hardvera i kontaktiranje poslužitelja, tako da NOS zahtijeva minimalnu količinu hardvera na klijentu. Sve ostale operacije izvode se na poslužitelju. Ovo je vrlo učinkovito u kontroli instaliranog softvera, jer klijenti nemaju mogućnost dodavanja ili uklanjanja softvera.

Primjeri mrežnih operacijskih sustava su:

- Microsoft [Windows Server](https://en.wikipedia.org/wiki/Windows_Server)
- Operacijski sustavi [slični Unixu](https://en.wikipedia.org/wiki/Unix-like), uključujući [GNU](https://en.wikipedia.org/wiki/GNU)/[Linux](https://en.wikipedia.org/wiki/Linux), [macOS](https://en.wikipedia.org/wiki/MacOS), [Berkeley Software Distribution](https://en.wikipedia.org/wiki/Berkeley_Software_Distribution) i derivate ([FreeBSD](https://en.wikipedia.org/wiki/FreeBSD), [NetBSD](https://en.wikipedia.org/wiki/NetBSD), [OpenBSD](https://en.wikipedia.org/wiki/OpenBSD), [DragonFly BSD](https://en.wikipedia.org/wiki/DragonFly_BSD))
- [Open Enterprise Server](https://en.wikipedia.org/wiki/Open_Enterprise_Server)
- [AppleShare](https://en.wikipedia.org/wiki/AppleShare)
- [LANtastic](https://en.wikipedia.org/wiki/LANtastic)
- [Banyan VINES](https://en.wikipedia.org/wiki/Banyan_VINES)

Postoje dvije vrste mrežnih OS-a, a to su:

- Peer-to-peer
- Klijent-poslužitelj

### Peer-to-peer mrežni operacijski sustav

Peer-to-peer (kraće P2P) mrežni operacijski sustav je operacijski sustav u kojem su računala ili čvorovi međusobno funkcionalno i operativno jednaki. Nitko nije superioran ili inferioran. Svi oni sposobni su obavljati slične vrste zadataka. Svi čvorovi imaju vlastiti prostor za pohranu podataka i resurse, koji dijele s drugim čvorovima. U ovom mrežnom sustavu nije potreban poseban poslužitelj za pohranu i dijeljenje resursa s drugima.

P2P je općenito najprikladniji za male (kućne mreže) do srednje lokalne mreže (male tvrtke). P2P mrežni OS zasnovani su na sličnim konceptima kao P2P mrežne aplikacije, primjeri kojih su BitTorrent, Skype (nekad), Bitcoin, Limewire, Kazaa, itd.
  
### Klijent-poslužitelj mrežni operacijski sustav

Klijent-poslužitelj mrežni operacijski sustav omogućuje korisnicima pristup resursima putem središnjeg poslužitelja. Poslužitelj je vrlo moćno računalo sposobno za izvođenje velikih izračuna i operacija te je odgovorno za pohranu i dijeljenje resursa s klijentskim računalom. Može biti višeprocesorsko računalo koje istovremeno obrađuje više zahtjeva klijenata.

Klijent upućuje zahtjev poslužiteljskom računalu, a poslužiteljsko računalo zauzvrat odgovara klijentu pružajući odgovarajuće usluge na siguran način. Ovaj OS je prikladan za velike mreže koje pružaju mnogo usluga i općenito je preskup za implementaciju i održavanje. Primjeri takvog mrežnog operacijskog sustava uključuju web poslužitelje, poslužitelje datoteka, poslužitelje ispisa, DNS poslužitelje, poslužitelje baza podataka i poslužitelje pošte.

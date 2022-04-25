---
author: Iva Fak, Vedran Miletić
---

# Strujanje multimedijskog sadržaja alatom VLC

[VLC media player](https://www.videolan.org/vlc/), kraće VLC ([Wikipedia](https://en.wikipedia.org/wiki/VLC_media_player)) je slobodni softver otvorenog koda za prikazivanje, kodiranje, emitiranje i snimanje multimedijskih sadržaja putem mreže. Ime VLC dolazi od **V**ideo**L**AN **C**lient, što je bilo inicijalno ime programa dok je bio samo klijent i dok je [VideoLAN Server](https://www.videolan.org/vlc/) bio odvojena aplikacija. Aktualna verzija može raditi i kao klijent i kao poslužitelj.

VLC je razvijen da bude prenosiv na različite platforme te se može prevesti i pokrenuti na više različitih operacijskih sustava i procesorskih arhitektura, ali većinom cilja na Linux, macOS i Windows. Podržava razne protokole za strujanje te mnoge audio i video kodeke i formate datoteka, može pretvarati multimedijalne datoteke u različite formate (transkodirati).

!!! note
    Sve verzije VLC-a do 1.0.x su bile imenovane po likovima iz filma [Golden Eye](https://www.imdb.com/title/tt0113189/), a iznad 1.1.x po [DiscWorldu](https://en.wikipedia.org/wiki/Discworld).

## Instalacija

Za instalaciju VLC-a je potrebno pokrenuti nekoliko repositorija, Epel, Remi and RPMFusion. Nakon toga, provjeravamo dostupnost VLC playera naredbom:

``` shell
# yum --enablerepo=remi-test info vlc
```

Ako je dostupan, instaliramo ga pomoću naredbe:

``` shell
# yum --enablerepo=remi install vlc
```

Zatim, pokrećemo kao normalan user (ne root), naredbom:

``` shell
$ vlc
```

## Strujanje

Strujanje videa se može pokrenuti pomoću GUI-a (`Media/Streaming`) koji nam uvelike olakšavaju posao, jer su sve opcije i metode [ponuđene na panelima](https://wiki.videolan.org/Documentation:Advanced_Use_of_VLC). No, u ovom radu ću prikazati streaming preko command line-a.

Pomoću naredbe `cvlc` pokrećemo VLC unutar terminala, zatim s modulom `STANDARD` omogućujemo streamanje preko mreže nakon što preko opcije `mux` enkapsuliramo metodu kojom ćemo streamat, u ovom slučaju `ogg` (ova opcija je obavezna).

Pomoću opcije `dst` određujemo na kojoj adresi i kojem portu se klijent može povezati. 192.168.104 je IP adresa računala. Naredba koju upisujemo na server računalu:

``` shell
$ cvlc -vvv /home/iva/stream/sintel_trailer-1080p.mp4 'standard{access=http,mux=ogg,dst=192.168.104:8080}'

```

Output/izlaz:

!!! todo
    Izlaz je potrebno ponovno snimiti.

Na klijent računalu pokrećemo naredbu:

``` shell
$ cvlc http://192.168.0.104:8080
```

i dobivamo sljedeći output/izlaz u VLC media playeru u terminalu:

!!! todo
    Izlaz je potrebno ponovno snimiti.

Ukoliko upišemo `vlc` ili `nvlc` umjesto `cvlc`, dobivamo i VLC-ovo grafičko sučelje ili ncurses sučelje (respektivno). Parametrom `--sout` možemo odlučiti kamo će se spremiti sadržaj preuzet strujanjem:

``` shell
$ nvlc http:// 192.168.104:8080 --sout=file/ogg: sintel_trailer-1080p.ogg
```

Pomoću VLC-a možemo spremiti stream na disk, a da bi to napravili koristimo VLC-ov Stream Output. Spomenuto možemo učiniti koristeći grafičko sučelje i VLC-ov record button ili putem Linux-ove komandne linije parametrom `--sout file/muxer:stream.xyz`, gdje je `muxer` jedan od podržanih formata. Format za spremanje streamova možemo mijenjati, podržani formati su:

- `ogg` za OGG format,
- `ps` za MPEG2-PS format,
- `ts` za MPEG2-TS format.

Mux može biti:

- `ts`: MPEG2/TS muxer
- `ps`: MPEG2/PS muxer
- `mpeg1`: standardni MPEG 1 muxer
- `ogg`: ogg muxer
- `asf`: Microsoft ASF muxer
- `avi`: Microsoft AVI muxer
- `mpjpeg`: multipart jpeg muxer

Razlikujemo nekoliko metoda streaminga:

- RTP/UDP Unicast: Streaming jednom računalu. Unesemo IP adresu klijenta u opsegu od 0.0.0.0 -- 223.255.255.255.
- RTP/UDP Multicast: Streaming više računala koristeći multicast. Unesemo IP adresu multicast grupe u opsegu od 224.0.0.0 do 239.255.255.255.
- HTTP: Streaming koristeći HTTP protokola.

Standardan UDP streaming istog videa u avi formatu iz direktorija `/home/iva/stream/` sa brojem UDP porta 1234:

``` shell
$ cvlc -vvv /home/iva/stream/sintel.avi --sout #std{access=udp, mux=ts, dst=:1234}'
```

Više detalja o strujanju u VLC-u može se pronaći na VideoLAN-ovom wikiju u dijelu dokumentacije [Streaming HowTo](https://wiki.videolan.org/Documentation:Streaming_HowTo): [Command Line Examples](https://wiki.videolan.org/Documentation:Streaming_HowTo/Command_Line_Examples), [Receive and Save a Stream](https://wiki.videolan.org/Documentation:Streaming_HowTo/Receive_and_Save_a_Stream) i [Advanced streaming with samples, multiple files streaming, using multicast in streaming](https://wiki.videolan.org/Documentation:Streaming_HowTo/Advanced_streaming_with_samples,_multiple_files_streaming,_using_multicast_in_streaming).

## Transkodiranje

Ako većina klijenata do kojih želim doprijeti koristi operacijski sustav Windows, znači da ću ja za VLC koristiti argument da on transkodira video u Windows Media Video (wmv) jer ću time moć doprijeti do najvećeg broja klijenata i bit ću sigurna da neće imati problema sa gledanjem mojih video snimki. Ako se na playlisti nalazi video u formatu avi, mpeg1, i sl., svaki taj video će se transkodirati u wmv format jer onda pouzdano znam da će većina korisnika (klijenata) bez ikakvih dodatnih kodeka moći gledati iste.

Primjerice, za Windows Media Video version 1, bitrate 512 with MP3 audio and ASFH muxer over MMSH protocol (`mmsh://192.168.0.10:30001`) parametri su:

```
'#transcode{vcodec=WMV1,vb=512,scale=1,acodec=mp3,ab=64}:std{access=mmsh,mux=asfh,dst=192.168.0.104:30001}'
```

Za DivX version 3, bitrate 1024 with Dolby Digital AC3 audio and ASFH muxer over MMSH protocol (`mmsh://192.168.0.10:30001`) parametri su:

```
'#transcode{vcodec=DIV3,vb=1024,acodec=a52,ab=512}:std{access=mmsh,mux=asfh,dst=192.168.0.104:30001}'
```

Za MPEG-4 with marquee filter, bitrate 1024 with MP3 audio and OGG muxer over HTTP protocol (`http://192.168.0.10:30001`) parametri su:

```
'#transcode{vcodec=mp4v,sfilter=marq,vb=1024,acodec=mp3,ab=512}:std{access=http,mux=ogg,dst=192.168.0.104:30001}
```

## Upravljanje VLC-om putem web sučelja

Kako bi omogućili da se [VLC kontrolira preko preglednika](https://wiki.videolan.org/Documentation:Modules/http_intf), potrebno je aktivirati Web interface putem kvačice na `Tools/Preferences/Interface/Main interfaces/Web`.

Zatim, otvaramo preglednik i na adresi `http://127.0.0.1:8080` se pokreće sljedeći prozor.

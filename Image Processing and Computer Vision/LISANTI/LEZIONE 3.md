https://betterexplained.com/articles/intuitive-convolution/ - 
## IMAGE FILTERS
Sono degli operatori che fanno IMAGE PROCESSING, computando il nuovo valore di un pixel _p_, date le intensita' dei pixels nel*neighborhood* di _p_ (chiamato **support**). Fanno **denoising**, **sharpening** (edge enhancement) sull'immagine.
Una sottoclasse importante di _image filters_ sono quelli **LINEARI** e **TRANSLATION-EQUIVALENT**, chiamati **LTE**.
Vengono applicati in _image processing_ tramite l'utilizzo di **2D CONVOLUTIONS** tra l'_immagine_ e il _kernel_ dell'operatore **LTE**.
## CONVOLUTIONS SPIEGATE PER BENE
Praticamente una convolution puo' essere vista in questo modo.
Mettiamo di avere un _piano di trattamento_ per dei pazienti di questo tipo : `[3, 2, 1]`
che indica che i pazienti al primo giorno di ospedalizzazione devono ricevere 3 pasticche,
quelli al secondo giorno 2, e quelli al terzo 1. 
Ho poi  un altro array che mi da' il _numero_ di _nuovi pazienti_ che arrivano *ogni giorno* all'ospedale in questa forma: `[1, 2, 3, 4, 5]`
Voglio avere come risultato un array, in cui per ogni giorno **della settimana** mi ritorna la quantita' di pillole somministrate: quindi saranno 3 il primo giorno, $3*2 + 2*1 = 8$ il secondo giorno (perche' ci sono 2 nuovi arrivati a cui si somministrano 3 pasticche, e un solo paziente che e' al secondo giorno di cura quidi gli si somministrano 2 pasticche), ecc....
**Come faccio a rendere questo calcolo organizzato?**
1. Flippo l'array del _numero_ di _nuovi pazienti_ che arrivano ogni giorno. 
2. Posiziono il calcolo in questo modo:
![[CV21.png]]
3. Faccio sliding delle rooms, slittando a ogni step l'array dei pazienti a DX:
![[CV22.png]]
4. Ripeto questo processo finche' non ho il risultato per i 7 giorni della settimana.
5. Il risultato e' una  **1D convolution** del _piano di pasticche_ e della _lista di pazienti_.
Chiamo il mio *piano di pasticche* **f(x)** e la _lista di pazienti_ **g(x)**.
La **convolution** e' espressa in questo modo: $f * g$
Nota che la prima operazione che viene fatta e' invertire _g_, facendo **g(-x)**.
Nota che se voglio prendere il valore corrispondente a uno specifico giorno devo fare **$g(-x + t)$**.
Ora, per avere il risultato in un **singolo giorno**, cio' che faccio e' moltiplicare ogni paziente col rispettivo piano, e sommare i risultati (quello che ho fatto prima). Matematicamente questo e' espresso con la formula:
![[CV23.png]]
1. $\tau$ in questo caso rappresenta questa cosa: immagina di girarti tutte le possibili stanze in cui vengono somministrate le pasticche (quindi ho $\tau$= 0, 1, 2, ecc... (nel mio caso ho solo 0,1,2 tra l'altro)), prendere il valore $f(\tau)$ del numero di pasticche da somministrare, di prendere il valore del numero di pazienti $g(t-\tau)$ a cui dover somministrare questo numero di pasticche e moltiplicarli tra loro. Poi, visto che $\tau$ varia, sommare le varie moltiplicazioni (tramite l'integrale che va meno infinito a piu' infinito ma che nel nostro caso consiste letteralmente di 3 valori che sono la lunghezza dell'array del _piano di pasticche_).
2. NOTA che t e' FISSO in questa equazione, quindi il giorno selezionato non cambia, ho solo che cambia $\tau$.
3. Il _piano di pasticche_ e' chiamato **KERNEL** e NOTA che NON e' lui a venire flippato (a differenza di quanto succede nelle lezioni del Prof.). 
### PROPRIETA' MATEMATICHE DELLA CONVOLUTION
1. La **convolution** e' COMMUTATIVA quindi: $f *g = g* f$. Questo risponde al dubbio di qui sopra. Volendo AVREI potuto flippare il KERNEL e lasciare la _lista di pazienti_ inalterata. Avrei ottenuto lo stesso risultato
**NOTA**: applica quello che hai visto qua sopra ma pensandola in ottica di $g* f$, in quanto nelle **2D convolutions** ho come standard ormai che flippo il kernel e lo posiziono su una zona dell'immagine. Anche la funzione di sopra, vedila in quest'altra ottica.
Riscritta per le **2D CONVOLUTIONS** con $g* f$ invece di $f * g$ ho:
$$(g * f)(x, y) = \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} g(\alpha, \beta)f(x-\alpha,y-\beta)d\alpha d\beta$$
Quindi ho che a differenza di prima invece di $\tau$ che cambia, rappresento il cambiamento con $\alpha$ e $\beta$, che coprono tutti i valori all'interno del **KERNEL**. Voglio trovare il nuovo pixel per il punto $(x,y)$ e lo faccio posizionando il kernel su quel punto e computando somme (i due integrali, uno per $\alpha$ e l'altro per $\beta$, per scorrermi tutti i valori del kernel) di moltiplicazioni tra singoli valori. **NOTA** che $\alpha, \beta$ e' il sistema di coordinate dell'immagine, (x,y) rappresenta semplicemente le COORDINATE del punto di cui voglio sapere il nuovo valore. Quindi variando $\alpha, \beta$, ho che g($\alpha, \beta$) ritorna i valori dei punti VICINI a (x,y) che e' il punto in esame. Questi g($\alpha, \beta$) si estendono FINO ALLA DIMENSIONE DEL KERNEL _f_ che vi e' sopra.
_f_ viene flippato e poi traslato in posizione (x,y), in modo da piazzarsi proprio sopra al punto di mio interesse. 
## CORRELATION
Nella correlation **NON ho il FLIP**, ho **SOLO la TRASLATION**.
La correlation **NON e' COMMUTATIVA**, a differenza della convolution.

**NOTA**: se ho che INVERTIRE IL KERNEL ritorna **lo stesso kernel**, ovvero quando _f_ e' una **even function** ovvero quando $f(x,y) = f(-x, -y)$, ho che la correlation risulta uguale alla convolution. Guarda:
![[CV24.png]]
Nell'immagine si usa _h_ ma noi abbiamo _f_. Posso fare la sostituzione data da $f(x,y) = f(-x, -y)$. In sto modo ho $g * f =  g \circ f$. (secondo me nell'immagine c'e' un errore, il risultato finale dovrebbe essere $i(x,y) \circ h(x,y)$ che corrisponde a $g \circ f$. L'inizio invece corrisponde a $g * f$, quindi c'e' palese errore dai).
Ma poi il prof dice che in questo setting di even function, la correlation rimane NON COMMUTATIVA, ma questa roba mi sembra un po' sbagliata non lo so.
## CONVOLUTIONS DISCRETE
$$(g * f)(x, y) = \sum_{\alpha = -\infty}^{+\infty} \sum_{\beta = -\infty}^{+\infty} g(\alpha, \beta)f(x-\alpha,y-\beta)$$
Per ottenere un output image sta operazione deve essere computata per ogni coordinata (x,y) nell'immagine. Per ognuna mi viene infatti ritornato un nuovo pixel.
**NOTA** quando computi le convoluzioni NON PUOI SOSTITUIRE i valori computati IN PLACE mentre stai facendo sliding. Senno' e' tutto falsato.
## IMPLEMENTAZIONE PRATICA
Devo settare k che e' la kernel size. E da questa calcolo la dimensione del **KERNEL** facendo $2k + 1$. Quindi ho che la size sara' $(2k +1)$ x $(2k + 1)$.
In questa slide si mostra come fare le convoluzioni in pratica. Mi infastidisce che qua si fa $(f *g)$  a differenza di come ho fatto fin'ora che era $(g *f)$.
Io continuo a scriverla cosi' come la avevo prima, cambiando solo la g con I (che rappresenta l'_immagine_) e la _f_ con K. Quindi in sto caso ho:
$$O(x, y) = \sum_{\alpha = -k}^{k} \sum_{\beta = -k}^{k} I(\alpha, \beta)K(x-\alpha,y-\beta)$$

**Quante OPERAZIONI computo?**
Faccio $(2k +1)(2k +1)$ moltiplicazioni e $(2k +1)(2k +1)$ somme.
Quindi $2(2k +1)^2$ operazioni in tutto.

**NOTA** non posso applicare i KERNELS sui BORDI.  Ci sono 2k colonne e 2k righe per cui NON ho i pixel per calcolare la convolution. quello che si fa e' fare **PADDING**:
1. Solitamente si fa **ZERO** PADDING.
2. Oppure posso usare **REPLICATE** PADDING (usando l'ultimo value del Pixel disponibile). 
3. Oppure posso fare **REFLECTION** PADDING ovvero  RIPETO gli ultimi n pixels (NON SOLO ripetendo l'ultimo come nel caso di prima, ma RIPETENDO L'ULTIMA SEQUENZA di pixels ).

### IMPLEMENTAZIONE DI LINEAR FILTERS
#### MEAN FILTER
Creo un filter in cui ogni elemento del KERNEL ha come valore $$\frac{1}{k^2}$$
.
Un esempio di **MEAN FILTER** con k=3 e':
![[CV25.png]]
E' chiamato **low-pass filtering**. Se faccio average infatti sto perdendo INFORMAZIONI, sto SIMPLIFICANDO l'immagine. Sto filter fa **image smoothing.** Non e' per forza buono, mi deteriora l'immagine.
Di base **RIDUCE il NOISE** ma rende anche **piu' BLURRATA l'immagine**.
Nell'esempio qua sotto cio' e' visibile:
![[CV26.png]]
Ho  aggiunto a questa immagine GAUSSIAN NOISE con 0 _mean_ e _std_ = 8. Il NOISE e' VISIBILE solamente nelle zone dell'immagine in cui **NON ho cambiamenti di density values.** In queste zone e' piu' visibile, in quelle in cui si hanno molti dettagli e' molto meno visibile.
Se applico uno smoothing 3x3 ho che ho levato il GAUSSIAN NOISE (l'ho ridotto almeno). L'immagine pero' risulta un po' piu' smooth, blurrata. Questo effetto aumenta se ALLARGO l'area che considero per performare la mean, tramite un kernel 5 x 5.
![[CV27.png]]
In un'immagine quando ci sono cambiamenti di intensita' di PIXELS ho un **salto**. Lo _zig-zag_ che noto in figura e' riferito al _noise_ sia nelle regionni di bassa intensita' che in quelle in quelle di alta intensita'. 
1. Quando applico il mean kernel nelle zone di bassa intensita', il _noise_ viene levato.
2. La stessa cosa quando applico il mean kernel in quelle di alta intensita'
3. Quando lo applico nella zona tra la bassa intensita' e l'alta intensita' avro' che il kernel rendera' meno drammatica la differenza tra le due zone. Questo comportera' a uno SMOOTHING **che non voglio**.


#### Gaussian Filter
Il migliore LINEAR FILTER e' il **GAUSSIAN FILTER**. Questa GAUSSIAN e' 2D in sto caso. Il kernel CREATO DALLA GAUSSIAN avra' DIFFERENTI WEIGHTS. Ogni weight e' legato alla GAUSSIAN FUNCTION. I pesi:
1. avranno VALORI PIU' ALTI per il CENTRO DEL KERNEL. 
2. avranno VALORI PIU' BASSI per zone del kernel piu' lontani. 
Con GAUSSIAN FILTER ho che il problema del _mean filter_ di sopra un po' migliora: PIXELS di diversa INTENSITA' da quella OSSERVATA che magari stanno nella parte piu' distante dal centro del KERNEL VENGONO meno considerate. Cio' rende lo SMOOTHING **MENO PRESENTE**.
In questo esempio ho:
![[CV28.png]]
**NOTA** Se aumento SIGMA la GAUSSIAN DIVENTA PIU' LARGA,  di conseguenza I VALORI DEL KERNEL DISTANTI DAL CENTRO AUMENTANO DI VALORE mentre quelli centrali diminuiscono di valore, accentuando lo smoothing. 
Selezionando $\sigma$ in un certo modo ho che genero UN'IMMAGINE  a una SPECIFICA SCALA.
**QUANTO DOVREBBE ESSERE GRANDE $\sigma$ affinche' io abbia un buon denoising senza rendere troppo smoothed l'immagine ?**
Quanto grande deve essere rispetto alla dimensione del **kernel**?
A livello logico ha senso usare filters con grandezze maggiori quando $\sigma$ e' grande, minori quando $\sigma$ e' piu' piccolo.
LA **RULE-OF-THUMB** e' che **k**, con kernel size pari a $(2k +1)$ x $(2k + 1)$ deve essere uguale a 3$\sigma$.
QUESTO PERCHE' il 99% della **variance** della _funzione gaussiana_ e' catturata da 
\[-3$\sigma$, +3$\sigma$]. 
Quindi ad esempio, con $\sigma$ = 3, calcolo k = 9, quindi ho _kernel size_ pari a $19$ x $19$ con $2*9+1 = 18$.

##### COME APPLICARE SEPARABILITY
Invece di applicare un _2D gaussian filter_, per computare **un numero di operazioni molto minore** applico due *1D gaussian filters* , visto che la 2D _gaussian_ e' scomponibile in due _1D gaussians_: $G(x,y)=G(x)G(y)$.
![[CV29.png]]
Nell'ultimo step viene mostrato che la convolution su G(x) e poi su G(y) e' uguale di fare la convolution su G(y) e poi su G(x). Questo e' dato dal fatto che la **convolution** supporta la proprieta' **associativa**.
Se calcolo ora il numero di operazioni:
1. Con _2D Gaussian filter_ ho $2 (2k+1)^2$, quindi con un kernel 7x7 avrei 49 moltiplicazioni e 49 somme, quindi 98
2. Con due _1D Gaussian filters_ ho $2 * 2  (2k+1)$ . Quindi nel mio caso ho 7 prodotti e 7 somme a ogni 1D _Gaussian filter_, e fa 14. Visto che ho due _1D Gaussian filters_ il risultato e' 28.

### Problema con MEAN FILTER/GAUSSIAN FILTER

![[CV30.png]]
Corrompo l'immagine con dell'**IMPULSE NOISE**, quindi randomicamente dei pixels vengono messi bianchi (o comunque a valori randomici).
USO _mean filter_ PER FARE **DENOISING** in questa immagine. L'immagine **PEGGIORA** in quanto questi punti vengono spalmati sull'immagine. Questo fenomeno PEGGIORA con l'aumentare del _kernel size._  IL GAUSSIAN FILTER **COMUNQUE** NON FUNZIONA, fara' magari meglio ma otteremmo sempre piu' o meno lo stesso risultato.

### NON-LINEAR FILTERS
Ora **NON** sto piu' computando una CONVOLUTION (la **convolution** e' possibile solo se il filtro e' **lineare** come avevamo visto sopra dalla teoria). 
NON ho linearita' in questi tipi di _filters_ perche' il _kernel_ non ha valori fissi, questi cambiano in base all'input dell'immagine che gli viene mostrato.

#### MEDIAN FILTER
E' uno dei piu' semplici.
Praticamente
1. prende una zona dell'immagine
2. Sorta i valori all'interno di quella zona
3. Prende il **MEDIAN** value.
Questo filtro e' utile, in quanto, nel caso in cui ci siano dei pixels nella zona che non c'entrano niente con gli altri, questi vengono ignorati (non si prende la mean come prima considerando anche quei pixels). Questo permette al filtro di essere piu' **SHARP** nelle zone con diverse intensita' e di **evitare** di fare _smoothing_. Mostro questa cosa con un esempio:
_ESEMPIO_:
Ho _pixel values_ pari a `10, 10, 40, 40`.Nota che esprimono una zona dell'immagine in cui ho un cambio di intensita'.
1. Con il mean filter avrei ottenuto `10, 20, 30, 40`
2. Con il median filter ottengo `10, 10, 40, 40`, quindi preservo lo sbalzo di intensita'
![[CV31.png]]
In questa immagine uso _median filter_ e cio' mi permette di levare  **QUASI TUTTI GLI INPUT NOISES**. 
**FACENDO DUE VOLTE IL MEDIAN FILTER** sembra che io abbia levato TUTTO IL NOISE.

**NOTA**: NON USO SEMPRE MEDIAN FILTER perche' ci sono IMMAGINI che hanno sia _GAUSSIAN NOISE_ che _IMPULSE NOISE_, e il median filter NON FUNZIONA MOLTO BENE CON il _GAUSSIAN NOISE_.
SOLITAMENTE SI APPLICANO SIA MEDIAN CHE GAUSSIAN FILTERS IN STI CASI, **ma in quale ordine**?
SE applicassi LA GAUSSIAN ALL'INIZIO , **spargerei** I PUNTI BIANCHI, cosa che voglio evitare.
Quindi la cosa che si fa e' : Prima levo sti punti bianchi con MEDIAN FILTER e poi applico GAUSSIAN FILTER.

## BILATERAL FILTER
Questo e' un *nonlinear filter* che **leva** **GAUSSIAN NOISE** **SENZA BLURRARE L'IMMAGINE** (edge preserving smoothing).
Praticamente in ogni _zona di pixels presa in esame dal filter_, vengono fatti dei calcoli.
Per ogni pixel _q_ all'interno della _zona_, calcolo $H(p,q)$, con _p_ ovviamente pixel preso in esame. Questa $H(p,q)$ internamente calcola due **distanze** : $d_s(p,q)$ e $d_r(p,q)$.
Ho che $d_s(p,q)$ = $||p -q ||_2$ e calcola la **SPATIAL DISTANCE di base**, $d_r(p,q)$ calcola la differenza tra le intensita' $|I_p - I_q|$. Queste due quantita' vengono messe come input in due diverse **GAUSSIANE** che stiamo la PDF di questi due valori. In questo modo due punti _p,q_ molto lontani avranno come PDF un valore molto piccolo; allo stesso modo se _q_ ha un'intensita' molto diversa rispetto a _p_ allora la PDF sara' piccola.
**NOTA** se queste due PDF sono piccole, allora nel calcolo della convoluzione $I_q$ avra' molta poca rilevanza nel calcolo del nuovo valore di _p_. **NOTA** queste PDF saranno piccole quando avro' DIFFERENZE DI INTENSITA' tra due aree della _zona_ presa in esame, quindi ho CHE LO SMOOTHING qua NON avviene. 
La formula di $H(p,q)$ e':![[CV32.png]]
Con 
![[CV33.png]]
Ho poi che l'output su _p_ viene calcolato, come gia' accennato e discusso sopra, cosi : ![[CV34.png]]

**PERCHE' NON E' LINEARE STO KERNEL?**
**Perche' NON PUO' ESSERE COMPUTATO CON CONVOLUTIONS?** 
Perche' _$H(p, q)$_ CAMBIA PER OGNI PUNTO p, di conseguenza **non** ho un FIXED KERNEL PER OGNI CONVOLUTION.  
NOTA CHE NON SERVE FARE UNA DNN cosi tosta per fare denoising. SEMPLICEMENTE APPLICO UN BILATERAL FILTER e sto a posto.

## NON-LOCAL MEANS FILTER
E' un altro filtro *non lineare* che preserva gli **edges** mentre fa smoothing (quindi riduce il noise).
Praticamente che fa:
dato un punto _p_, prende un insieme di punti _q_, intorno a _p_ (come si e' sempre fatto finora). Questo quadretto da cui prendere i punti _q_ e' annotato come S. Per ogni _q_ in _S_ si calcola _w(p, q)_, che e' un weight che verra' poi moltiplicato al valore del pixel in _q_ e mi dice ' *quanto il valore I(q) del pixel q e' importante di base nello stimare il nuovo valore del pixel p*.
_w(p, q)_ viene calcolato nel seguente modo:
si prendono delle patches, una intorno a _p_ e una intorno a _q_. Questi sono chiamati $N_p$ e $N_q$. Viene calcolata l'euclidian distance tra $N_p$ e $N_q$ (il che significa che per ogni pixel viene calcolata la differenza alla seconda e ogni differenza viene poi sommata, in modo da ottenere un solo numero per la distanza tra $N_p$ e $N_q$). 
La formula di *w(p, q)* e' la seguente:
$$w(p, q) = \frac{1}{Z(p)} e ^ {-\frac{\lVert N_p - N_q \rVert_{2}^ 2}{h ^ 2}}$$
Quindi ho che piu' e' grande la distanza tra $N_p$ e $N_q$ e meno e' grande il peso finale $w(p,q)$ (con la e alla qualcosa di molto negativo ho un valore MOLTO PICCOLO come risultato). 
Z(p) e' semplicemente un fattore di normalizzazione, per avere che la somma dei vari *w(p, q)* sia uguale a 1. 
La formula finale per computare il nuovo punto e':
$O(p) = \sum_{q \in S} w(p, q) I(q)$, questo significa che se il peso e' basso allora si dara' poca importanza al valore del pixel nel punto _q_. Il peso e' basso quando la patch di _p_ e quella di _q_ sono molto DIVERSE TRA LORO. E' alto altrimenti. In sto modo si preservano gli *edges* (non mi e' chiarissimo come, se ho delle patches simili intorno a me allora il valore del pixel in considerazione e' considerato come quello delle patches intorno SIMILI (solo di quelle)).
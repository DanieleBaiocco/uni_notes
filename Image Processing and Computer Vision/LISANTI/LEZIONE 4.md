# Edge Detection  
Gli edges sono features locali dell'immagine che catturano informazioni importanti riferite al *contenuto semantico dell'immagine*(quindi a cosa contiene effettivamente, quali oggetti ci sono e come questi sono rappresentati).
Gli **EDGES**  sono dei **pixels** che possono essere pensati come *POSIZIONATI  IN MEZZO TRA DUE ZONE DI DIVERSA INTENSITA'* o in altre parole  possono essere pensati come *SEPARATORI di DIFFERENTI REGIONI DELL'IMMAGINE*.
Ci sono diversi modi per fare *edge detection*. Uno di questi e'
## 1D STEP-EDGE
Posso concentrarmi su un 1D SIGNAL (quindi un'immagine composta da una sola dimensione), assunzione che mi sembra poco utile (io opero con immagini 2D no?). Ad ogni modo se ho questo 1D SIGNAL allora posso vedere come cambia il valore dei pixel man mano che vado avanti. Osservero' che a una certa il valore aumentera' rapidamente (in quanto sto passando da una *regione* dell'immagine a un'altra *regione* dell'immagine). Questo e' proprio un chiaro esempio che c'e' un *edge*. Per **detectare** questo edge, si calcola la derivata per ogni punto del 1D SIGNAL. Quando la derivata raggiunge un **peak** allora li' ho un edge. Io per detectarlo basta che faccio **THRESHOLDING** sui valori assoluti delle derivate. Per ogni valore assoluto della derivata in quel punto, se questo supera una **THRESHOLD** **allora il pixel legato a quel punto e' un edge.**

![[CV2 1.png]]

Nota, prendo sempre gli absolute values delle derivate perche' in questo modo posso detectare edges anche quando passo da zone di *alta intensita'* a zone di *bassa intensita'* (in quel caso il valore della derivata *scenderebbe* fino a raggiungere il minimo in corrispondenza dell'inflection point, e se **non** calcolassi l'**absolute value**,  il thresholding **non detecterebbe nessun edge**). Nell'immagine qua sotto si puo'  proprio vedere il comportamento della defrivativa quando si passa da valori alti a valori bassi. ![[1dkernel.png]]
L'algo e' qundi questo: ![[cv1 1.png]]
## 2D STEP-EDGE
In 2D Signals ho sia cambiamenti nella parte verticale che in quella orizzontale.
Ci serve in questo caso un modo di trovare l'edge **INDIPENDENTEMENTE DALLA DIREZIONE a cui e' orientato questo edge**. E' possibile fare cio' utilizzando il GRADIENTE, quindi calcolando le derivate parziali prima per l'asse _x_ (vedere come cambiano i valori di intensita' dei pixels per questo asse) e poi per l'asse _y_. In questo modo, guardando ai risultati di queste due derivate parziali per due particolari pixels, posso 
1. vedere se c'e' un *edge* (e questo lo faccio computando la *magnitude* del gradiente tramite formula $\lVert \nabla I(x,y) \rVert = \sqrt{I_x ^ 2 + I_y ^ 2}$ e poi facendo **THRESHOLDING** sul risultato della *magnitude*) e
2. qual'e' la direzione di questo *edge*. Questa e' indicata con $\theta$ e si calcola usando le seguenti formule matematiche:
	* $\theta \in [-\frac{\pi}{2}, \frac{\pi}{2}]$ = $\text{atan}(\frac{I_y}{I_x})$ questo mi da' la **direzione** date le derivate parziali 
	*  $\theta \in [0, 2\pi]$ = $\text{atan2}(I_x,I_y)$ questo mi da' la **direzione** e il **segno**
Si fa questo per ogni pixel conta (come nel caso di prima).
![[CV3 1.png]]
Qui sopra ho valori di $\theta$ calcolati usando la formula della *direzione e del segno* e la loro interpretazione.
Questa invece e' la schematizzazione di tutto il processo
![[cv4 1.png]]
### Discrete approximation of the GRADIENT
**NOTA**: le immagini con cui lavoro sono *discrete*. Devo quindi approssimare le derivate parziali.
Posso approssimare le derivate parziali in un modo molto furbo. Semplicemente se ho la derivata parziare di **$I(x,y)$** su x per due punti **_x, y_** allora fisso _x_, e muovo _y_ di poco. Calcolo quindi la differenza tra **_I(x,y)_** (il valore del pixel in quella particolare posizione _x,y_) e il pixel che sta di poco sopra (o di poco sotto). Nota che questo altro non e' che la definizione di derivata.

Applicando questo concetto in maniera piu' formale, posso scegliere se il punto da vedere con cui fare la differenza stia sopra/a destra o sotto/ a sinistra (a seconda se faccio derivata parziale su x o su y insomma). C'e' anche un terzo caso in cui prendo i due punti vicini orizzontalmente/verticalmente.
Quindi ho:
1. **backward**: prendo quello SOTTO nel caso di derivata parziale su x, quello a SINISTRA nel caso di derivata parziale su y![[cv5 2.png]]
2. **forward**: prendo quello SOPRA nel caso di derivata parziale su x, quello a DESTRA nel caso di derivata parziale su y![[cv6 1.png]]
3. **central**: si prendono il punto sopra e il punto sotto e si fa la differenza nel caso di derivata parziale su x, si prendono quello a sx e quello a dx nel caso di derivata parziale su y.![[cv7 1.png]]
	In questo specifico caso ![[cv8 1.png]]
### Different Approximations for the MAGNITUDE
Posso usare come approssimazione della magnitude NON solo quella vista sopra ma anche altre. Le possibili sono:
1. $\lVert \nabla I(x,y) \rVert = \sqrt{I_x ^ 2 + I_y ^ 2}$ gia' vista
2. $\lVert \nabla I(x,y) \rVert_+ = |I_x|  + |I_y|$
3. $\lVert \nabla I(x,y) \rVert_\text{max} = \text{max}(|I_x|, |I_y|)$
Nota che metti ho 3 immagini di questo tipo, in cui le intensita' passano da 0 a _h_, ma cio' succede con diverse orientazioni:
![[cv9 1.png]]
**NOTA** $I_x$ e' nel caso dell'immagine $E_v$ pari a _h_ nel pixel dove osservo la presenza di _edge_ (questo vale per ogni pixel in cui c'e' l'edge). Mentre $I_y$ e' pari a 0 (se faccio la differenza del pixel corrente con quello sopra/sotto a seconda se faccio backward o forward,  o tra i due di cui lui e' quello in mezzo, ho sempre come risultato _h_-_h_ = 0 (o 0-0=0 dipende quale pixel prendo in considerazione)).

Questo calcolo si puo' fare anche per $E_h$ e $E_d$:
1. in $E_h$ ho $I_x$ uguale a 0 e $I_y$ uguale a _h_   In $E_d$ ho che sia $I_x$ che $I_y$ sono pari a _h_.
Questo e' il risultato delle diverse _magnitude_ applicate sulle diverse foto $E_v$, $E_h$ e $E_d$ :
![[cv10 1.png]]
**NOTA**:  nel caso di $E_d$, in cui l'immagine e' **diagonale**,:
1. con $|\nabla I|$ , ho un valore molto basso rispetto a _h_ , quindi magari la threshold potrebbe NON percepire il punto in $E_d$  come edge, nonostante lo sia. 
2. con $|\nabla I|_+$ e' troppo alto,
3. con $|\nabla I|_max$ , in cui si prende il _max_, **ho lo stesso risultato che nel caso di edge verticale e orizzontale** quindi e' **isotropico**, e da' la stessa *magnitude* indipendentemente dalla direzione dell'edge.

# EDGES E NOISE
In real images an edge will not look as smooth as we have seen (due to noise). Taking derivatives of noisy signals is an **ill posed problem**…the solution is not robust wrt to input variations => *Derivatives amplify noise*.  To work with real images, an edge detector should therefore be **robust to noise**, so as to **highlight the meaningful edges** only and **filter out effectively the spurious transitions** caused by noise. *How do we get rid of noise?*

Solitamente come soluzione prima di fare edge detection, si fa **smoothing** del segnale. Questo ha come lato negativo che poi gli **edges** potrebbero essere blurrati cosi' tanto che l'*edge detector* non e' piu' in grado di riconoscerli.

Una soluzione piu' robusta e' fare **smoothing e differentiation** in un **singolo step**. Questo e' raggiunto facendo *differenze di averages* (invece di prima averaggiare l'immagine e poi calcolare le differenze come ho fatto finora, facendo prima denoising e poi calcolando i gradienti).
Quindi PRIMA calcolo per ogni punto di cui voglio calcolare la derivata l'average con i punti vicini. Poi applico la derivata con gli averages intorno.
![[cv2 2.png]]

![[cv3 2.png]]
Se dovessi calcolare $I_x(i,j)$ con la _forward_ in questo caso farei:
![[cv4 2.png]]

con
![[cv5 3.png]]
nelle slides c'e' anche il calcolo di $I_y(i,j)$ con la _forward_.
QUESTE DERIVATE SONO CHIAMATE **smooth derivatives**.

## Prewitt and Sobel
Posso utilizzare, invece della _forward_ o della _backward_ sulle averages, la _central_ difference (e questo e' detto il **Prewitt kernel**, molto famoso).
![[cv6 2.png]]
Praticamente sto kernel e' passabile su ogni pixel $(i,j)$ dell'immagine per computarne la $I_x(i,j) \text{ e } I_y(i,j)$.


Posso dare uno *weighting al pixel al centro*, e in questo caso ho il **Sobel kernel** cosi' fatto:

1. kernel per calcolare $I_x(i,j)$ per ogni pixel (i,j): 
![[cv10 2.png]]
3. kernel per calcolare $I_y(i,j)$ per ogni pixel (i,j): 
![[cv11 1.png]]
**NOTA** il kernel e' sempre centrato in $i,j$ eh

il modo in cui $I_x(i,j) \text{ e } I_y(i,j)$ vengono calcolate e' identico (le formule sono uguali), cambia solo il calcolo interno di $\mu_x \text{ e } \mu_y$.
![[cv7 2.png]]
![[cv8 2.png]]
**NOTA**: al pixel centrale viene data un'importanza doppia rispetto agli altri. Questo perche' i pixels vicini al centro dovrebbero avere piu' importanza nella computazione delle partial derivatives.


# Edges detection
Scoprire gli edges facendo gradient thresholding non e' accurato in quanto l'immagine puo' contenere al suo interno edges di diverso tipo: *edges molto marcati*, *come edges meno marcati* (quelli poco marcati sono quelli in cui il cambio di intensita' e' molto piccolo).
Se provo a trovare gli edges meno marcati (abbassando la threshold) ho una peggiore localizzazione degli edges piu' marcati, perche' la threshold sarebbe cosi' bassa nel caso dell'edge marcato che mi vengono selezionati una miriade di punti come edges (quando in verita' volevo che lo fossero solo quelli con edges piu' marcati). Ho una risposta TROPPO GRANDE quando ho edges marcati.
## Non maxima suppression (NMS)
Questa tecnica mi permette di risolvere il problema di sopra. La soluzione e' tramite l'uso di **local maximum**, qui qualcosa che localmente trova il massimo tra gli altri valori di magnitudes. Quindi praticamente guardo le magnitudes e localmente per ogni intorno di pixel detecto il massimo dell'intorno (quindi ho solo un pixel nell'intorno che viene determinato come edge).

**NOTA PERO'**, dobbiamo avere qualcosa che ci dica la *direzione su cui poi calcolare il massimo*. La direzione ci dice  l' insieme di magnitudes da prendere in considerazione da cui ritornare dopo il massimo valore.
![[cv12 1.png]]
Se non prendessi in considerazione la direzione dell'edge  (che nell'immagine di sopra e' deducibile essere da sx a dx, quindi orizzontale) IO non saprei in quale direzione comparare i valori di magnitudes:
1. potrei farlo lungo l'asse _x_: in questo modo ho che tra 9, 40 e 8 il maggiore e' 40, quindi metto 1, tra 11,42,12 il maggiore e' 42 quindi metto 1 anche li'.
2. oppure lungo l'asse _y_: potrei avere tra 40, 42, 40 come massimo 42 e li' metto un 1. Tra 9, 11, 10 avrei 11 come massimo, tra 8,12,10 avrei 12 come massimo ecc.. Questo mi porterebbe a detectare un edge orizzontale (quando in verita' non c'e' lol)
Come si vede dalla figura infatti, applicare l'NMS sull'asse x  e' corretto mentre averlo fatto sull'asse _y_ non lo e'.
![[cv13 1.png]]
**NOTA** We don’t know a differenza dell'immagine di sopra  in advance the correct direction to carry out NMS. The *direction* sul quale fare NMS has to be estimated locally (based on **gradient’s direction**, perche' la direzione del gradiente mi dice la direzione con la massima variabilita').

Quidni basta prendere la gradient's direction che ho gia' visto come calcolare e sto a posto.
**C'E' UN PROBLEMA**, la maggior parte delle volte la direzione del gradiente e' tale che non ha un angolo di 45 gradi, quindi NON ho le magnitudes di _A_ e _B_ da paragonare alla magnitude nel punto $(i,j)$ che sto considerando per decidere se questo punto sia maggiore di _A_ e _B_ (e quindi il massimo).
HO PERO' un modo per approssimare le magnitudes di A e B, nel mondo discreto, che e' il seguente:
![[cv14 1.png]]

Per approssimare $G_A$ e $G_B$ compio una linear interpolation con le magnitudes disponibili, supponendo che il gradiente cambi in modo **lineare**. Infatti per calcolare $G_A$ ad esempio, si prende il valore $G_1$ che e' legato alla magnitude nel punto $(i-1, j)$ e lo si somma alla differenza tra la magnitude nel punto a sx $(i-1, j+1)$ con $G_1$ che e' pesata con $\Delta x$, ovvero la coordinata del punto di intersezione di cui voglio sapere l'approssimazione. Per questo si assume che la magnitude decrementi in modo lineare da $(i-1,j) \text{ a } (i-1, j+1)$.


La **pipeline diventa quindi**
![[cv15 1.png]]

**NOTA** l'OUTPUT della NMS e' una mask. Se poi la applico dopo averla calcolata mi viene ritornata una matrice con tanti 0 e qualche valore maggiore di 0 che indica la magnitude del gradiente legato a quella posizione nella matrice (che vabe' sarebbe aka l'immagine). Posso POI applicare **thresholding** su questi valori maggiori di 0, per escludere delle magnitudes che sono troppo piccole, e che quindi non sono abbastanza alte per predictare un **edge**.
**NOTA**: senza la NMS, la pipeline prima consisteva in **thresholding** diretto sulle plain magnitudes.

# Canny's edge detector
Forse uno dei migliori edge detector esistenti ora.
Questo e' un detector che e' robusto wtr noise. 
Poi questo DETECTA un solo pixel per un edge (IO NON VOGLIO CHE UN EDGE DETECTOR MI PREDICTI THICK EDGES, MA solo una linea praticamente fatta di singoli punti)

praticamente Canny ha pensato di fare 
1. Gaussian smoothing
2. Poi computare il gradiente
3. e Poi computare NMS (non maximal suppression) su sto gradiente tenendo in considerazione della direzione
Invece di fare la **convoluzione** per applicare il *gaussian filter* e poi fare **partial derivatives** per *computare il gradiente*, posso usare la separabilita' della gaussian function per rendere piu' veloce il calcolo:
![[cv16 1.png]]
In questo caso, come si puo' vedere, inizio facendo *prima* la convoluzione e *poi* il calcolo della derivata.
Alla fine ho invece che vien performato *prima* il calcolo della deriva sul gaussian filter, e *poi* viene fatta la convolution. Nota che la convoluzione con la *gaussiana* e' stata splittata grazie all'utilizzo della formula $G(x,y) = G(x)G(y)$ che permette di separare le cose.

Quindi invece di usare un *2D Kernel* posso usare *1D Convolutions* per fare questo calcolo, che e' molto piu' veloce.
La cosa non finisce qui, Canny risolve anche un altro problema, oltre a quello del denoising. Lui risolve anche il problema del *thresholding*, dopo aver performato **NMS**. A volte edges potrebbero avere MENO INTENSITA' rispetto a altri edges, quindi potrebbe succedere che un punto, o piu' punti, NON VENGANO DETECTATI in quanto hanno un valore piu' basso. Si crea quindi il fenomeno chiamato edge **streaking** che consiste nel: per un oggetto MI VENGONO RITORNATI DEI **contorni** frammentati e spezzettati,  **NON** una *linea continua*, **NON** tutto l'*oggetto*.
Canny quindi introduce un'altra *threshold*. Ho ora due threshold $T_h \text{ e } T_i$, con $T_i$ piu basso. 
1. Se un pixel e' piu' alto di $T_h$ allora quello viene visto come edge.
2. Se un pixel e' piu' basso di $T_h$ allora
	1. Se e' piu' alto di $T_i$ e ha i pixel vicini maggiori di  $T_h$, allora e' considerato anch'esso un **edge**
	2. Non viene considerato un **edge** altrimenti
**Nel caso di CANNY viene ritornata una LISTA DI EDGES, con ogni elemento della LISTA che e' una CATENA DI PUNTI, di pixels che indicano un oggetto.**
Nei casi visti finora invece avevo un'immagine ritornata, al termine del processo.
![[cv17 1.png]]
![[cv19 2.png]]

La pipeline nel caso del **CANNY's edge detector** e' quindi la seguente:
![[cv21 1.png]]
**NOTA** L'immagine di partenza e' una **T** che e' illuminata da una fonte di luce che proviene da in basso a sinistra. Questo comporta che certi edges sono piu' marcati e altri un po' meno.  E' molto difficile fare edge detection qui data la diversa intensita'. Se infatti mostro il **NMS signal** ho che e' molto piu' weak nella parte meno illuminata. Se facessimo thresholding come abbiamo fatto finora, sicuramente l'immagine in output avrebbe **edge striking**, quindi sarebbe spezzettata (a meno che il valore della threshold non sarebbe settato a un valore bassissimo). Usando il thresholding visto col *Canny's detector* invece **POSSO TRANQUILLAMENTE DETECTARE TUTTI I PUNTI DEGLI EDGES**.
![[cv20 1.png]]
# Zero-Crossing
Cosa succederebbe se prendessi la *second derivative del valore dell'intensita' dei pixels*?![[cv22 1.png]]
Beh quando ho un *local maximum* nella *first derivative* ho 0 nella *second derivative*.  Questa informazione dello 0 puo' permettermi di *detectare* gli edges senza fare thresholding (nella seconda derivata non ho valori di  magnitudes su cui dover fare thresholding).
**Nota** pero' che usando la *second derivative*  ho 0 pero' anche quando non c'e' nessun cambiamento dell'intensita'.
Si ha un edge **solo quando** si passa da una *second derivative* positiva/negativa a una negativa/positiva, quindi solo quando c'e' un **cambiamento di segno nella second derivative**, da qui il nome **zero-crossing**.
**NOTA** Come ho gia' detto prima non ho bisogno di far thresholding con le *second derivatives*:  se osservo un zero crossing allora ho un **edge** E HO FATTO.

**PROBLEMA**: e' pesante calcolare la second derivative perche' devo calcolare la **HESSIAN MATRIX PER OGNI PUNTO**.
## Laplace operator
Marr & Hildreth hanno proposto di approssimare la *second order derivative* usando il **LAPLACE OPERATOR** che e' un'approssimazione. Approssimo la second order derivative facendo la somma dei diagonal values della HESSIAN MATRIX. ![[cv23 1.png]]
Questa approssimazione e' abbastanza buona.

POSSO APPROSSIMARE LA LAPLACE in modo discreto, prendendo i valori dei pixels. Posso usare le approssimazioni gia' viste per il FIRST ORDER PARTIAL DERIVATIVES. FACCIO DIFFERENZA TRA FIRST ORDER PARTIAL DERIVATIVES stimate come le stimavo prima, ottenendo quini SECOND ORDER PARTIAL DERIVATIVES. Posso usare o forward o backward per calcolare le first order partial derivatives che verranno poi sottratte per ottenere le second order partial derivatives usando anche in questo caso o forward o backward.
![[cv24 1.png]]
**NOTA**: qua sopra si usa *forward* per il first order e *backward* per computare la second order. Quando ho $I_{xx} \text{ e } I_{yy}$ allora posso SOMMARLE.

**UN ALTRO PROBLEMA** ho bisogno di un qualcosa che faccia smoothing, perche' la presenza del noise e' ancora piu' ingombrante e pervasiva quando si usano le *second order derivatives*.
## Laplacian of Gaussian (LOG)
L'algoritmo diventa quindi
1. levo il noise usando gaussian smoothing
2. faccio second order derivatives usando il *Laplace operator*
3. computo zero-crossing dell'output di queste second derivatives. (quindi se ho 0 come second derivative e prima ho negativo/positivo e poi ho positivo.negativo allora ho un edge)
**NOTA**: e' molto raro che io abbia $[10, 0, -20]$, ovvero un cambiamento del valore della *second derivative* con 0 all'interno (solitamente sara' $[10, -20]$), quindi  USO SOLTANTO Il fatto che ci sia un **CAMBIO DI SEGNO**.
Quale scelgo tra $10 \text{ e } -20$ per mettere un edge? Prendo il valore assoluto della partial derivative piu' basso. Nel nostro caso sara' il punto riferito a 10 il mio edge point.

Solitamente si usa un thresholding  finale per rimuovere qualche **sporious edge**.
Se faccio infatti poco noise filtering allora si mi si formeranno spurious edges. 

![[cv30 1.png]]
Nell'immagine qui sopra e' mostrato come  se cambio sigma (la sigma del gaussian smoothing ),quindi se uso un larger sigma, allora il laplacian of gaussian mi detecta oggetti a una granularita'maggiore, mi detecta sempre meno sporious edges dovuti al noise.
**NOTA** posso decidere il livello di smoothing (cosa che comunque era possibile fare anche con *Canny's edge detector eh*).
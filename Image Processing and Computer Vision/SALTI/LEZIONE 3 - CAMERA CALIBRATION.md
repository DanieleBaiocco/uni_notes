Ora faccio *CAMERA CALIBRATION*.
Questo permette di scoprire i **PARAMETRI** della $P$, dati punti nel 3D world e il loro map nell'image plane (quindi le cooridinate dei pixels di questi 3D points nel 2D).
![[cvsalti12.png]]
# CAMERA CALIBRATION
Date tante *coppie* di 2D e 3D points posso stimare gli *unknowns*. 
Come ottengo queste coppie pero' aspetta? 
Eh adesso vediamo.
Comunque la Calibration avviene prendendo 3 o piu' immagini di una scacchiera (comunque qualcosa che ha dei patterns ben definiti) vista da diverse angolazioni.
## Zhang's method
E' un metodo per appunto fare *calibration*![[cvsalti52 1.png]]
* Nei boxes in green sto minimizzando un errore algebrico
* Nei boxes gialli sto minimizzando un errore geometrico (quindi di loss legata alla norm)
OpenCV *cv.calibrateCamera* utilizza internamente questo metodo alla base.

## PRIMO STEP del Zhang's method - acquisizione immagini
Allora, partendo dal primo step, prendo delle immagini di una scacchiera posizionata su una superficie rigida. Questo perche' sono necessari dei *patterns ben chiari*.
Dato un pattern a *scacchiera*, e' devo ottenere 
1. il numero di *corners* interni: quindi quelli in cui due o piu' quadrati neri si incontrano. Solitamente voglio che il numero sia dispari in una dimensione e pari nell'altra per evitare delle ambiguita' legate alla *rotazione* della scacchiera. Voglio quindi un pattern che non sia ambiguo. Infatti se noti nella foto sotto, per ogni riga ho che la scacchiera FINISCE una riga con white square, una con black square (NON SEMPRE CON BLACK SQUARE COME IN QUESTA IMMAGINE: ![[cva.png]] che e' difatti molto ambigua perche' se la ruoto ottengo la stessa immagine e non va bene)
2. la grandezza dei quadrati che formano il pattern a scacchiera (in cm o mm insomma)
Questo e' il mio input diciamo, le foto che io scatto.
![[cvsalti314.png]]
Posso trovare la posizione e il numero degli internal corners utilizzando algoritmi quali l'*Harris corner detector* ad esempio (quelli che ho visto con Lisanti insomma). Posso invece ottenere la grandezza dei quadrati *semplicemente* MISURANDOLA irl.

La parte 2D e' risolta diciamo, con Harris. *NOTA* che devo risolvere la situa per i 3D points.

Definisco quindi il mio WRF. lo metto proprio sopra il pattern a scacchiera eheh. In un pattern *non ambiguo*, come quello che abbiamo di fronte della scacchiera, posso definirlo in modo tale che
1. per ogni foto abbia l'origine nello stesso corner 
2. il pattern e' presente in z=0. Quindi praticamente posso far come se z non esistesse (perche' i punti a cui sono interessato sono quelli in cui z e' sempre 0, ovvero i punti della scacchiera)
3. gli assi x e y sono allineati alla scacchiera (esempio magari con x lungo il lato piu' corto e y lungo quello piu' lungo)
Date queste regole e la grandezza dei **quadrati** e' possibile definire 3D coordinates per ogni **CORNER** nell'immagine del pattern a scacchiera.
![[cxvadfa.png]]
Ad esempio del punto in figura:
![[cvqsda.png]]
 le coordinate a esso legate sono $[3 \cdot 0.6, 6 \cdot 0.6, 0] = [1.8, 3.6, 0]$ 


Nota che questo WRF esula ovviamente dalle immagini. Posso calcolarlo partendo da un'immagine e questo **e' lo stesso** anche per le altre perche' e' legato a punti 3D (sembra una banalita').
**MA** la posizione del WRF, **CAMBIA** da un'immagine all'altra in base a dove sta la camera quando ho scattato la foto:
![[cvda.png]]
Perche' ovviamente la posizione relativa del WRF rispetto alla CRF e' **diversa**, e per ogni foto cambia. Questa cosa e' legata agli *extrinsic parameters* tra l'altro (non so se ricordi ma e' solo legata a loro, perche' si attua una *roto-translation che e' diversa ogni volta che scatto una foto a una determinata posizione della camera rispetto al WRF*). Gli intrinsic parameters non sono per niente condizionati dalla posizione della camera rispetto al WRF. Infatti nella figura qua sopra voglio STIMARE per la prima immagine $R_1, t_1$, per la seconda due differenti $R_2, t_2$. Ho quindi due differenti roto-translations (queste saranno sempre differenti in ogni immagine, ognuna ha la sua roto-translation wrt il WRF).
Quindi stimo tante extrinsic matrices quante sono le immagini prese per fare la calibrazione.
Quindi se ho 10 immagini, devo stimare **UNA** intrinsic matrix e **10** extrinsic matrices AHAHAH assurdo.

**NOTA** un'altra cosa assurda e' che la distortion varia a seconda di quanto sono lontano dall'image center. Quindi per stimare bene la distortion ho di prendere parecchie immagini magari (magari proprio 3 no diciamo), e i patterns e' meglio che siano **negli angoli dell'immagine acquisita dove la distortion e' piu' forte** (quindi lontano dal centro).



## SECONDO STEP del Zhang's method - guess iniziale sull'homography $H_i$
Per ogni immagine computo una initial homography $H_i$. 
Come si fa? Beh guarda qua sotto
![[cvplanar.png]]
Ho praticamente come passare da 3D a 2D, pero' nel mio caso z=0, di conseguenza posso levare una colonna in $P$. In questo modo arrivo a una matrice 3x3 che altro non e' che un'*homography*, ovvero una matrice che mi fa passare **da un projective PLANE** (quello legato al mio punto in 3D di cui son solo rimaste le coordinate $x,y$ ) e **un altro projective PLANE** (quello legato al punto nell'image plane 2D). Parlo di projective plane perche ho un piano che viene progettato nel suo projective space.

### Come stimare $H_i$
**NOTA** che io ho, sapendo la posizione degli scacchi, costruito delle COPPIE di punti 3D-punti 2D.
Uso l'algoritmo DLT.
Ricorda che io ho $m$  Frames ($m$ CRF, quindi $m$ immagini di base) e $c$ corners (sono sempre $c$ in ogni immagine). Quindi ho:
![[cvcv 1.png]]
Nota quindi che $\tilde{m_{i,j}}$ e' riferita all'i-esimo frame (all'i-esima immagine) e al j-esimo *corner*. Nota che $x,y$ non sono legate al frame, le loro coordinate sono semrpe le stesse indipendentemente dall'immagine (sono legate al WRF). Sono ovviamente solo legate ai corners, perche cambiano da corner a corner.

Nota pure che c'e' $H_i$, il che risultera' in $m$ homographies.
Nota che la moltiplicazione tra matrice e' riducibile come mostrato a 3 equazioni. Ho $c$ corners quindi potrei stacckare queste equazioni e risolvere un sistema lineare con $3c$ equazioni.

C'e' pero' un problema, io voglio risolvere un altro problema: 
* Non voglio portare il punto del projective space $\tilde{w}_j$ a essere progettato esattamente, tramite una trasformazione lineare, nel punto $\tilde{m}_{i,j}$ attraverso $H_i$.
* Voglio piuttosto trovare dei valori per $H_i$ che, facendo $H_i \tilde{w}_j$, mi portino a un punto della **retta** del projective space, perche' da li' poi posso tornare in $\tilde{m}_{i,j}$, dividendo per il k del punto trovato. 
Guarda qua 
![[cvcazzo.png]]
Come faccio a dire che questi due punti $\tilde{m}_{i,j}$ e quello che trovero' possono anche NON essere gli stessi ma **devono** stare sulla stessa linea?
Beh aggiungendo l'equazione che mi assicura il constraint che questi due punti debbono stare sulla stessa linea. Infatti due punti stanno sulla stessa linea se il loro **cross-product** e' uno zero vector. Il cross-product misura l'area del parallelogramma creato dai due vettori (sarebbe una specie di determinante tipo? Stava in 3Blues1Brown).
![[cvcvcvc 1.png]]Quindi ho alla fine un sistema di equazioni in cui c'e' questo constraint encodato praticamente. 
NOTA che il sistema di equazioni di prima e' da buttare e uso solo questo praticametne. 
Posso riscrivere il sistema nel seguente modo:
![[modo.png]]
Quindi per ogni frame risolvo questo, stacco $c$ frames e risolvo per tutti. Si e' visto pero' che sono solo 2 le equazioni linearmente indipendenti, non 3, quindi posso anche risparmiare computazione levando la terza riga:
![[res 1.png]]
Questo sistema di equazioni e' **underdetermined** (ha 2 equazioni e 9 *unknowns*, con 2<9).
Di conseguenza estendo questo sistema a *tutti i corners c*, in modo da ottenere un sistema lineare **overdetermined** e omogeneo (ovvero che ha un **vettore di 0s**).
Il sistema e' il seguente, da notare che l'*h* e' giustamente sempre la stessa, perche' la roto-translation performata e' la stessa (essendo che stiamo parlando di tutti i corners presi dallo stesso CRF) e gli intrinsic parameters ovviamente sono gli stessi.
![[cv1 2.png]]
Ho quindi **2c** equazioni. La matrice grande e' $2c\text{ x } 9$ di grandezza. Per quanto riguarda il vettore $h$, questo ha invece dimensione $9 \text{ x } 1$, e quindi ho un sistema **overdetermined**, perche' le equazioni sono $2c$ mentre gli *unknowns* sono $9$ (e $2c > 9$, in quanto i *corners presenti* in una foto sono piu' di 4 (9 diviso 2)).
**nota che visto che questo e' un sistema omogeneo, ho una soluzione banale che e' h=0**. Noi vogliamo trovare soluzioni con un constraint addizionale come ad esempio $||h||=1$. La norma del column vector deve essere 1. Quindi prendo un preciso *scale factor che e' 1*.
Quindi il problema che voglio risolvere e' il seguente:
La soluzione $h^*$ e' trovata minimizzando la norma del vettore $L_i\text{ } h_i$ (non chiaro, quindi non dico esplicitamente che debba essere il vettore 0, ma che la loro norma comunque deve essere la piu' piccola possibile), col constraint che $||h_i|| = 1$.

$$h_i^* = \text{argmin}_{h_i \in I\!R^9} ||L_i \text{ }h_i|| \text{ s. t. } ||h_i||=1$$

Non si puo' risolvere un **overdetermined** system in maniera **ESATTA**, a meno che io non abbia $2c = 9$. Quindi **devo** **RILASSARE** il requirement  per il quale voglio che $L_i \text{ }h_i$ siano **esattamente** 0 in uno piu' semplice che e': il risultato deve portare a un mapping di un World Point in 3D a un punto in 2D che sia il piu' vicino possibile al **corner in 2D legato a quel world point**. Quindi formulo il problema come uno di minimizzazione in cui voglio che si arrivi a $L_i \text{ }h_i$ piu' piccolo possibile.

La soluzione di questo problema puo' essere trovata usando **SVD** di $L_i$.
Praticamente e' risaputo che la soluzione di questo problema di ottimizzazione (che ha la forma di sopra) e' trovabile tramite SVD nei testi di *linear algebra*. 
La SVD risultante da $L_i$ e' la seguente:
1. divide in $U_i$ che e' da vedere come una rotation matrix di shape $2c \text{ x } 2c$ ortonormale
2. in $D_i$ che e' una *diagonal matrix* che scala ogni asse di shape $2c \text{ x }9$
3. in $V_i^T$ ho invece una matrice che e' sempre una *rotation matrix* ma di shape $9\text{ x }9$ ortonormale
In particolare, la soluzione e' $h_i^* = v_9$, che sarebbe l'ultima colonna di $V_i$,
Difatti questa colonna ha shape $9x1$, che e' la shape che volevo. 

Quindi da $m$ CRF, avro' $m$ homographies

## TERZO STEP - minimizzare il Reprojection Error
Per adesso quello che e' stato fatto e' stata la seguente cosa:
ho un punto nell'homogeneous space del WRF $\tilde{w}_j$ e un $m_j$ e ho imparato un mapping $H_i$ ,
![[cv2 3.png]]
Questa $H_i$  mappa $\tilde{w}_j$ in un punto vicino a $m_j$ (il piu' vicino possibile).
![[cvsalti.png]]
Nota che nell'immagine ho $w_j$ e non $\tilde{w}_j$. Voglio infatti **ALLA FINE** uscire dal *projective space*. Nota che le *lenti introducono una distorzione lineare*, mentre quello che ho imparato (almeno come first guess) e' una $H_i$ che fa una **trasformazione lineare**. Di conseguenza questa *omografia iniziale imparata* non sara' molto buona, proprio perche' non ho preso in considerazione l'effetto delle lenti, che creano distorzioni non lineari.

L'errore che si minimizza **risolvendo il sistema lineare di prima** e' chiamato **l'errore ALGEBRICO**, che non coincide perfettamente con quello che **veramente vorrei minimizzare**. L'errore che vorrei minimizzare in verita' e' che vorrei che $H_i$ crei un mapping tale che la distanza tra il **punto rosso e quello blu** sia minore possibile. Questo e' il vero errore che voglio minimizzare (voglio le coordinate dei pixels che siano le piu' vicine possibile).

Tra l'altro nota che $H_i \text{ }w_j$ non e' altro che
![[cvcv 2.png]] ovvero semplicemente l'outcome di sopra, in cui si divide per $k$ ovvero la terza coordinata $h_3^T \text{ }\tilde{w}_j$ trovata per passare da *projective space* a *euclidian space*.
Guarda questa figura per capire cosa sta succedendo
![[cvplanar.png]]
dove faccio $m = \tilde{m} / k$ 
Devo quindi **minimizzare** questo errore adesso, che e' il **reprojection error**, ovvero
$$||m_j-H_i\text{ }w_j||$$
per  ogni punto predetto di un determinato corner. Questa minimizzazione dell'L2 norm e' chiamata minimizzazione dell'**errore GEOMETRICO** anche chiamato reprojection error per l'appunto.

**DOMANDA**: perche' non l'abbiamo minimizzato da subito st'errore? Eh perche' rappresenta un errore **non lineare** (ha il quadrato) e quindi e' molto difficile da minimizzare senza una guess iniziale.

E come risolvo una ottimizzazione di una funzione *non-lineare*? BEH, con **iterative methods** come **Gradient Descent bro**.
In questo contesto si usa un altro iterative method che si chiama **Lebenberg-Marquardt algorithm**, non proprio il GD insomma.
Quello che voglio trovare messo in maniera matematica e':
![[cvcost.png]]
Visto che e'  un'ottimizzazione iterativa, e' molto importante avere una **buona INITIAL SOLUTION**. ECCO PERCHE' ho fatto lo **STEP 2**, l'initial guess per l'omografia, perche' e' un buon punto iniziale per fare un'ottimizzazione non lineare.

Nello Zhang's Method c'e' sempre questo alternarsi tra il trovare una buona stima di un qualcosa, ottimizzando un errore algebrico, per poi usare quella stima come input per un **non-linear refinement** minimizzando l'errore geometrico. Gli steps seguenti pure seguiranno questo pattern.

**NOTA** che alla fine di sto step ho che ho stimato 3 delle 4 colonne  della **PPM** P (quindi mi manca la quarta) e non ho ancora introdotto/trovato i parametri per le *lenti*.

## QUARTO STEP -  trovare una guess iniziale per A date le $H_i$
Per stimare la colonna tolta prima, devo prima dividere cio' che ho trovato in *intrinsic e extrinsic parameters*. So che esiste questa scomposizione in *intrinsic matrix* A e *extrinsic matrix* $[R_i|t_i]$.
![[cvso.png]]
**NOTA che** le H che ho trovato altro non sono che:
$$H_i = [h_{i1}, h_{i,2}, h_{i,3}]=[k\text{ }Ar_{i,1}, k\text{ }Ar_{i,2} k\text{ }At_{i}]$$
perche' ho levato la **terza colonna** (h ha la prima, la seconda e la quarta colonna della PPM).
Perche' metto la $k$? Perche' la PPM e' una trasformazione tra projective spaces, quindi lavora in homogeneous coordinates. Quindi quando dico che metti la prima colonna della PPM e' uguale alla prima della H, queste due sono uguali **secondo uno scale factor k**. Cioe' ho che $h_{i,1} = k Ar_{i,1}$ , ho l'UGUALE come vedi, non ho l'equivalence. Quindi devo introdurre uno scale factor *k* per avere l'equivalenza tra projective spaces.
Ottengo due equazioni (non so perche' non e' presente anche la terza con $t_i$)
![[cvcvcvc 2.png]]
Di queste ho come incognite $k, r_{i,1}, r_{i,2}$ e $A$. Conosco solo $h_{i,1}, h_{i,2}$.
Ma **so che r e' un vettore colonna di una rotation matrix**. Essendo di questo tipo deve seguire il constraint per cui ogni due vettori colonna di una rotation matrix sono **ortogonali** anche se li scalo di una quantita' *k* (rimangono ortogonali anche  se li scalo). Questo significa che $kr_{i,1}$ deve essere ortogonale con $kr_{i,2}$, perche' sono due colonne scalate (e non cambia niente) di una rotation matrix. Ne segue che anche le due quantita' a dx (i due Ah) devono essere ortogonali tra loro, quindi ho
![[cvragionando.png]]
Questo dot product si puo' scrivere come:
![[re.png]]
Ora ho un'equazione in cui conosco *h* e non conosco A (non ho piu' le fastidiose *r* e *k* che non conoscevo).

Un'altra cosa sulle colonne della rotation matrix e' che hanno *unit length*, quindi la loro norma e' 1, per TUTTE. Questa unit length vale solo se **non sono moltiplicate** per qualcosa tipo *k*. PERO' posso dire che , visto che k e' uguale e non cambia,  la norma di $r_{i,1}$ e quella di $r_{i,2}$ devono essere UGUALI.
Da questo ho $<r_{i,1}, r_{i,1}> = <r_{i,2}, r_{i,2}>$ e da qui se sviluppo ottengo un altro constraint:
![[cvsd.png]]
**NOTA** che **data una CRF ho SOLO questi due constraints a essa legati**. (non e' per ogni corner perche' ho imparato una omografia UNICA ovviamente che **vale per ogni corner**).
Quindi ho 2 constraints per ogni immagine.
Perche' ci siamo detti di fare almeno 3 foto per la calibrazione? Beh perche' se ne facessi solo una avrei solo 2 constraints in sto QUARTO STEP dell'algoritmo. L'UNKNWON $A^{-T} \text{ }A^{-1}$ e' SIMMETRICA (per motivi che non vedremo) e ha 6 entries indipendenti, 3 sulla diagonale  e 3 fuori dalla diagonale. Quindi avrei 2 equazioni su 6 incognite e avrei un sistema **underdetermined** (penso sia questo il motivo). Con 3 immagini avrei 6 constraints quindi 6 equazioni su 6 incognite e quindi posso trovare una soluzione.
In sto modo posso avere una guess iniziale per A (l'intrinsic param matrix)

## QUINTO STEP - DATO A e $H_i$ avere una guess iniziale per $R_i$ e $t_i$ (gli extrinsic parameters)
Una volta che computo $A$ e $h$, allora posso  trovare $r$.
Ricordo infatti che avevo
$$H_i = [h_{i1}, h_{i,2}, h_{i,3}]=[k\text{ }Ar_{i,1}, k\text{ }Ar_{i,2} k\text{ }At_{i}]$$
Da cui posso derivare $$r_{i,1}=\frac{1}{k}A^{-1}h_{i,1}$$
Ricordo che NON conosco $k$, pero' so che $r_{i,1}$ e' uno unit vector in quanto colonna di una ortonormal matrix. Di conseguenza $||r_{i,j}||=1$, quindi $||\frac{1}{k}A^{-1}h_{i,1}||=1$ sse $k=||A^{-1}h_{i,1}||$. 
Nota che k e' la stessa, quindi posso usarla per calcolare $r_{i,1}, r_{i,2} \text{ e }t_i$:
$$r_{i,2}=\frac{1}{k}A^{-1}h_{i,2}$$
$$t_i=\frac{1}{k}A^{-1}h_{i,3}$$
Mi manca solo $r_{i,3}$ e poi ho tutti gli *extrinsic parameters*.
Posso calcolare $r_{i,3}$ tramite il **cross product** tra $r_{i,1} \text{ e }r_{i,2}$. Primo perche' il crossproduct mi da' un vettore che e' perpendicolare al piano in cui questi due vettori $r_{i,1} \text{ e }r_{i,2}$ creano (e e' quello che voglio, voglio $r_{i,3}$ ortogonale a entrambi), inoltre  il cross-product e' l'area del quadrato creato dai due vettori. Essendo entrambi $r_{i,1} \text{ e }r_{i,2}$ di unit norm, **l'area sara' 1** e quindi $r_{i,3}$ sara' di unit norm anche lui. 
Il Prof. dice che PERO' ci saranno sempre dei problemi dovuti al noise, anche nella computazione di *k* partendo da $r_{i,1}$, che viene applicata poi a $r_{i,2}$. Quindi praticamente la matrice risultante $R_i$ calcolata **NON SARA' perfettamente ortonormale**, in quanto $r_{i,1} \text{ e }r_{i,2}$ non sono perfettamente ortogonali e $r_{i,2}$ non e' perfettamente di lunghezza 1 visto che $k$ e' stato computato partendo da $r_{i,1}$. Immagina il risultante $r_{i,3}$ che nasce da due vettori non proprio ortonormali.
Per far in modo che questa matrice, dopo l'aggiunta di  $r_{i,3}$, sia piu' ortonormale si utilizza **SVD**. Il modus operandi e' che $R_i$ viene scomposta in $U, D, V$ e viene sostituita la D con l'identita', in sto modo ho una matrice ortonormale data da $UIV$, perche' sia $U$ che $V$ sono ortonormali quindi sono sicuro di avere ortonormalita'.

## SESTO STEP - stimare i parametri delle lenti
Ho tutti i parametri della PPM ma mi mancano quelli della distorsione delle lenti.
Questi parametri sono ovviamente **intrinsic**, quindi esistera' solo un set di questi parametri **per tutte le immagini**, per tutti i CRF (come per l'intrinsic matrix A).
Zhang's method utilizza solo 2 coefficienti $k_1, k_2$.
Nota che io voglio stimare $k_1, k_2$ data la seguente formula per la *radial distortion*:![[cvss.png]]
Nota che $x,y$ sono le coordinate in millimetri perche' lo step della *lens distortion* e' il terzo step e avviene dopo *esser passati da WRF a CRF* e dopo aver fatto una ***canonical** perspective projection* dividento per $z$. Io posso andare dalle coordinate in pixels **trovate** dall'Harris Corner Detection a queste coordinate $x,y$ semplicemente applicando  $A^{-1}$ alle coordinate in pixels.
**NOTA PERO' che non ho** le coordinate PURE, quelle per cui non ho distorzione delle lenti.
Per approssimare questa purezza, utilizzo le omografie imparate: le coordinate predette dalle omografie imparate, partendo da punti del WRF, corrispondono alle coordinate ideali $m_{\text{undist}}$che non sono perturbate dalla distorzione delle lenti.
NOTA che questa e' una assunzione, ovviamente non e' cosi. 

Per passare da $u,v$ coordinate dei corners dell'Harris Corner Detector a $x,y$ non si usa proprio l'inversa ma si fa semplicemente cosi:
![[cvback.png]]
La stessa trasformazione avviene per $u_{\text{undist}}, v_{\text{undist}}$, che sarebbero le coordinate predictate usando la PPM imparata partendo dalle coordinate 3D del WRF. Queste vengono portate in $x_{\text{undist}}, y_{\text{undist}}$.
Quindi poi la formula diventa:
![[cvcvcvcvcvc.png]]
Posso applicare delle trasformazioni lineari
![[tras.png]]
per ottenere un sistema **lineare, non-omogeneo** di equazioni lineari $Dk = d$ con gli unknowns $k = [k_1, k_2]^T$. Con *c* corners in *n* immagini ottengo $2nc$ equazioni con 2 unknowns (quindi ottengo un **overdetermined system**), che puo' essere risolto risolvendo un *least squared error minimization problem*, pero' utilizzando la matrice pseudoinversa (perche' questo tipo di problema, oltre che con GD, puo' essere risolto in modo *esatto* diciamo, tramite una formula. Quindi **non** in modo iterativo)
![[least.png]]
**NOTA** che non ho stimato *p*. Lo skippo immagino.

## STEP 7 - approssimare meglio tutti i parametri minimizzando il reprojection error
Voglio minimizzare la differenza tra il punto computato da Harris Corner Detector e la predizione del mio *modello*, in cui ho tutto (sia PPM che lens distortion parameters tutti estimati).
![[dfasfd.png]]
Si utilizzano algoritmi iterativi per ottimizzare sta cosa. Solitamente il **residuo** (il MIN trovato) di questa loss diciamo e' usato come parametro per vedere quanto e' buona la calibration. Se alla fine lo ho alto ho fatto una brutta calibrazione.

# COSE CHE POSSO FARE UNA VOLTA CHE HO UNA CAMERA CALIBRATA
## COMPENSATE LENS DISTORTION
La prima cosa che faccio dopo aver calibrato la mia camera, e' **levare la distortion** dalle immagini. Questa cosa si fa perche' e' molto piu' facile lavorare con un modello **lineare** che non abbia al suo interno la **non-linearita'** che la distorzioni data dalle lenti introduce.
L'operazione che leva la distortion e' chiamata  **warping**.
Per ora ho gia' visto il **filtering**, quando faccio una convoluzione con un kernel su di una immagine. E' chiamata cosi' perche' lavoro sui valori rgb dell'immagine per cambiarli, senza muoverli in altri posti: la posizione spaziale dei pixels rimane la stessa, sto **solo cambiando l'intensita'** dei pixels.
### WARPING
Nel **warping** **MUOVO** i pixels, quindi ne cambio la loro collocazione spaziale (non ne sto cambiando i valori di intensita').

Il **warping** permette di computare una nuova immagine $I'$ data $I$.
Quindi devo sapere, dato un input pixel $u,v$, dove mettere il valore $u',v'$ in $I'$.![[arigato.png]]
Questo lo faccio utilizzando un $w$ che mi dica come passare da un pixel in un'immagine all'altro.
Posso vedere la **w** come un'omografia.![[xcvsll.png]]
Solitamente si fa **forward mapping**, ovvero 
1. si parte una coppia di coordinate **intera** di pixel,
2. questa viene mappata a un'altra coppia di coordinate che pero' non e' intera ma *float*
3. si fa un'approssimazione tramite rounding alla coppia di coordinate piu' vicina intera
4. si assegna a quel pixel il valore del pixel nell'immagine di partenza
Il problema e' che:
1. Piu' *pixels* potrebbero finire, a causa del **rounding**, nello **stesso** pixel (fenomeno chiamato **fold**)
2. Qualche *pixels* nell'immagine di destinazione potrebbero anche **non** essere messi (fenomeno chiamato **holds**). Certi pixels non sono mai il target di un input pixels.
Allora provo a fare **backward mapping**: si inizia da $I'$, per ogni pixel della target image (intero ovviamente), dove va questo pixel nell'immagine originale? Quindi invece di usare $w$ si utilizza $w^{-1}$. Ora ho pero' lo stesso problema di prima, ovvero che ho *NON INTEGER COORDINATES* come risultato della trasformazione. AD ogni modo ora il problema e' nella **source**. ![[cv2314.png]]
Come si puo' vedere dall'immagine il pixel in cui si arriva non e' intero. Per renderlo intero si fa  *o truncation, o Nearest Neighbour, o interpolation*(ad esempio nell'immagine si finisce su un pixel che e' blu, ma comunuqe il punto e' finito a margine di questo pixel blu, a confine con 3 pixels rossi. NON voglio che il valore ritornato sia  BLU, ma voglio quindi che il valore di intensita' ritornato sia una *interpolation* tra questi pixels). Un risultato da questa strategia verra' fuori e assegno quel risultato al pixel nell'$I'$. Pero' in sto modo ho PER OGNI PIXEL della target un valore ritornato (non ho piu' il problema degli HOLES). Capisci che prendendo singolarmente ogni pixel nella target image HO A OGNI PIXEL ASSEGNATO UNO E UN SOLO VALORE DI INTENSITA'. 

### Bilinear interpolation
Qui ho $u,v$ che e' dove finisco con il *warping inverso*. E poi ho dei punti $I_1,I_2,I_3,I_4$ che rappresentano i valori di **intensita'** dei pixels agli angoli, e identificabili con le coordinate $(u_1,v_1), (u_2, v_2), (u_3, v_3), (u_4,v_4)$.
Quindi questi $I$ sono i nearest neighborhoods del punto $u,v$ in cui finisco.
![[cvinterpolation.png]]
Calcolo $\Delta u = u - u_1$ e $\Delta v = v - v_1$.
Voglio fare *interpolazione lineare* per determinare il valore di intensita' di $u,v$, Vorrei che il risultato fosse una *interpolazione lineare* tra i valori di intensita' $I_a, I_b$, che ancora non so come calcolare.
![[iaib.png]]
Per calcolarli **faccio anche per loro una linear interpolation**, per $I_a$, faccio linear interpolation tra $I_1$ e $I_2$, per $I_b$ tra $I_3$ e $I_4$. Sotto e' presente questa linear interpolation per $I_a$:
![[dfa.png]]
La formula da derivare per fare questa linear interpolation e' la seguente:
$$\frac{I_a - I_1}{\Delta u} = I_2  - I_1$$
**Nota** e' stato omesso il diviso 1 nella parte di dx della formula. E' letteralmente una proporzione. Da questo calcolo $I_a$: $$I_a = (I_2 - I_1)\Delta u + I_1$$
A livello di interpretazione, piu' sono vicino a $I_2$ (indicato da un $\Delta u$ grande), piu' la parte di destra $(I_2 - I_1)$ contribuira'. Nota che se $\Delta u$ e' 1  (al limite), allora il risultato e' $I_2-I_1+I_1=I_2$.
Ottenute queste due posso calcolare 
$$I(\Delta u, \Delta v)= (I_b - I_a) \Delta v + I_a$$
Alla fine la formula finale sara':
![[bilinear_interpolation.png]]
Se la riscrivo meglio diventa:
![[bilinear_interpolation2.png]]
Ho quindi una **combinazione lineare** tra $I_1, I_2, I_3, I_4$. Pensa alle varie posizioni in cui $(\Delta u, \Delta v)$ puo' stare per capire come la contribuzione dei 4 pixels nell'intensita' finale cambia.
Se uso per zoommare un warp con Nearest Neighborhood ho una roba pixxellata, nel caso della *bilinear interpolation* ho invece una roba piu' smoothed.
![[bilinear_nn.png]]
Ogni volta in cui warpo qualcosa in cui la ***transizione** deve essere preservata*, allora Nearest Neigborhood deve essere usata. Ad esempio se ho una *binary mask* della mia immagine (magari dopo un segmentation algorithm). Se voglio poi fare **warping**, per conservare il fatto che ci sono SOLO 1 E 0, allora NON POSSO FARE **bilinear interpolation**, ma in generale nessun tipo di interpolation, perche' cambierebbe i valori e NON sarebbero piu' O 1 O 0, ma anche valori in mezzo a 1 e 0.

### UNDISTORT WARPING
Ora finalmente posso parlare di come fare un warping per levare l'effetto delle lenti imparato dallo step della *calibration*. 
Posso vedere le formule della distorzione delle lenti come un **warping**.
Quindi quello che si fa e' la seguente cosa, usando **backward warping**:
1. Per ogni pixel nell'immagine *ideale*, che e' quella che voglio creare, si fa backward warping, quindi andando a vedere dove $w^{-1}$ mi porta. Cio' che viene fatto e' la seguente cosa di base. ![[formula.png]]. Con $w^{-1}$ che e' 
 ![[radial.png]]
 Quello che viene fatto e' quindi:
![[cvcvcvcvcvvccvcvcvcv.png]]
1. Qui finiro' in un punto che NON ha coordinate intere e applichero' come visto prima una tecnica quale truncation, Nearest Neighborhood o bilinear interpolation


In sto modo ottengo: 
![[ottengo.png]]

## FARE WARPING DELL'IMMAGINE SU UNA CAMERA VIRTUALE
Posso ad esempio posizionare una nuova camera virtuale nello spazio e chiedermi come sarebbe l'immagine se fosse vista da quella camera.
Questo qua sotto e' un esempio di warping dell'immagine a una camera che e' posizionata SOPRA, quindi con visuale dall'alto.
![[cvsopra.png]]
Cio' e' possibile utilizzando il fatto che **ogni due immagini di una scena piana (quindi che ha la coordinata z del WRF a 0)** sono legate da un'**omografia** che permette di passare da una all'altra.
Nota che per far funzionare sta cosa c'e' bisogno che la lens distortion sia levata. 
Questa immagine qua sotto spiega il tutto:
Infatti ho che il WRF e' uguale per entrambe le Camera Reference Frames. Ho inoltre che la linea grigia rappresenta una scena PIANA, come magari puo' essere la strada. Il WRF e' messo in maniera tale che ogni punto espresso nel WRF ha 0 nella coordinata delle z (come ho fatto in calibration con i punti della scacchiera). Ho poi $I_1$ e $I_2$ che sono due image planes di due CRF $C_1$ e $C_2$. E' stimabile ovviamente un' omografia $H_1$ che permette di andare da $\tilde{M}_w$ a $\tilde{m}_1$. La stessa cosa con $H_2$. Posso quindi scrivere una equazione in funzione dell'altra e ottenere un modo per andare da $\tilde{m}_1$ a $\tilde{m}_2$ e viceversa. Quindi ho due **warpings** che mi fanno passare da una all'altra e viceversa, SENZA andare nel 3D. (ho una relazione tra 2D images).
![[fad.png]]
RICORDA SEMPRE CHE QUANDO VEDI IL TILDE E' PER INDICARE CHE SI STA PARLANDO DI HOMOGENEOUS COORDINATES.
Questo **MAPPING** e' corretto SOLO PER PUNTI SUL PIANO, quindi SOLO PER PUNTI DELLA STRADA, non anche per punti piu' alti della strada (quelli con z diverso da 0). Infatti dalla foto di sopra si vede come tutto cio' che e' proprio a livello della strada e' ben approssimato, il resto (le macchine gli alberi) e' **molto distorto**. 

*La domanda che mi chiedo pero' e': come ottengo $H_2$?* Di sicuro ottengo $H_1$ usando camera calibration. Per trovare $H_2$ , essendo il CRF della camera virtuale non calibrabile (visto che e' appunto virtuale), dovrei trovare dei punti salienti da matchare, delle corrispondenze. Poi risolvo usando l'errore algebrico e l'errore geometrico in maniera iterativa. Non chiarissimo eh. non che l'abbia spiegato proprio bene.

## FARE WARPING DELL'IMMAGINE SU UNA ROTAZIONE RISPETTO ALL'OPTICAL CENTER
Cio' e' possibile usando il fatto che **ogni due immagini che son prese da una camera ruotata rispetto al suo optical center sono legate da una omografia** (se la distorzione delle lenti e' stata rimossa ovviamente).

Per vedere questa cosa guarda qua:
![[cvrotation.png]]
Ho una CRF particolare, che e' quella che siede perfettamente dove e' presente la WRF, quindi NON ho roto-translation di alcun tipo.
Di conseguenza, la formula con cui un punto omogeneo nell'image plane $\tilde{m}_1$ e' espresso e':
![[eh minimal.png]]
Posso levare l'ultima coordinata omogenea 1 in quanto ho 0 nella parte della translation. Quindi posso direttamente lavorare con $M_w$.
Se faccio poi una rotazione, ho che la formula nel nuovo CRF sara':
![[nuovoroto.png]]
Questo punto e' esprimibile senza cambiare gli intrinsic parameters ma soltando aggiungendo la rotazione compiuta agli extrinsic parameters. 
Nota che anche qui non ho translation quindi ho 0 sull'ultima colonna, quindi posso lavorare con $M_w$. Ora e' possibile avere due warping, due omografie.
![[alla_fine_ho 1.png]]
Queste omografie sono valide **per ogni punto 3D**, non come prima, in cui andava bene SOLAMENTE per i punti che avevano coordinata z=0 nel WRF.
Sta cosa e' molto utile nelle *self driving cars*.

#### Caso d'uso
Ho un caso d'uso in cui ho una immagine, in cui il WRF e' posizionato proprio dove e' presente il CRF.
Ho anche delle linee parallele nel mondo reale (magari linee di una strada come in figura).
![[das.png]]
Devi sapere che il *pitch* e' di quanto e' ruotata la camera rispetto all'asse orizzontale.

Assumo che il veicolo stia guidando in maniera *dritta* rispetto alle linee. 
Ho che le linee della strada sono **parallele** all'asse delle z del mio WRF (cio' ha senso perche' immagina la z come la profondita' e se ci pensi e' vero, le linee della strada sono parallele).
Di conseguenza queste linee hanno un orientamento esprimibile tramite le coordinate $[0, 0, 1]$. Il loro punto all'infinito e' di conseguenza $[0,0,1,0]$. 

**Voglio trovare, dato questo setup, di quanto ammonta il pitch della foto di destra, voglio quindi stimare l'angolo secondo il quale la camera e' stata ruotata per ottenere poi la foto di destra**.
Posso esprimere il punto all'infinito nel CRF ruotato aggiungendo agli extrinsinc parameters una R che indica una rotazione lungo l'asse delle x (quello orizzontale) **parametrizzata**. Questa e' cosi' descritta ![[Rpitch.png]]
Di conseguenza il vanishing point e' calcolabile nel seguente modo:
![[msegreto.png]]
Ho che l'*unknown* e' $\beta$.
Nota che il calcolo del vanishing point riguarda **solamente** la terza colonna di $R_{\text{pitch}}$.
A questo punto posso proprio calcolare $r_3$, invertendo il risultato  e ottenendo
![[r3.png]]
Nota, qua visto che lavoro in *projective spaces* c'e' sempre un po' di ambiguita', quindi non posso fare direttamente $r_3 = A^{-1}m_{\infty}$. Essendo pero' $r_3$ parte di una rotation matrix che sappiamo essere ortonormale (so quindi che $r_3$ e' e deve essere uno *unit vector*) divido per la norma. A questo punto stimo $\beta$ come l'atan di sin e cos.
A sto punto, ho R, di conseguenza posso computare (COME VISTO SOPRA), l'omografia che lega l'immagine senza pitch di sinistra con l'immagine col pitch di destra. 
Basta infatti che calcolo l'omografia facendo $ARA^{-1}$.j
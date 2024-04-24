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

## PRIMO STEP
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
Quindi se ho 10 immagini, devo stimare UNA intrinsic matrix e 10 extrinsic matrices AHAHAH assurdo.

**NOTA** un'altra cosa assurda e' che la distortion varia a seconda di quanto sono lontano dall'image center. Quindi per stimare bene la distortion ho di prendere parecchie immagini magari (magari proprio 3 no diciamo), e i patterns e' meglio che siano **negli angoli dell'immagine acquisita dove la distortion e' piu' forte** (quindi lontano dal centro).



## SECONDO STEP
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


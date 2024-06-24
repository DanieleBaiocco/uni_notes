 # Differenza tra COMPUTER VISION E IMAGE PROCESSING
1. **IMAGE PROCESSING** ha come obiettivo quello di migliorare la qualita' di un'immagine
2. **COMPUTER VISION** ha come obiettivo quello di estrarre informazioni dall'immagine (classification, error detection, ecc...)
Molto spesso prima si fa IMAGE PROCESSING, per migliorare l'immagine, e poi COMPUTER VISION.
Lo scopo della **COMPUTER VISION** e' quello di INFERIRE INFORMAZIONI da immagini 2D come la profondita' ad esempio, che e' una proprieta' degli oggetti 3D che viene persa quando si scatta una foto.
# Pinhole camera model
E' uno dei modelli piu' semplici che esistano.
Nel _Pinhole camera model_, per ogni **PUNTO** della scena, un solo **RAGGIO** passa. Passando per il buco della camera, il raggio viene proiettato in un **IMAGE PLANE** (_al contrario tra l'altro_).
Se aprissi il **PINHOLE**, non avrei questo MAPPING di  un RAGGIO legato a un PUNTO, ma avrei un mapping del genere: a un raggio ho piu' PUNTI nell'image plane, a esso legati.
E' un modello SEMPLICE ma e' la base per come si modellano LE IMMAGINI nei dispositivi (di come passare da 3D a 2D).

# Perspective Projection
C'e' un MODELLO GEOMETRICO che regola le regole della **PINHOLE CAMERA**.
**NOTA**:
1. P (lettera maiuscola) indica un punto REALE, nella scena che sto osservando
2. p (lettera minuscola) indica un punto nell' **IMAGE PLANE**
La notazione che ho e' la seguente:
1. M: punto di scena che sto osservando
2. m: il corrispondente punto nell'IMAGE PLANE
3. I: l'IMAGE PLANE
4. C: l'**optical center**, ovver il PINHOLE
5. Optical Axis: la linea che passa attraverso C e e' perpendicolare a I
6. c: l'intersezione tra l'Optical Axis e l'**IMAGE PLANE**
7. f: **focal length**, ovvero la distanza tra C che e' il PINHOLE e la sua proiezione perpendicolare c sull'IMAGE PLANE. 
8. F: **focal plane**, e' il PIANO in cui sta C di base, e' parallelo al piano dell'IMAGE PLANE
![[CV1.png]]
**VOGLIO TROVARE una relazione geometrica tra M e m**. 
Per fare cio' ho bisogno di un sistema di coordiante, uno che ha come origine _C_,nel **focal plane** (chiamato CAMERA REFERENCE SYSTEM), e l'altro che ha come origine _c_, nell'**image plane**.
L'image plane ha coordinate _u_ e _v_. 
Per il primo ho due coordinate _x_, _y_, _z_, per il secondo ho coordinate _u_ e _v_.

L'equazione che mappa un punto dal 3D al 2D e' la seguente:
$$\frac{u}{x} = -\frac{f}{z}$$
e anche 
$$\frac{v}{y} = -\frac{f}{z}$$
Da queste ottengo il mapping in quanto poi ho:
$$u = -x\frac{f}{z}$$
e $$v = -y\frac{f}{z}$$**Come derivo queste equazioni?**
![[CV2.png]]
Ho che i rettangoli che si formano UMC e CZM sono **similar**, quindi posso usare **triangle similarity** e da questo derivo le equazioni di sopra.
C'e' un **trick** per fare in modo che le immagini NON risultino **flippate**, ovvero assumere l'image plane davanti al focal plane. Questo pero' necessita che si aggiunga un MENO per il calcolo di _u_ e _v_.
$$u = x\frac{f}{z}$$
 $$v = y\frac{f}{z}$$
 Se guardo _u_ nella formula, ho che x e' **scalato** di una quantita' $\frac{f}{z}$. 
 1. Se _z_ aumenta di conseguenza la risultante _u_ diminuira', il che significa che il PUNTO verra' mappato nell'image plane con una coordinata _u_ piu' piccola. 
 2. Se aumento la _focal length_ **f** , allora ho che la _u_ risultante sara' piu' grande, di conseguenza l'oggetto apparira' piu' grande nella risultante immagine. Con **f** piccolo ho invece un campo visivo molto piu' grande, di conseguenza coordinate nell'image plane molto piu' piccole (e' come se allontanassi cio' che vedo, prendendo una piu' ampia fetta di realta').

La **SCALE** dell'oggetto e' QUANTO L'OGGETTO e' grande nell'IMAGE PLANE.
La **DIMENSIONE** e' invece la reale dimensione nel mondo reale.
La SCALE e' influenzata da _z_ ,_f_ , _x_ 
La SCALE e' importantissima in COMPUTER VISION.
se fisso _x_ e _f_, STO SCALANDO IL MONDO in modo INVERSO rispetto alla DEPTH _z_.
![[CV3.png]]

**NOTA** io PERDO INFORMAZIONI mappando 3D in 2D. Tipo dall'immagine si puo' vedere che NON SI PUO' SAPERE quale dei PIANI del 3D e' stato mappato nell'IMAGE PLANE, in quanto tutti genererebbero la stessa immagine nell'image plane.
HO CHE DA 3D A 2D ho un MAP A UN PUNTO SOLO. **Da 2D A 3D MAPPO A UNA LINEA** (non a un punto). Quindi niente, non posso rendere REVERSIBILE la cosa. La soluzione non e' unica.

# COME STIMARE IL 3D
E' possibile farlo con STEREO IMAGES.
![[CV4.png]]
Non so se $p_L$ venga da **P** o **P'**. 
Se introduco una SECONDA CAMERA, ora ho 2 image planes $\pi_L$, $\pi_R$, 2 _optical centers_  $O_L$, $O_r$.
**C'e' un problema. Io non ho nessun modo per ora per capire se $p_L$ sia uguale a $p_R$ o a ${p_R}'$.**

Ai giorni d'oggi le telecamere hanno  questi **STEREO SENSORS** che capiscono la PROFONDITA' di cio' che vedono, ricostruendola scattando due immagini da camere differenti e trovando corrispondenze tra i punti di queste due immagini.
Vediamo ora piu' nel dettaglio questo problema.

## STANDARD STEREO GEOMETRY
In questo SETUP ho l'assunzione che le due camere siano PARALLELE sia sull'asse y, che sull'asse delle x, in cui sono traslate tra loro.
Hanno inoltre la stessa **FOCAL LENGTH f**, in sto modo i due image planes sono perfettamente allineati.
Assumo $p_L$ e $p_R$ come VETTORI con valori di coordinate 3D (uso il sistema di coordinate del real world x,y,z).

![[CV5.png]]
Ho che $y_l$ e $y_r$ sono uguali:  $y_L = y_R = y$, per come stanno messe le camere.
Anche $z_L$ e $z_R$, stessa situazione anche qui. Gli oggetti osservati sono alla stessa distanza _z_ difatti.
Per quanto riguarda $x_L$ e $x_R$ sono diversi, perche' le camere sono traslate l'una dall'altra sull'asse x. La differenza e' **b**.
![[CV6.png]]
So dall'equazione formulata prima  che $v_L$ e $v_R$ saranno le stesse , in quanto sono entrambe legate a _y_, che e' uguale in entrambi i casi (ovviamente anche _f_ e _z_ sono uguali). Mentre $u_L$ e $u_R$ sono DIVERSE. 
Ho che la loro differenza la chiamo **disparity** _d_.
La formula e' $d = b \frac{f}{z}$, e da questa posso derivare la formula $z = b\frac{f}{d}$. Di questa formula so quasi tutto. Conosco _f_ per come ho costruito le due camere, e conosco _d_ perche' conosco la distanza tra le due camere. L'unica cosa che non conosco e' b perche' NON so di preciso quali sono gli $u_L$ e $u_R$.
Questa quantita' mi serve per capire la DEPTH in 3D.
 L'unica cosa che mi manca e' un ALGORITMO che mi dica che $p_L$ e $p_R$ SIANO gli stessi, cioe' **che appartengano allo stesso punto osservato nel mondo reale**.

VOGLIO ORA QUALCOSA CHE dato $p_L$ trovo $p_R$.
Posso di certo **RIDURRE IL SEARCH SPACE  a una LINEA** perche' so che stanno sullo stesso x. Per poi matchare POTREI guardare agli **INTENSITY VALUES**. Questo potrebbe non essere robusto perche' se ho DIVERSI INTENSITY VALUES UGUALI eh continuo a non saperlo. LO FACCIO non con il singolo pixel MA CON MOLTI PIXEL, COSI' E' MOLTO PIU' ROBUSTO. (cosi se vedo le stesse intensita' allora posso matchare).
Questo vale solo quando si e' paralleli.

## EPIPOLAR GEOMETRY
Se le due camere **non sono parallele**, non posso cercare nella linea orizzontale come prima.
Questa non e' piu' un SETTING di **STANDARD STEREO GEOMETRY** ma  di **EPIPOLAR GEOMETRY**.
In sto caso NON DEVO COMUNQUE CERCARE IN TUTTA L'IMMAGINE.
Se CONOSCIAMO LA RELAZIONE TRA LE DUE CAMERE si puo' progettare la LINE che nasce da $p_L$ (perche da 2D a 3D il punto diventa una linea) e poi prendere la PARALLELA della linea sull'  IMAGE PLANE dell'altra camera. ![[CV7.png]]Quindi ho una linea ora come prima.
La trasformazione tra le due camera nelle posizioni deve essere una **ROTOTRANSLATION** (quindi devo  sapere quanto le due camere  sono ruotate e traslate tra loro), e devo anche considerare le focal lengths.


Questo pero' e' pesante da fare, perche' ogni volta ho delle *epipolar lines* che non sono mai orizzontali, ma sempre di diversa pendenza, come nel caso qua sotto.![[CV8.png]]
## RECTIFICATION
Applico alle immagini una trasformazione chiamata **RECTIFICATION** in modo tale che le due immagini siano ALLIGNED nell'asse x e POSSO ORA CERCARE SOLO TRAMITE PARALEL LINES, aka le epipolar lines sono SEMPRE E SOLO parallele adesso.

• Warp the images as if they were acquired through a standard geometry (horizontal and collinear conjugate epipolar lines) 
• Compute and apply to both images a transformation (i.e. homography) known as rectification
![[CV9.png]]
## STEREO CORRESPONDENCE
![[CV10.png]]
Certi MATCH sono semplici grazie al fatto che ci sono CLEAR PATTERS. Se pero' lo faccio su una FINESTRA, anche guardando i pixel intorno questa la MATCHERO' NELL ALTRA IMMAGINE CON TUTTE LE FINESTRE PRESENTI (questa cosa non va bene). E' per questo che spesso si aggiunge a queste camere una camera a infrarossi CENTRALE che permette di projectare su UNIFORM REGIONS in una maniera migliore perche' comprende bene i patters (praticamente).
**NOTA** che nella foto di sopra la **disparity** tra $u_L$ e $u_R$, nel caso del marciapiede, che e' un oggetto molto vicino alla telecamera, **e' molto grande** (nota lo shift nelle due foto). Invece la disparity di punti lontani, come la ruota della macchina e' molto piccola. Questo e' spiegabile guardando la formula di prima $z = b\frac{f}{d}$.

## Le PROPRIETA' DELLA PROSPETTIVA da 3D a 2D

POSSO ESTENDERE LA NOZIONE DI _u_ scalato da _x_ parlando ora di come le **linee** vengono progettate nell'image plane.
La formula e' $l = L\frac{f}{z}$. Ho quindi che la linea viene scalata. Ma questa formula VALE SOLO nel caso in cui _L_, ovvero la linea nel mondo reale, e' parallela all'image plane. Negli altri casi, non ho solo uno SCALING (quindi a livello di dimensioni dal 3D al 2D), ma anche proprio una distorzione della linea nel 2D.
Distorta significa che nei casi in cui _L_ non e' parallela le proporzioni NON vengono mantenute. Quindi se prendo tre punti in _L_ A, B e C e li progetto nell'image plane in _a, b_ e _c_, avro' che $\frac{AB}{BC} \neq \frac{ab}{bc}$.

## VANISHING POINT
GUARDA LA FOTO IN CUI LE LINEE SONO PARALLELE NEL MONDO REALE MA nell'immagine SONO STRANE sono OBLIQUE  (che e' letteralmente quello che vedo io umano, che distorco linee parallele solo perche non sono parallele al mio image plane)
Queste linee 3D convergono a un VANISHING POINT in 2D.
L'immage plane e' INFINITO nella nostra formulazione, nel nostro modello.

## DEPTH OF FIELD
Ho gia detto che il pinhole era cosi piccolo che passava solo un raggio per un punto.
SE QUESTO E' VERO allora ho **un INFINITO DEPTH OF FIELD**. Quindi **OGNI PUNTO NELL'IMMAGINE E' ON FOCUS**.
Essendo il Pinhole model un modello, il pinhole nel mondo reale NON e' mai cosi' piccolo da permettere questo mapping BIETTIVO tra punto di scena e punto nell'image plane.
Se il pinhole FOSSE cosi piccolo infatti NON avrei ABBASTANZA LUCE. Esistono comunque pinholes molto piccoli, in cui la luce acquisita e' molta poca.
**COL PASSARE DEL TEMPO PERO' LA CAMERA PUO' COLLEZIONARE LUCE** (anche se poca) e l'immagine MIGLIORA A LIVELLO DI LUCE (pero' devo aspettare che la camera collecti piu' luce).
Invece di integrare la luce per un lungo lasso di tempo, **posso invece aumentare la grandezza del pinhole**. Solitamente viene difatti fatto questo per avere immagini da subito illuminate, senza dover aspettare che la luce venga collezionata.
**HO PERO' CHE PER OGNI PUNTO NEL REAL WORLD ORA OCCUPO UN CERCHIO DI LUCE NELL'IMAGE PLANE.** Questo porta a **blur**, in quanto sto rappresentando un punto nel mondo reale con un **cono di pixels** nell'image plane.

## LE LENTI
Le uniche cose che ci possono aiutare sono le LENTI. Le lenti mi permettono di prendere piu' luce da un punto di scena e mettere in FOCUS su di esso. Le LENTI prendono il CONO DI LUCE che si genera avendo un _pinhole_ piu' grande e lo rimappa a un **solo punto** nell'image plane.
LE LENTI purtroppo **NON** permettono di avere **TUTTI GLI OGGETTI della scena IN FOCUS**. Hanno quindi come PROS
1. levano motion blur (e blur in generale)
2. collezionano molta luce 
E come CONS:
1. Riducono il DEPTH OF FIELD (non ho tutto a focus)
![[C11.png]]
## THIN LENS
E' un modello matematico che spiega le lenti. Questo e' mostrato in figura qui sotto:
![[CV11.png]]
**NOTA** qua _f_ e' la focal length delle **LENTI**, mentre v e' la _focal length_ dell'image plane. _f_ e' fisso e e' LEGATO alla lente. L'unica cosa che posso variare sono _u_ e _v_.
**NOTA** Ho che i raggi paralleli all'optical axis sono riflessi per passare attraverso F, mentre quelli che passano attraverso la pinhole C non sono riflessi e si comportano normalmente.

Se fisso la distanza dell'image plane dalla lente aka la _focal(che sarebbe v_), ho che posso calcolarmi la distanza a cui i punti di scena risulteranno A FUOCO, quindi posso calcolarmi _u_ praticamente.
**NOTA** esistera' solo un **PIANO** nel 3D che sara' a FUOCO nell'immagine.![[CV12.png]]
Se fisso invece la distanza degli oggetti di scena, devo mettere l'image plane a una distanza pari a ![[CV13.png]]
per avere quei punti ON FOCUS.
Data la posizione dell'image plane scelta, i punti di scena che saranno **DI FRONTE** o **DIETRO** l'image plane risulteranno fuori fuoco (produrranno i *Circles of confusion*  o i *Blur circles* che sono gli stessi di quando apro troppo il pinhole senza lenti). ![[CV14.png]]

Qua sopra mi viene mostrato cosa succede se osservo
1. Un punto a fuoco,
2. Un punto $P_1$ che e' piu' vicino rispetto al piano _u_, che e' l'unico piano che risultera' a fuoco
3. un punto $P_2$ che invece e' piu' lontano.
Negli ultimi due casi ho blur chiaramente.

## Diaphragm
Ad ogni modo finche questi *Circles of confusion* sono piu' piccoli della grandezza dei **fotosensori**, l'immagine sembrera' comunque in FOCUS. 
Se il BLUR CIRCLE e' PICCOLO ABBASTANZA da essere concentrato dallo STESSO FOTORECEPTOR che registra il valore del pixel nella camera allora NON VEDRO' BLUR in quanto **otterro' comunque un single value per quel pixel.**
IMPORTANTE:
The range of distances across which the image appears on focus - due to blur circles being small enough - determines the DOF (Depth of Field) of the imaging apparatus
Per questo dico che il pinhole camera model ha un DOF infinito, perche' il range delle distanze per cui ho i pixels a fuoco e' INFINITO (posso stare infinitamente lontano/vicino e avere comunque tutti i pixels a fuoco).
Se pero' il  *Circle of confusion*  e' TROPPO GRANDE, allora nell'immagine avro' BLUR.

C'e' comunque un MODO per regolare la quantita' di luce da prendere dalla scena e da far arrivare alla lente. Riducendo l'apertura, Con meno luce avro' un **piu' piccolo** BLUR CIRCLE. Aumentando l'apertura, Con piu' luce avro' un **piu grande** BLUR CIRCLE.


# Focusing Mechanism
Le camere piu' professionali, c'e' un mecanismo di FOCUSING.
C'e' un FOCUS ALL'INFINITIY per cui _u_ (la distanza di un punto di scena alla lente) tende all infinito quindi ho $1/v = 1/f$, quindi $v = f$ (la focal length della lente coincide con la focal length della camera). 
C'e' anche UN **MINIMUM FOCUSING DISTANCE** , che ottengo quando la lente e' molto lontano rispetto all'image plane (quindi _v_ e' molto grande). Cio' e' ottenibile mettendo a fuoco punti a distanza _u_ con _u_ il piu' piccolo possibile.![[CV15.png]]
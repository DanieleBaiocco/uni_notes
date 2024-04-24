## Intrinsinc Parameter Matrix Refinement
Per ora ho una linear transformation tra un 3D point e la sua proiezione nel 2D point, tramite l'*intrinsic matrix* vista fin'ora. 
Questa e' possibile dividerla in due parti:
1. nella parte piu' di sinistra, chiamata $P_{int}$ ovvero l'*intrinsic  parameter matrix*. Anche chiamata $A$ o $K$, e che modella le caratteristiche dell'*image sensing device*. Questa matrice e' una **upper right triangular matrix** se noti bene (e' una matrice quadrata con gli elementi sotto la diagonale che sono tutti 0). 
2. in quella rimanente, in cui ho un vettore di 0.

![[cvsalti1.png]]

**NOTA** pero' che c'e' uno 0 nella parte di sopra dell'*intrinsic parameter matrix*, cosa che mi permette comunque di riferirmi a questa matrice come upper right triangular eh, siamo chiari.
C'e' un modello in cui in quella posizione NON ho 0. In questo modello li' c'e' un parametro chiamato la **skew**. La presenza della "skew" permette di gestire casi in cui gli assi del sensore dell'immagine non sono perfettamente ortogonali tra loro. Possibili motivi per cui i sensori non risultano ortogonali sono:
1.  i sensori **non** sono stati montati in maniera **perpendicolare** rispetto all'optical axis (alla coordinata della profondita')
2. c'e' stato un problema durante la costruzione dei sensori, in cui magari le righe e le colonne dei pixels non sono state costruite perfettamente perpendicolari.

## Extrinsic Parameter Matrix Refinement
Posso vedere l'*extrinsic parameter matrix* come gia' visto attraverso una *rotation matrix $R$* e un *translation vector $t$*.
Posso gruppare $R$ e $t$ in una unica grande matrice.
![[cvsalti2.png]]
Cio' e' possibile utilizzando sempre il **projective space**, in maniera molto semplice.

## Metto tutto insieme 
Ho praticamente un **MODELLO LINEARE** per il **PROCESSO DI FORMAZIONE DELL'IMMAGINE** ora, quindi per passare da 3D point visto con il World Reference System a un punto 2D nell'image plane.
![[cvsalti3.png]]
Se ho due matrici lineari $P_{int} \text{ e } G$ posso combinarle tra loro, facendo moltiplicazione tra matrici,  creando la matrice $P$.
Questa $P$ finale e' chiamata la *Perspective Projection Matrix* (PPM). Questa ha *shape* 3x4: inizia da 3D homogeneous coordinates (4D) e va in 2D homogeneous coordinates (3D). E' **rank massimo**. Tra l'altro ogni matrice 3x4  *rappresenta* una possibile PPM (quindi una possibile camera), finche' ho che il rank e' massimo.
La piu' semplice PPM che esiste e' quella senza parametri, ovvero che ha una matrice di identita' nella parte a sx (quella 3x3) e un vettore di 0 nell'ultima colonna:
$$P \equiv \begin{bmatrix}I |0\end{bmatrix}$$In questa camera il WRF e il CRF sono **la stessa cosa** perche' NON ho roto-translation. Questa matrice fa *perspective projection* nella sua forma piu' pura, senza parametri (senza focal length, senza *pixellation)*. Di base scala soltanto le coordinate $(x,y)$ prese dal 3D della quantita' $z$ che e' la distanza dalla camera (effettivamente si fa solo questo alla fine, ti ricordi la divisione per $z$ che avveniva sempre alla fine?). FINE. Quindi questa $P$ e' chiamata la **canonical o standard PPM**.
E' possibile fattorizzare una generica PPM in una in cui ho $P \equiv A[I|0]G$. In questo modo posso vedere che una generica PPM performa le seguenti operazioni, una dopo l'altra:
1. Converte le coordinate da WRF a CRF
2. Fa la **canonical o standard perspective projection**, dividendo praticamente solo per la terza coordinata $z$
3. Applica trasformazioni relative e specifiche della camera (applica $A$ praticamente)
C'e' un altro modo per fattorizzare una PPM, e si puo' fare nel seguente modo:
![[cvsalti4.png]]
Basta fare moltiplicazione tra matrici per vederlo eh, niente di esagerato.


## Distorzione delle lenti
Con un piccolo pinhole, come gia visto con Listanti, mi serve molto tempo per fare entrare la luce. Quindi si allarga il pinhole solitamente e si utilizzano delle lenti. Loro servono per concentrare un sacco di luce in ogni pixel in un periodo di tempo piccolo  (minore rispetto a quello che ci sarebbe voluto con un pinhole molto piccolo). 
Le **lenti** introducono **DISTORZIONI**. Cambiano la matematica del *pinhole camera model*. Nell'*image formation process* quindi **cio' risulta in una distorzione delle linee** ad esempio: nel pinhole camera model una linea **RIMANE UNA LINEA** anche se va all'infinito magari, ma una linea rimane. Mentre con le lenti ho che le **linee non sempre rimangono tali, vengono distorte e rese curve**. Cio' e' dovuto dal fatto che le **LENTI** cambiano il **PERCORSO della luce**.![[cvsalti5.png]]
Nota come e' curvato sto pilastro.
Tra l'altro adesso la mia PPM non va piu' bene.
Cio' che faccio per rimediare e' **aggiungendo** nuovi parametri nel modello visto fin'ora.

Ci sono solitamente due distorzioni:
1. la **radial distortion**, in cui ho delle curvature nell'image plane. Due tipi diversi di *radial distortion* sono:
	1. la **barrel distortion**
	2. la **pincushion distortion**
![[cvsalti6.png]]
2. la **tangential distortion**, in cui ho dei disallineamenti

Avere una **radial distortion** significa che il pixel nell'image plane con un *pinhole camera model* si sarebbe trovato in una determinata posizione $(x,y)$, invece con l'utilizzo delle **lenti** ora si trova in una posizione ($\hat{x}, \hat{y}$) che sta lungo il **raggio** disegnato  collegando il centro dell'image plane con $(x,y)$. Questo shift e' di una quantita' $dr$ che indica la *radial distortion*.

C'e' anche la *tangential distortion* in cui c'e' anche un movimento nella direzione della **tangente**. Quindi sia una distorzione riferita al **raggio**, come prima, che anche quella riferita alla tangente. 
![[cvsalti7.png]]
La *tangental distortion* e' riferita come $dt$. La *tangental* distortion e difficile da modellare, perche' si sono punti della lente in cui questa e' diversa, e varia.![[cvsalti8.png]]
come possiamo vedere dalla foto, c'e' un asse in cui questa e' *minima*, e un asse in cui questa e' *massima*. Nella foto qui sopra ho *l'image plane* eh, non e' mica la lente questa ahahaha. E' l'image plane e come viene distorto. Come gia' detto ci sono zone dell'image plane che vengono piu' distorte dalla *tangental distortion* e altre che vengono meno distorte.

Nel 99% delle applicazioni **non posso ignorare** la *distortion delle lenti*, in particolare la *radial distortion*.

## Modellando la lens distortion
Di base e' modellato usando una trasformazioen *non lineare* che mappa **cooridnate non distorte dell'immagine** , nelle coordinate distorte.
Praticamente cio' che viene fatto e' che il risultato della PPM e' moltiplicato a una *funzione* che dipende (NOTA) dal raggio $r$, per modellare la *radial distortion*. La funzione dipende da r perche' piu' la distanza centro dell'immagine aumenta, piu' la distorzione sara' grande.
Per quantor iguarda la *tangential distortion*, questa e' calcolata tramite una funzione che prende il raggio sempre, e **dove mi trovo**, perche' abbiamo visto che dipende molto da dove mi trovo.
![[cvsalti9.png]]
$L(r)$ e' spesso modellata nel seguente modo:
come un polinomio pari (con esponenti pari) della seguente forma:
$L(r) = 1+ k_1r^2 + k_2r^4 + k_3r^6 + ...$ 
che altro non e' che una specie di taylor expansion. Quanti termini voglio? A seconda delle implementazioni delle librerie ho un numero di parametri diverso. In opencv per esempio l'espasione e' fino a $r^4$ (quindi 2 parametri). Nota che ho che $L(0)=1$, quindi il modello assume che nel punto con radius 0, ovvero nel centro dell'immagine, io *non abbia distortion* (perche' poi viene moltiplicato alle coordinate del pinhole camera model).
La tangental distortion e' modellata attraverso una funzione  che neanche mettero' perche' non serve capirla. Invece la metto LOL
![[cvsalti10.png]]
E praticamente il set degli *intrinsic parameters* per buildare un camera model che sia realistico e' ESTESO con l'aggiunta di $k_1, k_2,..., k_n$ e $p_1, p_2$. 

Questo **step**, di aggiunta della distortion avviene **tra** la *canonical perspective projection* e il *mapping da coordinate dell'immagine a pixels usando gli intrinsic parameters*.

Quindi ho:
1. Converte le coordinate da WRF a CRF
2. Fa la **canonical o standard perspective projection**, dividendo praticamente solo per la terza coordinata $z$
3. Fa non linear mpping dovuta a distorsione delle lenti
4. Applica trasformazioni relative e specifiche della camera (applica $A$ praticamente)
![[1.png]]![[2.png]]![[3.png]]![[4.png]]
Questa distortion delle lenti viene messa prima della pixellization perche' comunque e' una distorzione che avviene in **centimetri**, non e' legata ai pixels.


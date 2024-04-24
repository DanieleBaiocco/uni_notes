# Template matching
C'e' una similarity function che mi dice quanto il *template dell'oggetto* e patches prese dall'immagine.

## Similarity Function
![[cvsimilarity.png]]
Posso usare:
1. Sum of absolute differences: $\text{SAD}(i,j) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1}|I(i+m, j+n) - T(m,n)|$ che significa che di base, se prendo in esame una patch all'interno dell'immagine I, posso computare per ogni pixel tra I e T la differenza di valore.
2. Sum of Squared Differences: come prima ma metto alla seconda se non sbaglio

Queste due misure sono *invarianti* riguardo a **cambiamenti di luminosita'** tra la target image e la template image? Ovvero, anche se c'e' una differenza di luminosita' tra I e T, trovo comunque la patch che matcha il template? La risposa e' che NON sono invarianti rispetto a cambiamenti nella luminosita': in presenza di un match tra I e T (metti che comparo la patch giusta con il template a essa associato), la SAD o la SSD ritornerebbero dei valori comunque *molto alti*, perche' i pixels di I e T sono, se comparati, diversi in intensita'.

Serve quindi
3. Normalized Cross-Correlation: $NCC(i,j) = \frac{I(i,j) * T}{||I(i,j)|| \text{ }||T||}$, che se estesa diventa ![[ncc.png]]
Questa rappresenta il coseno dell'angolo tra i vettori $I(i,j) \text{ e } T$: immaginati di rendere $I(i,j) \text{ e } T$ dei vettori e rappresentarli come tali nello spazio. Il coseno dell'angolo tra questi due vettori e' la NCC. Nel caso in cui questi vettori fossero diamentralmente opposti, quindi uno e' perpendicolare all'altro, avrei che la NCC mi ritornerebbe 0, in quanto il coseno di 90 gradi e' 0. Se invece puntassero verso la stessa direzione, allora il NCC ritornato sarebbe alto.

Questa misura e' invariante rispetto a **cambi di luminosita'** per quanto riguarda la **moltiplicazione** perche', anche se si scala uno dei vettori di una grandezza, ho che cio' non influisce nell'angolo che si forma tra T e I (l'angolo rimane sempre lo stesso, anche se si scala uno dei due di una quantita').

Non e' pero' invariante rispetto all'**addizione**, perche' questa **cambia** la direzione del vettore, risultando in **un altro angolo tra I e T, diverso da quello che avrei avuto senza il cambio additivo di luminosita'**. Per evitare cio' si e' introdotto
4. Zero-Mean Normalized Cross-Correlation: in cui sia in I che in T vengono fatti dei cambiamenti. Viene prima calcolata la *mean* e poi sottratta nel seguente modo:
	 *  $\mu(I) = \frac{1}{MN}\sum_{m=0}^{M-1} \sum_{n=0}^{N-1}I(i+m, j+n)$ e $\mu(T) = \frac{1}{MN}\sum_{m=0}^{M-1} \sum_{n=0}^{N-1}T(m,n)$ 
	 * sottraggo poi le medie calcolate facendo $I(i+m, j+n) = I(i+m, j+n) - \mu(I)$ e $T(m,n) = T(m,n) - \mu(T)$
Quindi ho 
![[ZCC.png]]
in questo modo sono **ANCHE** invariante rispetto all'**additive component** (con $I(i+m, j+n) - \mu(I)$ e $T(m,n) - \mu(T)$.
Quindi sono invariante per **affine intensity changes** con questa misura (sia moltiplicative che additive).

## Shape-based template matching
Di base voglio matchare la patch che ha una shape simile a quella del *template*. Per fare cio' runno un algoritmo di Edge Detection sul *template*. Prendo i $k$ punti piu' interessanti e mi salvo la direzione del *gradiente*. A quel punto passo il template sull'immagine target, e per ogni patch, in corrispondenza di quei punti, calcolo il gradiente. Paragono i gradienti all'altezza dei control points *della target image* e della *template image*.
Piu' questi gradienti sono allineati, piu' e' probabile che io abbia un match.
![[cv341.png]]
Come calcolo quanto sono allineati? Eh. Di base prima mostro questi *vettori di derivate* , sia per  il template che per il target, che sono tradotti in *unit vector*, in modo da renderli comparabili:
![[cvunitvectors.png]]
Poi ne faccio il dot product dai facile (sono gia normalizzati ricordo). Se due gradienti sono perfettamente allineati (uno del template e uno del target) allora ho che l'angolo tra di loro e' 0, di conseguenza il coseno e' 1 (il massimo raggiungibile).
La formula e' la seguente:
$$S(i,j) = \frac{1}{n}\sum_{k=1}^{n} u_k(P_k) \cdot \widetilde{u_k}(\widetilde{P_k}) = \frac{1}{n}\sum_{k=1}^{n} \text{cos}\theta_k$$
Se PER tutti i gradienti da comparare ho coseno a 1, alla fine avro' che il risultato finale di $S(i,j)$ e' proprio 1, perche' sommo gli 1 per n volte e poi divido per n.
*C'e' un **PROBLEMA** pero'*, ovvero che potrei avere un template che ha un gradiente con una **polarita'** opposta rispetto a quella trovata nella target image, e per questo caso VORREI COMUNQUE avere un match, quindi vorrei essere *invariante rispeto a una inversione globale della polarita'* sugli oggetti delle patches prese in considerazionie.
Questi oggetti possono infatti apparire o piu' scuri o piu' chiari rispetto al background della patch, e se non sono invariante rispetto a questo io matchero' solamente uno dei due casi (o che sono piu' scuri o che sono piu' chiari, non entrambi. Cio' dipende dalla template image, in quale direzione ha i gradienti, cosa ha nel background). 
Risolvo questo problema facendo
$$S(i,j) = \frac{1}{n}|\sum_{k=1}^{n} u_k(P_k) \cdot \widetilde{u_k}(\widetilde{P_k})| = \frac{1}{n}|\sum_{k=1}^{n} \text{cos}\theta_k|$$
In sto modo sono invariante rispetto alla polarita' a livello globale. Ancora meglio se faccio a livello locale:
$$S(i,j) = \frac{1}{n}\sum_{k=1}^{n} |u_k(P_k) \cdot \widetilde{u_k}(\widetilde{P_k})| = \frac{1}{n}\sum_{k=1}^{n} |\text{cos}\theta_k|$$
## The Hough Transform
HT permette di detectare nell'immagine oggetti che hanno una *shape* conosciuta, che e' **ESPRIMIIBLE TRAMITE UN'EQUAZIONE**: linee, cerchi, elipsi. Cio' e' basato tramite una proiezione dell'*input data* su un altro *spazio* chiamato il **parameter space** o l'**Hough space**, che e' diverso dall'*image space*.

Nota che l'*input data* consiste nell'immagine iniziale a cui e' stato applicato un algoritmo di *edge detection*. Quindi la mia immagine consiste in *pixels* estratti dall'immagine originale dove ci sono degli edges. 

Praticamente differisce da quello visto finora perche' non si fa un match tra  la *shape* dell'input data e una template shape (che appunto puo' essere una linea o un cerchio), ma si guarda a *feature points* nello spazio dei parametri. 

Per capirci meglio, iniziamo a andare in profondita'.
L'HT ha come grossi **vantaggi** che e' 
1. **robusto** rispetto al noise
2. permette di detectare la shape in considerazione anche se e' **parzialmente oscurata/occlusa**
HT e' nata per detectare linee, poi si e' andati a altri tipi di shapes (cerchi, ellissi, shapes arbitrarie). Quando si parla di *shapes arbitrarie* allora li' si parla di **generalized hough transform(GHT)**

La versione base dell'**HOUGH TRANSFORM** e' la seguente:
1. dati degli edge points  $(x_i, y_i)$
2. la task e' quella di detectare una **linea** da questi edge points della forma $y = mx + c$
![[retta.png]]
Se considero un singolo punto in questa linea ($x_i, y_i$) ho che la formula che soddisfa e' $y_i = mx_i + c$, ma puo' anche essere scritta come $c=-mx_i+y_i$.

Se penso a una linea, questa ha la formula $y = \hat{m}x + \hat{c}$, con $m$ e $c$ **fisse**. MA invece io fisso ($\hat{x}, \hat{y}$) e cambio $m$ e $c$  (in formula e'  $\hat{y} = m\hat{x} + c$). In questo modo ho che dato un punto dell'immagine, rappresento nel **parameter space** tutte le possibili linee (al variare di $m$ e $c$) che passano attraverso ($\hat{x}, \hat{y}$).

$c=-mx_i+y_i$ e' una equazione di una linea vista pero' in $m, c$. Quindi posso passare dall'image space al parameter space.
![[image_parameter_space.png]]
Praticamente a un punto ($x_i, y_i$), corrisponde una retta di ($m,c$) punti. Ognuno di questi punti nella retta del parameter space si riferisce appunto a una coppia di valori ($m,c$), che indica un tipo di retta che passa per ($x_i, y_i$) nell'image space.

Se prendo un altro punto $(x_j, y_j)$ nell'image space, allora questo corrispondera' a un'altra linea nel *parameter space*. Nell'intersezione tra le due linee, vi e' il valore di ($m,c$) che produce l'unica linea nell'image space che passa per entrambi i punti ($x_i, y_i$),$(x_j, y_j)$.
![[cv2q4twq.png]]
Se prendo piu' edge points che sono in una linea allora questi si intersecano TUTTI nello stesso punto. Se pero' metti che nella mia immagine c'e' uno *spurious edge* che e' fuori dalla linea, questo viene mappato nel *parameter space* in una linea che non si interseca come le altre:
![[param_space_multilines.png]]
Quindi invece di guardare a una global shape nell'immagine, guardiamo a livello locale, nel parameter space delle linee, la presenza di una particolare feature, ovvero dove le linee si intersecano tra loro.
Intersezioni delle curve del parameter space (nel caso di sopra delle linee del parameter space) indicano la presenza di tanti image points che sono **spiegabili** da una certa shape (una linea nel caso di sopra).
Piu' curve (o linee nel nostro caso) si intersecano nel *parameter space*, piu' image points esistono, e quindi piu' e' marcata l'evidenza della presenza di quella determinata shape (una linea nel nostro caso) nell'immagine.
E' necessario un modo per **QUANTIZZARE il parameter space**. Questo viene fatto attraverso un **Accumulator Array (AA)**. Praticamente per ogni punto nell'image space, ho una linea nel *parameter space*. E questa linea comunque attraversa l'**AA** in diversi bins, ognuno di questi bins viene incrementato di 1.![[cvquantized.png]]
Questa immagine spiega molto bene, perche come si puo notare, nel bin (0.1,9) ho che li' passa una sola linea, quindi ho 1. Nel bin (0.1,8) invece, passano 3 linee, quindi ho 3 nell'**AA** e via dicendo.
L'HT e' robusto al noise perche' dei voti sporious dati dal noise, difficilmente saranno cosi' tanti da accumularsi dentro un bin in modo poi da triggerare una *line detection*.
Un oggetto parzialmente occluso inoltre puo' comunque essere detectato, abbassando il valore della threshold dei voti minimi che servono per triggerare una *line detection*.
*Algoritmo*:
1. Quantizzo il *parameter space*, con degli scacchi (guarda figura di qua sopra, il *continuous parameter space*)
2. Creo un *accumulator array* A(m,c)
3. Metto A(m,c) =0 per ogni (m,c)
4. Per ogni edge point ($x_i, y_i$), $A(m,c) = A(m,c)+1$ se ($m,c$) sta sulla linea $c = -mx_i + y_i$ (questo e' proprio il processo descritto qua sopra e visualizzabile nel grafico *Quantized AA*)
5. Trovo i local maxima in A(m,c)
C'e' quindi un **voting scheme** che ritorna i local maxima. Nell'immagine di sopra, verranno ritornati come local maxima i punti (0.4, 5), (0.3, 5), (0.5, 5).
Qua sotto metto un esempio comunquue, nel caso di piu' linee:
![[esempio.png]]

**NOTA pero'**, io non posso parametrizzare la retta come l'ho sempre vista con m e c, perche' m ha dei valori che possono andare fino all'infinito, quindi NON saprei come fare nel creare un **AA** che abbia come numero di colonne un numero infinito di valori. Quindi, per esprimere una linea si utilizza la formula $x \text{sin}\theta-y\text{cos}\theta+\rho =0$. Ho che $\theta$ rappresenta l'orientamento, e va da 0 a $\pi$, mentre $\rho$ rappresenta la distanza della linea in esame dall'origine, e questa e' finita e e' al massimo la lunghezza della diagonale dell'immagine di partenza, ovvero metti che ho NxN pixels, allora avro' $\rho_{\text{max}} = N \sqrt{2}$. Posso costruire il $(\rho, \theta)$ space.
Un punto nell'*image space* e' una *sinusoide* nel parameter space:![[cvcv.png]]
Ho due match che sono $\theta$ e $\theta+\pi$ che rappresentano LA STESSA RETTA.
Questo qua sotto e' un esempio di risultato di questa cosa:
![[liunes.png]]
Questo qua sotto e' l'algoritmo per il GHT invece:
![[boh.png]]con la tabella fatta nel seguente modo:

![[bohpt2.png]]
# LOCAL INVARIANT FEATURE
Questa parte e' legata ai *CORRESPONDING POINTS*. Abbiamo gia' parlato del problema di trovare corrispondenze in punti tra due immagini.
Lo affrontiamo per bene ora.
Pensa alle immagini *panorama* che vengono prese quando ho il camera center FISSO e muovo solo lungo l'x axis e faccio diverse foto.
C'e' da fare *homography* che e' una 3x3 matrix che puo' mappare ogni punto in un'immagine in un altro punto nell'altra immagine. Per computare questa homography abbiamo bisogno di CORRESPONDENCES, matching points. HO pero' bisogno matching points che siano RILEVANTI (tipo se prendo due punti nel cielo , questi non sono molto rilevanti). Si prendono punti che sono motlo piu' discriminativi, devono avere del CONTESTO che li descriva per bene in modo caratteristico. ci sono algoritmi che mi dicono quali sono i salient points. Poi ce n'e' un altro che invece li DESCRIVE: che estragga da questi pixels delle info che li descriva bene  (che non e' solo una descrizione legata ALL'INTENSITA' DEI PIXELS, questa non e' reliable, qeusta cambia, io sto cercando INVARIANCE wrt intensity, illumination, rotation, ecc...)
Una volta che ho queste descrizioni e' possibile poi fare matching pairs, calcolando le distanze tra le descrizioni dei salient pixels. Se questa distanza e' piccola abbastanza allora c'e' un match.
Il descriptor e' un vettore di numeri, ogni numero e' una proprieta' computata dai neighborhood di quel salient pixel.
Quando ho almeno 4 CORRESPONDENZE tra immagine di dx e di sx allora posso COMPUTARE L'homography e a quel punto ho vinto.

LE CORRESPONDENZE POSSONO ESSERE USATE PER ALTRE TASKS:
1. INSTANCE LEVEL OBJECT DETECTION: posso praticamente prendere un'oggett in un immagine e prendere una target image e alla fine POSSO VEDERE se quella target image e' proprio quell'oggetto in quell'immagine. Nota che una DNN per fare una roba del genere avrebbe bisogno di una marea di elementi.
2. ROBOT NAVIGATION
3. AUGUMENTTED REALITY: posso computare cose che non ci sono nell'immagine, devastante
4. 3D RECONSTRUCTION: pazzo sgravato.
RIPETO COME LHO GIA SCRITTO SOPRA, il description deve essere invariante rispetto alla maggior parte delle trasformazioni possibil : illumination, scale, viewpoint changes. O anche cambiamenti dovuti al tempo (se c'e' nebbia in una foto e nell'altra no, devo avere un descriptor che mi ritorna lo stesso risultato piu' o meno per lo stesso pixel nelle due immagini che matcha).

Un DETECTOR deve rispettare due proprieta':
1. REPEATABILITY
2. SALIENCY
un descriptor invece deve essere 

vedi su slides.
DEVE ESSERE PIU VELOCE IL DETECTOR O IL DESCRIPTOR? il detector, perche' devo computare la detection di ogni punto dell'immagine il descriptor invece e' impiegato solo perpunti che  sono silence.


# Interest points vs Edges
Che informazione posso estrarre dal punto preso in considerazione qui sotto, per avere qualcosa che e' poi paragonabile nell'altra immagine?![[cv31 1.png]]
Il **gradiente** ad esempio, la **direzione** del gradiente in particolare.
Se ho lo stesso gradiente in due immagini su un punto allora *potrei dire di avere un match*? Beh no, non posso usare solo l'informazione del gradiente, perche' ho *piu' matches sull'altra immagine se uso **solo** questa info (tutti i punti con quel gradiente direzione, quindi tutti quelli che sono su quel lato)*:
![[cv32 1.png]]
Gli edges sono difatti **localmente ambigui** proprio per questo motivo. Quindi non posso usarli come *salient points/interest points*, ovvero punti **interessanti** che mi possono permettere di fare un match univoco tra le due immagini.

Posso usare invece i **CORNERS** perche' questi mostrano variazioni lungo *tutte le direzioni*.
Uno dei piu' famosi detectors di *interest points* che mi trova proprio i *corners* si chiama
## Moravec Interest Point Detector
Praticamente si prendono 8 neighborhoods del punto _p_ preso in considerazione, di cui voglio calcolare la **cornerness** (quanto e' un corner). Si fa poi la *squared difference*, per ogni neighborhood _q_, tra una patch centrata in _q_ e una centrata in _p_. Si prende il **minimo** tra le varie *squared differences* e questo sara' il valore della **cornerness** di _p_. L'idea di fondo  e' che avro' un valore alto di **cornerness** alto quando ho che _p_ ha intorno punti **molto differenti da _p_**. Si prende il minimo perche'  se ho un valore alto di *squared difference* con il minimo, allora lo avro' sicuramente ancora piu' alto con gli altri.
![[moravec_1.png]]
**NOTA**, se prendessi _p_ al centro di un'immagine completamente *nera*, avrei che **C(p)** sarebbe 0 praticamente (ricorda che prendo il minimo, quindi basta una patch che abbia gli stessi valori della patch su _p_ e ho che **C(p)** e' 0).
Se prendo _p_ edge avrei che **C(p)** sarebbe molto piccolo: potrei addirittura avere 0 se prendo come neighborhood un punto orizzontale al punto _p_ (guarda l'immagine). Sui *corners* invece ho **sempre** una **C(p)** alta (o almeno sicuramente non 0), perche' per ogni neighborhood non ho mai un match perfetto tra patches, ho sempre qualcosa che cambia tra la patch in _p_ e le altre 8 patch di _q_.![[cv33 1.png]]
Dopo aver computato la **cornerness** per ogni punto, si fa solitamente *thresholding* sui valori e poi si applica **NMS**. Nota: non c'entra nulla il gradiente in questo caso nel NMS. 
Nel codice del Prof. infatti, prima viene fatto *thresholding*, e viene ritornata una immagine di *True e False* values (*True* se il valore della *cornerness* e' sopra una *threshold* specificata, *False* altrimenti). Viene poi fatto NMS, usando una window 3x3 che viene slidata su tutta l'immagine *thresholdata*. Si gira questa sliding window sull'immagine. Ogni volta si prende il pixel centrale di questa sliding window. Se questo e' *True* e e' il massimo, il che significa che fino a prima ho avuto solo *False pixels*, allora viene messo un 1 nella NMS mask da outputtare (il che significa che in corrispondenza di quel pixel ho un corner).Nel caso in cui queste due condizioni non vengoano rispettate allora ho che quel pixel non rappresenta un corner.

Un esempio di pixel che viene visto come corner e' il seguente: ```
[[False False False] 
[False True True]
[False True True]]```
Ho che il pixel centrale e' il **massimo** nel senso che prima ho avuto solo *False pixels* e e' il primo pixel a diventare *True*. 
La NMS mi permette di avere **un solo pixel legato a un corner**. 

```python
# Non maxima suppression

nms = np.zeros(img.shape)

for i in range(1,corners_filtered.shape[0]-2):
  for j in range(1,corners_filtered.shape[1]-2):
    window = corners_filtered[i-1:i+2,j-1:j+2]

    # if the element in the centre of the 3x3 window is not zero and is the max
    if(window[1,1] != 0 and window[:].argmax()==4):
      nms[i,j] = 1
```
![[cv34 1.png]]
Il risultato finale e' il seguente (con w = 3, ovvero la gradnezza delle patches da comparare di _p_ e delle _q_ 's). Nella prima immagine c'e' la figura iniziale, nella seconda ci sono i valori calcolati della **cornerness**, nella terza si e' applicato *thresholding*, nella quarta si e' applicato NMS. Nota che l'output della NMS sono difatti 4 pixels, uno per ogni corner presente nell'immagine iniziale (un quadrato).

![[amarcord_cornerness.png]]
Questo e' il risultato con una immagine di una locandina di Amarcord.
## Harris Corner Detector
**NOTA**, *Moravec Detector* non e' invariante rispetto alla **rotazione e scale**. Forse e' invariante rispetto alla **luminosita'** al massimo, ma neanche troppo.

L'idea e' di muovere l'immagine di uno **shift** infinitesimale, invece di vederne i neighborhoods.
https://www.baeldung.com/cs/harris-corner-detection
https://www.baeldung.com/cs/harris-corner-detection
https://www.baeldung.com/cs/harris-corner-detection
vedi questo link e video su youtube messi in playlist CV

Let’s consider a two-dimensional image ![I](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-14b16a74c9ddcc6f9be3e94b9c8d8f08_l3.svg "Rendered by QuickLaTeX.com") and a patch ![W](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-183777ab9133546b80b6f342c6ec9919_l3.svg "Rendered by QuickLaTeX.com") of size ![m \times m](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-9a088a308369746912e014aa5d10d610_l3.svg "Rendered by QuickLaTeX.com") centered in ![(x_0,y_0)](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-213d1e90dd8b2845bcd66ae9a5eb13c5_l3.svg "Rendered by QuickLaTeX.com").
We want to evaluate the intensity variation occurred if the window ![W](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-183777ab9133546b80b6f342c6ec9919_l3.svg "Rendered by QuickLaTeX.com") is shifted by a small amount ![(u, v)](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-2ab9e2f4d2b648908df4133879394778_l3.svg "Rendered by QuickLaTeX.com"). Such variation can be estimated by computing the Sum of Squared Differences (SSD):

  $$SSD(u, v)=\sum_{(x, y)\in W} g(x,y) \left[I\left(x, y\right)-I\left(x+u, y+v\right)\right]^{2}$$
  where ![g(x,y)](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-6888cabd4c14c6d6a27f7cebee652937_l3.svg "Rendered by QuickLaTeX.com") is a window function that can be a rectangular or a Gaussian function. We need to maximize the function ![\hbox{SSD}(u, v)](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-48f3f26ae68ccef93590c1973a750145_l3.svg "Rendered by QuickLaTeX.com") for corner detection. Since ![u](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-e817933126862db10ae510d35359568e_l3.svg "Rendered by QuickLaTeX.com") and ![v](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-796872219106704832bd95ce08640b7b_l3.svg "Rendered by QuickLaTeX.com") are small, the shifted intensity ![I(x+u, y+v)](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-7c44acb690553825c5df4920de979573_l3.svg "Rendered by QuickLaTeX.com") can be approximated by the following first-order Taylor expansion that says that 
  $f(x+\Delta x) = f(x)+ f'(x)\Delta x$. So we have:
  ![\begin{equation*} I(x+u, y+v) \approx I(x, y)+u I_{x}(x, y)+v I_{y}(x, y) \end{equation*}](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-6cbe166ec45ef7f21e26d6d3a547ecc3_l3.svg "Rendered by QuickLaTeX.com")
  where ![I_x](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-3b075cf4826d2b10ccc59eb2512b5d62_l3.svg "Rendered by QuickLaTeX.com") and ![I_y](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-1f0e754de319c59e7d61a73fe2bc71c5_l3.svg "Rendered by QuickLaTeX.com") are partial derivatives of ![I](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-14b16a74c9ddcc6f9be3e94b9c8d8f08_l3.svg "Rendered by QuickLaTeX.com") in ![x](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-7e5fbfa0bbbd9f3051cd156a0f1b5e31_l3.svg "Rendered by QuickLaTeX.com") and ![y](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-38461fc041e953482219abf5d4cce1cb_l3.svg "Rendered by QuickLaTeX.com") direction.
  By substituting I have that:
   $$SSD(u, v) \approx \sum_{(x, y) \in W} g(x, y)\left[u^{2} I_{x}^{2}+2 u v I_{x} I_{y}+v^{2} I_{y}^{2}\right]$$
   Posso trasformare questa equazione in una equivalente della forma:
$$SSD(u, v) \approx[\begin{array}{ll} u & v \end{array}] M[\begin{array}{l} u \\ v \end{array}]$$
where ![M](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-27d6692c77760dc1111628e74a6d272f_l3.svg "Rendered by QuickLaTeX.com") is a ![2 \times 2](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-2f24fb7fb627da49dd5e446c9afdfd6c_l3.svg "Rendered by QuickLaTeX.com") matrix computed from image derivatives:

$$ M=\sum_{(x, y) \in W}g(x, y)[\begin{array}{ll} I_{x} I_{x} & I_{x} I_{y} \\ I_{x} I_{y} & I_{y} I_{y} \end{array}]$$
The matrix ![M](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-27d6692c77760dc1111628e74a6d272f_l3.svg "Rendered by QuickLaTeX.com") is called _structure tensor._
![[cv35.png]]
Come possiamo vedere da questa immagine se calcolo la distribuzione di $I_x$ e $I_y$  a seconda del punto considerato, posso vedere che nel caso di una regione uniforme, ho che i valori sono bene o male tutti vicini a (0,0) perche' non ci sono grandi cambiamenti di gradiente. Nel caso di un punto sopra un edge ho che i valori cambiano molto verso una determinata direzione. Nel caso di un corner in cui si hanno cambiamenti di intensita' da diverse parti ho che i valori cambiano in due direzioni.

Se fitto un elipse su questa distribuzione di punti, e calcolo i *semi-major axis* $\lambda_1$ e *semi-minor axis* $\lambda_2$, ho la seguente shit:
![[cv36.png]]
Nel video si mostra come e' possibile calcolare questi due $\lambda$, io mi accontento dell'intuizione. Ad ogni modo per calcolarli devo far riferimento a M e calcolarne gli *eigenvalues*.
Posso poi classificare la regione basandomi su $\lambda1$ e $\lambda2$.
![[cv37.png]]Questo mi permette di avere proprio una classificazione basandomi sui loro valori, molto nel preciso. 
![[cv38.png]]
**The Harris detector uses the following response function that scores the presence of a corner within the patch**:

(6)   ![\begin{equation*} R = \operatorname{det}(M)-k \operatorname{tr}(M)^{2} . \end{equation*}](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-e2436307fba8f3371c600d02070ca742_l3.svg "Rendered by QuickLaTeX.com")

![k](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-d42bc2203d6f76ad01b27ac9acc0bee1_l3.svg "Rendered by QuickLaTeX.com") is a constant to chose in the range \[0.04, 0.06]. Since ![M](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-27d6692c77760dc1111628e74a6d272f_l3.svg "Rendered by QuickLaTeX.com") is a symmetric matrix, ![\operatorname{det}(M) = \lambda_1 \lambda_2](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-fb37f3eb3c161d4588fc36c31c0c849b_l3.svg "Rendered by QuickLaTeX.com") and ![\operatorname{tr}(M) = \lambda_1 + \lambda_2](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-505b4f133cc1fbc3e5a525345a6f029e_l3.svg "Rendered by QuickLaTeX.com") where ![\lambda_1](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-9113ad7dd2a409d951e5d541d6b97502_l3.svg "Rendered by QuickLaTeX.com") and ![\lambda_2](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-7007550fecf7810e69c75f1f35da4bb3_l3.svg "Rendered by QuickLaTeX.com") are the eigenvalues of ![M](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-27d6692c77760dc1111628e74a6d272f_l3.svg "Rendered by QuickLaTeX.com"). Hence, we can express the corner response as a function of the eigenvalues of the structure tensor:

(7)   ![\begin{equation*} R=\lambda_{1} \lambda_{2}-k\left(\lambda_{1}+\lambda_{2}\right)^{2} . \end{equation*}](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-22e135ab746ae9160a29251117f7a317_l3.svg "Rendered by QuickLaTeX.com")

Ho che quello che ne risulta di R e' questo:
![[cv39.png]]
Posso poi usare una threshold per capire cosa e' un corner e cosa non lo e':
![[cv40.png]]

The Harris corner detection algorithm can thus be summarized as follows:
1. Compute R at each pixel
2. Select all pixels where R is higher than a chosen positive threshold (T)
3. Within the previous set, detect as corners only those pixels that are local maxima of R (NMS)
It is worth highlighting that the weighting function w(x,y) used by the Harris corner detector is **Gaussian** rather than Box-shaped, so to assign more weight to closer pixels and less weight to those farther away. It is like the second one below:
![[cv41.png]]
### Invariance properties of Harris corner detector
1. it is invariant wrt **rotation**: gli eigenvalues trovati sono sempre gli stessi (letteralmente gli stessi numeri), indipendentemente dalla rotazione dell'immagine presa in considerazione. Questo perche' M e' **simmetrica** e quindi e' diagonalizzabile nel seguente modo:
![[cv50.png]]
2. non e' invariant wrt **intensity**: ho che e' invariante quando aggiungo una quantita' all'immagine perche' questa si cancella quando calcolo le derivate. Ma nel caso in cui moltiplico all'immagine una quantita', questa moltiplicazione poi e' presente anche nella derivata. E quando faccio Thresholding puo' succedere che la threshold accetta un punto come corner, mentre dice che l'altro punto (quello dell'immagine in cui non ho moltiplicato nessun fattore) non e' abbastanza alto a livello di R per essere considerato un corner.![[cv51.png]]
3. Ovviamente non e' scale invariance, perche'  la window e' di fixed size e mi detecta completamente punti diversi tra l'altro. Mi detecta punti come corners ma che poi hanno descriptors che sono totalmente diversi e non posso matchare.
## SCALE INVARIANCE
si e' lavorato a lungo per ottenere detectors che fossero scale invariance, perche' e' molto difficile essere invarianti rispetto a questa property.
NOTA: the neighbourhoods surrounding **large scale features** (quindi features acquisite da piu' vicino) are far richer of details than those around **small scale ones**.
• To make it possible to compute similar descriptors – and so match them- the details that do not appear across the range of scales should be cancelled-out by means of image smoothing. (importantissimo)
### SCALE SPACE
A Scale-Space is a one-parameter family of images created from the original one so that the **structures** at smaller scales are successively suppressed by **smoothing operations** (quindi le strutture/features diminuiscono e NON ne vengono introdotte di nuove).

Several researchers have studied the problem and shown that a Scale-Space must be realized by **Gaussian Smoothing**.

**A Scale-Space is created by repeatedly smoothing the original image with larger and larger Gaussian kernels**.  NOTA: scale space e' definito in base alla posizione(x,y) e alla scala in cui si e' $\sigma$.

DOMANDA: As features do exist *across a range of scales*…**how do we establish at which scale a feature turns out maximally interesting and should therefore be described**? La soluzione trovata e' stata:
compute suitable combinations of **scale-normalized derivatives** of the *Gaussian Scale-Space* and find their **EXTREMA**.
* As we filter more (higher sigma) derivatives tends to become weaker
* to compensate Lindberg proposes to **multiply/normalize derivatives** by sigma (scale-normalized per l'appunto, perche' l'applicazione di un filtro gaussiano sempre piu' largo porta a derivative piu' piccine).

## SCALE NORMALIZED LOG (Laplacian of Gaussian)
L'algoritmo e' di base il seguente:
1. Data un'immagine $I(x,y)$
2. convolvo l'immagine usando NLog (normalized LOG), a molte scale $\sigma$, nel seguente modo: $F(x,y,\sigma) = \sigma^2 (\nabla^2 n_\sigma * I(x,y))$
3. trovo $(x^*, y^*, \sigma^*)= \text{argmax}_{(x,y,\sigma)} |\sigma^2 (\nabla^2 n_\sigma * I(x,y))|= \text{argmax}_{(x,y,\sigma)} |F(x,y,\sigma)|$
4. ottengo quindi $(x^*, y^*)$ che sono le posizioni dei *blobs* , e $\sigma^*$ che sono le sizes dei *blobs* trovati.
Data la figura: 
 ![[cv56 1.png]]
 si ha che $\frac{\text{grandezza del blob A}}{\text{grandezza del blob B}} = \frac{\sigma_A^*}{\sigma_B^*}$. Quindi c'e' una relazione tra la grandezza del diametro effettivo di un blob in un'immagine e la scale del gaussian filter alla quale si ha il punto di massimo nella derivata.  

 **NOTA**: la grandezza della *gaussian* che massimizza la scale-normalize derivative e' legata alla grandezza del Blob (questo lega il diametro del blob alla grandezza della gaussian che massimizza la risposta sulla derivata)
![[cv56.png]]

## Difference of Gaussian (DoG)
Lowe ha proposto un'altro modo, attraverso la differenza di *gaussiane*. Invece di approssimare la derivata seconda della gaussiana attraverso il **Laplacian operator**, utilizzo come approssimazione della derivata seconda la differenza tra due gaussiane successive (tra una gaussiana e la gaussiana successiva che aumenta la scale col $\sigma$).
Ho quindi una approssimazione dell'NLog operator.
Ho infatti che:
$$(n_{s\sigma} - n_{\sigma}) \approx (s-1)\sigma^2\nabla^2 n_\sigma$$
ricordandoci che $\text{NLoG} = \sigma^2\nabla^2 n_\sigma$.
![[cv57.png]]
Guarda che approssimazione vicina.
$$DoG \approx (s-1)NLoG$$
Posso ora prendere due immagini consecutive nel mio scale space e computare la differenza di due immagini consecutive (NOTA: le immagini sono ovviamente gia' smoothate). ![[cv53 1.png]]

**NOTA** mi serve una rule of thumb pero' per capire come variare $\sigma$ nel mio scale space.
Se volessi avere dei $\sigma$ che si succedono nel seguente modo: 
$\sigma, k\sigma, k^2\sigma, k^3\sigma$ e volessi chiudere questa serie con $2\sigma$, allora dovrei avere 
$2\sigma = k^4\sigma$, che porta a $k = 2^{1/4}$. Quindi ho che $k = 2^{1/s}$ con s che e' il numero di scales (di dog images in output) che voglio nella mia *ottava*.
![[cv452.png]]
In sto modo ho  qualcosa su cui poter calcolare il massimo (sono gia' allo step 3 dell'algoritmo presentato qua sopra).
A questo punto come detto possiamo trovare **extremas** in ogni 3x3x3 grid :3. In sto modo trovo peaks. e; Proprio NMS praticamente sto approccio.
**NOTA** pero'. fin'ora ho utilizzato NMS sull'x,y, ora pero' ho 3 dimensioni. E' proprio per questo che utilizzo una window cubica in questo caso.
Per prima fare NMS devo pero' aggiungere due altre scale all'*octave* che sono quella con $\sigma/k$ e quella con $k^{s+1}\sigma$. In sto modo ho lo stesso numero di differenze (di DoG), rispetto alle scale iniziali (non chiara sta cosa).![[cv452 1.png]]
Il problema qua qual e', che se io voglio TANTE scales, devo ogni volta *doublare $\sigma$, quindi raddoppiarlo*. E questa cosa dovrebbe essere per qualche motivo molto dispendiosa a livello computazionale. Cio' che si puo' fare invece, e' *downsamplare* l'immagine, renderla piu' piccina, invece di aumentare $\sigma$. Quindi usando menta' delle colonne  e meta' delle righe.
**NOTA**, il kernel non viene cambiato, si utilizza lo stesso, ma in una versione piu' piccola della mia immagine.![[cv4524.png]]
In sto modo trovo immagini a scale maggiori, senza effettivamente aver aumentato la scala.

ORA, solo ORA posso computare con la 3x3x3 grid il punto massimo con NMS (si prendono gli abs values).
![[cv314.png]]
Prendo come sigma ottimale $\sigma^*$ quello in cui ho il valore massimo.
Da quello che i ricercatori hanno trovato esce fuori che 
* il miglior numero di scale entro un'ottava e' s=3
* il sigma di base di ogni ottava migliore e' $\sigma = 1.6$ (quello di partenza capiamoci)
* l'immagine in input all'inizio proprio e' ingrandita (maxpool) di 2 in entrambe le direzioni (poi ovviamente si creano diverse ottave da questa, ognuna con valore iniziale di sigma di 1.6 e con un downsampling diverso ogni volta, sempre piu' piccino immagino. Le ottave create sono 4 e hanno: l'immagine upsamplata, l'immagine originale, l'immagine downsamplata, l'immagine ancora donwsamplata un'altra volta).

L'output di tutto sto proceso, anche dopo NMS, e' il seguente:
![[cv313.png]]
Si applica quindi una threshold sui punti che mostrano valori poco alti di derivate. Si fa anche un'altro pruning in cui si levano DoG responses che stanno sugli **edges**.![[cv435.png]]
Nella figura viene fatto prima l'uno e poi l'altro.
Guarda che risultato, spesso i blobs non rappresentano effettivamente cerchi.
![[cv.png]]

* HO ora SCALE INVARIANCE, perche' posso vedere lo stesso punto in entrambe le immagini solo che questo viene detectato a una scale differente. Cio' che devo fare e' solamente riscalare una delle due immagini computando la frazione tra il sigma della prima col sigma della seconda, che mi dice di quanto a livello di scala le due immagini differiscono.
* HO rotation invariance per quanto riguarda il detectare uno stesso punto saliente a diverse rotazioni dell'immagine, perche' utilizzo dei *blobs* e quindi non ho problemi. Il problema nasce quando voglio che ci sia rotation invariance anche a livello dei descriptors: ovvero voglio che due descriptors dello stesso punto saliente, uno in un'immagine ruotata, l'altro nell'immagine non ruotata, se comparati, mi dicano che c'e' un match. C'e' un modo, posso capire l'orientamento di uno dei due **punti salienti**, guardando i gradienti e il loro *orientamento*.
Ne parlo adesso un po' piu' nel dettaglio:
### Canonical orientation
Posso computare l'orientamento del punto saliente vedendo dove va la maggior parte dei gradienti che stanno nel neghborhood del mio punto saliente (identificato dal sigma ottimale trovato).
Dato il punto saliente:
1. la magnitude e l'orientation del gradiente sono computati a ogni pixel dentro il neighborhood dato dal sigma ottimale trovato della gaussian smoothed image.
2. costruisco un' *orientation histogram* creando dei bins con bin size uguale a 10 gradi. Qui accumulo per ogni bin tutti quei gradienti che hanno l'orientamento che cade in quello specifico bin.
3.  Il contributo di ogni pixel su ogni bin a cui questo appartiene (causa il suo orientamento) non e' di 1 (non e' statico a 1)ma e' dato dalla magnitude di quel pixel, pesata da una Gaussian (che e' piu' forte per pixels vicini al salient point, meno per pixels lontani dal salient point) con $\sigma = 1.5 \sigma^*$ con $\sigma^*$ che denota la scale ottimale trovata del punto saliente.
4. Questo istogramma potrebbe avere come output il seguente output:![[cv31 2.png]]
Questo mostra che c'e' una canonical orientation , di questo istogramma si prende il massimo, fittando una parabola sui neighborhood del peak nell'istogramma e prendendo il punto piu' alto risultante dal fit:
![[cv3441.png]]
Quella sara' la canonical orientation. In sto modo, prima di calcolare il descriptor, il salient point viene orientato (tutta la patch in verita') verso la direzione di questa canonical orientation. Solo dopo si calcola il descriptor. In sto modo risulto invariante rispetto alla rotazione.

Nota pero' che a volte si ha una situazione in cui un punto saliente ha **due** direzioni prominenti, come nel seguente caso:
![[cv131.png]]
Ho questo caso solo quando ci sono altri peaks che sono sopra l'80% del main peak.

In sto caso? Si computano **DUE** DESCRIPTORS, uno in cui seguo la prima canonical orientation e poi calcolo descriptor, l'altro in cui seguo la seconda canonical orientation.

Si e' calcolato che questa cosa e' abbastanza rara, nel 15% delle volte  si hanno punti salienti di questo tipo. 
NOTA posso avere anche piu' di due peaks eh, ho detto due cosi per dire.

## SHIFT DESCRIPTOR
Ora devo computare lo shift descriptor.
Quello che si fa e' la seguente cosa:
1. una grid di 16x16 pixels intorno a ogni punto saliente e' presa in considerazione, orientata wrt la **canonical orientation** prima trovata. Nota che viene anche levato il sigma ottimale legato a questa patch presa in considerazione (non so come ma evidentemente e' possibile). Questo mi permette di poter computare un descriptor che sia rotation invariant e scale invariant.
2. Questa grid e' divisa in  grids di 4x4
3. Per ogni 4x4 grid, computo un istogramma. Ho solo 8 bins perche' la bin size che considero e' ogni 45 gradi 
4. Anche in questo caso ho che ogni pixel nella 4x4 grid contribuisce al suo bin designato attraverso la sua magnitude che viene weightata con un gaussian weighting centrato nel punto saliente (con $\sigma$ uguale a meta' della grid size).
5. Si concatenano i risultanti istogrammi uno dietro l'altro e si computa cosi un feature vector che rappresenta il mio descriptor![[cv242.png]]
Nota che come detto prima posso avere piu' canonical directions per ogni punto saliente, e quindi piu' descriptors, ognuno orientato verso quella canonical direction.
C'e' un altro modo per il calcolo dell'istogramma. C'e' un modo che usa un **SOFT ASSIGNMENT** quando si parla di pixels assegnati a bins. Se ho un pixel che ha orientamento a 40 gradi e come ho detto sopra si hanno bins ogni 45 gradi, questo pixel andra' nel PRIMO bin (hard assignment), anche se e' vicino al CONFINE col secondo bin . Facendo un soft assignment invece i bins vengono entrambi aumentati della contribuzione di quel pixel per ogni bins.
Questo calcolo si fa dando $d_k$ (la distanza dal centro del bin all'orientamento del pixel) del contributo al PRIMO bin e $d_{k+1}$ (distanza dall'orientamento del pixel al centro del secondo bin) del contributo al secondo.![[cv1362 1.png]]
Questa cosa del soft assignment e' sia fatto entro un istogramma in ogni grid 4x4, che sia **tra regioni** (la contribuzione e' diffusa in modo bilineare tra le 4 regioni adiacenti). Quindi lo schema e' chiamato *trilinear interpolation*.
Il descriptor e' poi **normalizzato** a *unit length* per avere invariance wrt affine intensity changes (quindi cambiamneti enll'intensita'; che riguardano somme e moltiplicazioni). 

## MATCHING PROCESS
Nel matching process POSSO SOLO USARE IL DESCRIPTOR. NON posso usare la posizione dei salient points perche' appunto l'immagine potrebbe esssere ruotata, scalata ecc.. invece i descriptors sono such that sono invarianti rispetto a tutto praticamente a sto punto.
Computo le distanze tra i descriptors ora.
Posso usare come approccio **NEAREST NEIGHBORHOOD search**: ovvero per ogni punto saliente $x_i$, per ogni altro punto $x_j$, computo la distanza tra i descriptors $D(descr(x_i), descr(x_j))$ e matcho $x_i$, con il punto $x_j$ che mi da' la minor distanza. Questo e' un **BRUTE FORCE APPROACH** che non e' efficiente a livello computazionale. Non sempre questo approccio e' applicabile, capisci questa cosa.  NOTA che con questo approccio ogni punto saliente nella target image T avra' un match nell'altra immagine(la reference image R), anche se la cardinalita' dei punti salienti nelle due immagini e' diversa. 
Abbiamo bisogno di un criterio per **ACCETTARE O RIFIUTARE UN MATCH**.
1. La cosa piu' semplice e' utilizzare un threshold del tipo 
   $d_NN \leq T$. Ma questa threshold e' difficile da trovare perche' la dimensione su cui viene calcolata la distanza e' molto grande. Questo non mi permette di esser bravo nella scelta della threshold
   2. Si usa quindi la Lowe's Threshold
   ![[cv42124.png]]
   Si possono avere situazioni come nel caso 1 in cui i 2 descriptors migliori rispetto al descriptor preso in considerazione della target image, abbiano distanze dal target descriptor MOLTO DIVERSE. In questo caso non ho ambiguita', posso dire con abbastanza certeza che il puinto da matchare sia quello piu' vicino. Nel caso 2 ho invece che i due piu' vicini hanno un ratio di distanza dal target descriptor molto piccolo. Ho quiondi che qui la situazione e' ambigua e con difficolta' riesco a dire quale dovrebbe essere il punto da matchare.
   Lowe's threshold funziona infatti seguendo questa idea nel seguente modo:
   $\frac{d_{NN}}{d_{2-NN}}\leq T$
   Con $d_{2-NN}$ ratio tra la distanza del primo descriptor piu' vicino e il target descriptor e il seconod descriptor piu' vicino e il target descriptor. Quindi piu' questa distanza e' PICCOLA, piu' la frazione da' come risultato un valore alto, piu' NON CI SARA' MATCH, perche' si superera' il valore di T. 
   Lowe mostro' che T=0.8 permette di rigettare il 90% di matches **sbagliati**, missando soltanto il 5% di quelli corretti.
   ![[cv42.png]]
   Per capirlo da questa PDF, guarda l'area della PDF dei matches corretti da 0.8 fino alla fine e vedi quanto e' piccola (5%) e l'area della PDF dei matches sbagliati da 0.8 in poi e vedi quanto e' grande (90% piu' o meno).
   Posso usare *indexing techniques* per velocizzare la NN-search.
   C'e' un algoritmo chiamato **Best bin First** che e' molto piu' veloce di NN search e scala motlo bene con l'aumentare delle dimensioni. Pero' non sempre trova l'opt 
   solution.

   
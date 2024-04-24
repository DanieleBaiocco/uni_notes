Ricordi tutto quello detto sulla *pinhole camera?* Beh in verita' in pratica tutto il discorso della *perspective projection* da 3D a 2D, del fatto che le coordinate $(u,v)$ sono delle versioni scalate di rispettivamente $x$ e $y$ di una quantita' $z$, che e' la coordinata della profondita' nel 3D reference system.
![[fgfgf.png]]
Ho molte cose stravolte, infatti devo avere che:
1. solitamente si usa un *World Reference Frame* per i punti in 3D che e' diverso dal *camera reference frame* del pinhole camera model per i punti 3D.
2. le immagini sono una *griglia di pixels*, non sono un piano di valori *continui*, quindi devo implementare qualcosa che faccia **pixelization**.
3. le coordinate della mia immagine devono essere espresse seguendo un *image reference frame* che abbia l'origine nella parte in alto a sinistra, quando nel caso della *pinhole camera* ho che l'origine e' presente all'altezza della  $c$ (quindi vengono ammessi anche valori negativi).

Quindi in pratica le cose cambiano e il modello diventa un po' piu' complesso, ma non di tanto.
# 1. World Reference Frame
Prima di tutto introduco lui.
Praticamente si passa dal 3D al 2D in una maniera un po' piu' complicata:
1. si inizia con un punto in 3D espresso secondo il *World Reference Frame*. Questo punto e' $M_w = [x_w, y_w, z_w]$. 
2. Questo, attraverso **extrinsic parameters**  ovvero parametri che cambiano a seconda di quale WRF scelgo e come  **posiziono** con la camera rispetto al WRF, viene trasformato in $M_C = [x_c, y_c, z_c]$.
3. Attraverso degli **intrinsic parameters**, che si chiamano cosi' perche' sono fixati appena scelgo i sensori praticamente, passo finalmente a $m = [u, v]$.
# 2. Image Pixelization
Senza di questa, quando faccio *perspective projection* per ottenere $u \text{ e } v$, otterrei dei valori in **millimetri**, questo perche' $f, x_c \text{ e } z_c$ sono delle quantita' espresse in millimetri. Ma io non voglio, voglio che ci siano dei pixel con ognuno legato a un sensore. Quindi praticamente le formule per ottenere $u \text{ e } v$ diventano tali che si divide alla fine per $\Delta u$ e $\Delta v$, che sono rispettivamente
1. la grandezza orizzontale di un pixel in mm
2. la grandezza verticale di un pixel in mm
Queste due quantita' sono diverse perche' e' permesso in questa formulazione di avere dei pixel che **non** siano *quadrati*.
Le formule quindi diventano
$$u = \frac{1}{\Delta u} \frac{f}{z_c} x_c$$
$$v = \frac{1}{\Delta v} \frac{f}{z_c} y_c$$
Nota che il risultato e' comunque un *float*, quindi con la virgola. Se approssimo ho precisamente il pixel in cui il raggio di luce passa (con il float sono ancora piu' preciso btw).

# Origine dell'Image Reference Frame
Si sposta l'origine dalla $c$ al punto in alto a sx dell'image plane. Cio' comporta che le equazioni per $(u,v)$ diventano:

$$u = \frac{1}{\Delta u} \frac{f}{z_c} x_c + u_0$$ (1.0)
$$v = \frac{1}{\Delta v} \frac{f}{z_c} y_c + v_0$$ (1.1)
con $u_0$ e $v_0$ che sono le coordinate del piercing point $c$.
![[cvcvcvc.png]]
C'e' anche un modo per capire sta cosa per bene basta pensare alle coordinate di un punto che finisce in $c$. 

# Intrinsic parameters
Io non sono interessato nel calcolare precisamente $f \text{ e } \Delta u$ o $f \text{ e } \Delta v$, rispettivamente in  (1.0) e (1.1), ma sono interessato nel calcolare gli scaling factors $f_u$ e $f_v$.
Questi sono rispettivamente 
1. la focal length calcolata contando il numero di  pixels (usando la lunghezza di quelli orizzontali)
2. focal length calcolata in pixels (usando la lunghezza di quelli verticali)
Se ho che $\Delta u \text{  e } \Delta v$ sono uguali (ho pixels quadrati), allora sto $f_u=f_v$.
Il numero totale di **intrinsic parameters** e' quindi 4:
1. $f_u$
2. $f_v$
3. le coordinate del piercing point $c$ che sono $(u_0, v_0)$
FINE, questi 4 rappresentano la *camera geometry, che e' indipendente dalla sua posizione nel mondo*.

# Extrinsic parameters
E' solo una trasformazione tra due 3D reference systems. Qua non ho nessuna *perspective projection*. E voglio che si continui a misurare in millimetri, quindi non voglio nessun *scaling factor* (come prima erano $f_u$ e $f_v$).
C'e' un modo per mappare l'uno nell'altro. E' una trasformazione lineare chiamata **ROTO-TRANSLATION**, che consiste in
1. una rotazione intorno all'*optical center* espressa tramite una 3x3 **rotation matrix R**
2. una traslazione espressa tramite un vettore 3x1 $t$, che altro non e' che uno scaling factor (una volta che ho ruotato nel giusto verso poi applico soltanto uno scaling per fermarmi al punto giusto).
La relazione e' quindi 
$M_C = [x_c, y_c, z_c]= RM_w + t$
![[rototranslation.png]]
 Devo stimare R, nota che **NON** ogni matrice 3x3 e' una **ROTATION MATRIX**.
 Per essere tale, la matrice deve essere **ORTONORMALE**.
Ho che due vettori sono ortonormali se
1. il **dot product** tra di loro e' uguale a 0: $<a,b>= a^Tb= 0$ 
2. la loro norma e' uguale a 1 (unit length): $||a||_2 = ||b||_2 = 1$
Una matrice ortonormale e' tale che ha come vettori colonna vettori ortonormali
Di conseguenza si ha che $RR^T = R^TR = I$, perche' quando questi sono uguali, il risultato del dot product e' 1 (per il punto 2), mentre quando sono diversi il risultato del dot product e' 0 (per il punto 1). 
In pratica **l'inversa e' la trasposta** quindi in questo caso.
Posso ora secondo questo setting sapere le COORDINATE nel WRF di $C_M$ (dell'Optical center C in WRF), quindi del punto  $C_M =[0,0,0]$.
Voglio scoprire $C_W$ dato $C_M$
$$ C_M = RC_W+t <=> 0 = RC_W+t => RC_W =-R^Tt$$
Di quanti **extrinsic parameters** ho bisogno adesso?
Mi verrebbe da dire 12 perche' e' 9 di R e 3 di t, ma R ha solo 3 **parametri indipendenti** che corrispondono agli angoli di rotazione intorno agli assi del Reference Frame, uno per ogni coordinata $x,y,z$. Quindi in tutto ne ho 6 (3+3).

**POSSO ORA** esser tentato nel mettere tutto in un sistema di equazioni, per il calcolo di $u \text{ e }v$.
![[intring.png]]
Guarda bene come avvengono le sostituzioni, nulla di impossibile. Ho pero' che il risultante sistema e' **non lineare**. Questo e' dovuto al fatto che l'**intrinsic model** era **NON LINEARE**, quindi continua a esserlo. 
Il problema e' che per trovare la soluzione di un sistema non lineare devo **ottimizzare**.

# Vanishing point
Devo avere un modo per capire il vanishing point. Mi serve un modo per mappare un punto con coordinate $[+\infty, +\infty, +\infty]$, in un 2D point nel mio image plane, che corrisponde proprio al *vanishing point* (cioe' io vorrei comprendere le coordinate a livello di pixels di questo vanishing point).  Con il sistema di equazioni sotto , se pluggo $[+\infty, +\infty, +\infty]$ mi ritorna che $u \text{ e } v$ sono  anch'esse piu' infinito. Il che non e' molto informativo. Ho bisogno di qualcos'altro.

# Projective Space
Sono in un setting in $R^2$, in sto spazio ho che le linee parallele non si intersecano e i punti a infinity non possono essere rappresentati.
Posso appendere un'altra coordinata a un punto $m=[u,v]$ che diventa $\hat{m} =[u,v,1]$.
Posso poi dire che, per ogni scale factor diverso da 0, ho un infinito set di EQUIVALENTI 3D VECTORS che rappresentano lo stesso 2D point:
$\hat{m} =[u,v,1] \equiv [2u,2v,2] \equiv[ku,kv,k]\forall h\neq 0$  
Ho cosi creato un *projective space*, da $R^2 \text{ a } P^2$. 
Quindi un punto nel piano puo' essere rappresentato da una classe **equivalente** di **triplette**, in cui queste triplette equivalenti differiscono le une dalle altre di un *fattore*.![[projectiveSpace.png]]
Quello che e' stato fatto e' spiegato guardando questa immagine. Io sono partito con un piano (quello blu) con un  sistema di coordinate in $R^2$. Ho il punto $m = [u,v]^T$. Posso pero' dire che questo punto e' **equivalente** a tutti gli altri punti che si trovano sulla retta tratteggiata. Questi non sono altro che i punti visti prima, ovvero$\hat{m} =[u,v,1] \equiv [2u,2v,2] \equiv[ku,kv,k]\forall h\neq 0$. Per creare il *projective space* infatti basta creare un altro sistema di coordinate sotto il piano a **una unita'** di distanza. Ho li' l'origine del *perspective space* $P^2$. $m$ e' infatti da subito a distanza 1 (per l'asse delle $z$ ovviamente) dall'origine del perspective projection space. Nota che posso vedere l'origine del *perspective space* $P^2$ come il punto in cui ho la pinhole camera, e il piano in $R^2$ come l'image plane.

Ora, pensa a una linea in un piano. Come la definisco? Si puo' definire in diversi modi una linea. Io la definisco come 
$m = m_0 + \lambda d$, con $m_0$ che e' il punto in cui la linea  inizia, $d$ e' la direzione che e' moltiplicata a un parametro libero $\lambda$ che puo' prendere qualsiasi valore in $R$.
Praticamente e' variando $\lambda$ che ottengo tutti i punti della linea che passa per $m_0$ e segue la direzione $d$. Quindi ho
$$m = m_0 + \lambda d = \begin{bmatrix}u_{0} \\ v_{0} \end{bmatrix} + \lambda \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix}u_0+\lambda a\\ v_0+\lambda b\end{bmatrix}$$
Ora vedo questa linea generica $m$ in $P^2$:
$$\tilde{m}=\begin{bmatrix}m\\1\end{bmatrix}=\begin{bmatrix}u_0+\lambda a\\ v_0+\lambda b\\1\end{bmatrix} =\begin{bmatrix} \frac{u_0}{\lambda}+ a \\ \frac{v_0}{\lambda}+ b\\ \frac{1}{\lambda}\end{bmatrix}$$
Ottengo ORA un punto all'*infinity* , che e' quello che volevo trovare, quando $\lambda$ tende all'infinito. 
Ho, in questo setting, che $\tilde{m}_\infty = \text{lim}_{\lambda -> \infty} \tilde{m} = \begin{bmatrix} a\\ b\\ 0 \end{bmatrix}$  . Questo vettore e' un punto valido nel mio *projection space*. Non e' un euclidian point, questo perche' secondo quanto detto prima non posso passare da un punto in $R^2$ a un punto in $P^2$ che sia tale a  $\begin{bmatrix} a\\ b\\ 0 \end{bmatrix}$. Questo perche' non si puo' ammettere $h\neq 0$ come detto nella formula per trovare tutti i vettori equivalenti da $R^2$:
$$\hat{m} =[u,v,1] \equiv [2u,2v,2] \equiv[ku,kv,k]\forall h\neq 0$$
Questa cosa che $\tilde{m}$ non e' un valido euclidian point nel piano in $R^2$ e' ancora piu' visibile dall'immagine qua sotto: 
![[cvinfinity.png]]
Qui si ha che come si puo' vedere $\tilde{m}_\infty$ non interseca il piano in cui e' presente $m_0$, in quanto e' un vettore che giace nel piano con k = 0. E sto $\tilde{m}_\infty$ mi permette di rappresentare **punti all'infinito**. E' utile perche' una linea infinita nell'euclidian space viene mappata in un vettore in cui vengono preservate $a \text{ e }b$, che rappresentano la $DIREZIONE$ della linea. Vedo che questo vettore rappresenta un punto all'infinito in quanto ha 0 nella terza coordinata. **NOTA** che piu' linee parallele nell'euclidian space con diversi $m_0$ verranno mappate allo stesso vettore di $P^2$.

Faccio tutto questo perche' cosi posso:
1. prendere linee parallele nel mondo reale
2. computare il loro punto all'infinito
3. da questi, computare il vanishing point nella mia foto 2D
DI CONSEGUENZA voglio estendere tutto cio' che ho detto sui punti all'infinito a una linea 3D.
![[3dline.png]]
I *projective spaces* mi permettono anche di rendere la **perspective projection** **LINEARE**, che era un altro problema che volevo risolvere.
Dato  un 3D point $\tilde{M_C}$, e la sua proiezione nell'image plane $m = [u, v]^T$ ho che se esprimo questa operazione di trasformazione da 3D a 2D (perspective projection) utlizzando i **projective spaces** invece degli euclidian spaces ottengo un sistema lineare: 
![[cvcambia.png]]
**NOTA** posso moltiplicare tutto per $z_c$ e ottenere comunque un vettore **equivalente** solo perche' sono nell'*homogeneous space* e quindi per ogni k (in sto caso k=$z_c$) ho che il vettore risultante e' equivalente a quello precedente. Nota che alla fine $\tilde{m}$ non e' altro che moltiplicazione di una matrice lineare con l'*homogeneous representation* con k=1 del 3D point del CRF (della  camera reference frame).
**nota che** posso, una volta risolto il sistema di equazioni per $u, v \text{ e } k$ e una volta aver trovato $\tilde{m} \equiv \begin{bmatrix}ku\\ kv \\ k \end{bmatrix} \equiv \begin{bmatrix}u \\ v \\ 1 \end{bmatrix}$, tornare al vettore nell'euclidian space facendo $u = \frac{ku}{k}, v = \frac{kv}{k}$.
*nota che k e' $z_c$ no?*, e' la quantita' per cui ho moltiplicato di sopra nell'immagine.Infatti posso rappresentare l'immagine di sopra anche nel seguenete modo (con un k = $z_c$ che moltiplica $u$ e $v$)
![[different repres.png]]
Ora posso calcolare le *euclidean coordinates* del vanishing point nell'immagine partendo dalle coordinate di un punto nello spazio che va all'infinito.
Ho infatti che il vanishing point di un set di linee parallele in 3D e' la proiezione del loro punto all'infinito. Prendo infatti di una linea in 3D,  il suo punto all'infinito, che altro non e' che quello visto prima: 
$\tilde{M}_\infty = \text{lim}_{\lambda -> \infty} \tilde{M} = \begin{bmatrix} a\\ b\\ c\\ 0 \end{bmatrix}$, in cui solo la direzione e' preservata (non le coordinate del punto di inizio $x_0, y_0, z_0$). Questa e' l'*homogeneous coordinate* del 3D point di cui voglio sapere il map nel sistema di coordinate dell'immagine 2D. Trovando l'immagine 2D legata a questo 3D point ottengo le coordinate del vanishing point.
Nota che il vettore $\tilde{M}_\infty$ continuo a ricordare essere SEMPRE lo stesso per linee che sono parallele nel mondo reale, e quindi a un vanishing point trovato sono associate TUTTE LE LINEE che sono parallele nel mondo reale perche' queste ritorneranno TUTTE sempre la stessa $\tilde{M}_\infty$.
Il modo per trovare il vanishing point e' semplicemente:
![[cvvanishing.png]]
NOTA che NON ho un vanishing point **quando** $c$ e' uguale a 0, perche' dividerei per 0. MA QUANDO E' CHE c e' uguale a 0?

C'e' un set di linee 3D che non risulta poi in un *vanishing point* nell'immagine 2D, che e' il set di linee che hanno $c$ uguale a 0 per l'appunto.
Quando $c$ e' zero, quindi ci sono due validi componenti riguardanti la lunghezza e la larghezza, ma il terzo componente, legato alla profondita' e' 0. Le linee di questo tipo solo quelle **parallele** rispetto all'**image plane**. Queste NON hanno un *vanishing point* nella 2D image. Se infatti ho linee perfettamente parallele, allora queste NON convergeranno in nessun modo nell'immagine fino a un *vanishing point*, ma RIMARRANNO parallele in questa (nell'immagine finale saranno ancora parallele).

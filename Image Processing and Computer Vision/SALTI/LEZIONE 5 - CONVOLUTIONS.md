Se ho un'immagine in input di dimensione $3\text{ x } 32\text{ x } 32$, e' applicabile un **filtro, o kernel** all'immagine. La dimensione spaziale della convoluzione e' un *hyperparameter*, quindi lo posso scegliere io. Cio' che non posso scegliere e' la prima dimensione: un filtro in sto caso deve esser composto da 3 patches di pesi. Quindi , se scelgo la patch size a 5, dovra' avere una dimensione pari a $3 \text{ x } 5 \text{ x } 5$. Praticamente per ogni patch della convoluzione, cio' che viene outputtato e' un singolo numero, dato dalla somma delle 3 convoluzioni performate su quella patch. L'output quindi avra' 1 come prima dimensione.
![[convol.png]]
Nota che un filtro $3 \text{ x } 5 \text{ x } 5$ ha $3 \text{ x } 5 \text{ x } 5 = 75 + 1$ parametri perche' c'e' anche da contare il bias term che e' unico per l'intero filtro.
Posso creare **piu' di un singolo filtro**, quindi piu' $3 \text{ x } 5 \text{ x } 5$ filtri. In sta immagine ad esempio ne ho due.
![[convol2.png]]
Nota che il numero di filtri che voglio e' anch'esso un *hyperparameter*. Io posso scegliere quanti averne.
![[4filters.png]]
Nella forma generale ho la seguente cosa:
![[conv.png]]
Posso stacckare piu' convolutional layers una dopo l'altra.
Ti ricordo che se li staccko sto staccando delle trasformazioni lineari che sono constrained a essere di un determinato tipo (ma comunuque sto staccando delle trasformazioni lineari di base). Quindi devo introdurre non linearita' tra un layer e l'altro.
![[nonlin.png]]
**NOTA**: se io applico una convoluzione a una immagine, l'immagine viene resa piu' piccola. Questo perche' la convoluzione, senza un padding (**quindi quando si dice che il padding='valid'**), riduce la grandezza dell'immagine.
Si chiama *valid* perche' sto applicando il kernel in varie posizioni dell'immagine in input.
La regola di base e' $H_{\text{out}}= H_{\text{in}} - H_{\text{k}}+1$ , $W_{\text{out}}= W_{\text{in}} - W_{\text{k}}+1$. Con $H_k$ e $W_k$ la dimensione del kernel (della patch).
# Zero Padding
Se non voglio diminuire la grandezza dell'immagine, faccio zero padding. Quindi si applica la convoluzione su un'immagine in cui aggiungo  una cornice di 0s. 

La formula generale per il calcolo di $H_{\text{out}}$ e $W_{\text{out}}$ e' la seguente:
 1. $H_{\text{out}}= H_{\text{in}} - H_{\text{k}}+1 + 2P$
 2. $W_{\text{out}}= W_{\text{in}} - W_{\text{k}}+1 + 2P$.
 Per ottenere un'immagine che sia uguale all'originale devo porre P (ovvero quante cornici di 0 aggiungere all'immagine) pari a  $P = \frac{(H_k -1)}{2}$, in modo tale che cosi se sviluppo i calcoli ho $H_{\text{out}}= H_{\text{in}}$. Stessa cosa con $W_{out}$ (perche' tanto solitamente $W_k$= $H_K$).
 In sto caso si dice che utilizzo la strategia **padding = 'same'**.
 ![[zeropadding.png]]
 Ha senso usare degli 0 per la cornice? Beh avere 0 non e' molto informativo. Potrei magari avere i pixels sul bordo copiati nella cornice, solitamente e' meglio.
**NOTA UNA COSA CHE IL PROF RIPETE DA TEMPO**: nelle convoluzioni io ho introdotto l'**INDUCTIVE BIAS** della **locality**. Un inductive bias e' qualcosa che io programmatore voglio imporre al modello; la convolution e' un inductive bias perche' e' l'equivalente di un *linear mapping* W (fully connected praticamente) in cui i pesi sono diversi da 0 solo in alcuni pixels di volta in volta (in cui praticamente si hanno GLI STESSI PESI che vengono applicati a patches diverse dell'immagine). Infatti introducendo il convolutional layer io forzo il modello a imparare dei filtri che operano su delle patches dell'immagine ( e le degrees of freedom sono molte di meno rispetto a un *linear mapping* che rappresenta una full connection e in cui ogni pixel contribuisce. Qui, invece, solo i pixel considerati localmente (nelle patch) contribuiscono.).

# Receptive Field
Sarebbero gli *input pixels* che hanno un impatto nel calcolo del valore di una **hidden unit**.
Guarda questa immagine da dx a sx:
1. il valore della terza attivazione dipende da un kernel 3x3 sulla seconda attivazione
2. tutti i valori arancioni dipendono da una patch di grandezza 5x5 nella prima attivazione
3. tutti i valori in quella patch di 5x5 dipendono da TUTTA L'IMMAGINE IN INPUT di base (da un'area di 7x7)
![[receptivefield.png]]
Ho bisogno di 3 convoluzioni per avere come receptive field di un valore della terza attivazione TUTTA l'immagine in input.
La formula per la dimensione del receptive field, che dipende da dove mi trovo (dal livello L dell'attivazione: primo, secondo, ecc...) e'
![[r_l.png]]
Il probelma qua e' che $r_L$ cresce in maniera lineare, quindi molto lentamente. Cio' significa che se voglio un'attivazione in cui il *receptive field* legato a valori di quell'attivazione SIA ABBASTANZA GRANDE (ora non dico l'intera immagine ma quasi) devo **staccare** una marea di convolutional layers. Devo arrivare alla 20esima attivazione, alla 30esima. Cio' e' infattibile, o comunque molto poco efficiente.

**Per far crescere piu' velocemente il receptive field** solitamente si *downsampla l'attivazione* dentro la rete (quindi tipo con la stride a 2 invece che a 1 immagino). In sto modo mi muovo piu' velocemente quando scorro i valori dell'attivazione e osservo i corrispettivi receptive fields (anche se la definizione di receptive field e' solo legata all'input image, vabe 
pero' hai capito) e copro piu' velocemente aree piu' grandi.


Di base se utilizzo una STRIDE A 2 ho questi calcoli qua sotto:
![[stride.png]]
In sto modo con molti meno layers staccati ho un receptive field che e' abbastanza grande (grande tanto quanto magari anche l'intera immagine).
Guarda bene e studia la regoletta per calcolare $H_{\text{out}}$.
Cio' che succede e' la seguente cosa:
![[expreceptive.png]]
Nota come con stride a 2 il receptive field sia molto grande se prendo un pixel della *terza attivazione*.
Nota che la grandezza del receptive field aumenta **esponenzialmente** rispetto alla stride.
Questa e' la formula che hanno derivato con l'aggiunta della stride
![[stridereceptive.png]]
Praticamente devo mettere a ogni attivazione $l$ che scorro con la sommatoria il prodotto di tutte le strides che ho avuto fino a quell'attivazione (quindi le strides utilizzate nei convolutional layers concatenati). Posso anche non avere strides magari per certi conv layers quindi nella produttoria per quei layers metto un valore pari a 1 (assenza di stride).
Se abbiamo una **stride costante S che e' sempre quella e e' applicata a tutti i conv layers** allora la formula diventa:
![[stridereceptive2.png]]
Ho un aumento esponenziale del receptive field con la stride.

# Parametri della convoluzione
Se ho questo conv layer
![[convparcount.png]]
Allora avro' per la convoluzione un numero di parametri da imparare pari a $16( 8 \text{ x } 5 \text{ x }5 + 1) = 3216$.
Per calcolare $H_{\text{out}}$ guardo la formula scritta prima, quindi ho
$32 - 5 + 2*2 = 32$, la stessa cosa per $W_{\text{out}}$.
La grandezza totale dell'output e' quindi $16 \text{ x }32\text{ x }32$ = 16384. Ho 16 mila valori in output :).

VOGLIO VEDERE QUANTO CI VUOLE A LIVELLO DI OPERAZIONI IN FLOATING POINT per computare questi 16mila numeri.
Praticamente per ogni valore dei 16mila presente nell'output activation, ho che questo e' stato calcolato attraverso una convoluzione $8 \text{ x }5\text{ x }5$. In questa convoluzione ci sono state $8 \text{ x }5\text{ x }5=200$ prodotti e anche 200 somme (perche' comunque cosi' funziona la convoluzione). Quindi avro' per un singolo valore nell'output activation $200*2$ operazioni. Per trovare questa cosa per ogni pixels moltiplico per questi 16384 valori.
![[outputactivation.png]]
La formula generale e' 
![[genform.png]]
Nota che le convoluzioni sono comunque molto pesanti eh nonostante siano una constrained version del linear layer.
Nota pure che i flops sono legati alla dimensione di output.
# Layers comuni in una CNN
1. fully connected layers
2. non linear activation function
3. batch norm layers
4. pooling layers

## POOLING
E' molto simile alla convoluzione, perche' slida sull'input activation. La differenza e' che 
1. i numeri NON SONO IMPARABILI, sono gia' scritti, come nel caso in cui faccio **average pooling**
2. oppure c'e' una funzione NON LINEARE diversa dalla convoluzione come nel caso del **max pooling** in cui viene preso il pixel col valore piu' grande.
Non ha molto senso fare un **average pooling** solitamente, perche' il modello potrebbe imparare una convoluzione in un conv layer che faccia proprio average pooling. Emulare *max* con una convoluzione e' invece **impossibile**, a meno che non ci sia una combo con un conv layer e la ReLU.
**IMPORTANTE** che per fare downsample uso la STRIDE, ma POOLING non per forza viene usato per fare downsampling, puo' anche avere la stride a 1 tranquillamente. Si e' sempre pensato che il pooling servisse per downsamplare, ma cio' non e' vero. Difatti posso donwsamplare anche usando un conv layer basta che la stride sia >1.
Questo e' il *max pooling*:
![[pooling.png]]
Un'altra differenza e' che la *convoluzione* processa TUTTI GLI INPUT CHANNELS, mentre qui col pooling ho che si processa un channel alla volta. Quindi la dimensione degli input channels e degli output channels DOPO il pooling e' **la stessa**.
![[poolingex.png]]
Il lato positivo e' che il **pooling** permette di ottenere **invariance** rispetto a shifts spaziali.
Mi permette di avere la stessa risposta finale anche se le features compaiono in posizioni leggermente diverse. Cioe' il 6 e il 7 possono essere SOPRA in qualsiasi altra posizione all'interno di quelle griglie 2x2, HO comunque lo stesso output. Vista in una maniera migliore, se magari in sto layer sto detectando degli occhi, se nell'immagine in input l'occhio compare in una posizione leggermente diversa (quindi cio' risulta in un input channel con un 6 magari spostato leggermente rispetto a quello di qua sopra), ad ogni modo l'output del pooling mi dara' la stessa risposta.
Nota che stiamo ragionando secondo il receptive field, in verita' l'invarianza e' molto grande nell'input image.
**NOTA** pero' che sto perdendo un sacco di informazione eh prendendone solo il massimo.

## Vanilla CNN
Quello che posso fare ora e' questo
![[vanillacnn.png]]
Nota che a una certa flatteno l'immagine e torno a fare *linear mapping*. Nota che l'ultimo layer e', come ho detto le scorse volte, un LINEAR CLASSIFIER scemo che separa le classi di base, perche' ormai sono linearly separable dopo tutti sti conv layers. 
Il modello si divide in due parti:
1. la head (ovvero sto ultimo linear classifier)
2. e la backbone (ovvero quello che viene prima). Chiamato anche il *feature extractor*, perche' computa high level image representation. In $r_3$ infatti, quello che si ottiene (o che spero di ottenere) e' che in questo spazio i gatti siano in una parte di esso, i cani in un'altra, le tartarughe in un'altra ancora. Distribuiti in maniera tale che sono separabili linearmente con hyperplanes lineari.
Perche' a una certa passo da convoluzioni a linear layers che sono fully connected?
Perche' voglio guardare l'interezza dell'immagine. Talvolta non riesco a arrivare a global receptive fields (non riesco a arrivare al receptive field che tenga tutta l'immagine all'interno). Inoltre sappiamo che i linear layers fanno pattern matching praticamente. Qua ho che si fa pattern matching sulle features che sono presenti in $r_2$. Quindi si imparano dei templates, magari per il gatto che presuppone una certa distanza degli occhi e del naso e della bocca. E' pero' un  template in un feature space $r_2$ (prima lo facevamo direttamente sui pixels).

**E' pero' molto difficile trainare sto CNN, soprattutto coi linear layers alla fine**.

## Internal Covariance Shift
Il covariance shift solitamente e' quando traino su una distribuzione e a test time ho un'altra distribuzione, altre immagini diverse.
L'internal covariance shift e' dovuto al fatto che la distribuzione cambia ogni volta. h cambia mentre sto facendo training perche' sto cambiando W1 e b1. Quindi quando faccio un update su W2 e b2 sto facendo questo update sulla *vecchia* distribuzione di h (l'ultimo linear layer fa l'update sulla vecchia distribuzione di h, la quale cambia in continuazione).
![[covshift.png]]
La distribuzione di h cambia drasticamente tra uno step e l'altro, e per il linear layer finale a sta dietro a tutti sti cambiamenti e' pesante.
Voglio fare in modo che la distribuzione di h non cambi in modo cosi' libero.
## Batch norm layer
E' legato alle minibatches. E' l'unico layer legato alle minibatches, che conosce la minibatch size.
Suppongo che ho 3 immagini nella minibatch. Per ognuna di queste immagini computo ovviamente un'activation layer dopo il linear layer. PRIMA DI RUNNARE L'ATTIVAZIONE (con la ReLU)
![[batchnorm.png]]
Cio' che viene fatto e' prendere i tre vettori riferiti alle tre immagini, calcolare la mean tra righe di questi vettori e la varianza(l'std) tra le righe sempre.
Faccio poi standardization: quindi sottraggo per la mean e divido per la std (voglio che sia zero mean e unit variance). Questo si e' notato essere troppo restrittivo (tiene la distribuzione delle attivazioni troppo sotto controllo), quindi si e' aggiunta un po' di flessibilita'. Si aggiunge una *scale* e un parametro di *shift* che sono imparabili dal batch norm layer. E' un po' contraddittorio che sto agiungendo il constraint della distribuzione e che poi lasci variare gamma e beta per non constrainare troppo no? eh.

In teoria SGD puo' anche *ignorare* la presenza della batch norm perche' puo' fissare ![[identity.png]]
e in sto modo e' come se non fosse successo nulla. E' come se non avessi applicato niente.
![[slidebatchnorm.png]]
NOTA CHE A TEST TIME HO UN BEL CASOTTO
perche' di base ho che la mia predizione dipende da quali altre immagini sono presenti all'interno della minibatch del test set che sto processando in quel momento. Questa cosa non va bene, vorrei essere invariante rispetto a sta cosa. Sarebbe un processo completamente stocastico.
E' per questo che prima ho calcolato gli intermedi $\mu_j^{(t)}, v_j^{(t)}$ a tempo di training. A ogni tierazione di SGD ho queste running averages di $\mu_j^{(t)}, v_j^{(t)}$ in cui queste vengono aggiornate prendendo i valori precedenti $\mu_j^{(t-1)}, v_j^{(t-1)}$ e gli attuali $\mu_j, v_j$ calcolati allo step corrente. Alla fine del training SI USANO QUESTI $\mu_j, v_j$ **finali** come  valori COSTANTI per calcolare gli $s_j$,
![[batchnormtest.png]]
Nota che a test time ho che anche $\gamma_j$ e $\beta_j$   che SONO COSTANTI, in quanto imparate. Queste possono essere fuse con il linear layer precedente che anche quello e' costante. Quindi a test time e' come se non ci fosse. Questo mi fa dire che NON ho piu' potere espressivo con la batch norm.
Se lo disabilitassi a tempo di test, avrei che il layer successivo ottiene un vettore diverso da quelli passati a train time, perche' e' completamente diverso.

L'inception model e' un modello in cui NON ci sono BatchNormLayers. Nota che i modelli con batchNorm convergono a una accuracy alta in MOLTI MENO STEPS rispetto all'inception model. Inoltre l'accuracy e' addirittura piu' alta rispetto a quella dell'inception.
![[batchnormfast.png]]
Nota che le reti **BN x5** e **BN x30** si riferisce a **quanto aumento il learning rate**. 
E quando traino il BatchNorm model con un learning rate alto ho che converge al 72.2% di accuracy molto piu' velocemente rispetto alla baseline.
Inoltre ottiene dei valori di **accuracy** addirittura piu' alti (arriva a 74.8% nel caso in cui aumento il **lr** per 30)
![[Pasted image 20240521110912.png]]
Il training e' MOOOLTO STABILE.


Ho che i pro e i contro quindi sono:
![[prosconsbatchnorm.png]]
L'inizializzazione non e' cosi importante perche' INDIPENDENTEMENTE da come sono inizializzati i pesi, vengono  ricalcolati come valori con zero mean e 1 di std.
Il training non e' deterministico perche' le attivazioni dipendono da quale batch si utilizza: la stessa immagine considerata CON DUE DIVERSE BATCHES genera delle attivazioni che sono DIVERSE per ogni batch (in quanto il calcolo e' legato alle batches).
L'ultimo punto dei cons: quando uso piccole batches io sto calcolando delle *statistiche* come la mean e la variance su POCHISSIMI ESEMPI e ovviamente sono sbagliate queste statistiche, sono troppo noisy. Questa cosa e' poco democratica perche' io studente USERO' sempre micro-batches non avendo le GPUs di Google, e quindi BatchNorm non sara' cosi' efficace con batches cosi piccole.

### Batch Norm con Convolutions
Ho per ora visto batch norm con fully connected layers. Vedo ora con le convoluzioni come opera.
Col fully connected abbiamo visto che praticamente ho B x D con B batch size e D dimensione di output del fully connected layer.
E qui prendo ogni riga della batch e computo una *mean* e una *variance*. Computo statistiche per ogni riga e i vettori risultanti sono $\mu$,$v$ di dimensioni 1 x D.
Ho poi legati a essi i  vettori $\gamma$, $\beta$ che sono di dimensioni 1 x D.
Poi da sti 4 vettori computo il vettore $s$ che torna a essere B x D.
![[fullyconnectedbatchnorm.png]]
Nel caso della convoluzione invece cio' che viene fatto e'
![[convolutionbatchnorm.png]]
In sto caso infatti ho che ho in input B x $C_{\text{out}}$ x H x W. E $\mu$,$v$ sono di dimensione legata a $C_{\text{out}}$. Ovvero se ho 16 immagini nella mia batch e 4 channels, calcolo 4 statistiche, una per ogni channel, prendendo per ognuna in considerazione tutte e 16 le immagini della batch (nella loro totalita') su quel channel li'.
Lo stesso per $\gamma$, $\beta$. Quindi le statistiche non vengono computate solo secondo la minibatch dimension ma anche secondo la spatial dimension (perche' praticamente vado a fare avg dei pixels delle immagini in un determinato channel tra di loro).
# Architetture di successo per risolvere ImageNet
## LeNet5
![[LeNet5.png]]
Nel primo conv layer, ho come risultato 6 feature maps con dimensioni 28x28.
So pure che il kernel aveva dimensione 5x5. Quindi quanto era la stride? e il padding? e gli input channels?
il kernel era di  dimensione 6 x 1 x 5 x 5 (perche' 6 sono le feature maps prodotte, 1 e' la dimensione dell'input channel perche' le immagini in input sono grayscale images)
La stride e' 1 e il padding pure perche' secondo la formula ho $H_{\text{out}}= \lfloor\frac{H_{\text{in}} - H_{\text{k}} + 2P}{S}\rfloor + 1$ , con il termine della frazione arrotondato per difetto.

Subsampling nell'immagine e' *avg pooling*.
![[lenet5specifics.png]]
Nota che il numero di featuremaps, di channels aumenta man mano che andiamo giu' nella rete, mentre la spatial dimension diminuisce. Voglio infatti che il numero di features estratte AUMENTI SEMPRE DI PIU'. All'inizio ho low level features, e servono meno channels. Cosa inversa succede per la spatial dimension, che inizia grande e va a diminuire.

NOTA CHE QUESTA ARCHITETTURA non e' legata a imagenet, e' stata trainata su hand written digits. ALEXNET riprende questa architettura.

## AlexNet
![[alexnet.png]]
Hanno preso la LeNet architecture, l'hanno resa piu' grande e hanno cambiato la sigmoid con la ReLU **FINE**.
Il concetto di far crescere il numero di channels e di far diminuire la spatial dimension e' seguito.
NOTA che ci sono due gruppi di convoluzioni, questo perche' c'e' stato il bisogno di splittare il modello in due GPUs per il training. Si e' fatto *model parallelization*. Solitamente si fa *data paralelization*, con un unico modello in memoria ma con i dati che vengono allocati in piu' GPUs parallelamente durante il training.
Praticamente all'inizio ho che si creano 48 channels in una GPU e 48 altri output channels nell'altra GPU.
Le convoluzioni successive processano SOLO i channels disponibili su una singola GPU, quindi tipo nel conv layer successivo ho un kernel che e' 48x5x5 perche' processa SOLO l'attivazione legata alla GPU di sopra. Lo stesso per la GPU di sotto. Se non avessi avuto parallelizzazione del modello, li' avrei avuto una convoluzione con dimensione 96x5x5.

E quando vedo poi nei layer successivi le linee che si intersecano li' ho che le due GPUs iniziano a comunicare.

### Architecture breakdown
In questo breakdown suppongo che il modello sia in un'unica GPU
Ho che ogni minibatch pesa 75 MB
![[attivazioni.png]]![[stemlayer.png]]
Lo stem layer permette di dimiuire la spatial dimension e si mette all'inizio solitamente.
Penso ci sia un errore comunque nella slide di qua sopra perche' c'e' scritto che le attivazioni iniziali sono 227 sia per H che per W quando nell'immagine di sopra invece c'e' scritto che le immagini sono di grandezza 224. SUPPONGO CHE L'ERRORE SIA NELL'IMMAGINE DI SOPRA e continuo.
![[calcolo.png]]
Cosi ottengo 55 attivazioni, vado poi a calcolare il numero TOTALE di attivazioni espresso come 55 x 55 x 96 e ottengo 290400. Per calcolare invece il numero di parametri che la rete deve imparare in questo primo conv layer basta che faccio
![[numpar.png]]
Nota che il +1 e' per il bias term e ce l'ha ogni pacco di convoluzione. Ho che e' $(11*11*3 +1) * 96= 34944$.
Per i flops calcolo:
![[flops.png]]
E' una grande bestia, devo computare 27Gflops solo per la prima attivazione.
Per calcolare invece la memoria occupata la calcolo nel seguente modo
![[Pasted image 20240521131646.png]]
![[calc.png]]

Ho poi un layer di maxpooling che e' 3x3 con stride a  2, quindi ho overlap nel maxpooling. Qua per calcolare la nuova spatial dimension uso la formuletta $H_{\text{out}}= \lfloor\frac{H_{\text{in}} - H_{\text{k}} + 2P}{S}\rfloor + 1$ come sempre, invece per il numero di feature maps in output beh questo e' lo stesso dell'input perche' maxpooling agisce sul singolo channel di base, quindi avro' che rimane 96.
Nota pure che il **numero di parametri** e' 0, non ho parametri da imparare.
![[finale.png]]
Nota che quando poi sto nei fullyconnected layers, io devo imparare un sacco di parametir.
Di fatti ho che fc6 ha in output 4096 unita', prima ne avevo 9216, quindi dovro' imparare una matrice W con 4096 righe e 9216 colonne a cui sommare 4096 bias terms.
![[par.png]]
Nota il numero di parametri COME E' GRANDISSIMO rispetto alle convoluzioni.
![[dfa.png]]
ORA ANALIZZO I RISULTATI:
![[breakdown.png]]

1. Prima cosa da notare e' che io ho dimiuito la spatial dimension da subito per controbilanciare l'aumetno del numero di kernels (di feature maps). Se l'avessi tenuta alta i flops e l' activations memory sarebbero schizzati alle stelle![[fff.png]]
Diciamo infatti CHE LARGHE ATTIVAZIONI sono responsabili di grande consumo di memoria, per quello e' buono tenerle basse fin da subito. Poi ovviamente la memory consumption NON E' PIU' UN PROBLEMA. LO STEM LAYER QUINDI LO AGGIUNGO SOPRATTUTTO SE HO PROBLEMI DI MEMORY CONSUMPTION.
2. La maggior parte dei parametri e' nel *fully connected layer* (il 95% dei parametri NON BENISSIMO DAI)
3. La maggior parte dei FLOPS li ho nelle **intermediate convolutions** come e' mostrato in rosso nella figura. Questo perche' si generano TANTI CHANNELS DA TANTI CHANNELS anche se le attivazioni sono piccole
4. Le attivazioni e i parametri occupano piu' o meno la stessa memoria a livello di MB. Questa cosa solitamente non va cosi nelle moderne architetture di modelli perche' le attivazioni tendono a essere molto piu' grandi e i parametri molto piu' piccoli. Cio' e' legato al fatto che TUTTI QUEI PARAMETRI NELLA PARTE FINALE DELLA RETE DEL FULLYCONNECTEDLAYER non sono poi cosi importanti e verranno diminuiti nelle architetture che verranno negli anni
In tutto ho 60milioni parametri, e' una grossa bestia eh ALEXNET.
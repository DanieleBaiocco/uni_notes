 ![[Pasted image 20240607191038.png]]
 L'asse delle x indica la model capacity. Non si riferisce alla training history attenzione. 
 Indica quanta capacita' ha in termini di numero di neuroni (e non solo). 
 Solitamente si ha che piu' il modello e' potente mi vien da dire, allora piu' e' probabile che si OVERFITTI. Questo perche' il modello e' cosi' potente che impara anche **spurious correlations** del training data tra i dati e le labels, che NON sono presenti nel test data insomma.
 Metti che nel training set ho che i gatti sono visti da una posizione LATERALE, un modello con grande capacita' potrebbe imparare che c'e' una correlazione tra la posizione del soggetto (LATERALE in sto caso) e la classificazione di quell'oggetto in gatto. Questa e' una *spurious correlation*. Generalizzera' male perche' se poi vede gatti visti di fronte allora non sara' in grado di detectarli perche' non rispettano la correlazione imparata.

Nel contesto delle NN la *model capacity* corrisponde a NETWORK WIDTH, NETWORK DEPTH, input resolution (che ha un effetto sulla capacita' del modello), l'ARCHITETTURA del modello (due reti con lo stesso numero di parametri HANNO una CAPACITY diversa se ci pensi, ad esempio basandosi sulla presenza o meno delle skip connections).
Non pensare alla capacity SOLO come il numero di parametri.
Ci sono modi per calcolare la CAPACITY di un modello tra l'altro.

# Esempio con  LINEAR REGRESSION GENERALIZZATA
Praticamente si puo' fittare qualsiasi tipo di curva sui dati prendendo una feature e calcolando altre features da essa legate a POLINOMI sempre piu' grandi.
Posso infatti scegliere K features partendo dalla x, e in base a quanto e' grande K, definisco la capacita' del mio modello.
![[Pasted image 20240607233520.png]]
La ground truth e' espressa tramite la formula polinomiale $y = (ax^3 + bx^2 + cx + d) + \epsilon$. Io posso da x generare dei polinomi in numero K. Con k uguale a 3 ottengo la capacita' del modello ottimale.
Con k=1 ho troppo poco potere espressivo e quindi underfitto i dati. Con K = 9 ho una capacita' del modello troppo grande che mi porta a overfittare sui training data. Di conseguenza non generalizzo PER NIENTE BENE.
Nota infatti che con DEGREE k = 9 (ho una degree pari al numero di training samples) ho che la metrica dell'accuracy e' a 1.0. Posso infatti interpolare/fittare perfettamente tutti i punti.  

NOTA che nelle NN non ho questo discorso di DIVERSI MODELLI, ho un solo modello CON UN POTERE distruttivo LOL.
NOTA CHE la mia ResNet e' piu' paragonabile al modello con Degree a 9, rispetto a modelli con Degree bassa come 1 o 3. Infatti ha **milioni e milioni di parametri** per un datastet di **150mila immagini**. 
Potrebbe difatti imparare anche a fittare PERFETTAMENTE le training images come succede nell'esempio di sopra. Potrebbe teoricamente imparare quello, data la sua alta **capacity**. Dobbiamo fare in modo di **evitare questa cosa**.

In una SVM, che ha una loss che e' CONVESSA ho che il risultato a cui convergera' l'ottimizzazione SARA' sempre uno. So quindi a PRIORI IL RISULTATO A CUI CONVERGERA'. Nelle reti neurali NON ho questa certezza, di conseguenza l'OTTIMIZZAZIONE della loss landscape gioca un ruolo fondamentale. 

E' importante pensare alla CAPACITY del modello come EFFETTIVE CAPACITY. Se prendo in esempio ResNet con i valori inizializzati a random, la traino per poco e faccio evaluation su train e test set avro' che l'error rate sara' MOLTO alto, indicando UNDERFITTING. Quindi si ha un'**effective capacity** di merda/bassa (il modello UNDERFITTA) anche se la theoretical capacity della rete e' molto larga/grande (in verita' e'  cosi alta che potrebbe approssimare qualsiasi funzione e quindi potrebbe teoricamente overfittare pesantemente il training set, come nel caso visto di Degree = 9).

Cio che voglio vedere e' "**DATA UNA ARCHITETTURA FISSA, COME SI PUO' LAVORARE SULLA SUA EFFECTIVE CAPACITY, in modo da ottenere una OPTIMAL MODEL CAPACITY** (come quella nella figura iniziale di questa Obsidian Valut)".

## LEARNING RATE
Si inizia da questo, come fa il lr a influire sull'effective capacity? Perche' se lo metto troppo basso, allora il mio modello NON si muovera' di tantissimo dalla random initialization che aveva il modello all'inizio. Quindi continuero' a avere underfitting.

![[Pasted image 20240607235959.png]]
Un learning rate alto mi permette di avere un buon modello inizialmente, e poi NON migliora. Ha senso quindi magari iniziare con un LR alto e poi abbassarlo. Alla fine arrivo a un buon valore della LOSS anche se non sapevo come settarlo.
![[Pasted image 20240608000900.png]]
A una certa CONTINUO a andare a DX e SX, sono stacckato.
INFATTI IN ResNet, quando si va a vedere la training history, si ha che l'errore e' a SCALA
![[Pasted image 20240608001102.png]]
Quindi si inizia a un valore di lr, poi quando si fanno 30 epoche, si fa diviso 10. Guarda il PLOT a DX.
Per la precisione, ogni volta che si ha un plateau allora si fa DIVISO 10.
Si e' migliorato di 15 punti di accuracy (da 45 a 30% dell'errore del modello) attraverso questo lr schedule.
PERCHE' non ho overfitting in questo caso? L'intuizione e' che LR schedulers permettono di trovare dei MINIMA nella error loss curve che GENERALIZZANO bene perche' SONO LARGHI. Questi MINIMA sono CONVEX IN UNA AREA MOLTO GRANDE, o meglio quando la LOSS e' PIATTA vicino al MINIMO.
![[Pasted image 20240608001616.png]]
Quindi nella figura si atterrera' nella conca piu' a DX.
Questo perche' se inizio con un piccolo LR allora mi fermo al primo minimo che trovo e mi STACCKO perche' inizio a bounchare:
![[Pasted image 20240608001711.png]]
Con la lr SCHEDULE ho che la step size e' cosi' alta che sorpasso il primo local minima e mi muovo in altre zone della loss landscape. E HO che CI SI FERMA, ovvero ci si trova in una situaazione di STACK quando si trova una CONCA che abbia una larghezza UGUALE a quella del LEARNING RATE (che e' molto alta come gia' detto):
![[Pasted image 20240608001906.png]]
Quindi mi fermero' a un MINIMO che e' MOLTO LARGO, skippando tutti i MINIMA CHE SONO STRETTI, in quanto continuo a esser sbalzato fuori da questi USANDO QUESTO ALTO LR.
A sto punto usando LR schedule, ho che appena c'e' una situazione di STUCK, questo viene diminuito, e quindi l'errore continua a scendere. Appena mi ri-stuccko si ridivide e si ricontinua a scendere finche' non ottengo il local minimum (o comunque una sua buona approssimazione)
![[Pasted image 20240608002117.png]]
OVVIAMENTE POSSIAMO VEDERE COME E' LA LOSS LANDSCAPE anche per il test set, e e' stato dimostrato PIU' VOLTE che c'e' uno **SHIFT** tra train e test set. Quindi il valore ottimale per il train set NON e' lo stesso per il test set. Si ha quindi uno SHIFT che in figura e' verso DX ma in pratica e' composto da distorzioni della loss landscape. 
Solitamente l'optimal value della loss del test set e' VICINO al valore ottimale della loss del training set MA NON ESATTAMENTE LI'. HO QUINDI CHE MINIMA LARGHI, come quello che viene trovato da una lr schedule SONO MOLTO PIU' ROBUSTI RISPETTO A MINIMA STRETTI (con conche strette), perche' nei minima stretti, un piccolo shift della loss landscape risulta in un GRANDE ERRORE DI GENERALIZZAZIONE:
![[Pasted image 20240608002557.png]]
 , perche' **LA LOSS E' SIMILE NEI VALORI VICINI DEL PARAMETER SPACE**. Quindi se si ha uno shift ho un impatto MOLTO MENO DRAMMATICO. I valori dei parametri trovati comportano comunque una loss abbastanza bassa, perche' anche dopo uno shift quel valore della loss e' comunque all'interno della conca nella test loss landscape.

E' PER QUESTO CHE NON SI OVERFITTA.

SI E' ANCHE PENSATO DI CAMBIARE IL LR IN MANIERA CONTINUA, levando quindi le design decisions per determinare QUANDO ABBASSARE IL LR.
![[Pasted image 20240608002934.png]]
In sto modo non ho scelte di design da fare. Solo il valore del LR iniziale
![[Pasted image 20240608003011.png]]
Anche questo puo' funzionare.
PROBLEMA pero': INIZIARE CON UN GRANDE LR porta a **instabilita' del TRAINING**, in particolare se il modello e' DEEP, come una ResNet.
C'e' infatti da dare al modello degli steps in cui si muove FUORI dalla random initialization INIZIALE. Quindi c'e' bisogno di un piccolo lr per le prime epoche in modo che il modello CONVERGA in qualcosa di okey, e da li' poi si parta con questo lr schedule, con un lr alto.

![[Pasted image 20240608003357.png]]
GUARDA L'IMMAGINE A DX, ci sono poche epoche in cui il lr e' molto basso, tipo 6. All'initialization voglio muovermi in zone MIGLIORI da quella terribile della random initialization.
IN PRATICA UNA COSA CHE SI FA E':
![[Pasted image 20240608003507.png]]
Questi segmenti sono lineari o cosine. NOTA che il LR viene cambiato DOPO OGNI MINIBATCH, non dopo ogni epoca. Quindi ogni volta che faccio un GD step.

## REGULARIZATION
E' qualsiasi  modifica che apporto che riduce il test error senza tener conto di come cambia il training error.
![[Pasted image 20240608011827.png]]
Tramite regularization posso ottenere la situazione in figura, in cui il training error e' piu' alto ma il test error e' piu' basso. Ho migliorato la GENERALIZZAZIONE.

### PARAMETER NORM PENALTIES
Aggiungo dei constraints sui pesi che ho. Tipo posso mettere un constraint per cui la somma di TUTTI I PARAMETRI CHE HO non puo' eccedere una threshold. Quindi il mio modello MAGARI ha 20 MILIONI di parametri, MA NE PUO' USARE SOLO LA META', perche' se tutti hanno dei valori, allora la loro somma eccede la threshold e il constraint viene violato. 
POSSO QUINDI TENERE SOTTO CONTROLLO L'EFFECTIVE CAPACITY del mio modello.
SOLITAMENTE il constrain NON E' COSI' HARD, utilizzo dei SOFT CONSTRAINTS.
Solitamente PENALIZZO modelli che hanno valori troppo alti nei loro weights
![[Pasted image 20240608012357.png]]
Do' una preferenza a modelli che usano POCHI pesi (valori bassi dei pesi).
TORNANDO ALL'ESEMPIO DI PRIMA![[Pasted image 20240607233520.png]]
traslare questo paragone a una NN come detto sopra e' pari a avere un modello di General Linear Regression con Degree a 9. 
Applico quindi a questo regressor con DEGREE 9 una $l_2$ norm regularization.
![[Pasted image 20240608012741.png]]
Con lambda = 0  non ho regularization quindi HO LO STESSO RISULTATO DI SOPRA. OVERFITTO PERFETTAMENTE il training data.
Con lambda ALTISSIMO, ho che ho un'approssimazione del LINEAR MODEL con Degree = 1, perche' solo uno/due coefficienti hanno valori diversi da 0, o comunque i valori imparati sono PICCOLISSIMI, molto piccoli. Quindi NON FITTA PER NIENTE BENE il dato.
NOTA che mettendo lambda ALTISSIMO sto LIMITANDO l'EFFECTIVE CAPACITY DEL MIO MODELLO, sto diminuendo la sua potenza, che puo' portarmi a overfittare
Con lambda giusto ho che allora TENGO perfettamente sotto controllo l'effective capacity.
![[Pasted image 20240608013715.png]]
NOTA infatti che il weight dello step precedente VIENE portato veramente verso l'origine ( viene moltiplicato per qualcosa compreso tra 0 e 1) prima di fare l'update. In base a quanto e' grande LAMBDA ho che $\theta^{(i)}$ viene piu' o meno cambiato. NOTA che una grande lambda porta sto weight letteralmente quasi a 0 (1-lr * altolambda).
NOTA CHE NEI PASSAGGI HO IL GRADIENTE DEL PENALTY TERM che da' come risultato esattamente $\theta^{(i)}$.
**NOTA CHE** la grandezza del DECAY **dipende anche dal LR** ASSURDO.
Quindi c'e' un problema se uso un lr schedule, quando cambio LR, se voglio mantenere lo stesso decay devo moltiplicare il weight_decay dell'inversa per cui ho diviso l'LR. ASSURDO.

QUESTA COSA DEL WEIGHT DECAY DERIVATA CON LA FORMULA SOPRA E' VERA SOLO CON PLAIN SGD.
QUINDI SE USO UN ALTRO OPTIMIZER, **LE EQUIVALENZE DI SOPRA NON VALGONO PIU'**.

**NOTA** gli optimizers (come ADAM, MOMENTUM) vengono usati per migliorare l'effective capacity del mio modello. Quindi quando ho underfitting questi mi possono portare a convergere a valori di parametri migliori che mi portano a fittare meglio i dati. REGULARIZATION e' usata per il problema opposto: il mio modello e' troppo fittante, voglio che generalizzi meglio.
Posso usare REGULARIZATION insieme agli OPTIMIZER come scritto sopra (TIPO e' possibile usarlo con Adam, momentum ecc..)

### EARLY STOPPING
![[Pasted image 20240608015045.png]]
POSSO FARE EARLY STOPPING anche sulla loss della validation, ma ha molto piu' senso fare early stopping sull'accuracy di cui sono interessato, vista sul validation.
LA LOSS INFATTI puo' essere BASSA ma magari l'accuracy NON CAMBIA, perche' come gia' detto la loss e' un proxy dell'accuracy.

### LABEL SMOOTHING
![[Pasted image 20240608020404.png]]
Sta cosa funziona perche' DI BASE io NON VOGLIO PUSHARE lo score della classe corretta a esser SEMPRE PIU' ALTO e NON voglio pushare gli altri scores a essere SEMPRE PIU' BASSI.
Rischio in sto modo di overfittare. Per ottenere 0 come loss infatti, ho che
1. la classe predictata deve avere uno score altissimo, in sto modo facendo l'exp ho che sia il numeratore che il denominatore sono + INFINITO (gli altri valori della sum al denominatore sono trascurabili visto che ho +infinito) e quindi il risultato e' 1, di conseguenza -log(1) = 0 e ho loss uguale a 0
2. le altre classi devono avere scores BASSISSIMI, in modo tale che io abbia una frazione in cui il numeratore e il denominatore sono UGUALI. Ho poi quindi -log(1)=0 con loss a 0.
MA cio' porta quindi come detto a PUSHARE lo score della corretta classe a valore ALTO e gli altri a valori SEMPRE PIU' BASSI. Questa cosa NON E' NECESSARIA, alla fine mi basta che lo score della classe giusta sia piu' alto degli altri.
Ho quindi che sta cosa potrebbe portare a overfitting
Quindi introduco LABEL SMOOTHING per cui le target labels sono smoothate, si e' meno convinti di quale e' la classe corretta.
![[Pasted image 20240608021036.png]]
### DROPOUT
![[Pasted image 20240608021435.png]]
NOTA CHE IL DROPOUT MASKA LE ATTIVAZIONI. Gli weights CI SONO ma il risultato prodotto dagli weights VIENE IGNORATO per computare l'output. Quindi l'output, che sono le attivazioni, viene IGNORATO per il calcolo dei successivi output. Il DROPOUT e' applicabile in ogni layer, addirittura nell'INPUT,tranne che nell'output layer ovviamente.

Ho sta situa
![[Pasted image 20240608021903.png]]
Praticamente il dropout rende robusta la predizione, perche' levando l'attivazione di una high level feature, tipo la presenza di occhiali,  la mia rete impara COMUNQUE a ritornare un alto face score. Quindi e' come se levassi dall'immagine in input gli occhiali di quella persona, e il modello imparasse LO STESSO a riconoscere quell'immagine come una faccia di una persona.
Questa cosa vale per tutte le high level features, come magari la presenza di occhi, o del naso, ecc...
Rende la rete meno confidente del fatto che HA BISOGNO di tutte quelle features per determinare lo score della faccia.
![[Pasted image 20240608022723.png]]
OPPURE POSSO VEDERE LA COSA SECONDO UN'ALTRA INTERPRETAZIONE:
![[Pasted image 20240608022817.png]]
Praticamente ho un ensemble
Sto limitando la mia effective capacity del MIO MODELLO perche' a ogni step sto trainando UNA RETE molto piu' piccola e con **una capacity molto minore rispetto all'originale (proprio a livello teorico - theoretical capacity)**.
DROPOUT A TEST TIME, come faccio?
HO stocasticity nelle predizioni a training time, ma a test time IO NON VOGLIO STOCASTICITY.
![[Pasted image 20240608023053.png]]
SOMMO tutte le possibili masks, NELLA PRATICA un centinaio. Faccio forwarding con queste diverse masks e faccio avg delle predizioni.
Solitamente posso avere CONFIDENCE INTERVAL sulle predizioni, COL DROPOUT CE L'HO A GRATIS lol (ho mean e variance). STA COSA E' COSTLY comunque eh, si fa solo se ho bisogno di confidence intervals.
Solitamente si fa un'approssimazione chiamata **WEIGHT SCALING**

![[Pasted image 20240608023657.png]]
Cioe' sta cosa funziona SOLTANTO NEL CASO LINEARE. Per questo e' un'approssimazione. Ovviamente nella mia rete ho non linearita', ma e' una buona approssimazione comunque. Solitamente si utilizza la tecnica di INVERTED DROPOUT, a training time, cosi' che a test time io NON DEVO FARE NIENTE, devo solo fare forward pass e that's it.

**NOTA** che IO non posso permettermi di fare forward direttamente senza applicare niente perche' cio' risulterebbe IN UN CAMBIO della distribuzione dei valori dei neuroni, che e' completamente diverso da quella a train time (per il batch norm layer da' problemi, quando ho non linearita' mi genera risultati DIVERSI e' un macello).

HA SENSO UTILIZZARE IL DROPOUT QUANDO HO TANTI PARAMETRI.
tipo in alexnet c'e' dropout negli ultimi layers dei FC.
Solitamente dropout non e' utilizzato nei CONV LAYERS, perche' gia' hanno inductive bias al loro interno, quindi sono gia' molto constrained.

**NOTA** che batchnorm introduce STOCHASTICITY perche' a seconda della MINIBATCH in cui si e' ho un risultato diverso. Quando a test time fisso le avgs e le variances, sto levando quella stocasticita' dalla rete, PROPRIO COME HO FATTO COL DROPOUT.

Batch norm e' considerato REGULARIZATION method perche' segue il pattern di
1. rendere il training artificialmente piu' difficile, causa stocasticity introdotta dalla batch nella quale si e'
2. a test time questa variabilita'/stocasticita' viene rimossa
### DATA AGUMENTATION
E' la migliore regularization tecnique. 
Se ho una network che overfitta ALLORA la cosa migliore  che posso fare e' trovare nuove immagini.
IMMAGINA INFATTI TRAINARE IL MODELLO SULLA DISTRIBUZIONE REALE DEI DATI, Li' NON avrei overfitting no? perche' il train set spiegherebbe perfettamente il test set, NON ci sarebbe uno SHIFT come l'ho osservato prima. Quindi ottimizzare per il train set SIGNIFICA ANCHE GENERALIZZARE.
Di conseguenza **PIU' DATA RIDUCE L'OVERFITTING**.
NON POTENDO prendere NUOVE IMMAGINI, posso trasformare quelle che ho.
**NOTA** le trasformazioni DEVONO PRESERVARE LA LABEL
UN esempio in cui cio' non e' vero e':
![[Pasted image 20240608025135.png]]
GUARDA SEMPRE L'AUGUMENTED DATA che hai generato.
POSSO FARE HORIZONTAL FLIPPING DI BASE QUASI SEMPRE
SPESSO VOGLIO RENDERE UNA RETE **ROBUSTA RIGUARDO A SCALE CHANGES** e cio' e' possibile augumentando in una certa maniera:

![[Pasted image 20240608025546.png]]
Di base si ingrandisce l'immagine  e poi si prende una patch della grandezza usata per trainare come nuova istanza del dataset.
Inoltre sta cosa di S che viene presa randomicamente mi permette di ingrandire l'immagine a differenti risoluzioni.
SI PUO' ANCHE QUI RISCHIARE DI NON AVERE LABEL PRESERVING TRANSFORMATION, tipo nel caso della patch gialla nell'esempio di sopra in cui cho la zampetta che non me dice niente, non preserva la label di uccello. DAMN
-8.57
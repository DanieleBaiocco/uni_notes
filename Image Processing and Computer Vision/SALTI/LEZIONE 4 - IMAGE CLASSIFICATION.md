Da un'immagine categorizzo l'immagine in *una* categoria presa da **un set di categorie**.
Questa cosa di poter scegliere solo da un set predefinito di categorie e' limitante.
La cosa difficile e' che certe immagini hanno una categoria all'interno ma ci sono problematiche legate all'immagine che impediscono una corretta identificazione.
![[difficolta.png]]
Machine Learning/DL riesce a gestire tutte queste variazioni dell'immagine. Ancora ci sono problemi con *cambiamenti di luminosita'* in ML/DL, per gli altri tipi di variazione funziona molto bene invece. Potrebbe avere anche problemi con immagini molto strane, come l'ultima foto, se questa non e' presente nel dataset allora non riuscira' a capire che quello e' un gatto (il mondo e' pieno di foto strane!).

C'e' un assunzione sul *training* in ML.
Io ho come assunzione che i samples del training set e del test set sono *indipendenti* e *distribuiti in maniera identica* dalla **STESSA** distribuzione $p_{\text{data}}(x,y)$ non conosciuta.
E' importante, quando a livello aziendale si crea un dataset, che questa assunzione sia rispettata.


Datasets famosi:
1. MNIST, non utilizzabile come benchmark perche' ormai e' un problema risolto 
2. CIFAR 10, e' un MNIST un po' piu' complicato, che utilizza vere foto scattate. Ad ogni modo anche questo viene usato solo come sperimentazione.
3. CIFAR100, ha 100 classi. Ci sono 500 immagini per ogni classe. 
4. ImageNet/**ImageNet21k**, (ha 21mila classi) qui c'e' ogni ricercatore di CV. Questa e' una bestia molto grande. Ha 14 milioni di immagini RGB, con risoluzioni **diverse**, come lo erano in  internet (questo e' un problema), con grandezza media 400x350, che e' una grande immagine MA NON e' la risoluzione del mio telefono, neanche lontanamente. Questo dataset ha una struttura gerarchica: ci sono immagini di mammiferi, dentro le quali ci sono immagini di carnivori, dentro i quali ci sono immagini di cani, dentro i quali ci sono immagini di cani domestici, dentro i quali ci sono immagini di huskies.
5. ILSVRC: e' un subset di ImageNet. In verita' e' **ImageNet1k**, in quanto ha 1000 classi, con 1300 immagini per classi. Visto che ho 1000 classi (o 21mila nel caso di prima), la performance e' calcolata in questo modo: ho una corretta classificazione se la label (la classe corretta) di un'immagine compare nelle **prime 5 predizioni del modello**. Nota che se ottimizzo il mio problema secondo questa metrica (cosa che NON si fa no? si ottimizza in base a una loss function, ma magari il Prof. intende che faccio validation con la metrica, o la utilizzo per fare grid search), e ho che il modello performa bene, NON posso dire di aver risolto ImageNet. Io penso che il mio modello sia in grado di classificare in modo corretto una particolare specie di tartaruga, MA invece, se viene ottimizzato secondo questa metrica, SARA' SOLO IN GRADO DI riconoscere una **generica tartaruga**, in quanto basta che assegna a ogni specie di tartaruga un valore di predizione abbastanza alto da rientrare nei primi 5 e ha fatto.

LIMITAZIONI DI UN LINEAR CLASSIFIER
Il linear classifier fa *template matching*, di base. Perche' per ogni W ho n righe con n numero di classi e m colonne con m numero di pixels flattenati.
Per ogni immagine io la flatteno e poi faccio il dot product con ogni riga di W per avere come outptut uno score per ogni classe. Ogni riga di W ha quindi dei pesi riferiti a ogni pixel che sono imparati. La limitazione e' che vengono imparati dei pesi per riconoscere magari una particolare tipo di macchina vista in una certa prospettiva e posizione, mentre io voglio che venga matchata qualsiasi tipo di macchina. Pratiamente il linear classifier fa template matching, e questo FUNZIONA solo se si parla di un particolare template, NON per tutti gli oggetti di una determinata classe.

riscritto:
  
Il linear classifier si basa fondamentalmente sul _template matching_. In pratica, per ogni immagine, viene effettuata la conversione dei pixel in un vettore e quindi calcolato il prodotto scalare tra questo vettore e ciascuna riga della matrice dei pesi 𝑊, generando uno score per ogni classe. Ogni riga di 𝑊 contiene pesi associati a ciascun pixel, appresi durante il processo di addestramento del modello. Tuttavia, questa metodologia ha una limitazione: i pesi appresi possono essere specifici per riconoscere un particolare tipo di oggetto solo nelle condizioni in cui è stato addestrato il modello, come una macchina vista da una certa prospettiva e posizione. Questo significa che il linear classifier tende a fare matching solo con un determinato tipo di template, e non è efficace nel riconoscere tutti gli oggetti di una determinata classe in maniera generale.
![[linrst.png]]

# LOSS e ACCURACY
Io NON posso utilizzare una 0-1 Loss per trainare una rete neurale : sarebbe una loss in cui di base conto il numero di classificazioni SBAGLIATE DEL MIO MODELLO, e MINIMIZZO su questo numero di  classificazioni errate. Questa loss e' praticamente l'ACCURACY. Alla fine dei conti voglio un modello che abbia il numero di predizioni sbagliate MINIMO. Il fatto e' che esistono INFINITE DECISION BOUNDARIES per lo stesso NUMERO di questa loss: posso avere che il numero di matches con le true classes sia LO STESSO per INFINITE DECISION BOUNDARIES DIVERSE. Non e' una loss che mi dice la direzione in cui andare, magari seguendone il suo gradiente
![[01loss.png]]
Come e' visibile dalla figura, per tutte quelle decision boundaries HO lo stesso valore di 01Loss (faccio lo stesso numero di errori), nonostante il fatto che IN VERITA' queste decision boundaries sono molto diverse tra loro  e servirebbe una loss che sia piu' informativa su quale di queste e' migliore/peggiore, in modo da capire la direzione **per lo spazio dei parametri** del modello in cui **andare**.
In un problema di ottimizzazione continuo, come in questo caso, la 01Loss NON e' utilizzabile.
![[01loss2.png]]
Nota che se continuo verso giu', abbassando la decision boundary, a una certa ho che la 01Loss passa da 4 predizioni sbagliate a solo 2 predizioni sbagliate. MA non c'e' NULLA che mi dice che questa e' la direzione **giusta**.
Questa loss puo' essere usata in **combinatorial optimization problem**, decidendo per ogni esempio dove dovrebbe stare, dicendo che devo imparare una linea in cui ogni mio esempio positivo e' minore di quella linea e ogni mio esempio negativo sia maggiore. Quindi aggiungo un constraint alla linea per ogni elemento/sample del mio training set. Il problema e' che ci mettono tantissimo a convergere i **combinatorial optimization problem** con l'aumento dei samples.

VOGLIO QUINDI QUALCOSA, un **proxy**, CHE SOSTITUISCA QUESTA 01Loss, qualcosa che **indirettamente alla fine minimizza** la 01Loss.
Quello che voglio e' qualcosa da **minimizzare** che come effetto collaterale abbia il fatto che anche la 01Loss venga minimizzata (la oroxy-loss viene minimizzata nella SPERANZA CHE CIO' ACCADA).
Nota pero' che posso avere un decremento della proxy-loss SENZA un cambio effettivo della 01Loss. Se infatti questa proxy-loss e' una funzione continua che e' in grado di dire nell' ESEMPIO DI SOPRA-SOPRA che magari la decision boundary piu' in basso e' meglio di quella piu' in alto ANCHE SE NON C'E' UN VERO CAMBIAMENTO DELLA 01Loss (dell'accuracy) (di base questa rimane uguale se utilizzo la decision boundary piu' in basso). Alla lunga avro' un cambiamento della 01Loss (dell'accuracy) perche' se continuo a scendere, seguendo il gradiente della proxy-loss, allora di conseguenza avro' anche una diminuzione del valore della 01Loss.

## Idee di loss nel caso di problema di image classification 
1. RMSE: calcola la differenza tra il vettore predictato e il true vector. Pero' non e' molto efficace, perche' sto forzando  gli scores outputtati dal modello di essere 1 per il true label e 0 per gli altri. Io voglio solo che lo score della CLASSE GIUSTA sia MAGGIORE rispetto agli altri scores, FINE. NON voglio che GLI ALTRI SCORES siano pari a 0. RMSE sta imponendo un constraint di cui il modello non ha bisogno: sta forzando il modello a outputtare 0 NELLE ALTRE CLASSI, quando a me sta cosa non interessa. Mi interessa solo che la giusta classe sia la maggiore
2. Si ragiona in termini di probabilita' di distribuzione, volendo trovare il miglior set di parametri che Massimizza la Likelihood del mio training set. Quindi invece di outputtare gli scores, questi vengono resi una *probabilita' di distribuzione* utilizzando la **softmax function**, formata da
	1. $exp$(.): che rende tutti gli scores positivi. Inoltre ha come side effect che **amplifica** la differenza tra gli scores
	![[expsideeffect.png]]
		dalla figura si puo' vedere come la differenza tra lo score di plane e di car sia piccola, dopo l'$exp$ diventa molto grande.
	2. $norm$(.): per ogni risultato, lo divido per la somma degli altri risultati, in sto modo ottengo una *probability distribution* (un set di numeri che sono positivi e sommano a 1). Che poi questa e' una Probability Mass Function perche' sono nel campo discreto 
	Dopo si calcola la *negative cross entropy loss*.
A livello implementativo c'e' una chicca, per cui per migliorare la stabilita' numerica durante le operazioni si sfrutta una proprieta' della **softmax function**, che e' la seguente:
![[softmaxprop.png]]
Ovvero che se sommo un $c$ allo score di cui voglio calcolare la **softmax**, allora il risultato e' LO STESSO di calcolare la **softmax** su quello score da solo.
Devo ovviamente sommare *c* a tutti gli altri scores.
In sto modo CALCOLO a livello implementativo ![[stablesoftmax.png]]
In sto modo il massimo degli scores sara' 0 e tutti gli altri saranno **negativi**. In sto modo, facendo poi l'**exp** ho che l'exp di un numero negativo e' tra 0 e 1. Quindi tutti i risultati saranno tra 0 e 1 dopo l'exp. Questo mi garantisce stabilita' numerica (ho numeri gestibili nel mio PC) e mi evita di avere numeri **troppo grossi** dopo l'exp da poter essere rappresentati nel PC.


A sto punto si calcola la **NEGATIVE CROSS ENTROPY LOSS**. Questa e' derivata formulando il problema di ottimizzazione come un problema di **maximum likelihood estimation**.
![[mle.png]]
Nota che il secondo uguale e' dato dal fatto che le training instances si suppongono essere **indipendenti tra loro**, senza questa assunzione non posso sviluppare una joint probability in quel modo ma lo sviluppo e' mooooolto piu' lungo.
Ragiono sempre in somme di *logaritmi* perche' in sto modo sono molto piu' stabile a livello numerico. Nota che *log* e' un operatore monotonico quindi **NON cambia il massimo** (un massimo prima rimane un massimo adesso). Nei problemi di massimizzazione/minimizzazione non cambia il risultato e rende tutto molto piu' stabile.
L'ultima equazione e' letteralmente la **negative crossentropy loss**. Si prende per ogni sample la probabilita' outputtata dal modello legata alla true label.

Nota che implicitamente se massimizzo la probabilita' della classe corretta ad ogni modo sto indirettamente dicendo di minimizzare la probabilita' nelle altre classi, perche' la probabilita' crea competizione (ovviamente se da una parte e' alta NON puo' essere alta anche dalle altre parti, quindi come side effect si abbassa per le altre classi).

**IMPORTANTE**. E' importante capire quanto il valore della loss e' buono. E' importante capire quando ho una buona loss e quando ho una brutta loss, calcolandola con degli esempi.
Ad esempio, se ho questa PMF con true label = **2**, 
![[losscalculus.png]]
allora la negative crossentropy loss sara': $-log(0.9) = 0.1$. Quindi ho come riferimento che una *buona loss* deve essere simile/uguale a 0.1
Nel caso in cui invece la true label fosse stata = **1**, allora avrei avuto $-log(0.09) = 2.4$. In sto modo se il mio modello mi outputta una loss simile a 2.4 capiro' che sta performando malissimo.
Voglio anche vedere **quale sarebbe la loss se usassi un classifier RANDOMICO**. Quindi praticamente da' a ogni score  il valore **1/n** con **n** numero di classi. Quindi nel caso in cui avessi 10 classi, allora avrei per ogni classe lo score 0.1. Di conseguenza il valore della loss sara' $-log(0.1)$ che e' simile a quello calcolato prima di 0.09, ovvero 2.4. Se quindi ho una loss del genere nel mio modello **so** che sto performando **COME UN RANDOM CLASSIFIER** , il che e' pessimo e mi dice che magari qualcosa non sta andando bene. E' uno dei pochi modi per debuggare un modello.
Nota che se sviluppo questa Negative Cross-Entropy ho
![[losssviluppo.png]]
Praticamente il log di una frazione puo' essere scomposto in una differenza. Ho poi che il $log(exp(.))$ si annulla. Di conseguenza finisco con l'espressione sulla destra. Questo mi mostra che la **cross-entropy** minimizza il - dello score della classe giusta + l'altro termine, che e' chiamato il **logsumexp**.
Questo e' come internamente **pytorch** ottimizza. 
Ora mostro che questa loss e' una buona proxy per l'ACCURACY (per il 01Loss).
Nota infatti che quando faccio il *logsumexp*, ho che dopo l'exp lo score piu' alto SARA' ANCORA PIU' ALTO RISPETTO AGLI ALTRI scores. Gli altri scores saranno piccolini rispetto a loro e se uso solo il piu' alto invece di sommare il piu' alto con gli altri NON HO UNA GRANDE DIFFERENZA. Quindi posso APPROSSIMARE il *logsumexp* come il $log(exp(\text{score piu' grande}))$. Cio' si semplifica e rimane solo *score piu' grande*. Quindi posso approssimare la crossentropy in questo modo:
![[logsumexpapprox.png]]
In sto modo si mostra che 
1. se lo score della classe corretta e' il piu' grande allora ho 0 come differenza.
2. Se invece NON e' cosi', allora minimizzando questa  quantita' si puo' 
	1. o aumentare lo score della classe corretta
	2. o diminuire lo score della classe che lo ha massimo
Quindi questa loss penalizza **the most active incorrect prediction**.  Questo mi mostra che la crossentropy e' un buon proxy per l'accuracy. Perche' abbassa la predizione non corretta, alza quella corretta, e' proprio cio' che poi porta a avere una accuracy (01Loss) che migliora (faro' sempre meno errori se l'ottimizzazione **seguita** dal modello e' la seguente).
Quindi minimizzare la crossentropy e' un proxy dell'accuracy.

# GRADIENT DESCENT
Vabe' lo sai dai.
![[gd.png]]

Io ho che le losses si decompongono come somma delle singole loss di ogni training instance. Lo stesso per il gradiente che sara' la somma dei singoli gradienti calcolati.
![[loss.png]]
Metti che ho come dataset ImageNet1k, ogni volta devo calcolare un gradiente come la somma di 1.3 milioni gradienti, uno per ogni training instance.
Ci metto quindi **tantissimo** e tutto questo sforzo per fare poi alla fine uno **STEP MOLTO PICCOLO** dato dal valore del **learning rate**.

Quello che posso pero' dire e' che questo calcolato e' il **true gradient**. In verita' NON E' NEANCHE VERO perche' il training set e' un proxy della popolazione reale, del mondo reale. Quindi **NEANCHE E' IL TRUE GRADIENT**.

## SGD
Si usa quindi lo Stocastic Gradient Descent.
Quindi calcolo il gradiente per ogni immagine nel training set (per e epoche). Quindi faccio *n* updates dei parametri per ogni epoche (con n numero di esempi nel training set).
Qua uso una sola immagine per aggiornare i parametri, questa cosa e' molto **noisy**.
![[sgd.png]]
# SGD with Minibatches
Una via di mezzo che prende in considerazione minibatches.
La cosa importante e' scegliere la **B** giusta, che mi dice **QUANTE VOLTE HO AGGIORNATO I MIEI PARAMETRI DURANTE L'EPOCA**.
![[SGDminibatch.png]]
Se ho B=1 ho SGD, se ho B=n allora ho GD classico.
**NOTA** e' importante che ci sia uno *shuffle* all'inizio di ogni epoca perche' voglio che ci sia **indipendenza delle istanze** a ogni epoca. Anche se le istanze del training sono le stesse, voglio che le minibatches create siano diverse a ogni epoca rispetto alle precedenti.

**NOTA** un'altra cosa: reporta cio' che succede **dopo un certo numero di updates dei parametri**, non al termine di un'epoca. Perche' importa solo cio' che accade dopo un po' che ho cambiato i parametri.
## Tradeoffs nelle minibatches
Se uso una minibatch size grande riduco il noise. Pero' se ne utilizzo una piccolina ci potrebbe essere anche come lato positivo (in una maniera un po' controintuitiva) che non si overfitti quindi il modello performa meglio alla fine. Serve ovviamente piu' tempo per convergere con una minibatch piu' piccina, pero' potrebbe essere meglio.

# Risultati del modello lineare
Ho come risultato del modello lineare, dopo il training usando *negative cross entropy loss*, i seguenti templates che sono stati imparati dal modello:
![[linearmodelresults.png]]
Ricordo infatti che il modello lineare non fa nient'altro che template matching e facendo un *reshape* dei pesi legati a ogni classe posso vedere il template che ha imparato.
Nota che nel caso del cavallo, ha imparato due cavalli, due punti di vista (sia col muso rivolto verso dx che verso sx). Ovviamente esistono TANTISSIMI ALTRI punti di vista (anche nel training set) ma il modello ha imparato quelli piu' presenti evidentemente.
Guarda anche la macchina, ha imparato solo macchine viste da **davanti**. Inoltre guarda il **background**, lo sfondo. Ha imparato che una *barca* sta solo in un background blu (il mare), MA cio' non e' detto: potrei avere la foto di una barca magari scattata dentro un magazzino di produzione e li' sto modello NON riuscirebbe a classificarla come tale.

**DEVO IMPARARE DELLE IMAGE REPRESENTATIONS che non sono legate ai PIXELS perche' qui ho un sacco di errori** e i pixels sono MISLEADING.
Io non voglio rendere il modello piu' complicato (voglio comunuque usare un modello lineare) ma voglio avere una rappresentazione migliore delle immagini rispetto a quella dei pixels.
Esempio:
in $x_1$, $x_2$ non c'e' modo di separare con una linea (quindi usando comunque un modello lineare).
Se pero' utilizzo un altro feature space in cui prendo $p$ che indica la distanza dal centro di ogni punto e 
$\theta$ che indica l'angolo che ogni punto ha con l'asse delle x, allora ho che ho qualcosa di separabile attraverso una linea.
![[featurespace.png]]
Avevo una **nonlinear decision function** nel feature space iniziale. Questa cosa non la ho piu' nell'altro feature space. Quindi ho **migliorato la rappresentazione dei dati**, SENZA AVER CAMBIATO IL MODELLO (voglio sempre usare un modello lineare).
![[nonlinear.png]]
Tipo in NLP si e' utilizzata la rappresentazione della BoW (Bag of Words) in cui si conta la frequenza di ogni parola nel testo in esame. Quindi in base a quali parole sono presenti e alla loro frequenza poi decido se classificare quel testo come un commento **positivo** o **negativo** per esempio.Tutto in base a cosa vedo, senza prendere in considerazione l'ordine delle parole, il significato logico del discorso diciamo.

Questo concetto e' stato traslato in CV, creando una grid/rettangolini. Per ognuno di questi rettangolini, NON AVENDO UN VOCABOLARIO COME INVECE AVEVO IN NLP, devo matchare questo rettangolo con delle **CODEWORDS**, che sono delle patches prefissate. La codeword che piu' assomiglia a quel rettangolino viene scelta per lui e viene messo un +1 alla frequenza. Alla fine ho un array di frequenze di queste patches legato all'immagine
![[BoWCV.png]]
Cio' che si e' fatto a livello implementativo e' stato dalle grids/rettangolini estrarre dei *sift descriptors* che erano poi da matchare con i sift descriptors dei codewords (questi erano stati samplati dalle classi, sono stati samplati 1 milione di sift descriptors da usare come codewords). Poi si applicava SVM a queste rappresentazioni dei dati diverse.

Il problema comunque e' che prima dell'avvento di DL, gli umani hanno dovuto creare **buone features**, buone rappresentazioni del dato, come nel caso di qui sopra con BoW applicato alle immagini con i SIFT descriptors. Grosso lavoro. La qualita' del modello POI ERA LEGATA DALLA RAPPRESENTAZIONE DEL DATO. Non c'era grosso margine di miglioramento. La rappresentazione del dato rimaneva STATICA, NON cambiava.
In DL ho che il modello IMPARA una rappresentazione del dato che CAMBIA nel corso del training che poi permette di portare l'immagine in uno spazio delle features in cui e' linearmente separabile (alla fine ho un linear model per separare non pensare, pero' solo alla fine).
E quando ottimizzo i parametri NON SOLO MIGLIORO IL CLASSIFIER MA ANCHE LA RAPPRESENTAZIONE DEL DATO (che prima era vista come STATICA).
![[dlrep.png]]

# Neural Network
Di base ho questa cosa qua:
![[NNN.png]]
Ho bisogno di un'activation function perche' altrimenti avrei SOLAMENTE un sistema lineare alla fine della fiera e nulla cambierebbe:
![[linear.png]]
		Avrei quindi un linear classifier.
		Ci serve l'activation per imparare una non-linear representation

Per molto tempo la gente ha usato la *sigmoid* come activation funcion. Ma questa HA UN GRADIENTE MOLTO BASSO quando l'input ha un valore assoluto alto (sia in positivo che in negativo). Con un gradiente cosi' basso servono molti steps per migliorare i parametri:
![[sigmoidgrad.png]]

Nel caso della ReLU invece ho che il gradiente e' 1 nel caso in cui l'input sia positivo quindi imparo. Quando l'input e' negativo pero' ho che il gradiente e' 0, di conseguenza NON IMPARO NULLA. Se tutti gli input sono tutti minori di 0 allora ho che NON IMPARO NULLA.
![[relugrad.png]]
Si sono inventati il LeakyReLU
![[leakyReLU.png]]
per evitare sto problema che input negativi risultavano in gradiente pari a 0, in sto modo ho comunuque un gradiente, anche se piccino.

Di base se vedo da vicino cosa fa l'applicazione di un sistema lineare, semplicemente **"sposta", "allunga", "accorcia" o "ruota"** i punti in un modo specifico e prevedibile, mantenendo le linee rette e la proporzionalità delle distanze tra i punti. Mantenendo le distanze e le proporzionalita' tra punti, DOPO UNA TRASFORMAZIONE LINEARE, se prima i miei dati NON erano separabili con una linea (linearly separable) beh **non** lo saranno neanche **dopo**. Cio' che succede e' che riscrivo le coordinate delle mie features $x_1, x_2$ (a sx nella foto) in quelle riferite al nuovo sistema di basi $h_1$ e $h_2$, calcolato usando i pesi della matrice W:
![[lineartransformation'.png]]
Nota che si ha un *mapping affino* per la precisione (non lineare perche' si ha comunque un bias term che non obbliga la trasformazione lineare a avvenire rispetto al centro).
A cosa serve quindi fare questo linear mapping? Di base serve **in vista dell'applicazione della ReLU**, infatti con sto linear mapping ho che i punti vengono spostati e alcuni cambiano di segno (se guardo le coordinate, alcuni che prima magari erano nel terzo quadrante, poi saranno dopo la trasformazione nel primo quadrante, ovvero quello positivo).

Di base la ReLu infatti fa la seguente cosa: se il punto (la training instance dopo che le e' stato applicato un mapping affino) e' presente nel quadrante POSITIVO allora lo lascia cosi come sta. Se invece ha una coordinata negativa, questa viene portata a 0 (cio' si fa per tutte le sue coordinate). In sto modo si agglomerano  tutti i punti all'interno del quadrante positivo. ReLU fa quindi un folding dello spazio.
Questo e' cio' che accade di base, se combino mapping affino(un buon mapping affino IMPARATO DALLA RETE) e ReLU:
![[ReLUapplication.png]]
Ho che adesso ho **linear separability**.
NOTA che un linear classifier in $h_1,h_2$ dopo la ReLU **EQUIVALE A una NON LINEAR DECISION BOUNDARY in $x_1, x_2$**.
![[linearseparable.png]]

# Edge detection verticale con Fully Connected Layer
Voglio implementare un edge detection verticale utilizzando la W del linear layer. Voglio quindi che dato un input aka un'immagine, la W sia in grado di fare edge detection su edges VERTICALI.
Un modo per implementarlo e' fare la seguente cosa:
![[tempsnip.png]]
	Posso implementarlo come una matrice W che prende in input un'immagine flattenata di dimensione 9  (ho immagini in input di dimensioni 3x3) e computa come risultato un array di dimensione 6, perche' per ogni riga dell'immagine in input da' come risultato solo due numeri. Praticamente fa la differenza tra i pixels consecutivi nella stessa riga  perche' cio' indica la presenza di un edge verticale. Nota che la matrice ha ovviamente un sacco di ZERI perche' il primo risultato e' solamente b-a (non riguarda gli altri pixels).
Se ho una immagine in input di grandezza H x W, ho come risultato una matrice W con $(H \text{x} W) \text{x} (H \text{x} (W - 1))$ parametri (perche' considero solo paragoni a due a due sulla stessa riga, da questo viene il meno 1) che e' approssimabile a $H^2 W^2$. Tutti sti parametri devo tenermeli in memoria eh...
Con un'immagine di 224 x 224 devo tenere in memoria e imparare $2.5 \text{ x } 10^9$ parametri.

Inoltre devo fare per ogni riga  **una moltiplicazione e una somma** (dot product tra riga e immagine flattenata). Cio' significa che ho $2 H^2 W^2$ operazioni **in virgola mobile** da fare per ogni immagine per calcolare poi l'output. 
In un'immagine di 224 x 224 mi da $5 \text{ x }10^9$ operazioni (TANTISSIMO).
Nota che la matrice ha la stessa struttura, lo stesso set di parametri per ogni riga. 
**NOTA** che qua il fatto di avere una matrice W **fully connected** come lo e' quella di sopra (infatti ho che come gia' visto **in ogni riga**, **a ogni pixel legato a quella riga** e' legato un peso quindi ho un contributo richiesto DA TUTTI (che e' 0 per tutti tranne per due pixels per cui e' diverso da zero che sono quelli presi in esame)) e' **negativo e limitante in questo caso**.
Questo e' un problema **locale**, non deve riguardare tutti gli altri pixels, tutti gli altri inputs.

# CONVOLUZIONI
A differenza dei linear layers 
1. Ho che la convoluzione preserva la struttura spaziale dell'immagine, infatti qua NON FLATTENO PIU' L'IMMAGINE
2. Ho che una convoluzione processa solo un piccolo set di pixels vicini in ogni zona in esame. In altre parole ogni output della convoluzione e' legato a pochi inputs (si chiama **local receptive field difatti**).
3. I parametri della convoluzione sono gli stessi SEMPRE E COMUNQUE in ogni patch in cui vengono applicati
Di base adesso se ho una immagine $H \text{ x }W$, basta che imparo **2 PARAMETRI** (ti rendi conto).
Il numero di operazioni che faccio sono molte meno (non ho ben capito perche' faccio $3 H W$, forse perche' faccio per ogni patch due moltiplicazioni e una somma).

**NOTA UNA COSA IMPORTANTISSIMA**: le convoluzioni NON hanno maggiore potere espressivo rispetto agli **operatori lineari W**, in quanto ogni convoluzione puo' essere traslata in un grande operatore lineare legato a essa. Il motivo per cui utilizzo le convoluzioni e' che implicano meno calcoli e meno parametri da salvare in memoria.
In questa foto viene mostrata proprio questa cosa:
![[convnotmoreexpressive.png]]
Nota addirittura che IN VERITA' un fullyconnected linear layer e' addirittura piu' potente rispetto a una convoluzione in quanto HO MOLTE PIU' DEGREES OF FREEDOM: i parametri sono molti di piu' (ad esempio in foto a posto degli 0 il modello potrebbe imparare qualche altro valore diverso da 0 come peso legato a quel pixel).
La convoluzione introduce **un constraint** sul linear operator.
Tra l'altro il fullyconnected linear layer e' cosi potente che introducendo la correlazione io evito che il modello mi impari delle **spurious correlations** tra le immagini del training. Voglio limitare questa cosa. Infatti le possibilita' di overfittare aumentano.

Ho come proprieta' della convolution inoltre l'**EQUIVARIANCE**, OVVERO LA PROPRIETA' PER CUI
1. se ho un'immagine e faccio prima una TRASLAZIONE dell'immagine e poi applico la convoluzione
2. ho lo stesso risultato rispetto a fare la convoluzione e poi TRASLARE l'output della convoluzione
![[equivariance.png]]
Graficamente e' questo:
![[equivarianceexample.png]]
Questa proprieta' e' buona per il training perche' significa che posso **imparare** la mia convoluzione per non so *detectare degli edges* o qualsiasi altra feature, INDIPENDENTEMENTE da dove oggetti appaiono nelle training images.
Per esempio NON devo osservare un gatto IN OGNI POSIZIONE nella foto per imparare una convoluzione che identifichi il gatto.
Non ho equivariance con **rotazione e scale** purtroppo. A volte si fa data agumentation ruotando le immagini o scalandole per cosi rendere il modello equivariance anche riguardo a questo tipo di immagini scalate e ruotate.

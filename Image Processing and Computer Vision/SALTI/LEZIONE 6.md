Il notebook deve essere piu' conciso  possibile per il SECONDO ASSIGNMENT RICORDALO.
QUANDO FAI HYPERPARAM TUNING SULLA SECONDA PARTE DELL'ASSIGNMENT VUOLE CHE UTILIZZI I TRAINING HYPERPARAMS (quindi solo il LR, la bacth_size, ecc...)

# ZFNet/Claifai: una AlexNet migliore
Ha usato dei kernels piu' piccini con strides piu' piccine. AlexNet aveva un primo kernel 11x11 con stride a 4. Qui si hanno: un kernel 7x7 con stride a 2 , a seguire un 5x5 con stride 2. Alla fine ottengo la stessa *spatial resolution* di AlexNet, ma con due convlayers invece di uno.
Hanno visto cosa era stato imparato nel primo pacco di kernels perche' questi kernels hanno individualmente dimensione 3 quindi possono essere visualizzati come una rgb image. Questi sono tanti quante le feature maps prodotte in output che nel caso di ZFNet c'e' scritto essere 96. E' lo stesso numero di feature maps in AlexNet infatti li posso comparare:
![[kernelsrgb.png]]
In quello a sx ci sono quelli di AlexNet, NOTA CHE CI SONOD DEI KERNELS CHE NON FANNO NIENTE DI IMPORATNTE. Sono chiamati *dead neurons*.
Hanno visto che se si va a una piu' gentile riduzione della spatial dimension nello stem layer (come si e' fatto in ZFNet) allora non ho dead neurons (immagine a dx).

# VGG
Volevano vedere se aumentare i layers avrebbe migliorato la situa. Inoltre volevano ripetere operazioni **regolari** a ogni layer. Hanno voluto provare a usare soltanto:
1. 3x3 conv con stride 1 e padding 1, in sto modo non riduco mai la spatial resolution della mia attivazione
2. 2x2 maxpooling per ridurre la spatial resolution
3. Quando faccio downsample con maxpooling allora **raddoppio le feature maps**
Questo significa che il numero di **flops** rimane costante : perche' raddoppio le feature maps e dimezzo la spatial resolution
La regola del raddoppiare le feature maps non vale per l'ultimo layer in cui rimango a 512. 
La rete piu' lunga che son riusciti a trainare e' di 11 layers (si conta soltanto quelli trainabili)
![[Pasted image 20240528164618.png]]
![[Pasted image 20240528164756.png]]
Ci sono 8 convoluzioni e (non e' mostrato ma) 4 fullyconnected layers. BatchNorm ancora non era stato inventato.
Dopo aver trainato la A, hanno fatto architetture piu' lunghe in cui si aggiungevano convlayers per rendere piu' deep la rete ma usando i pesi imparati per gli altri layers che erano gli stessi.
![[Pasted image 20240528164948.png]]
Alla fine le ultime due hanno preso dei nomi
![[Pasted image 20240528165016.png]]
VGG ha introdotto il concetto di stages: ovvero di insiemi di layers che processano l'immagine alla stessa *spatial resolution*. Un esempio di stage e' nel VGG-19 i due conv3 all'inizio della rete da 64 feature maps in output (con il maxpool finale)
Uno stage con dei piccoli kernels come lo e' in questo caso e' molto buono per diversi motivi:
1. Prima si tendeva a usare come abbiamo visto un grande kernel okey? Adesso ho uno stage in cui si staccano kernels piu' piccini.
2. Il receptive field di staccare piccoli kernels e' lo stesso rispetto a considerare un grande kernel.
3. Il numero di flops e' minore se stacco due kernels, anche il numero di parametri
4. Ci sono due ReLU in sto modo, invece di una. Tanta roba, ho ancora piu' nonlinearita'.
![[Pasted image 20240528165755.png]]
Il prezzo sta che devo computare un numero di attivazioni maggiore (di base il doppio delle feature maps). Quindi a livello di memoria magari soffro.

## VGG 16 SUMMARY
Allora prima di tutto e' infinita sta rete
Ha un numero di parametri gigantesco, di cui la maggior parte e' nei FULLYCONNECTED LAYERS, come nel caso di AlexNet
Ha un numero di FLOPS abnorme, questo e' dovuto dal fatto che **NON E' PRESENTE UNO STEM LAYER CHE RIDUCA LA SPATIAL DIMENSION**. Quindi serve molta computazione per i primi layers.
Sta VGG16 e' usata per catturare lo **stile delle immagini** proprio perche' non ha una riduzione della spatial dimension, quindi si concentra su piccole differenze tra pixels che con stem layer avrei perso.
![[Pasted image 20240528170150.png]]
# Inception 1 (GoogLeNet)
Volevano fare una rete efficiente.
C'e' una struttura:
1. C'e' uno stem layer per evitare FLOPS ALTI all'inizio
2. Ci sono un sacco di inception modules
3. NON CI SONO PIU' I FULLY CONNECTED LAYERS alla fine, perche' NON VOLEVANO TUTTI QUEI PARAMETRI. C'e' un GlobalAVGPooling Classifier.
Ho che viene fatto uno stem layer di questo tipo 
![[Pasted image 20240528171209.png]]
In cui ho 7x7 con stride 2 (c'e' da andarci piano all'inizio con lo stem come abbiamo gia' visto), poi max pool poi altre conv leggere poi maxpool. Arrivo a una spatial resolution dopo lo stem layer che e' di **28x28** (PICCOLISSIMA). Con 192 feature maps.

Nota la differenza tra questa e VGG. VGG ci mette tre anni a arrivare a una spatial dimension di 28x28.
![[Pasted image 20240528171459.png]]

Il modello e' questo:
![[Pasted image 20240530152539.png]]
L'inception module ha 22 trainable layers.

### Naive inception module
Si processa la stessa input activation con diverse convoluzioni in parallelo.
![[Pasted image 20240530152940.png]]
Si concatena l'output sulla dimensione della *profondita'*.
Ho quindi TANTISSIMI CHANNELS per colpa del maxpool per la maggior parte.
Ci sono 2 problemi:
1. il numero di channels PUO' SOLO CRESCERE perche' sicuramente ne ho 192 causa maxpool, e poi aumentera' causa altri convlayers. Se stacco tanti inception modules ho una crescita disarmante.
2. Il numero di FLOPS e' altissimo ![[Pasted image 20240530153456.png]]
	con i conv che ho in figura.
Il motivo per cui l'inception e' buono e' perche' ci sono **MOLTI PATHS** per raggiungere l'output (guarda la figura la' sopra). Quindi si scelgono paths ottimali. 

Perche' introdurre una conv con kernel 1x1?
![[Pasted image 20240530155002.png]]
![[Pasted image 20240530154933.png]]
Questo perche' ho effettivamente una matrice di dimensione 128x3.
E' un linear layer che e' slidabile, perche' la applico a ogni pixel dell'attivazione iniziale.
le 1x1 conv sono molto buone per cambiare il numero di channels delle activations, senza cambiare la spatial resolution. Ora stanno dappertutto. Nell'inception model di sopra fa passare da 192 a 64.

Come gia' detto i problemi del naive sono che il maxpool fa aumentare troppo la profondita' dei channels e che i 3x3 e i 5x5 convs sono troppo costosi a livello di operazioni (FLOPS).

E' qui che entrano in gioco i 1x1 CONVS:

![[Pasted image 20240530155920.png]]
Come puoi vedere qui, ho 16x192x1x1 che mi dimiuisce la *profondita'* a 16. Un altro 1x1 che me la diminuisce a 96. A sto  punto su sti due output posso rispettivamente applicare un kernel 5x5 e 3x3. In sto modo ho molti meno flops. 
Inoltre dopo il maxpool faccio una 1x1 con che diminuisce la profondita' a 32.
In sto modo ho un aumento contenuto pure della profondita' e un numero di flops ridotto.


Un'altra novita' introdotta da GoogleNet e' l'assenza di fullyconnected layers alla fine.
Hanno proposto **global average pooling**.

![[Pasted image 20240530162023.png]]
Questa ocosa e' data dall'assunzione per cui in un filtro magari ho quanto e' gatto l'immagine. E loro dicono che posso buttar via la spatial dimension e fare un average dei valori della spatial dimension, in sto modo ottengo una unica rappresentazione dell'intero filtro.
![[Pasted image 20240530162253.png]]
HO in GoogleNet ad esempio che alla fine degli inceptions ho 1024x7x7, e dopo avgpooling con global kernel 7x7 diventa 1024x1, e ho che c'e' un unico fullyconnected layer di dimensione 1000x1024 che outputta 1000 valori che sono le mie probabilities. Ho quindi solo 1 milione di parametri nel Fullyconnected layer a differenza di VGG in cui avevo 124milioni di parametri negli ultimi 3 FullyConnected layers.
Questo mostra che non c'era bisogno di avere tutti quei parametri negli ultimi FC layers.

NOTA CHE CON UN CONVOLUTION MODEL io posso finche' ho conv layers trainare la rete su qualsiasi tipo di foto, di qualsiasi dimensione. 
Il PROBLEMA con GoogleNet e' che c'e' alla fine un average poooling in cui devo specificare 7x7.
Se uso pero' 
![[Pasted image 20240530163223.png]]
Allora non c'e' bisogno di specificare esplicitametne 7x7 e lui, in base alla spatial resolution che viene ritornata AL TERMINE DEGLI INCEPTION LAYERS mi fa global average pooling.
Poi il fully connected layer e' sempre di dimensione 1000x1024 facci caso (perche' la spatial dimension viene comunque mandata a 0 tramite il global average pooling, incredibile).
Sta cosa tipo con le reti passate non potevo farla, perche' flattenando li' ho che la dimensione da feeddare nel FC layer deve essere statica, sempre la stessa.

Nota che queste networks NON sono scale invariants quindi se magari uso in evaluation un'immagine con una spatial resolution MAGGIORE rispetto a quella con cui e' stata trainata la network, beh avro' che le strutture spaziali che i kernels hanno riconosciuto in fase di training ora sono molto piu' grandi/ sono diverse (ho una risoluzione piu' grande abbiamo detto.)

![[Pasted image 20240530163817.png]]
Questa network ha solo 7 mils params e ha battuto VGG. TIPO ANCHE SULLA MEMORIA ,  grossi passi avanti, molto ottimizzata.
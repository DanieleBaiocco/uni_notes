![[Pasted image 20240604130658.png]]
Hanno fatto sta rete alternativa, in cui ho diversi tipi di *inception modules* all'interno della rete. Inoltre ho anche **factorization** in cui magari una conv che avevo prima di 5x5 diventa 3x3 seguita da 3x3 come si puo' vedere nel primo grafo.
NON c'e' da sapere i dettagli di questo inception module per l'esame.

Inception v2 e' stato definito nello stesso paper in inception v3, ma e' stato mostrato che inception v3 e' meglio dell'inception v2.


# Residual connections
Per adesso sono stato abituato al fatto che PIU' la network e' deep meglio e'.
![[Pasted image 20240604131514.png]]
Hanno staccato 56 layers invece di 20 e hanno visto che le performances sono **diminuite**. 
Quindi hanno visto che staccare piu' layers non ha portato a risultati migliori.
Per diverse ragioni:
1. Aggiungere tanti altri layers aumenta il numero di parametri e quindi c'e' il rischio di overfittare
2. Pero' NON e' un problema di overfitting, perche' anche nel train set ho un errore piu' alto. Quindi c'e' un problema di **underfitting** in verita'.
Si e' ragionato sul fatto che si puo' **ottenere** una rete piu' deep  di 56 layers in cui ho un train e test error che siano uguali a quelli della rete con 20 layers. 
Questa rete e' ottenibile staccando sopra i 20 layers, 36 **identity** layers, che non cambiano niente. Trainando questa rete l'errore prodotto sara' UGUALE a quello prodotto usando solo i 20 layers (per costruzione cio' e' vero). **NOTA** pero' che nel caso di sopra ho che **SGD** non e' stato in grado di trovare i parametri per ottenere degli **identity layers** nei layers dal 20esimo al 56esimo, perche' il modello ha trovato una soluzione peggiore di quella **ottenibile** per costruzione.

L'idea e' quella di far in modo che SGD riesca almeno a **partire** vicino alla soluzione identita'.
## Residual block
![[Pasted image 20240604132705.png]]
Si impone un constraint su un block, prima della ReLU.
NOTA che se F(x) e' inizializzato per essere **vicino a 0** il training iniziera' col blocco che da' come risultato l'**identita'**. In sto modo se metto dei residual blocks dal 20esimo al 56esimo, ho che SGD inzia per il modello con 56 layers da un risultato che e' UGUALE per costruzione a quello del modello con 20 layers. E da li' va migliorando. Quindi non si crea piu' il problema di prima. Per ogni blocco dal 20esimo al 56esimo si inizializzano i parametri per ottenere F(x) simile a 0. In sto modo ho sempre l'identita'.
Ho quindi aggiunto un constraint nel modello, con l'idea di facilitare e trainare in modo effettivo la rete neruale.
In sto modo forzo ogni blocco per essere **una piccola variazione dell'IDENTITA'** (F(x) + x).
Per avere residual blocks ho bisogno di **BATCH NORM** nel blocco in considerazione.

Con sti residual blocks posso trainare deeper networks.

# ResNets (Residual Networks)
![[Pasted image 20240604133424.png]]
In ResNet-18 ho 18 trainable layers e li puoi pure contare.
Ci sono degli stages che sono stacks di residual blocks (in sto caso 2).
Ogni stage raddoppia il numero di channels e dimezza la resolution (sta cosa e' fatta dal PRIMO BLOCCO di ogni stage).
NON c'e' maxpooling. 
La ReLU e' fuori perche' se fosse dentro, prima della skip connection, ho che F(x) sarebbe sempre e solo positiva, quindi avrei sempre e solo un AUMENTO DI x (non voglio sta cosa).

ResNet solitamente e' un compromesso, infatti
1. downsamplano con STEM ma non in maniera aggressiva
2. fanno GlobalAvgPooling invece di usare i FC layers.

La struttura di sopra puo' essere replicata quante volte voglio.

### Problema
![[Pasted image 20240604134052.png]]
Ho questa struttura in cui dopo esser entrati in stage n+1 ho che la skip connection di grandezza C x 2H x 2W deve essere sommata a un qualcosa di grandezza 2C x H x W (perche' il numero di channels appena entro in uno stage viene raddoppiato e la spatial dimension viene dimezzata). Questa cosa non e' possibile. Come si fa?
Gli autori del paper hanno pensato a due soluzioni:
1. una soluzione che non ho compreso
2. la seconda in cui si fa una 1x1 conv con stride  a 2 e con output channels a 2C

![[Pasted image 20240604141056.png]]

Con questi cambiamenti ho che staccando piu' layers ho migliori risultati
![[Pasted image 20240604141258.png]]
dotted line per il train set, continous line per il test set.
NOTA dall'immagine che passsare da 20 a 32 ho un grande aumento, pero' dopo un po' rendere la rete piu' deep NON ha un cosi' grande impatto/aumento. Questo per dire che a una certa posso anche non farla coisi' deep.

La ResNet e' una baseline che voglio sempre provare in ogni dataset al giorno d'oggi.

## Bottleneck residual block
E' una variante del residual block visto fin'ora. Di base quello che succede qua e' DIVERSO. E sto bottleneck residual block verra' poi utilizzato in altre reti.
Fin'ora ho visto questo residual block:
![[Pasted image 20240604172342.png]]
Che ha un numero di flops e di parametri rispettivamente $36C^2HW$ e $18C^2$.
Ogni blocco di questa forma incrementa la depth(la profondita' della rete) calcolandola come il numero di steps di **non linearita'** di 2. Quindi se voglio 50 layers ho bisogno (ignorando lo stem layer e il globalavgpooling) di 25 di questi blocchi. 
Hanno proposto un altro blocco in cui **il numero di steps di non linearita' aumenta piu' velocemente, e di conseguenza aumenta la depth**, da 2 a 3, SENZA aumentare il numero di paramentri e SENZA aumentare i flops.
![[Pasted image 20240604172723.png]]
Ho 3 convoluzioni, utilizza 2 convoluzioni con kernel a 1x1. In mezzo hanno una 3x3 convolution.
Parto da 4C channels, riduco a C e poi alla fine con l'ultima 1x1 conv riporto il numero di channels a 4C. In sto modo posso fare la somma con la residual connection (con l'input che ha 4C channels). Per questo sto blocco e' chiamato bottleneck perche' si riduce la sua feature dimension per poi ripristinarla (a imbuto). Ha leggermente meno numero di parametri e di flops quindi TOP  e ho 3 non linearity steps invece di 2. Tutto di guadagnato. Ho anche che il numero di CHANNELS in input e' 4 volte piu' grande dell'altro residual block.

**NOTA** importante: comunque la PRIMA convoluzione che e' 1x1 , per quello DETTO fino ad ora, seguendo quindi la regola vista fin'ora, c'e' una STRIDE A 2 che dimezza la spatial dimension. Non e' presente in figura perche' la figura si riferisce a un BLOCCO dello stage DIVERSO DAL PRIMO, che si occupa di dimezzare la spatial e di raddoppiare le attivazioni. 

Nel primo cio' che succede e' che nella PRIMA 1x1 conv si ha una stride a 2. Questo dimezza la spatial dimension. Non mi e' chiaro quando avviene l'aumento delle feature maps al DOPPIO (immagino in prossimita' del secondo 1x1 conv (invece di 4C magari ritornando 8C)). Parallelamente la residual connection fara' una 1x1 conv con stride a 2 e con output channels a 8C.

Sto bottleneck block e' usato quando SI VA MOLTO IN BASSO CON IL NUMERO DI LIVELLI.

![[Pasted image 20240604173147.png]]
ResNet-34 ha la stessa struttura di ResNet-18 ma ha piu' res-blocks all'interno degli Stages.
Nota che quando vado in 50 in SU si usano bottleneck blocks. Si hanno 4C channels nel numero di channels rispetto alle precedenti archietture che usano standard residual blocks. Questo e' coerente con quello visto sopra.

Nota che ResNet-34 e ResNet-50 hanno stessa struttura e stesso numero di res blocks, eppure il secondo conta 50 layers proprio per il fatto che i res blocks di ResNet-50 sono bottleneck blocks e hanno 3 non linear activations invece di 2.

I risultati sono i seguenti:
![[Pasted image 20240604173627.png]]
NOTA CHE ResNet-152 HA lo stesso numero di parametri di AlexNet, e ha 152 layers mentre l'altro 8. Inoltre funziona meglio.
Nota che dai risultati,  piu' depth si stacca MENO E' LA GAIN (c'e' comunque una gain ma e' sempre minore).

## ResNet v2
Nella ResNet con bottleneck block ho che come primo step ho una 1x1 conv con **stride 2**. Ma sta cosa non ha molto senso perche' una 1x1 conv con stride 2 usa SOLTANTO UN PIXEL ogni 4 attivazioni per computare l'output. Non ha molto senso. Quindi ho usato un sacco di FLOPS per computare le attivazioni e NE STO USANDO soltando 1/4.

Quindi in ResNetv2 la **stride=2** e' messa soltanto nel 3x3 conv. Questo e' fatto nella ResNet-B
NOTA CHE LA STRIDE A 2 PERO' RIMANE NEL 1X1 della residual connection, sta cosa e' un po' sussy.
![[Pasted image 20240604183602.png]]
Nella ResNet-C si cambia lo stem layer che diventa questo 
![[Pasted image 20240604183646.png]]
In ResNet-D viene cambiato il modo di matchare la residual connection con la dimensione in output del bottleneck block. **Questo per lo stesso motivo che la 1x1 conv con stride 2 sulla skip connection evitava alcuni pixels. In sto modo ho che l'avg pool aggiusta la dimensione e la conv invece raddoppia il numero di channels**
![[Pasted image 20240607180138.png]]

![[Pasted image 20240607181018.png]]
## INCEPTION V4 E INCEPTION-RESNET-V2
INCEPTION v4 e' un inception v3 con uno STEM PIU' COMPLICATO
Gli autori nello stesso paper hanno anche provato ad aggiungere la residual connection attorno all'**Inception module**. Quello che ne e' uscito fuori e' L'INCEPTION-RESNET-V2. 
CI SONO DUE BLOCCHI CHE SONO STATI IMPLEMENTATI
1. Inception Resnet A
2. Inception Resnet B
![[Pasted image 20240607182911.png]]
Se prendiamo in esame la Inception Resnet A si ha come idea quella di avere dei BOTTLENECK BLOCKS  e renderli simili all'INCEPTION MODULE. All'inizio di quasi tutti i path di un inception layer ho difatti una 1x1 conv che puo' avere il ruolo della prima 1x1 conv del BOTTLENECK BLOCK. **NOTA** che ho una 3x3 e una 5x5 (che viene fattorizzata in due 3x3 come avevo nell'Inception Module difatti). Ho poi una concatenation come nell'Inception module. C'e' infine una 1x1 conv che porta il numero di channels a un numero minore pari a quello della skip/residual connection.
Ha la stessa struttura del bottleneck block con PIU' PATHS possibili e con la concatenazione dell'Inception module.

# TRANSFER LEARNING
Prima si RITRAINAVANO I MODELLI DA CAPO, se si cambiava DATASET.
Ci sono delle rappresentazioni IMPARATE da modelli gia' trainati come quelli visti, che possono essere prese (i pesi di quei neuroni) e utilizzate nel mio modello.
Questi pesi sono una **buonissima inizializzazione** per il mio modello sul nuovo DATASET.
Questa cosa e' sempre beneficial e funziona sempre molto bene, almeno per i primi layers delle CNN che contengono lower level features, anche se il DATASET e' COMPLETAMENTE DIVERSO da quello su cui il modello (tipo ResNet) e' stato trainato (che nel caso di ResNet e' ImageNet).

POSSO fare cosi:
1. Prendere la ResNet trainata su ImageNet
2. Prendere i pesi FINO al Global AVG Pooling+FC e inizializzare una copia di sta ResNet con questi pesi
3. Per quanto riguarda l'Global AVG Pooling+FC, i pesi sono inizializzati RANDOMICAMENTE 
4. Faccio il training sul nuovo dataset, TENENDO FREEZZATI i pesi delle CONVOLUZIONI e imparando solo quelli del Global AVG Pooling+FC
![[Pasted image 20240607184853.png]]
QUESTA COSA E' UNA COSA BUONA DA FARE quando
1. ho poche immagini nel nuovo dataset
2. se il domain del nuovo dataset e' MOLTO SIMILE a quello di ImageNet

Se invece
1. ho molte immagini
2. ho un dominio COMPLETAMENTE diverso, che ne so tipo foto di tumori al cervello ok?
allora qui ha piu' senso fare **fine tuning**, ovvero far in modo che possano cambiare anche i pesi delle CONVOLUZIONI.
![[Pasted image 20240607185042.png]]
Quando si fa fine tuning HA SENSO
1. fare QUALCHE EPOCA in cui si tengono FREEZZATI i pesi delle CONVOLUZIONI (che sono i miei feature extractors) (le prime epoche almeno).
2. usare un lr piu' basso di quello usato da ResNet nel suo training (usare un ordine di magnitude minore rispetto all'originale lr)
3. usare **progressive lr**: ovvero usare un piccolo lr per trainare Global AVG pooling layer e usarne UNO ANCORA PIU' PICCOLO per i pesi delle CONVOLUZIONI. A volte ha addirittura senso FREEZZARE PROPRIO i PRIMI layers (legati agli edge extractors, ecc...) e proprio NON RENDERLI MAI TRAINABILI ecco. 
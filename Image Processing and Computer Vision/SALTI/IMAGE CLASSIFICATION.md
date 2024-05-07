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
# AI, algorithmic decision making and Big Data (Risks and Opportunities)
AI e l'INTERNET hanno portato a DATA-DRIVEN AI systems, perche' ho molti piu' dati no? grazie all'internet.
Queste due cose insieme permettono di:
1. Avere una generazione e distribuzione a livello mondiale di soluzioni e conoscienze
2. Avere migliore produttivita', fare *cost saving* e *value creation*, infatti le aziende possono anticipare i trends del mercato tramite AI e scegliere decisioni migliori. I consumatori possono fare delle scelte migliori e anche piu' informate, e ottenere servizi personalizzati.

### Opportunita' dell'AI
AI puo' aiutare nell'affrontare i problemi/le grandi challenges degli esserei umani come
1. fornire le risorse per mantenere in vita una popolazione (energia, cibo, acqua),
2. mantenere un ambiente sostenibile e sano, 
3. affrontare malattie
4. eliminare la poverta'
L'AI 
1. puo' addirittura sostituire *attivita' umane* tipo con self-driving cars;
2. nella maggior parte dei casi pero' *sono strumenti ausiliari* che possono aiutare nel migliorare le capacita' umane, le *complementano*: es. supportando *creativita'* (magari tipo usando una *generative* AI per generarmi delle immagini particolari da cui poi io artista prendero' spunto) o aiutando nel *prendere decisioni* come nel caso di AI in campo medico che data una lastra dicono se c'e' o meno un *tumore*(ma e' solo un parere che il medico tiene in considerazione).
**Si puo' quindi raggiungere un livello di *cooperazione tra AI e umani*.** Si puo' arrivare a un punto in cui l'AI performa delle *repetitive tasks*, i cui outcomes vengono poi utilizzati dagli *umani*,che poi agiscono (tenendo anche in considerazione questi outocmes ovviamente).
**Nota** pero', ci sono problemi nella *generative AI*, perche' 
1. rende l'arte un qualcosa di molto piu' *cheap e accessibile a tutti*: se devo creare una copertina posso chiedere a un'AI invece che a un artista. 
2. Inoltre l'artista puo' sentirsi **copiato** da una AI, che copia il suo stile e lo riproduce (quindi c'e' anche un problema di copyright e di imitazione, che non fa mai piacere a un artista no?).

### Rischi dell'AI
Ci sono pero' ovviamente diversi rischi legati all'AI:
1. puo' elimiare o devalutare dei *lavori* che sono rimpiazzabili da *AI SYSTEMS*: i chatbots possono eliminare il lavoro del *segretario* ad esempio. Cio' porta a *poverta'* e *esclusione sociale* per quelle persone che avevano questi lavori
2. puo' inoltre portare a un'economia in cui *chi arriva prima prende tutto*, quindi chi prima sviluppa modelli di AI cosi' forti da essere in grado di rimpiazzare la forza lavoro umana, ha il *monopolio* sull'economia. L'AI quindi concentrerebbe il *benessere/i soldi* SOLO in queste aziende.
3. puo' essere usata a scopi molto *illegali e pericolosi*, come *il caso delle armi autonome*.
4. puo' essere usata per *sorvegliazna*, con riconoscimento facciale come avviene gia' in Cina. Ma anche viene usata nel *manipolare* le scelte altrui, magari capendo le mie prefertenze immagino.

 Certi pericoli possono essere incentivati dalla presenza di compagnie che hanno un *two sided market* ovvero che hanno servizi riferiti a **due diversi tipi di utenti, magari uno che consuma il contenuto e uno che lo vende**. E la vendita credo sia riferita alla vendita di informazioni dell'utente che consuma. In questo modo le applicazioni di queste compagnie consigliano all'utente consumatore cose che sono **fidelizzate**, che lo porteranno a stare nell'app piu' tempo, che cattureranno di piu' la sua attenzione. 

Questa cosa **porta** a **POLARIZATION** e frammentazione della *sfera pubblica*. Infatti AI e' molto legata alla diffusione di **fake news**. E queste vengono utilizzate per catturare l'attenzione dell'utente e lo portano poi a persuaderlo sulle sue idee (anche politiche magari).

Tanti politici (come aveva fatto Trump) possono utilizzare AI in campagna politica per anticipare e controllare il pensiero degli elettori/dei cittandini per portare il voto dalla loro.

#### Profiling, influence e manipulation
Ci sono AIs che fanno profiling e poi manipolazione delle persone. 
##### PROFILING
L'AI puo' inferire informazioni riguardo a *persone*,  e questo e' proprio il profiling. Questo inferire e' possibile collezionando prima informazioni, preferenze dell'utente, per poi trovare **correlazioni** da questi dati e derivare diciamo un **profilo** dell'utente. Si puo' fare anche su gruppi di persone.
PROFILING permette anche di capire che se un individuo ha una determinata feature F1 allora ha anche con una determinata probabilita' un'altra feature F2 (si fa inference in questo senso).
Quindi i modelli di PROFILING, appena arriva un nuovo individuo con una determinata feature F1, associano subito questo individuo **con un'altra feature F2**: esempio che ne so arriva un carcerato con features riguardanti la sua storia criminale, il suo carattere, e il suo background e da queste features (che rappresentano F1 diciamo) il PROFILING inferisce la probabilita' che questo possa **ricommettere un crimine** (che sarebbe la feature F2). Sembra *sbagliato* affidarsi a qeuste predizioni per cose importanti.

Il fatto e' che sta feature F2 potrebbe NON essere vera ovviamente.E' spesso vista in questo modo: cio' che viene inferito indica la **propenzione** a reagire in un certo modo (tipo la propenzione a ricommettere il crimine) dati certi inputs (che sarebbero le F1). Ma e' solo una propenzione, non e' mica certo. Cio' puo' essere usato quindi per **manipolazione e influenza**, perche' vado a giocare su qeusta **propenzione** provando a influenzare verso la F2 a quelli che hanno la F1.![[profiling.png]]
DEFINIZIONE di PROFILING (per bene): e' una tecnica di *automated processing* di dati riferiti alla persona e non, che ha l'obiettivo di *conoscere* la persona, inferendo correlazioni da questi dati nella forma di **profili** che possono essere poi utilizzati per fare *decision-making* : tipo consigliando un determinato AD a uno che viene profilato in un determinato gruppo.

#### PROFILING in GDPR
GDPF: general data protectional regulation, e' una serie di norme
In questa regulationc'e' una definizione di  *profiling* che e' definito solo a livello di **singola persona**, non di **gruppi di persone**. E' una definizione sicuramente diversa da quella che abbiamo visto finora.
![[profiling2.png]]
natural person: a person that is an individual human being, distinguished from the broader category of a legal person, which may be a private or public organization.
legal effect: significa che una **decision** affects someone’s legal rights
significantly affects him or her: tipo nella reputazione di una persona

#### I pericoli del Profiling - Cambridge Analytica
![[cambridgeanalytical.png]]
![[cambridgeanalytical2.png]]
![[cambridgeanalytical3.png]]
![[cambridgeanalytical4.png]]Lo **scandalo dei dati Facebook-Cambridge Analytica**[[1]](https://it.wikipedia.org/wiki/Scandalo_Facebook-Cambridge_Analytica#cite_note-1) è stato uno dei maggiori scandali politici avvenuti all'inizio del 2018, quando fu rivelato che [Cambridge Analytica](https://it.wikipedia.org/wiki/Cambridge_Analytica "Cambridge Analytica") aveva raccolto i dati personali di 87 milioni di account [Facebook](https://it.wikipedia.org/wiki/Facebook "Facebook") senza il loro consenso e li aveva usati per scopi di [propaganda](https://it.wikipedia.org/wiki/Propaganda "Propaganda") politica.
I dati collezionati vennero usati per le elezioni negli USA invece HAHAHA.

#### Surveillance capitalism
E' il nuovo *leading economic model*. Karl Polanyi ha osservato che il *capitalismo industriale* tratta cose che non sono **prodotti** del mercato **come** prodotti del mercato: 
1. la vita umana viene vista come **lavoro**
2. la natura viene vista come **terra in cui edificare**
3. gli scambi vengono visti come **scambio di soldi**
Di conseguenza queste dinamiche del capitalismo producono delle tensioni distruttive come la *distruzione dell'ambiente e del pianeta, crisi finanziarie, sfruttamento (uomo visto come lavoratore da sfruttare)*. Questo viene mitigato ovviamente dalla legge, dalla politica e da organizzazioni sociali (movimenti dei lavoratori) che intervengono e mitigano i lati negativi del capitalismo.

Il *surveillance capitalism* che e' quello di adesso **ESTENDE** questa mercificazione (che prima era riferita come visto sopra a persone, natura e scambi) introducendo adesso all'interno anche la *human experience*. Infatti l'esperienza umana (quella di vivere, navigare in internet, comprare ecc...) viene vista come un **comportamento da salvare e analizzare** (per farci appunto profilazione). 
**NOTA** le prime tre **commodities** (le ha sempre chiamate cosi fin dall'inizio ma e' la prima volta che le scrivo in questo modo usando sto termine) che sono quella del lavoro, delle terre e dei soldi (partendo da persone, natura e scambi) sono soggette alla **legge**. Cosa che **NON** e' ancora avvenuta per la quarta **commodity** che e' questa riferita all'human experience.
![[surveliancecapitalism.png]]
#### Surveillance state
In questo caso la situazione e' la parallela with respect to il survelliance capitalism. 
Nel National Survelliance State, il governo utilizza la *sorveglianza*, la collezione di dati, e la loro analisi, per intervenire tempestivamente in situazioni di *potenziale pericolo e minaccia* (nota, cio' puo anche significare soppressione di movimenti che insorgono contro lo stato eh), per *governare le popolazioni* e per fornire *social services* utili a tutti.

Il National Survelliance State e' una forma speciale di Information Sstate, che e' uno stato che cerca di risolvere i problemi di una popolazione tramite la collezione, il confronto, l'analisi e la produzione di informazioni utili per far fronte ai problemi e risolverli.

Se usato bene permette di:
1. supportare l'efficienza nella gestione delle attivita' sociali
2. coordina il comportamento dei cittadini
3. cerca di far in modo che non ci siano pericoli sociali
Se usato male permette di:
1. avere un modo di influenza e controllo sulla popolazione
2. promuovere valori che vanno contro i principi base della democrazia
3. diminuire/eliminare la *human autonomy*
Esempio peggiore di *Surveillance State* e' quello della CINA, col **chinese social credit system**.
![[cina.png]]
Sta cosa e' assurda
Cioe' i comportamenti delle persone si omologano allo standard richiesto. Non si hanno pensieri diversi, contrastanti, non c'e' dibattito, non c'e' un modo diverso di percepire la vita. Si vive tutti allo stesso modo. Oltre al fatto che cio' genera terrore (cioe' io mi  sentirei in ansia diocane).
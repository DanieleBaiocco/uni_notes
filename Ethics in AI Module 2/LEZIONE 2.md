# Individual and social costs of AI & Big Data applications
In some cases and domain, AI & Big Data applications—even when accurate and unbiased— may have individual and social costs that outweigh their advantages.

Consider, for instance, systems that are able to recognise sexual orientation, or criminal tendencies from the faces of persons. Should we just ask whether these systems provide reliable assessments, or **should we rather ask whether they should be built at all?**

The key aspect of ML systems is the ability to engage in **differential inference**: different combinations of *predictor-values* are correlated to *different predictions*.

**The general problem of social scoring and differential treatment** is that
a new dynamic of **stereotyping and differentiation** takes place:
1. The individuals whose data support the same prediction, will be considered and treated in the same way.
2. The individuals whose data support different predictions, will be considered and treated differently
## Problemi nell'utilizzo dell'ML nel predirre la salute di un umano
This provision has become hugely relevant in the context of the Coronavirus disease 2019 (COVID-19) epidemics. In particular a vast debate has been raised by development of applications for tracing contacts.
Such processing should be viewed as legittimate as long as it effectively contributes to limit the diffusion and the harmfulness of the epidemics, assuming that the privacy and data protection risks are proportionate to the expected benefit, and that appropriate mitigation measures are applied.

C'e' pero' un piccolo problema in questa questione: 
se applico ML per predirre la salute di un paziente va bene,  se poi questa informazione viene utilizzata magari nel contesto dei **prestiti** per **non cedere** un prestito a una persona che, secondo il modello di ML, morira' tra poco, allora c'e' un grosso problema. 
In sto modo infatti si avrebbe un mondo in cui se  una persona sta male, viene ulteriormente svantaggiata anche in quest'altro contesto, causando emarginazione e maggiori sofferenze.

Immagina ancora cosa significa applicare queste predizioni **sulla salute**, invece nel contesto delle *assunzioni lavorative*. Sarebbe tremendo, in quanto questa persona sarebbe tagliata fuori **addirittura anche dal contesto lavorativo**. 

## Problemi della Discriminazione di Prezzo
Un'altra cosa che potrebbe avvenire e' la *discriminazione di prezzo*. Quindi il fenomeno per cui verrebbero consigliati diversi prezzi di prodotti a seconda della loro **voglia e disponibilita' nel comprare un determinato prodotto**. Ad esempio alle donne potrebbe essere mostrato un prezzo piu' alto per le scarpe, avendo il modello di ML imparato che alle donne piacciono motlo, mentre ai maschi potrebbe esser mostrato un prezzo basso per le stesse scarpe e un prezzo alto magari per delle *griglie*.![[ethics23.png]]
Questo porta a qualcosa di poco etico, in quanto certi individui **verrebbero privati di certe opportunita'** solo perche' appartengono a un determinato gruppo imparato dal modello.

## I sistemi di AI sono meglio degli umani  nel valutarci?
In tanti *domains* si ha che **predizioni automatiche** date da una AI non solo sono piu' **economiche** rispetto a avere un gruppo di persone che fa quel tipo di scelte, ma spesso sono anche **piu' precise e imparziali** rispetto agli umani:
1. L'AI puo' evitare i tipici errori della **psicologia umana** come essere over-confidenti, avere dei bias e evitare l'**inabilita' da parte degli umani di processare dati statistici**, oltre all'evitare **i pregiudizi umani** (riguardo ad esempio al sesso, all'etnia).
2. In molti campi si e' proprio visto che sistemi algoritmici (come AI systems) hanno spesso performato meglio rispetto a esperti umani

I problemi dell'umano nel valutare sono i seguenti:
* Over-confidenza: quando gli umani hanno piu' confidenza del dovuto nelle proprie abilita', nel fatto di sapere e saper agire
* Avversione verso la perdita: gli umani tendono a preferire la mancanza di perdite, rifiutano la possibilita' di una perdita, anche se questa potrebbe poi portare a un guadagno maggiore nel  lungo tempo
* Ancoraggio: le decisioni degli individui sono **influenzate** da un punto di riferimento particolare, anche detto ancora, AD esempio un individuo potrebbe essere invogliato a comprare una macchina se questa e' messa vicina a una macchina di un modello molto piu' costoso (quindi l'individuo vede un'offerta, basandosi sul reference point, sull'ancora rappresentata dall'altra macchina piu' costosa)
* Bias di conferma: gli individui tendono a fare scelte perpetrando le loro idee e le loro credenze (che sono spesso biased e legate al proprio penisero)


Ci sono stati casi pero' di AI che erano *discriminatorie*. In pochi casi c'e' stata una discriminazione esplicita, chiamata *disparate treatment*, in cui l'AI basava i suoi outcomes su features proibite come la razza, l'etnia, il gender.
Si tende piu' a associare un outcome a una razza se il dataset e' fatto per cui la razza e' legata a un certo tipo di outcome **causa discriminazione razziale, causa contesto sociale INGIUSTO**.

**IMPORTANTE** infatti precisare che:
Dei sistemi basati su *supervised learning* possono infatti esser trainati su *giudizi umani passati* e quindi possono *riprodurre* i punti di forza ma soprattutto quelli di **debolezza** degli umani che hanno dato quel tipo di giudizio, che includono ovviamente la propensione di alcuni di esser pregiudizievoli verso determinate etnie,ecc...

## Pregiudizi nel training set
Il pregiudizio e la discriminazione all'interno del dataset puo' anche essere presente nonostante le *prohibit features* vengano escluse (quindi features riguardanti il sesso, la razza, la regione di appartenenza, ecc...).
Infatti basta che ci siano **correlazioni**, anche NON dirette, tra features discriminatorie e predizioni del training set per avere discriminazioni. 
Ad esempio si potrebbe avere un *dataset* in cui NON e' presente la feature della razza a tempo di training MA che e' riferito a assunzioni fatte da un manager con grossi pregiudizi. Questo manager NON ha mai assunto persone negre. Queste persone abitavano (e continuano a abitare) nello stesso quartiere. 
La feature del quartiere di provenienza e' pero' presente nel dataset utilizzato per il training. Di conseguenza quando si faranno predictions, dato l'input, si sara' discriminatori verso il quartiere di provenienza (si tendera' a non assumere persone provenienti da quel quartiere, perche' nel dataset persone provenienti da li' non vengono mai assunte) e di riflesso lo si sara' verso le persone nere , in quanto maggiormente risiedono in quel quartiere.

## Biases del sistema verso determinati gruppi
Ci sono casi in cui un training set e' biased contro un particolare gruppo di persone, in quanto l'outcome che viene predictato e' approssimato attraverso un **proxy**, quindi una sola feature che risulta poi essere discriminatoria.

Per fare un esempio:
Immagina che si utilizzasse un'AI per dare uno score a ogni persona, che indica, dati degli inputs, quanto quella persona e' **produttiva**. Mettiamo che questa prediction sia solo legata al **numero di ore lavorate in ufficio**. Il dataset e' ovviamente legato al passato, e si avra' che le donne tenderanno a avere uno score piu' basso. Cio' e' dovuto dal fatto che le donne lavorano in generale meno ore causa problemi legati al travaglio, alla gestazione ecc...
Quindi il modello predicterebbe che alle donne e' giusto associare uno score piu' basso, in quanto sono meno performanti rispetto agli uomini.

## Biases del sistema presenti nelle input features
A volte ci sono casi in cui il sistema performa in maniera discriminatoria perche' ha come input feature una categorical feature magari che e' **SEMPRE** settata a False per membri di un determinato gruppo, mentre per altri gruppi varia (o e' per la maggioranza True). La feature potrebbe essere *e' stato iscritto a una scuola superiore*. Ovviamente e' piu' probabile che per persone di colore, poco abbienti e con condizioni sociali precarie, questa feature sia settata a False, ma in questo modo il dataset rispecchia una discriminazione che avviene nel mondo reale. La prediction non dovrebbe essere basata su una feature su cui le persone di colore non hanno avuto scelta magari (tanti di loro avrebbero voluto andare a scuola ma per poche disponibilita' economiche non e' stato possibile).

## Biases dovuti a datasets che non rispecchiano la composizione statistica della popolazione
Metti che c'e' un'AI che predicta la possibilita' di pagare una cauzione date delle features su una persona. E metti che una delle features in input sia *quanti reati hai commesso*. Per le persone nere questo numero sara' piu' alto, NON solo per motivi magari legati a disagio sociale, situazioni precarie ecc..., ma anche a causa del fatto che **si tende, soprattutto in America** a fare **controlli piu' stringenti a persone nere, mentre per quanto riguarda i bianchi, anche se hanno commesso qualcosa, magari non si scopre niente proprio perche' i controlli per loro invece solo piu' blandi**. Conta che in America la probabilita' che un nero venga fermato per dei controlli dalla polizia e' 8 volte piu' alta rispetto ai bianchi.

C'e' anche da pensare al fatto che membri di un gruppo potrebbero esser discriminati dall'AI causa fatto che nel dataset la porzione di dati riferita a loro sia molto esigua e quindi non si riesca a imparare per bene a predictare per quei gruppi, creando non affidabilita' nella predizione dell'AI e quindi possibile ingiustizia.

## Considerazioni su come affrontare l'ingiustizia di sistemi AI che performano decision-making 
Si e' visto che spesso questi sistemi AI sono **comunque**, nonstante **tutto**, piu' controllabili delle decisioni prese dagli umani. Le discriminazioni possono difatti essere identificate con precisione e questi outcome discriminatori possono esser migliorati per evitare outcomes ingiusti.

Un'altra cosa importante da dire e' che: un algoritmo biased puo' comunque essere **piu' giusto** rispetto a un umano che prende una decisione nello stesso contesto.

La cosa migliore da fare consiste nell'integrare outcomes dell'AI con revisione da parte degli umani. Quindi dando la possibilita' agli individui vittime di potenziali discriminazioni da parte dell'AI di richiedere una **revisione umana** della predizione fatta dal sistema.

**IL GDPR, prendendo in considerazione tutte queste problematiche, ha introdotto dei constraints**:
1. Hanno introdotto il bisogno di far leggi per gestire e limitare l'utilizzo di *dati personali* degli utenti
2. Hanno introdotto obblighi legati alla trasparenza di un sistema AI
3. Hanno introdotto limitazioni riguardanti il *profiling* e riguardanti sistemi *di automated decision making*
4. Hanno introdotto dei requisiti rugardanti l'anonimizzare i dati

## Social Empowerment
L'utilizzo della AI e' utilizzato gia' in domini in cui c'e' gia' una grossa differenza di potere (persone gia' ricche che aumenterebbero la disparita' e hanno ora la possibilita' di far ancor piu' il cazzo che gli pare).
Serve la **societa'** per scovare *abusi, informare il pubblico, attivare delle contro-misure*. E l'AI puo' essere utile appunto a cittadini di questa societa' per sviluppare sistemi che tutelino e proteggano l'utente dalle minacce di altri sistemi AI fatti da gente cattiva magari (come detto sopra).

Ad esempio, i cittadini hanno nel tempo sviluppato degli strumetni come *ad-block systems, anti-spam softwares, anti-phishing techniques, ecc...* che possano controbilanciare queste cose.
Ci sono stati due progetti che seguono qeusta scia che sono:
1. Claudiette: veditelo
2. PDA/CDA che sta per Privacy digital assistants/consumer digital
assistants: sarebbe una proposta per estrarre autocamticamente, categorizzare e riassumere informazioni legate a *documenti sulla privacy*, in modo da assistere l'utente nel processarli e capirne i contenuti. Il motivo e' prevenire la collezione di dati personali eccessivi, non voluti e illegali , inoltre to protect users from
manipulation and fraud, provide them with awareness of fake and
untrustworthy information, and facilitate their escape from “filter
bubbles”.
# AI in the GDPR
l'AI viene vista sotto diversi punti di vista nel GDPR. La prima cosa sotto la quale viene vista e'
## AI nel conceptual framework del GDPR
Nota che nel GDPR non viene mai nominata la parola *Artificial Intelligence*. Il GDPR principalmente si concentra sulle challenges dell'Internet. Ad ogni modo ci sono cose che comunque sono legate all'AI anche non volendo.
### Articolo 4
In questo articolo sono presenti delle *definizioni*:
1. C'e' la definizione di **personal data**, che indica qualsiasi informazione relativa a una persona fisica identificata o identificabile (chiamata **data subject**). Una persona fisica identificabile è quella che può essere identificata, direttamente o indirettamente, in particolare mediante un identificatore come un nome, un numero di identificazione, dati di localizzazione, un identificatore online o uno o più fattori specifici relativi all'identità fisica, fisiologica, genetica, mentale, economica, culturale o sociale di quella persona fisica.
2. C'e' la definizione di **profiling**, che indica qualsiasi forma di elaborazione automatizzata di dati personali che consiste nell'uso di dati personali per valutare determinati aspetti personali relativi a una persona fisica, in particolare per analizzare o prevedere aspetti riguardanti la prestazione lavorativa, la situazione economica, la salute, le preferenze personali, gli interessi, l'affidabilità, il comportamento, la posizione o i movimenti di quella persona fisica.
3. C'e' la definizione di **data subject**, che indica la *natural person* le  cui informazioni sono a essa legate
4. C'e la definizione di **processing**, che indica ogni operazione o set di operazioni che e' performata su dati personali o su sets di dati personali, in modo automatico e non
5. C'e' la definizione di **controller**, col quale si indica la persona *naturale* o *legale*, l'autorita' pubblica, agenzia o altri corpi i quali, in modo indipendente o collaborando, determinano i **motivi e i mezzi** del processare i dati personali.
6. C'e' la definizione di **processor**, che e' una persona *naturale* o *legale*, l'autorita', l'azienda, o altri corpi che *processa* **i dati personali** per conto del **controller**.

### Articolo 3
Questo articolo dice **per chi** le regolamentazioni presenti in questo GDPR saranno effettive.
Si applica al:
1.  **processing** dei dati personali nel contesto di attivita' di un'*istituzione* di un **controller** o di un **processor** nell'Unione Europea, indipendentemente dal fatto che il **processing** avvenga effettivamente o meno nell'Unione Europea.
2.  **processing** dei dati personali dei **data subjects** che stanno nell'Unione Europea da parte di un **controller** o di un **processor** che non sono stabiliti in Unione Europea, dove le attivita' di **processing** sono legate a:
	1. l'offerta di beni o servizi, non curanti del fatto che magari un pagamento al **data subject** e' necessario, a questi **data subjects nell'Unione Europea** (perche' si sta usando i loro dati quindi e' importante constatare se c'e' da pagarli o meno)
	2. il controllo dei loro comportamenti (dei data subjects), solo se questi comportamenti avvengono entro l'Unione Europea (se non avvengono nell'Unione, allora non  si applicano le regolamentazioni del GDPR).

### Articolo 5
L'articolo 5 parla di tante cose riassunte in questa slide:
![[ethics31 1.png]]
### Articolo 6
Partla della **legittimita' del processing**.
Dice che il processing puo' essere legittimo **solo se** **almeno** uno dei seguenti punti si applica:
1. i **data subjects** hanno dato il **consenso** per il **processing** dei loro personal data
2. quando il **processing** e' necessario per l'**esecuzione di un contratto**.
3. quando il **processing** e' necessario per **conformita'** a un obbligo legale (per attenersi a questo obbligo) al quale il **controller** e' soggetto
4. quando il **processing** e' necessario per **proteggere** interesti **vitali del data subject** o di un'altra **personal naturale**.
5. quando il **processing** e' necessario per l'**esecuzione di una task portata avanti nell'interesse pubblico** o se voluta da una autorita' officiale data poi al **controller**.
6. quando il **processing** e' necessario per *interessi legittimi* portati avanti dal **controller o da terze parti**, tranne quando questi interessi si scontrano con **gli interessi o i diritti fondamentali**del *data subject*, che magari richiede protezione dei suoi dati personali, in particolare quando il *data subject* e' un bambino.
## Articolo 4 - Identificabilita' 

```
'personal data'
 qualsiasi informazione relativa a una persona fisica identificata o identificabile (chiamata **data subject**). Una persona fisica identificabile è quella che può essere identificata, direttamente o indirettamente, in particolare mediante un identificatore come un nome, un numero di identificazione, dati di localizzazione, un identificatore online o uno o più fattori specifici relativi all'identità fisica, fisiologica, genetica, mentale, economica, culturale o sociale di quella persona fisica.
```
Vorrei spendere un po' piu' parole sull'**identificabilita'**. Rectal la esprime come le condizioni sotto le quali un pezzo di dato, che non e' esplicitamente collegato a una persona, conta comunque come **personal data** in quanto c'e' la possibilita' da quel pezzo di dato di risalire alla persona.
Quindi l'**identificabilita'** dipende dalla disponibilita' di **mezzi** in grado di essere usati per fare una **re-identificazione con successo**. Questa disponibilita' **dipende** dallo stato tecnologico e sociotecnico dell'arte.
Per determinare se una persona fisica e' identificabile, partendo da quei pezzi di dati, si dovrebbero usare **tutti i mezzi che sono ragionevolmente possibili**.
Per dire se  l'utilizzo di un determinato mezzo sia **ragionevole o meno** per identificare la persona fisica, si dovrebbero considerare tutti i fattori oggettivi, come i costi e la quantità di tempo necessari per l'identificazione, tenendo conto della tecnologia disponibile al momento del trattamento e degli sviluppi tecnologici.

Se ad esempio io faccio **speudonymisation** del dato personale, ovvero cio' che identifica una persona e' sostituito con uno **pseudonimo**, HO comunque che questo **pseudonymised data** e' ancora considerato un dato personale. Questo perche' **e' possibile per quanto detto qui sopra, risalire alla persona**, magari trovando una tabella in cui sono salvati i link tra **pseudonimi e nomi reali** o attraverso  *chiavi di crittografia* per decodare i nomi encriptati.

Per questo come abbiamo visto prima c'e' da vedere lo stato dell'arte della tecnologia al momento della discussione, per vedere se partendo da un dato e' possibile identificare la persona a esso legata o meno (e quindi definire o meno quel dato un dato personale).

Infatti nella *Regula)on (EU) 2018/1807 of the European Parliament and of the Council of
14 November 2018 on a framework for the free flow of non-personal data in the
European Union* c'e' proprio sto link tra progresso tecnologico e la natura **personale** del dato:
*se gli sviluppi tecnololgici rendono possibile trasformare un dato anonimizzato in un dato personale, allora quel dato deve esser trattato come dato personale*

### AI e la definizione di dato personale nel GDPR 
La AI e' una nuova tecnologia che rende il dato praticamente quasi sempre dato personale, perche' anche se anonimizzato e' possibile deanonimizzarlo no?
Infatti la AI e' in grado di:
1. fare la *re-personalization* di dati anonimi, chiamata anche la **reidentificazione** degli individui che sono legati a questi dati
2. fare **inferenze** su informazioni personale aggiuntive date le informazioni personali (i dati personali) acquisiti grazie allo step precedente
#### Reidentification
  
L'intelligenza artificiale aumenta l'identificabilità dei dati *apparentemente* anonimi, poiché consente  di collegare i dati non identificati (inclusi i dati che sono stati **anonimizzati** o **pseudonimizzati**) alle persone interessate. La reidentificazione e' spesso basata su *statistical correlation* tra i **dati non-identificati** e **dati personali** che riguardano lo stesso individuo (chissa' come si ottengono i dati personali che si hanno a priori su ogni individuo).
Un esempio e':*dei politici sono stati re-identificati in un dataset anonimizzato di cronologia di browsing di 3 milioni di persone tedesche*. In sto modo e' stato possibile risalire alle loro preferenze sessuali e alle loro informazioni legate alla salute. TERRIBILE. Tutto grazie a un match tra cronologia e dati personali.

A volte si puo' reidentificare in maniera molto semplice. Ad esempio se ho un dataset de-identificato con informazioni riferite all'ammissione in ospedale e un dataset identificato con informazioni sulla persona, posso scovare un match tra quelle features che sono condivise da entrambi i datasets come *la data di compleanno, il sesso, lo ZIP code*. In sto modo posso linkare i due datasets e *reidentificare le persone*.
![[ethiscs.png]]
Le misure da prendere a riguardo possono essere:
1. assicurarsi che il dato sia de-identificato in maniere che lo rendono piu' difficile da *reidentificare*
2. fare in modo che ci siano delle *misure di sicurezza* che limitino o vietino il rilascio di dati **identificati** che contribuiscono nella reidentificazione di una persona.
#### Inferire dati personali
I sistemi di AI sono in grado di inferire nuove informazioni riguardo a un data subject, utilizzando come input personal data. 
La domanda e': "Queste nuove informazioni che sono **inferite** possono essere considerate come nuova *personal data*?". Conta che comunque e' un output di un modello di ML, quindi e' affidabile fino a una certa.
Perche' se la risposta e' si', allora questo meccanismo di *inferenza* tramite AI verrebbe classificato come **processing** del personal data, e tutte le consegueneze scritte nel GDPR verrebbero applicate a questo.

Alcuni indizi su se le informazioni dedotte automaticamente sono considerabili *personal data* possono essere ottenuti considerando cosa la legge dice delle informazioni **dedotte dall'uomo**: c'è **incertezza** su se le affermazioni riguardanti gli individui, **risultanti da inferenze e ragionamenti umani**, possano essere considerate dati personali.

Questa questione è stata esaminata dalla Corte di Giustizia dell'Unione Europea nei casi congiunti C-141 e 372/12, dove è stato **NEGATO** che l'*analisi legale*, effettuata dall'**ufficiale** competente, su una domanda di **permesso di soggiorno** potesse essere considerata *dati personali*. 
Secondo la Corte di Giustizia dell'Unione Europea, **SOLO i dati su cui si basava l'analisi** (i dati di input relativi al richiedente) così come **la conclusione finale dell'analisi** (la constatazione che la domanda doveva essere respinta) **dovevano essere considerati dati personali**.

Nell'articolo 29 WP viene detto che nel caso *di automated inference* (quindi di profiling) i data subjects hanno il **diritto** di accedere sia **l'input data** che le **conclusioni** che sono state automaticamente inferite dall'input data.


## Articolo 4 - Profiling
```
'profiling'
qualsiasi forma di elaborazione automatizzata di dati personali che consiste nell'uso di dati personali per valutare determinati aspetti personali relativi a una persona fisica, in particolare per analizzare o prevedere aspetti riguardanti la prestazione lavorativa, la situazione economica, la salute, le preferenze personali, gli interessi, l'affidabilità, il comportamento, la posizione o i movimenti di quella persona fisica.
```
E' legato ai dati personali.
Nota che il profiling  riguarda aspetti legali, in quanto mi puo' predirre aspetti che riguardano la prestazione lavorativa, la situazione economica, ecc...
Quindi e' molto pericolosa. Entra in sfere tutelate a livello legale.

L'Articolo 29 WP dice che la profilazione ha come obiettivo quello di classificare le persone in categorie di gruppi che condividono le features che vengono *predictate, inferte*. Quindi praticamente permettono di clusterizzare le persone in gruppi.

### AI  e profiling
L'AI ovviamente si presta a questa cosa.
Un esempio di profiling con un AI system e':
* quanto e' affidabile prestare dei soldi a una persona basandosi su *records* legati alla salute, alle abitudini di quella persona e alle sue condizioni sociali
AI potrebbe anche far *profiling* imparando correlazioni riguardo alla propensione di una particolare persona di reagire in un determinato modo sotto certi stimoli. 


## Dato inferito come dato personale
E' importante notare questa cosa ad ogni modo: non c'entra molto ma c'e' una slide chiamata  *Inferences as personal data* che ne parla.
Si dice che c'e' da distinguere le *correlazioni che sono catturate da un modello allenato* e *i risultati dell'applicare questo modello su dei personal data*.
Ho che il modello imparato non contiene piu' **personal data** una volta finito l'allenamento, ma solo pesi. Le correlazioni che sono **embeddate** all'interno del modello non sono **dati personali** in quanto si applicano a TUTTI gli individui che condividono una simile caratteristica. Posso accedere dei *group data* piuttosto che dei *personal data*. Dei *group data* perche' le correlazioni encodate mi linkano un determinato gruppo (magari quello che *vive in un quartiere povero*) a un determinato outcome del modello (un prestito piu' basso della banca).
Quando pero' **applico** il modello a una specifica persona partendo dai suoi personal data, allora li' si che l'inferenza ritornata dal modello e' considerabile **personal data**.

## Diritti sulle inferenze
Praticamente, visto che abbiamo visto (non so come siamo arrivati a questa conclusione pero' ok, prima era un DUBBIO) che il *data inferito* che riguarda un individuo e' comunque considerabile **personal data** sotto il GDPR, almeno quando sono utilizzati per derivare delle conclusioni su cui e' possibile fare azioni, i **diritti sulla protezione dei dati dovrebbero applicarsi anche con queste inferred data**.

Anche se abbiamo gia' detto sta cosa ho che i *data subjects* hanno il diritto di accedere difatti sia i personal data usati come input per l'inference sia l'inferred data.

### Diritto alla rettifica
C'e' anche il diritto a rettificare l'*inferred information*, quidni cambiandola nel caso in cui giudicata errata, da parte dei *data subjects*. Questo diritto esiste sia nel caso in cui l'informazione inferred e' **verificabile**, quindi quando la sua correttezza puo' essere oggettivamente determinata, ma anche quando l'informazione e' l'outcome di una probabilistic inference non verificabile. Non e' chiaro come e' possibile rettificare questa cosa, la Prof dice **fornendo piu' data, additional data, che fornisca dellle evidenze per dire che io sono in grado di ridare il prestito dalla banca anche se la AI system dice di no**. Non ho il diritto di cambiare l'OUTCOME direttamente fra, NO. Posso farlo solo se fornisco delle evidenze che cambiano l'input data, che contestino l'input data perche' inesatto.

### Diritto alla REASONABLE INFERENCE
Il diritto che ogni inference che riguarda una persona sia ottenuta tramite una inference automatica che sia **ragionavole e che rispoetti standard etici e epistemici**. E' stato discusso che una inferenza e' **ragionevole** quando soddisfa i seguenti criteri:
1. **accettabilita'**: l'input data deve essere composto da features che siano accettabili a livello normativo come basi per inferenze legate a **individui** (quindi con features proibite che vengono escluse, come l'orientamento sessuale)
2. **rilevanza**: l'informazione inferta dovrebbe essere rilevante per **la decisione che poi ne deve conseguire** (esempio, la AI predicta uno score per una persona, questo score viene poi utilizzato per vedere se dare o meno un prestito (azione )a quella persona e cio' va bene. Ma se tipo l'AI predice l'etnia QUESTA COSA NON PUO' ESSER poi usata per fare una decisione su se dare o meno il prestito)
3. **affidabilita'**: l'input data, il training set, i metodi per processarli devono essere statisticamente affidabili. Tipo l'input data non puo' esser legata a inferenced data lol.
Al contrario, dovrebbe essere **proibito** ai **controller** di basare le proprie decisioni su **inferenze non ragionevoli** e dovrebbero avere l'**obbligo** di dimostrare che queste inferenze in verita' lo **siano** (mostrando che sono *accettabili*, *rilevanti* e *affidabili*).

Infatti la **proibizione** delle *inferenze automatiche non ragionevoli* dovrebbe applicarsi solo alle inferenze che porteranno a **decisioni che riguarderanno/avranno un effetto su il data subject**. La proibizione **non** deve avvenire nel caso in cui si faccia ad esempio **ricerca scientifica**.

## Cosa dice il GDPR sul Consenso
Secondo l'articolo **4, nell'11-esimo punto** (che non e' presente sopra perche' l'articolo era troppo lungo), vien detto che:

* il *consenso* del *data subject* significa  qualsiasi *indicazione* liberamente data, specifica, informativa e non equivoca dei desideri del **data subject** mediante la quale lui, con una dichiarazione o con un'azione affermativa chiara, manifesta l'**accordo** al **trattamento dei dati personali che lo riguardano**.*


Nota che c'e' scritto che il consenso deve essere *dato liberamente*, questo e' uno dei requisiti. Dato liberamente significa proprio quello che fa il Washington Journal:
Il Washington Journal da' il servizio giornalistico **gratis** pero' solo a patto che dai il **consenso** per il trattamento dei tuoi dati. Questi dati vengono poi utilizzati per fare *advertising* su quella specifica *data subject*. Altrimenti **devi pagare per leggere le notizie**.
Quindi e' dato liberamente perche' c'e' un'alternativa (e' possibile mantenere la propria privacy **pagando**). Quando ad esempio c'e' un *muro di cookie* che *non mi permette di accedere al sito*, senza un'alternativa, allora li' il **consenso NON e' liberamente dato**.

C'e' anche scritto che il consenso deve essere **specifico**, il che significa che l'utente deve essere informato della categoria specifica di dato che sto collezionando (tipo la religione che professo) e del motivo per cui questo *personal data* verra' utilizzato (boh il motivo per cui e' utile collezionare la religione professata).

Deve anche essere **informativa**. Solitamente si dice che spesso *si collezionano informazioni dell'utente per migliorare il servizio, customizzare l'esperienza in base all'utente, ecc...* (anche se su Whatsapp non ho mai visto nulla di customizzato in base  all'utente). Spesso manca questo tipo di informazioni, riguardo a come verranno usati i miei dati

L'indicazione deve essere *non ambigua* e legata ai desideri del *data subject*. Non ambigua significa che il consenso **non puo' essere dedotto/inferto** (inferred) dal comportamento del *data subject* ma deve essere chiaramente firmato/accettato/cliccato dall'utente. Spesso nelle privacy policies c'e' scritto che *anche solo stando spesso nel sito per molto tempo, l'utente sta consentendo all'utilizzo dei suoi dati personali*, il che e' **assurdo**.

Nell'**Articolo 7** sono scritte le **condizioni** per il *consenso*:
1. Quando il *processing* e' basato sul consenso, allora il controller deve essere in grado di dimostrare che quel *data subject* ha acconsentito al processo dei propri *personal data*
2. Se il *data subject* da' il consenso nel *contesto* di una dichiarazione scritta in cui e' scritto **anche altro**, la richiesta per il consento deve essere presentata in un mood che e' chiaramente distinguibile dall'**altro** trattato nella dichiarazione scritta, utilizzando una forma accessibile e parole chiare per descriverlo.
3. Il *data subject* deve avere la possibilita' di **andarsene e togliere il proprio consenso in ogni momento**. Il togliersi non rende illegale l'aver processato i suoi dati personali **prima** del *withdraw*. Questa azione di *withdraw* deve essere facile da compiere
4.   Quando si valuta se il consenso è dato liberamente, si terrà massimamente conto, tra l'altro, se l'esecuzione di un contratto, compresa la prestazione di un servizio, è condizionata rispetto al consenso dato sul *processing* dei dati personali che **non sono necessari** per l'esecuzione di quel contratto. Cioe' se praticamente nel consenso sono presenti dei consensi *inutili, non necessari*, allora il consenso non e' dato liberamente.
## Informazioni da fornire al Data Subject
Secondo L'Articolo 13-14, o secondo il rectal 42 GDPR, o secondo l'articolo 29 del WP Guidelines on conset (la pensano tutti alla stessa maniera, o e' una summa di tutte queste cose) le informazioni da fornire al data subjects DEVONO ESSERE LE SEGUENTI:
1. **identita' del controller** e i **contatti**. Nel caso di Facebook, cio' che questo deve fornire al *data subject* e': come controller la societa' dietro Facebook, l'indirizzo dello stabilimento principale di Facebook, l'email, il numero di telefono per assistenza ecc... Comunque  i contatti che vengono messi **dipendono** dal servizio fornito. Se sono una piccola compagnia, basta magari un numero di telefono. Nel caso di Facebook invece servono molti altri dettagli di contatto.
2. **dettagli di contatto** del *Responsabile della Protezione dei dati*
3. **Gli scopi** del processing del personal data collezionato dall'utente. Qua serve il link specifico tra ogni categoria di data con lo scopo per quella categoria di data. Non posso scrivere scopi generici eh, devo essere molto specifico.
4. Le basi legali secondo cui e' possibile fare *processing*. Come per sopra, non devono esserci dei motivi generici. 
5. Le categorie del personal data che viene collezionato (se e' inferto/dedotto o meno)
6. I **recipienti** (quelli che condividono i dati personali con altri) come ad esempio Whatsapp e Facebook che condividono informazioni tra di loro: magari Whatsapp vede che io ho mio cugino nella lista dei contatti e manda questa info a Facebook. Lui poi mi *consigliera'* come amicizia, come *persona che potrebbe interessarmi*, proprio questo mio cugino. In sto caso ho dei recipienti. E quindi e' giusto che mi vengano mostrati a me *data subject*. Oppure si dice di mostrare le **categorie di recipienti**, quindi specificando non l'azienda specifica con cui io controller voglio condividere il personal data del *data subject* ma la **categoria** in cui questa azienda rientra.
7. il periodo per il quale il *personal data* verra' salvato o il criterio per determinare questo periodo, nel caso in cui non fosse possibile sapere il periodo esatto
8. l'esistenza del diritto di richiedere al controller l'**accesso** al proprio personal data, ma anche la **rettifica** del personal data o l'**eliminiazione** addirittura. Anche l'esistenza della **restrizione** del *processing* (quindi levando il processing di una determinata feature per un determinato scopo, perche' non mi piace piu' che questa sia usata per quello scopo li), e il **diritto della portabilita' del dato** (se vuoi portarla su un'altra piattaforma puoi farlo)
9. l'esistenza del diritto a **presentare un reclamo**
10. la sorgente da cui il dato personale ha avuto origine. Magari io controller non ho collezionato il data da te, ma l'ho collezionato da una terza parte (da qualcun altro), e te lo devo dire. Devo anche dirti se c'e' dell'inferred data ovviamente tra i tuoi dati personali, perche' riguardano l'origine del dato personale.
11. L'esistenza di *automated decision making*, che include anche la possibilita' di **profiling**.

## Articolo 17 - Il diritto a Eliminare (il diritto all'oblio)
Ho il diritto di ottenere dal controller l'eliminazione del mio *personal data*, senza **delay**, 

1. Il controller e' obbligato a cancellare un personal data quando uno dei seguenti punti si applica:
	1. quando il dato personale non e' piu' necessario in relazione ai motivi per cui e' stato collezionato  o processato. Questo viene constatato e se e' vero allora viene eliminato il dato.
	2. quando il *data subject* richiede un *widthdraw del consenso* sul proprio *personal data* **E** non c'e' piu' una base giuridica per fare *processing*.
	3. quando ho il data subject che si oppone al processing e non ci sono basi legittime per contnuare il processing. Un esempio e' magari il fatto che un minore si scarica e fa un profilo tiktok senza il consenso dei genitori. I genitori lo vengono a sapere e chiedono di cancellare i dati del minore, a quel punto **la cancellazione** deve avvenire perche' non ci sono basi legittime per tenere quel dato (quello e' un minore e non posso collezionare il dato sul minore)
	4. il dato personale e' stato processato in maniera **illegale**
2. il controller a cui e' richiesto di cancellare i dati, deve **informare**, se richiesto dal *data subject*, gli altri controller a cui magari ha condiviso questi personal data (o che hanno preso il dato/l'info personale da te, perche' l'hai reso disponibile al pubblico magari) e dir loro di **cancellare** qualsiasi **link, copia o replica dei dati personali in loro possesso**. 
Questi due punti riguardanti la cancellazione del dato **non  si possono applicare quando il preprocessing e' necessario per i seguenti motivi**:
1. per esercitare il diritto alla **liberta' di espressione e di informazione**. Ad esempio un giornale che pubblica un articolo su una persona (un personaggio pubblico), utilizzando i suoi dati personali, per informare gli  altri su questa persona. In quel caso questa persona **non** puo' richiedere la cancellazione dei suoi dati.
2. per motivi legati a **interessi pubblici**. Tipo nel caso della pandemia, se una persona avesse chiesto di eliminare il proprio dato medico sulla contrazione del Covid, non sarebbe stato possibile eliminarlo perche' e' legato a un interesse pubblico di vitale importanza (uscire dalla pandemia).
3. per motivi legati a motivi **di ricerca scientifica e storica o per motivi statistici** (tipo l'evoluzione della pandemia nel corso degli anni a seconda dello stato).
4. per la difesa in caso di azioni legali. Esempio se sono un servizio di taxi e tu hai preso il mio taxi, e tu vuoi cancellare l'informazione di aver preso un determinato taxi a una certa ora, allora io servizio posso dirti di no perche' avere quella informazione mi tutela da magari te che poi dici *ma no, in verita' in quella situazione io non l'ho mai preso il taxi*.

## Articolo 22 - decision-making individuale automatizzato attraverso profiling
Questo articolo ha all'interno i seguenti paragrafi:
1. il *data subject* deve avere il diritto di non essere **soggetto a una decisione basata solamente su un processo automatico** (**profiling** incluso) che produce **effetti legali** che riguarda il *data object* o che comunque vanno a *affectarla* a livello significativo. Se ad esempio uso la AI per vedere se tu hai un alto rischio di *recidivita'*, allora questa predizione ha un effetto legale. Per quanto riguarda invece l'affecting a livello significativo e' ad esempio un'AI che ti predice quanto sei affidabile per chiedere un prestito.
2. il paragrafo (1) non si applica se la decisione:
   1. e' necessaria per entrare in un **contratto** tra il *data subject* e il *data controller*.
   2. e' autorizzato dall'Unione Europea
   3. e' basato sul consenso esplicito del *data subject*
Questo e' un articolo fondamentale per l'AI, per me. :3.


C'e' stato un dibattito dietro questo articolo, perche' **quasi mai c'e' una decisione che e' SOLAMENTE basata su un sistema di AI automatico**, c'e' sempre un umano nel mezzo che supervisiona.
Quindi ci sono delle condizioni per far si che una decisione automatica sia proibita:
1. deve essere presa una decisione verso una persona
2. deve essere basata solamente su automated processing: quindi richiede che l'umano non eserciti nessuna influenza sull'outcome del processo di decision making. Questa condizione non e' soddisfatta quando il sistema e' usato come un *decision-support tool* che viene revisionato da un umano (per l'appunto il motivo del dibattito)
3. deve includere profiling: quindi richiede che il processo automatico di determinare una decisione includa il *profiling*.
4. deve avere un effetto legale o comunque significativo

## Articolo 21 (1) e (2)
Questo articolo si concentra contro il profiling e il *direct marketing*.
Il 21 (1) dice che c'e' il diritto a **obiettare** anche nel caso del profiling, quidni il data subject ha il diritto di obiettare in ogni momento al *processing* del *personal data* che lo riguarda, incluso il profiling (come processing si intende).
Il profiling e' poi trattato nel contesto del **direct marketing** nell'Articolo 21 (2) e anche qui ho un diritto all'**obiettare**: ogni volta che il personal data e' processato per scopi legati al **direct marketing**, allora il data subject ha il diritto a **obiettare** in ogni momento al processing di questo data in questo contesto di *direct marketing*. Cio' significa che puo' obiettare **anche** per il **profiling** che e' stato utilizzato per performare questo **direct marketing**.
I controllers devono fornire dalla loro dei modi facile per facilitare l'**esercizio** di questo **diritto** da parte dei *data subjects*.

La Prof. dice: "Anche se c'e' una base legale secondo cui un sito puo' portare avanti con un data subject forme di  **direct marketing**, magari mandando mails al *data subject* secondo il principio dell'**interesse legittimo** (e quindi secondo questo principio il sito puo' processare i personal data del data subject), **in ogni momento** il data subject ha il **diritto** di fermare questo *direct marketing* e quindi anche il *profiling* che serve per fare quel direct marketing."

## Informazione su automated decision making
Nell'Articolo 13 (2) e 14(2) e' presente una regolamentazione sull'*automated decision making*.
Il controller ha l'obbligo di fornire:
1. L'informazione dell' **esistenza di automated decision-making**, incluso il *profiling*
2. almeno un'informazione significativa sulla **logica coinvolta** e delle **conseguenze** di questo processing sul *data subject*.  Per quanto riguarda la *logica coinvolta*, e' per dire che il controller mi deve dire come funziona l'*automated decision-making* a livello di logica sotto. C'e' un problema di **explainability**, nel caso in cui il modello sia un **black box model**.
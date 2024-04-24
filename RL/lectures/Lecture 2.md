# n armed decision problem
Ho praticamente tante slot machines e il problema e' capire qual'e' la migliore di queste  (in verita' e' di massimizzare l'expected reward dopo un certo numero di timesteps).
Qui c'e' tutto il dramma del balance tra exploration e exploitation, perche' per ogni slot machine non ho un reward deterministico ma questo e' scelto con una certa probabilita' dalla slot machine: che ne so magari la slot machine 1 e' tale che il 20% delle  volte mi da' reward 1 e l' 80% mi da' il reward 10.
Quindi magari mentre provo a stimare l'action value della slot machine 1 questa mi da' come reward 1 la prima volta e potrei credere che fa cagare, invece in verita' da' come reward 10 il piu delle volte quindi non e' malaccia.
Ad ogni modo per stimare gli action values faccio una average sui reward ricevuti da una determinata slot machine e quindi alla fine li stimo tutti in maniera accurata dopo molti steps e ho risolto il problema.
Ma come faccio se il numero di steps a mia disposizione e' piccolo?
Devo inventarmi qualcosa.
Diciamo che c'e' da capire quale slot machine scegliere a ogni step, per collezionarne il reward.
C'e' la scelta non greedy che sceglie sempre la slot machine che ha l'action value piu' alto (fa solo exploitation).  Non e' detto che sia sempre la stessa perche' magari facendo exploitation mi rendo conto che quella slot machine era un pacco. Quindi ne scelgo un'altra con l'action value piu' alto e cosi' via.
C'e' poi la scelta $\epsilon$-greedy che con probabilita' $\epsilon$ a ogni step si scegle un'azione randomica in modo tale da fare exploration pure (con probabilita 1- $\epsilon$) si sceglie la migliore invece.
Solitamente l'$\epsilon$-greedy  performa meglio del non greedy, soprattutto quando i reward di una slot machine hanno una varianza alta (tipo che ne so la slot machine 3 ha come rewards possibili 1 e 30000). Nel caso della non greedy magari appena esce fuori 1, sta strategia pensa che sta slot faccia cagare, invece butta fuori rewards da 30000 sta piccola, e non lo scopriro' mai. 
La non greedy invece potrebbe performare meglio nel caso in cui i rewards sono deterministici (quindi a ogni slot ho un numero e basta), quindi qua la non greedy tira solo una volta la leva di ogni slot, vede quella col reward piu alto e niente ha finito (seleziona quella fino alla fine dei tempi) (produce quindi un expected reward maggiore). Ma anche in sto caso potrei avere un caso deterministico ma non stazionario, quindi i rewards dellle slot potrebbero cambiare nel tempo, e con il non greedy non lo sapro' mai.

## iterative method
c'e' un modo per non dover tenere tutti i reward ottenuti fino a quel momento in memoria  per una slot machine ma di doversi salvare soltanto la stima fino a quel momento e il reward ottenuto a tempo t (sta sulle slides).

## Come affrontare il non stationary problem
Se i rewards cambiano nel tempo in una slot, devo trovare un modo di dare piu peso ai rewards incontrati ora rispetto a quelli passati.
Lo faccio non mettendo 1/n come step size (guarda formula iterative method di prima), che cambia nel tempo e da' sempre meno peso ai nuovi reward, ma mettendo una step size costante $\alpha$ .
C'e' una formula che mi mostra che con $\alpha$  costante ho una weighted average (n average resulting from the [multiplication](https://www.google.it/search?sxsrf=APwXEddbS_arJ3H5FbROH4t3Ggbtv224eg:1682684559251&q=multiplication&si=AMnBZoGo5JO1-RMo7Hmdi0jk7E03rYJk3srYZrtPE7Kei-IyZVK_7TRqIgs2GpBsKqIzlMfRSq_1s3BYEAQAJKa4Qqc5e6P7_JC3jEnLSTMnYzt1lFwUEi4%3D&expnd=1) of each component by a factor reflecting its importance) sui passati rewards e sulla stima iniziale Q1. I pesi sommati danno come risultato 1.

## Optimistic initial values
C'e' una differenza tra avere 1/n come step size e $\alpha$, ovvero che nel primo caso la prima stima che faccio dell'action value di una slot (Q1), appena samplo il primo reward, viene cestinata.
Nel secondo caso invece rimane, anche se col tempo perde di importanza perche' e' moltiplicata a (1-$\alpha$)<sup>k</sup> con k numero di steps.
Quindi qui e' importante dare un buon valore a Q1:3.
C'e' un trick che permette di avere molta explorationa anche se uso non greedy strategy che e' chiamata optimistic initial values. Praticamente do' ai Q1 un valore di action value molto alto (ottimista). In sto modo la strategia sara' inizialmente sempre delusa dalla scelta che fa, perche risultera' in un Q stimato sempre piu piccolo rispetto agli altri (anche usando la greedy selection). Overall performa meglio di una realistic epsilon greedy in cui pero' do' ai Q1 valore 0. (non molto buono pero' per i non stationary problems).

## Upper-confidence-bound action selection
Praticamente a noi da' fastidio che nell'epsilon greedy search l'azione che viene selezionata quando c'e' la scelta (nel caso in cui si sta nella probabilita' dell'epsilon) e' **randomica**. Vorremmo qualcosa di piu' intelligente, tipo selezionare un'azione di una slot machine promettente, non di una a caso.
Quindi i ragazzi si sono inventati questa cosina:
![[upper_confidence_bound.PNG]]
Praticamente si sceglie l'azione seguendo questo argmax (non c'e' piu' l'epsilon come puoi notare). Nota che la radice e cio che c'e' dentro rappresenta l'incertezza sulla stima del action value dell'azione _a_. 
Ogni volta che seleziono _a_ l'incertezza diminuisce: N_t (a) aumenta, di conseguenza la radice da' un valore piu' basso, quindi e' meno probabile che venga selezionata sempre st'azione al passo successivo (si da' spazio a azioni piu incerte insomma). Il fatto che ho t invece al nominatore serve perche' ogni volta che scelgo un'azione che non e' _a_, t aumenta al numeratore e quindi aumenta l'incertezza. Di conseguenza e' piu' probabile che al prossimo giro _a_ venga selezionata.
GENIALE

## Contextual bandits
Qua voglio imparare una policy a differenza di prima dove mi interessavano solo gli action values. Qui infatti ho stati e voglio un mapping (la policy per l'appunto) da stato a distribuzione di prob. sulle azioni. A differenza di un problema di RL pero' qua l'azione che scelgo non va a modificare lo stato.
Contextual/associative because the action i choose depend on the state i'm in in that moment.
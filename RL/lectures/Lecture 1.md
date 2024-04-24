In RL voglio arrivare a un goal che consiste **sempre** nel massimizzare l' **_expected cumulative reward_**.
Quindi io voglio che l'AI selezioni le azioni in modo tale che massimizzi il **reward totale futuro**.
# History e State
## History  
E' la sequenza di osservazioni, azioni e rewards
> H<sub>t</sub> = A<sub>1</sub>, O<sub>1</sub>, R<sub>1</sub>, A<sub>2</sub>, O<sub>2</sub>, R<sub>2</sub>,  ... , A<sub>t</sub>, O<sub>t</sub>, R<sub>t</sub>

Cosa avviene dopo **dipende** dalla storia.
Ma non e' poi cosi' utile da solo
## State
E' una cosa che mi permette di capire cosa accadra' all'enviroment nel/i prossimo/i step/steps.
E' una **funzione** della history:
> S<sub>t</sub> = f(H<sub>t</sub>)

Ci sono diversi tipi di stati:
* **enviroment state** S<sub>t</sub><sup>e</sup>: e' la rappresentazione interna dell'enviroment.  E' quello  a cui l'agente _vorrebbe_ avere accesso, ma non sempre e' possibile. Questa mi dice in maniera precisa cosa accadra' prossimamente. Il problema e' che l'agente osserva un mondo a lui spesso non accessibile completamente (la rappresentazione interna del mondo non e' **totalmente** osservabile).
* **agent state** S<sub>t</sub><sup>a</sup>: e' la rappresentazione di cio' che e' accaduto fino a quel momento all'agente, che cattura tutto cio' che all'agente puo' servire per fare la prossima azione. Questa e' la rappresentazione interna dell'agente, in cui questo ha processato tutte le osservazioni, i reward e le azioni fatte fino a quel momento. Proprio per lui vale che  **S<sub>t</sub><sup>a</sup> = f(H<sub>t</sub>)**.
* **information state / Markov state**: questo tipo di stato e' tale che contiene tutte le informazioni utili che sono prendibili dalla **history**.
> Uno stato S<sub>t</sub> e' **Markov** sse P\[S<sub>t+1</sub> | S<sub>t</sub>\] = P\[S<sub>t+1</sub> | S<sub>1</sub>, ... , S<sub>t</sub>\]

   cio' significa che S<sub>t</sub> deve avere all'interno tutto cio' che e' successo prima (in un qualche modo compatto), in modo tale che il futuro e' indipendente da cio' che e' successo nel passato, dato il presente. **Non ho quindi bisogno della history da 1 a t, se ho S<sub>t</sub>**.
L' **enviroment state** e' Markov, perche' per definizione caratterizza in modo completo cio' che avverra' prossimamente (ho solo bisogno di esso a tempo t, non ho bisogno degli stati precedenti).
Inoltre  **H<sub>t</sub>** e' anch'essa Markov (se uso come stato tutta la history fino al tempo t, ho che e' Markov, perche' ho tutto cio' accaduto fino a quel momento).
Questa cosa solitamente non e' molto utile, perche' vorrei una rappresentazione piu' piccina come agent state (e' quello che mi interessa fare).

# Fully Observable Enviroments
In questi tipi di enviroment, all'agente e' concesso di vedere l'**enviroment state** (molto fortunato). Quindi in questo caso ho che cio' che viene osservato a tempo t coincide con l'enviroment state, che concide con l'agent state:
> O<sub>t</sub> =  S<sub>t</sub><sup>e</sup>  = S<sub>t</sub><sup>a</sup>

Inoltre essendo un enviroment state un Markov state ho che env state = markov state = agent state.
In questo caso ho che questo problema di RL e' chiamato di conseguenza **Markov decision process** (MDP), perche' lavoro con l'enviroment state che e' markov.

# Partially Observable Enviroments
L'agente in questo caso non ha accesso alla rappresentazione interna totale dell'enviroment (gli manca qualcosa della full picture). Tipo un agente che gioca a poker non ha accesso alle carte che non vede :).
Qui **agent state != enviroment state**. Chiamo questo problema **partially observable MDP** (POMDP).
Quindi l'agente deve costruire il suo proprio **stato** in qualche modo, perche' non ce l'ha gratis dall'enviroment.
Si possono usare diverse strategie:
* Utilizzare la **history completa**: S<sub>t</sub><sup>a</sup> = H<sub>t</sub>
* Utilizzare un approccio probabilistico in cui a ogni step t, creo una distribuzione di probabilita' sugli  **enviroment states**  (con probabilita' 0.1  S<sub>t</sub><sup>e</sup>  = s1, con probabilita' 0.4 S<sub>t</sub><sup>e</sup>  = s2, ecc...). Questo vettore di probabilita' definisce il mio stato S<sub>t</sub><sup>a</sup>.
* Utilizzare recurrent neural network (non chiaro lol) : tipo dallo stato passato creo il  nuovo stato tramite combinazione lineare con dei pesi e una funzione che aggiunge non linearita' esterna.

# Dentro un RL agent
* **policy**: come l'agent prende le decisioni (il behaviour)
* **value function**: mi dice quanto e' buono stare in un particolare stato (a livello di reward che avro' se sto/se vado in sto stato)
* **model**: e' la rappresentazione che l'agente ha dell'enviroment in cui sta operando (e' una cosa che l'agente crea per capire come funziona l'env).
## Policy
e' una map dallo stato all'azione. Solitamente e' stocastica (non greedy/deterministica).
* deterministica:  a = $\pi$(s)
* stocastica: $\pi$(a|s) = P\[A = a | S = s\]
## Value function
e' fondamentalmente una predizione attribuita a uno stato, dell'expected future reward stando in quello stato. Quindi fondamentalmente mi dice quanto e' buono stare in un determinato stato (**ricorda che e' legato al reward sempre**).
Ovviamente la value function e' legata alla policy (come mi comporto quando scelgo azioni) che sto utilizzando nel momento in cui calcolo la value function.
>v_$\pi$ (s) = E_$\pi$ \[ R<sub>t</sub>  +  $\gamma$ R<sub>t + 1</sub>  +  $\gamma$<sup>2</sup> R<sub>t + 2</sub> + ...  | S<sub>t </sub> = s]

Quindi la value function di s e' l'expected value della somma dei reward futuri partendo da sto stato s e seguendo la policy $\pi$ (con discounting $\gamma$ ).

## Model
posso creare un modello definendo 
* le **Transitions** P, che danno una distribuzione di probabilita' sul prossimo stato dato lo stato corrente e l'azione corrente
* i **Rewards** R, che dicono il valore del prossimo reward (quello **immediato**) dati lo stato e l'azione correnti.

Importante: non per forza devo creare un **model**: ci son molti **model free problems**.
Solitamente se lo creo, e' legato ai reward e basta (almeno questo e' cio' che ho visto fin ora), in quanto le transitions P difficilmente sono disponibili/modellabili in maniera accurata e soddisfacente.
Nota che se conosco l'enviroment posso definire le transitions in maniera chiara (dando ad esempio alla probabilita di andare in uno stato s_t+1, nel caso in cui voglio andare dritto ma c'e' un muro e non si puo proseguire,  il valore 0).

 Un problema RL puo' essere:
 * Value Based: quindi utilizzo la **value function** e basta, come policy ho  quella greedy che sceglie l'azione che porta nello stato con value function maggiore. Praticamenta non salva nessuna policy, non lavora con le policy.
 * Policy Based: invece di rappresentare la value function, rappresento esplicitamente la policy (quindi gioco con la policy, senza dover calcolare/ricordare i value function)
 * Actor Critic: salva e utilizza entrambe.
Posso avere problemi RL di questa natura:
* Model Free: non conosco/non modello le dinamiche dell'enviroment (con quale probabilita' andro' nel prossimo stato dato lo stato in cui sono e l'azione scelta, ecc...). Vado direttamente a vedere policy e/o value function. NON c'e' modello.
* Model Based: all'inizio modello le dinamiche dell'enviroment.
# Problemi in RL
* **LEARNING** : L'agente non sa come funziona l'enviroment in cui viene messo (chiamato Reinforcement Learning problem). La soluzione e' interagire con esso (try and error approach solitamente).
* **PLANNING** : Stavolta l'agente sa tutte le regole dell'enviroment, sa come funziona e tutto (conosce quindi il modello dell'enviroment). Facendo computazione stavolta (senza interagire) l'agente puo' migliorare la propria policy. Semplicemente facendo calcoli. Qui posso fare tree search, thinking ahead (perche so gia dove andro partendo dallo stato in cui sono) e far reasoning senza dover interagire.
Io prima faccio learning e poi planning se non conosco l'enviroment.

**Exploration e Exploitation**
Voglio capire qual e' una buona policy. Lo posso fare facendo exploitation (tipo andando in una zona magari gia visitata in cui so che il reward e' alto e approfondirla, aggiustare i valori del value function). 
Ma serve anche exploration per capire meglio dell'enviroment, cosi vedo anche se ci sono altre zone che hanno reward magari  piu alti di quelli trovati prima.

**Prediction and control**
Voglio con la prediction valutare la policy che ho, calcolando magari i value function per vedere come e'.  
Con control voglio migliorare la policy.
Solitamente si fanno insieme nel senso che valuto una policy che cambia nel tempo (sembra illogico).
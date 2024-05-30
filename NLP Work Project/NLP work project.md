Cio' che mi e' richiesto di fare e' la seguente cosa:
Voglio capire quali sono i papers che sono piu' interessanti. C'e' una intuizione per cui i papers piu' interessanti sono quelli che contengono una quantita' maggiore/piu' alta di *argumentative topics*. Per approcciare questo problema ci sono almeno due modi:
1.  Fare **classificazione** degli *argumentative topics* e poi contarli
2.  Potremmo evitare di capire per ogni frase se questa e' un *argumentative topic*, potremmo piuttosto  voler solo **stimare direttamente** l'*ammount* dell'*augumentative content*. In sto modo quantifico il numero di argomenti dentro il testo
Ho due papers in cui si fa classificazione di *augumentative topics* che sono i seguenti
[https://www.sciencedirect.com/science/article/abs/pii/S0933365721000919?via%3Dihub](https://www.sciencedirect.com/science/article/abs/pii/S0933365721000919?via%3Dihub)
[https://www.ijcai.org/proceedings/2022/0859.pdf](https://www.ijcai.org/proceedings/2022/0859.pdf)

Mentre ho una libreria che mi permette di quantificare il numero di *augumentative topics* in un testo:
[https://ceur-ws.org/Vol-3180/paper-146.pdf](https://ceur-ws.org/Vol-3180/paper-146.pdf)

Per fare un'analisi che metta a paragone i due approcci mi posso concentrare sugli stessi datasets che sono citati in Cabrio e Villata.

## RIASSUNTO del paper di Villata
**Enhancing evidence-based medicine with natural language argumentative analysis of clinical trials**

Hanno sviluppato
1. Un modello di *Augumentative Mining* per estrarre e classificare *componenti argomentativi* (i.e., evidenze  e affermazioni (evidences and claims) durante il processo)
2. Un altro modulo legato all'analisi dell'outcome che identifica e classifica gli effetti di un intervento (credo legato a componenti argomentativi a sto punto) nel risultato del **processo**
Citano un paper per la **detection** di elementi PICO che si trova al link
https://aclanthology.org/W18-2308/
e che si chiama  [PICO Element Detection in Medical Text via Long Short-Term Memory Neural Networks](https://aclanthology.org/W18-2308.pdf)
Mi dice che questo paper e' legato a questo use case: 
il modello in esso sviluppato ha come obiettivo quello di estrarre da documenti di test informazioni da presentare poi in una maniera strutturata, facile da analizzare.

Ci sono 3 papers (da 7 a 10) che citano che sono legati a Agumentation Mining, ovvero alla task per cui si deve *detectare, classificare e valutare* la qualita' di strutture agumentative nel testo. Tasks standard che si fanno in AM sono:
1. detectare argument components (evidence and claims)
2. predictare le relazioni che tengono tra gli argument components

Quello che stiamo vedendo e' qualcosa che e' diverso.

Il component detection, che immagino sia proprio la task di trovare nel testo componenti argomentativi, e' un **multi-class sequence tagging problem**.
Ora metto il focus su 
# Tecnical robustness and safety
![[ethics.png]]
Siamo qui, a questo punto.
Questo requirement e' molto grande.


E' importante che un modello, se subisce dei cambiamenti nell'input a causa di *malicious attacks*, debba essere in grado di mitigare l'impatto di questi cambiamenti sulla predizione che viene poi data in output.

E' importante capire la differenza tra:
1. **Reliability**: il modello performa in modo **simile** in ogni *test set*, finche' questi sono samplati dalla stessa distribuzione
2. **Robustness**: il modello performa SENZA **un fallimento** (quindi senza la possibilita' che avvenga una catastrofe) anche quando **le istanze del test set** sono leggermente diverse dalla *train distribution*.
3. **Resilience**: modello e' in grado di **adattarsi a inputs inaspettati** che vengono presi da *unknown distributions*, oppure il modello e' in grado di **rifiutare** una predizione quando ha una **bassa confidenza**.
Vorrei poterli ottenere tutti nel mio modello.
**NOTA** che 
1. se voglio che il design sia legato alla *robustness* allora voglio un modello che sia *insensibile* rispetto a piccoli cambiamenti nell'input
2. se voglio che il design sia legato alla *reliability* allora voglio che il modello abbia una **probabilita' di fallimento** che e' minore di una determinata *soglia*.

Ci sono dei **metodi** per fare in modo che il modello sia:
1. **Technical Robustness**
2. Safety
3. Reproducibility
## Robustness levels
Posso avere diversi livelli di *ROBUSTNESS*:
1. Livello 0: non e' presente robustness. In questo caso ho solo la generalization che nasce dalla capacita' di astrazione del modello trainato.
2. Livello 1: si e' robusti se ci sono shifts delle covariate. Con il livello 1 e' possibile *generalizzare under input shifts*
3. Level 2: il modello e' in grado di essere robusto contro un singolo attacco (tipo adversarial input)
4. Level 3: quando il mio modello e' robusto contro piu' attacchi
5. Level 4: universal robustness: il modello e' robusto contro TUTTI i RISCHI esistenti
6. Level 5: qua sono Dio, non ho ben capito


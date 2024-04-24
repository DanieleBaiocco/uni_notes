E' un optimization topic
USECASE: donazione di RENE
Spesso non si hanno i DONATORI. 
Il problema e' che deve esserci COMPATIBILITA' di sangue.
Spesso si fa un exchange spesso: KIDNEY PAIRED DONATION : io lo do' al partner di uno straniero e lo straniero lo da' al mio partner.
Ci possono esser delle chains piu' grandi di due.
CAPIRE CHI DEVE DONARE A CHI in maniera intelligente e' complicato.
Abbiamo bisogno di un DECISION SUPPORT TOOL.
Il MATCHING PROBLEM e' chiamato KIDNEY EXCHANGE PROBLEM. E' QUI che OPTIMIZATION e' TOP, perche' in sto problema ho CONSTRAINTS di compatibility e devo ottimizzare il matching del trapianto degli organi
## Formulazione del problema
Posso mettere tutto sotto forma di GRAFO
in cui ho
1. Recipient-donor _pairs_ $(r_i, d_i)$ in the programs can be seen as _nodes_ in a graph
2. The graph contains an _arc_ from pair $i$ to pair $j$ iff the $d_i$ is _compatible_ with $r_j$
In the example there are four pairs
The donor in pair 1 is compatible with the recipient in pair 2, and so on
![[dpair.png]]

Posso fare un EXCHANGE TRA 1 E 4 TRA 2 E 3. Non tra 1 e 3. Posso fare anche chain tra 1,2 e 3. Ma in generale HA PIU' SENSO fare 1-4 e 2-3 in modo tale da SALVARE TUTTI, anche se 1,2 e 3 e' una chain PIU' GRANDE

**This is enough to start defining a _combinatorial optimization problem_**
* We want to select _groups of nodes_
* No node can be _included in two groups_
* _Too large_ groups/cycles should not be considered (perche' sotto un punto di vista pratico performare tutte ste operazioni cliniche una dopo l'altra diviene troppo complicato)
* **Every group should _correspond to a cycle_**
* A group/cycle with _$n$ nodes lead to $n$ transplants_
* **We want to maximize the _total number of transplants_**
Questo e' un COMBINATORIAL OPTIMIZATION PROBLEM
Abbiamo bisogno di un modello per questo. Che mi invento? Che costruisco? Ma anche una roba semplice tipo che chiude i cerchi. 
Mi concentro su **DECISION VARIABLES**, come RAPPRESENTO le decisioni?

**Whem building a CO model, this is usually a good approach:**
* **Start by choosing how to model the _decisions_**
* **Then, consider the _constraints_ one by one**
	  - Define how to model then with the chosen variables
	  - Introduce additional variable as needed
* Then, do the same for the problem objective
**During this process, it is very common to have difficulties**
When that happens, try thinking about:
* Alternative ways to formulte the constraints
* ...But even more, _alternative ways_ to represent decisions
(RICORDI CDMO?)
**Our decision variables need to identify groups of nodes**. 
**We could use binary variables $x_{ij}$**
* $x_{ij} = 1$ iff node $i$ is part of the $j$-th cycle
* For $m$ nodes, we can have at most $n = \lfloor ^m/_2 \rfloor$ cycles
Now we can attempt to formulate the constraints
1. **No node can be included in two groups":**
$$\begin{align}
& \sum_{j=1}^n x_{ij} \leq 1 & \forall i = 1..m
\end{align}$$Ho che al massimo e' 1 SU TUTTI i possibili cycles, e 0 nel caso in cui NON VENISSE SCELTO.
2. **"Too large groups/cycles should not be considered":**
$$\begin{align}
& \sum_{i=1}^m x_{ij} \leq C & \forall j = 1..n
\end{align}$$3. **"Every group should _correspond to a cycle_"**
Questo E' VERAMENTE DIFFICILE da ensurare come CONSTRAINT. Ma in generale eh, non dipende da questa rappresentazione. Sarebbe sempre stato difficile. Invece di modellare un constraint con le decision variables che HO, CAMBIO LA MIA DECISION VARIABLE che tiene into account i CYCLES. In sto modo NON MI STO A SCERVELLARE, mi sarei bloccato altrimenti. Ora ho  **a binary $x_j$ variable for every cycle in the graph**:
* $x_{j} = 1$ iff the $j$-th cycle is chosen for surgery
* With this formulation, **groups** are **cycles by construction**
ORA pero' devo ridefinire gli altri
**"No node can be included in two groups":**
$$\begin{align}
& \sum_{j = 1}^n a_{ij} x_{ij} \leq 1 & \forall i = 1..m
\end{align}$$

* $a_{ij} = 1$ if node $i$ is in cycle $j$
* This is basically a mutual exclusion constraint
PENSA AL CONSTRAINT NON E' IMPOSSIBILE
**"Too large groups/cycles should not be considered":**
Per questo NON ABBIAMO BISOGNO di un'equazione perche' BASTA calcolare TUTTI I POSSIBILI CYCLES  e si puo fare e quelli CON TROPPI NODI, con un numero di nodi maggiore di C semplicemente NON LI INSERISCO COME VARIABILI DECISIONALI di cui voglio calcolare il valore.

**"We want to maximize the _total number of transplants_":**

$$\begin{align}
\max \ & \sum_{j = 1}^n w_{j} x_{j}
\end{align}$$

* $w_j$ is the number of transplants associated to cycle $j$
* This is our objetive function

**Therefore, the _cycle formulation_ consists in the following Integer Program**

$$\begin{align}
\max & \sum_{j=1}^n w_j x_j \\
\text{s.t. } & \sum_{j=1}^n a_{ij} x_j \leq 1 & \forall i = 1..m \\
& x_j \in \{0, 1\} & \forall j = 1..n
\end{align}$$

* $m$ is the number of pairs, $n$ of cycles
* $w_j$ is the weight of cycle $j$ (i.e. its number of nodes)
* $a_{ij} = 1$ iff node $i$ belongs to cycle $j$ (and $a_{ij} = 0$ otherwise)
* The maximum length constraint is handle when generating the set of cycles
PROBLEMA: quanti CYCLES ci sono in un GRAPH? TANTISSIMI bro. Per fortuna ho un CAPPING nel numero dei nodi nel CYCLE. Come faccio a addressare questo numero altissimo di POSSIBILI CYCLES? Risolveremo questa questione
PER ORA prendo un dataset di 12 elementi:
```python
pairs, arcs, aplus = util.generate_compatibility_graph(size=12, seed=2)
```
Mi genera 12 pairs e il COMPATIBILY GRAPH costruibile da queste pairs (andando a mettere un link se una pair ha un donor che ha lo stesso gruppo sanguigno di un recipient di un'altra pair)
* Compatible pairs would not need to go through a KPD program
* The blood type prevalence reflects the Italian distribution
* In the pairs, **we are neglecting all other factors that impact compatibility**
**Arcs are first determined based on blood type compatibility**
...***Then a small (random) fraction of them (5%) is removed***
* This simulated the other compatibility factors (tipo che anche se hai lo stesso gruppo sanguigno potrebbero comunque esserci ALTRI PROBLEMI che non permettono il trapianto)
* ...Which are therefore accounted for at the graph level

## Enumerating Cycles
**We enumerate cycles using simple Depth First Search with limited depth**
```python
def cycle_next(seq, nsteps, aplus, cycles, cap=None):
    node = seq[-1]
    successors = np.array(aplus[node]) # Consider all possible successors
    np.random.shuffle(successors) # ...in randomized order
    for dst in successors:
        # Early exit if the capacity has been exceeded
        if cap is not None and len(cycles) >= cap: return
        if dst == seq[0] and dst == min(seq): # close the cycle
            cycles.add(tuple(seq))
        elif nsteps > 0 and dst not in seq:
            cycle_next(seq+[dst], nsteps-1, aplus, cycles, cap) # recursive call
```
* Cycles are stored as tuples, which mean that the node ordering matters
* ...So we take only the ordering that starts with the minimum index
* There is a capacity parameter to limit the number of enumerated cycles
**We use a second function to start the enumeration from all possible sources**
```python
def find_all_cycles(aplus, max_length, cap=None, seed=42):
    cycles = set()
    roots = np.array(list(aplus.keys()))
    np.random.seed(seed)
    np.random.shuffle(roots)
    for node in roots:
        if cap is None or len(cycles) < cap:
            cycle_next([node], max_length-1, aplus, cycles, cap)
    return list(cycles)
```
**We can now enumerate the cycles for our graph (HP: max length of 4)**

## Cycle Formulation - Implementation
**Once we have all cycles, we can build the Cycle Formulation model**
```python
def cycle_formulation(pairs, cycles, tlim=None, verbose=1):
    infinity, ncycles, npairs = slv.infinity(), len(cycles), len(pairs)
    slv = pywraplp.Solver.CreateSolver('CBC') # Build the solver
    cpp = {i:[] for i in range(npairs)} # group cycles by pair
    for j, cycle in enumerate(cycles):
		# per ogni PAIR, gli vado a associare TUTTI i possibili CICLI legati a quella 
		# PAIR
        for i in cycle: cpp[i].append(j)
    # Creo la decision variable x_j che e' 1 se il CYCLE viene selezionato, 0 altrimenti
    x = [slv.IntVar(0, 1, f'x_{j}') for j in range(ncycles)] # variables
    # AL MASSIMO solo uno dei cycles che passa per un determinato nodo puo' essere settato a 1 
    for i in range(npairs): # constraints
        slv.Add(sum(x[j] for j in cpp[i]) <= 1)
    # voglio che ci sia piu' cover possibile dei trapianti (che in tutto io salvi piu' gente possibile)
    slv.Maximize(sum(len(c) * x[j] for j, c in enumerate(cycles))) # objective
    if tlim is not None: # time limit
        slv.SetTimeLimit(1000*tlim)
    status = slv.Solve() # solve
    # Extract results and return
    ...
```

**We use [the CBC solver](https://github.com/coin-or/Cbc), via [Google OR-Tools](https://developers.google.com/optimization)**
* It's the fastest MIP solver with a fully permissive license
**Variables are built with `IntVar`, constraints posted with `Add`**
The `cpp` dictionary contains cycles, grouped by the pair/node they use
Time limits are enforced with `SetTimeLimit`

```python
pairs, arcs, aplus = util.generate_compatibility_graph(size=12, seed=2)
cycles = util.find_all_cycles(aplus, max_length=4, cap=None)
sol, tme, _ = util.cycle_formulation(pairs, cycles, tlim=10, verbose=1)
print({k for k, v in sol.items() if v != 0 and k != 'objective'})
```
*Risultato: Solution time: 0.037 sec, objective value: 6.0 (optimal)
{'x_2', 'x_1', 'x_8'}*
Posso salvare 6 persone quindi HOccapito. 

##  Column Generation for Better Scalability
Questo e' il problema che dicevo prima
**The main drawback of the cycle formulation is the limited scalability**
* The number of cycles grows with the graph size as $O(n^{\text{max length}})$
* The enumeration becomes _more expensive_ and the model becomes _larger_
**Both can quickly become major bottlenecks**
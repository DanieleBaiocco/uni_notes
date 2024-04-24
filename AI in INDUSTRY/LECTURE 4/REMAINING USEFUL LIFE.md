Nel mio dataset ho una colonna chiamata RUL, che mi dice appunto la remaining useful life. Questo valore decresce a ogni timestep di 1 finche' non arriva a 0.

# RUL Prediction with REGRESSION

**Let's start from _the simpler formulation_ of a RUL-based policy**

* We will predict the RUL using a regression approach
* ...And trigger maintenance when the estimated RUL becomes too low, i.e.:

$$
f(x, \theta) \leq \varepsilon
$$

* $f$ is the regressor, with parameter vector $\theta$
* The threshold $\varepsilon$ must account for possible estimation errors

**STEP 1*
1. *standardizzo il dataset* (almeno le colonne usate per il training) 
```python
trmean = tr[dt_in].mean()
trstd = tr[dt_in].std().replace(to_replace=0, value=1) # handle static fields

ts_s = ts.copy()
ts_s[dt_in] = (ts_s[dt_in] - trmean) / trstd
tr_s = tr.copy()
tr_s[dt_in] = (tr_s[dt_in] - trmean) / trstd
```
3. metto il target RUL tra 0 e 1 (normalizzo RUL)
```python
trmaxrul = tr['rul'].max()

ts_s['rul'] = ts['rul'] / trmaxrul 
tr_s['rul'] = tr['rul'] / trmaxrul 
```

**STEP 2**
Traino un SEMPLICISSIMO NN regressor, che e' un LINEAR REGRESSOR (NON METTO ne' hidden layers ne' non-linearity con activation function).

**STEP 3**
Traino un NN regressor con 2 hidden layers. Tengo i risultati di questo.
Questo performa meglio del linear regressor, pero'
1. Ho un risultato di r2 score di 0.5, che e' pessimo
2. Ho che i RUL points da 500 a 100 sono tutti classificati  come 200 piu' o meno. Questo PERCHE' l'MSE pusha le PREDIZIONI verso la MEAN del target "RUL", quando l'input data e' STABILE e quindi non ci sono variazioni di alcun tipo.
3. Ho che la variance della mia prediction e' molto piu' alta quando la RUL e' alta (quindi quando non si e' certi), mentre va a diminuire quando la RUL e' bassa (quindi quando le turbine si stanno rompendo. Li' il modello e' molto piu' sicuro sul valore predictato). QUINDI ho che l'assunzione dell'MSE, che il componente deve avere piu' o meno la stessa variance dappertutto, e' VIOLATA.
4. A volte la predizione e' NEGATIVA
5. Nonostante la prediction brutta nella parte iniziale (quando la RUL e' ALTA), quando la RUL si abbassa (QUANDO SERVE quindi),  il modello E' BRAVO a capire che qualcosa non va.
6. Il modello infatti PREDICTA che le TURBINE si stanno ROMPENDO, solo quando NOTA UN CAMBIAMENTO NELL'INPUT DATA.

**STEP 4**
NON mi interessa avere una HIGH ACCURACY per sto modello. Voglio piuttosto valutare la POLICY del mio modello che decide QUANDO e' il caso di mandare il prodotto in MANUTENZIONE.
Devo allora definire un COST MODEL:
**We will assume that:**
We consider one step of operation as our value unit
* ...So we can express the failure cost in terms of operating steps
Every run end with either failure or maintenance:
* Assuming that the failure cost is higher than maintenance cost
* ...We can diseregard the maintenance cost 

**Slighly more formally:**
* One step of operation brings 1 unit of profit
* A failure costs $C$ units more than maintenance
* We only count what happens after $s$ steps

**Formally, let $x_k$ be the times series for machine $k$, and $I_k$ its set of time steps**
* The time step when our policy triggers maintenance is given by:
$$
\min \{i \in I_k \mid f(x_{ki}) < \varepsilon \}
$$
* A failure occurs if:
$$
f(x_{ki}) \geq \varepsilon \quad \forall i \in I_k
$$
$I_k$ in sto caso e' il set di timesteps fino al FAILURE  (che ovviamente coincidera' quando RUL e' 0).



**The whole cost formula _for a single machine_ will be:**

$$
\mathit{cost}(f({x_k}), \varepsilon) = \mathit{op\_profit}(f(x_k), \varepsilon) + \mathit{fail\_cost}(f(x_k), \varepsilon)
$$

Where:

$$
\mathit{op\_profit}(f(x_k), \varepsilon) = -\max(0, \min \{i \in I_k \mid f(x_{ki}) < \varepsilon\}-s) \\
\mathit{fail\_cost}(f(x_k), \varepsilon) = \left\{
\begin{align}
& C \text{ if } f(x_{ki}) \geq \varepsilon \quad \forall i \in I_k \\
& 0 \text{ otherwise}
\end{align}
\right.
$$

* $s$ units of machine operation are guaranteed
* ...So we gain over the default policy only if we stop after that
* Profit is modeled as a negative cost

**For the total cost, we need to sum over all machines**

**Normally, we would proceed as follows**

* $s$ is determined by the preventive maintenance schedule
* $C$ must be determined by discussing with the customer

In our example, we will derive both from data

**First, we collect all failure times**
```
failtimes = dt.groupby('machine')['cycle'].max()
failtimes.head()
```
**Then, we define $s$ and $C$ based on statistics**
```
print(failtimes.describe())
safe_interval = failtimes.min()
maintenance_cost = failtimes.max()
```
* For the safe interval $s$, we choose the minimum failure time
* For the maintenance cost $C$ we choose the largest failure time

**STEP 5**
Threshold optimization is DONE. AS ALWAYS (come abbiamo sempre visto fin'ora), quando si fa threshold optimization (che non e' legata all'overfitting del training set), non c'e' bisogno di farla sul VALIDATION SET (che in sto caso non e' stato neanche creato APPUNTO),MA LA FACCIO DIRETTAMENTE SUL TRAINING SET.

*Avg. cost: -95.88 (training), -96.87 (test)*
*Avg. fails: 0.0 (training), 0.015873015873015872 (test)
Avg. slack: 20.34 (training), 16.92 (test)*

* Slack = distance between when we stop and the failure
* The results are actually quite good!
* ...And we also generalize fairly well
## Sequence Input in Neural Models
We use regression but with Sequence Input now.
**Feeding more time steps to our NN might improve the results** (per ora ne metto solo uno di INPUT, non ho windows)
* Intuitively, sequences provide information about the _trend_
* This may allow a better RUL estimate w.r.t. using only the current state
* E.g. we may gauge how quickly the component is deteriorating

Avendo macchine diverse non posso creare le sliding windows come ho sempre fatto (se prendo due punti consecutivi questi potrebbero APPARTENERE a due esperimenti diversi: potrei mergiare la fine di un esperimento con l'inizio del successivo). Uso delle contromisure per evitare questa cosa e cambio il DATASET in un altro di shape *(n_windows, window_length, num_dims)*.
A questo punto uso una 1D Convolution in Keras
* We have a single convlution with 32 kernels
* Then a hidden layer with 32 ReLU neurons
* ...And finally the output layer

IL RISULTATO FINALE, dopo il fit, la threshold optimization ecc.. e' che NON HO GROSSE DIFFERENZE con l'approccio di sopra.
Quindi la morale e': 
Le sequenze nella pratica non funzionano spesso.  
1. se NON ci sono PATTERN VISIBILI allora NON LE USO.
2. SE CI SONO ALLORA prima provo approcci di Probability Density Estimation  (approcci statistici comunque). 
3. SE PROPRIO QUESTI NON VANNO, allora PROVO CON LE SLIDING WINDOWS (quindi un sequence based model).

**Just because you are dealing with time series**
...Do not assume that sequence input is useful!

* Sequences matter only if the output is correlated with patterns
* ...That involve multiple time steps
* 
**In many practical problems**
...A single "state" encodes most of the useful information
* You can think of that as sort of Markov property

**Therefore, before using sequences, it makes sense to _think_**
Do you expect sequences to provide useful information?
* E.g. is there seom kind of inertia?
* ...And does it matter for the considered problem?

# RUL prediction as CLASSIFICATION

**RUL-based maintenance can also be tackled _using a classifier_**
* We build a classifier to determine whether a failure will occur in $\varepsilon$ steps
* We stop as soon as the classifier outputs (say) a 0, i.e.

$$
f_\varepsilon(x, \theta) = 0
$$
* $f$ is the classifier, with parameter vector $\theta$
* $\varepsilon$ is the horizon for detecting a failure

**In a sense, we are trying to learn _directly_ a maintenance policy**
* The policy is the form "stop $\varepsilon$ units before a failure"
* The classifier tries to learn it

Invece di fare Regression voglio fare Classification.
Il modello e' trainato con target del training set tale che se il RUL e' sopra i 20  allora ho 0 (che indica che NON C'E' FAILURE), se e' sotto i 20 remaining steps allora ho 1 (fallimento).
Quindi scegliere la "soglia" per la creazione della target label computata dalla RUL "sostituisce" la threshold optimization del caso della regression qua sopra.
POSSO PERO' mettere ora una **threshold sulla probabilita' predetta** (il valore predetto sara' tra 0 e 1 e posso mettere una threshold per dire quando il valore sta predictando un 1 o uno 0 (un esempio e':  SOPRA 0.65 e' 1, sotto e' 0)).

L'architettura utilizzata e' una NN senza hidden layers e con una sigmoid activation function: GUARDA CASO se noti QUESTA E' UNA LOGISTIC REGRESSION POGGERS:
```python
nn1 = util.build_nn_model(input_shape=(len(dt_in), ), output_shape=1, hidden=[], output_activation='sigmoid')
util.plot_nn_model(nn1)
```
**Classification problems _tend to be easier_ than regression problems**
* On the other hand, learning the whole policy
* ...May be trickier than just estimating the RUL 
Ne creo un'altra con 2 hidden layers e la traino. Questa mi da' un risultato migliore rispetto alla semplice logistic regression.

**The model prediction can be interpreted as a probabilities of _not_ stopping** (una predizione e' quanto e' la probabilita' di CONTINUARE (perche' va ancora tutto bene)). Se le cose non vanno bene nei dati allora questa probabilita' SCENDERA'.
![[RULclassification.png]]
Questo e' il risultato. Nota come i valori delle predizioni sono di un'altra scala (sono tra 0 e 1) rispetto i valori del RUL (vengono plottati comunque su scale comparabili per rendere il plot piu' chiaro).

**In practice, we'll need to convert the predictions into integers via rounding**
...Unless we want to deal with one more threshold (in addition to $\theta$)
Quindi possiamo qui decidere se fare ROUNDING a 0.5 banale (quindi sopra => 1, sotto => 0) oppure se ottimizzare la threshold. Decidiamo di fare rounding banale.

Ho dei risultati abbastanza buoni, MA COMUNQUE molto peggiori dell'approccio con REGRESSION.
Questo perche':
1. Non ho calibrato la soglia (che e' fissa a 0.5)
2. Non ho calibrato la s INIZIALE che decide lo STEP della RUL in cui fare il CUT.

**NOTA** 
1. L'ottimizzazione del REGRESSION MODEL e' la seguente:
$$\begin{align}
\mathop{\text{argmin}}_{\varepsilon} & \sum_{k \in K} \mathit{cost}(f(x_k, \theta^*), \varepsilon) \\
\text{ s.t.: } & \theta^* = \mathop{\text{argmin}}_\theta L(f(x_k, \theta), y_k)
\end{align}$$
Quindi voglio trovare il $\theta ^ *$ (best params per il regressor) e l' $\epsilon$ (la threshold sotto il quale predicto MANTAINANCE) tali che $\theta ^ *$ e' quello che mi produce la loss minima tra prediction e target e $\epsilon$ la soglia che mi produce il COSTO MINIMO, calcolato usando come input il regressor GIA' FITTATO.
Sono due problemi che sono INDIPENDENTI TRA LORO.
2. L'ottimizzazione del CLASSIFICATION MODEL e' la seguente invece: 
$$\begin{align}
\mathop{\text{argmin}}_{\varepsilon} & \sum_{k \in K} \mathit{cost}(f(x_k, \theta^*), 1/2) \\
\text{ s.t.: } & \theta^* = \mathop{\text{argmin}}_\theta L(f(x_k, \theta), \mathbb{1}_{y_k \geq \varepsilon})
\end{align}$$
* We use a canonical threshold in the cost model (i.e. 0.5)
* $L$ is again the loss function (binary cross entropy)
* $\mathbb{1}_{y_k \geq \varepsilon}$ is the indicator function of $y_k \geq \varepsilon$ (i.e. our class labels)

Qua ho che il calcolo di $\theta ^ *$ e' LEGATO a come ho scelto $\epsilon$, che mi dice il numero di  steps SOTTO IL QUALE ho ANOMALIA.

**Unlike the previous one, this problem _cannot be decomposed_**
...Because $\varepsilon$ appears in the loss function!
* This means we need to _optimize $\varepsilon$ and $\theta$ at the same time_

**Let's sketch a possible optimization approach**
1. We search over the possible values of $\varepsilon$
2. For the given $\varepsilon$ value, we compute $\mathbb{1}_{y_k \geq \varepsilon}$ (i.e. the class labels)
3. We train the model to compute $\theta^*$
4. Then we compute the cost
5. ...And finally we repeat, for the next value of $\varepsilon$
At the end of the process, we choose the configuration with the best cost

**In principle we could use grid search again, but...**
* Evaluating the cost is _slow_, since it requires retraining 
* The search space _grows exponentially_ with the number of parameters (non mi e' chiaro questo punto)
We need a better optimization method!
Secondo me voleva solo farci vedere qualcosa di nuovo (secondo me gridsearch non sarebbe stata male dai).

# SURROGATE-BASED BAYESIAN OPTIMIZATION
**We will use an approach known as _Surrogate-Based Bayesian Optimization_**
* It is designed to optimize _blackbox functions_
* I.e. functions with an unknown structure, that can only be evaluated
* Utilizzato quando le FUNZIONI sono COSTOSE DA CALCOLARE (la mia cost function e' difatti molto costosa a livello computazionale, ma immagino anche il training della rete di base no?)
**Formally, they address problems in the form:**
$$
\min_{x \in B} f(x)
$$
* Where $B$ is a box, i.e. a specification of bounds for each component of $x$
* In our case, the decision variable $x$ would be $\epsilon$ (quindi IL CUT sul TARGET 'RUL')
* ...And the function to be optimized would be the cost (GIUSTAMENTE DAL CUT TRAINO IL MODELLO E poi COMPUTO IL COSTO)
Quindi voglio trovare l'$\epsilon$ tale che il COSTO (che e' la funzione PESANTE da valutare), che e' _f_, sia MINIMO.

**Since evaluating $f$ is expensive, it should be done _infrequently_**
The main trick to achieve this is using a _surrogate model_:
* After each evaluation we train _a Machine Learning model_
* ...Then we perform **optimization** _on the ML model_
* ...Since it can be evaluated much more quickly
Quindi invece di fare OTTIMIZZAZIONE su _f_ la facciamo su sto _surrogate model_.
**This is where the name stems from**
* Since we use the ML model instead of the function, we call it a _surrogate_
* Moreover, we optimize over _prior_ information (i.e. the current model)
* ...And we refine the model based on the evaluation (_posterior_)
* Hence we call it _Bayesian Optimization_
Credo che sia come IMPARARE LA LANDSCAPE della COST FUNCTION trainando un modello di ML per impararla, utilizzando SOLO POCHI PUNTI (che sono i punti risultanti dall'evaluation di questa _f_, sono per l'appunto POCHI PERCHE' E' UN' OPERAZIONE MOLTO PESANTE).


Come imparo questa landscape? Ovvero COSA USO COME surrogate model?  Cioe' il modello che fa OTTIMIZZAZIONE senza fare GRIDSEARCH (senza valutare _f_ per ogni valore della gridsearch)?
GP ovviamente.
**STEP 1** SELEZIONO DEI PUNTI INIZIALI DI CUI  CALCOLO LA _f_
**STEP 2** FITTO LA GP su questi PUNTI INIZIALI (che sono x e y)
**STEP 3** After HAVING FIT the GP on the initial points, for which I HAD TO COMPUTE the _f_, how do I choose the NEXT POINT x (ovvero $\epsilon$, il CUT nel mio caso)?
**We need to account for both the _predictions_ and their _confidence_**
* Area with _low predictions_ are promising
* ...But so are also areas with _high confidence_

**This issue is solved in SBO by optimizig an _acquisition function_**
...Which should balance _exploration_ and _exploitation_.
* Examples include the Probability of improvement, the Expected Improvement
* ...And the Lower/Upper confidence bound

**We will use the Lower Confidence Bound, which is given by:**
$$
\mathit{LCB}(x) = \mu(x) - Z_\alpha \sigma(x)
$$
* Where $\mu(x)$ is the predicted mean, $\sigma(x)$ is the predicted standard deviation
* ...And $Z_\alpha$ is multiplier for a $\alpha\%$ Normal confidence inteval

**NOTA** questa funzione e' una funzione ESEMPIO che rappresenta nel mio caso la FUNZIONE DI COSTO banalmente (che a me NON E' NOTA OVVIAMENTE).
![[lowerconfidencebound_sample.png]]
La linea in rosso e' il RISULTATO di aver computato LCB per ogni punto (tanto di ogni punto conosciamo mean e STD).
Si prende QUINDI la x col valore PIU BASSO DI LCB che sarebbe il punto con la BEST ACQUISITION FUNCTION.
**STEP 4** 
**Now wen update our surrogate model**
First, we evaluate $f$ for the new point and grow our training set (fatto dai punti precedenti scelti inizialmente piu' sto punto aggiunto)
**STEP 5** RITRAINO la GP con il punto aggiunto.
**STEP 6** Si ripetono da STEP 3 a STEP 5 fino a che non voglio io di base (o fino a che una condizione diventa True).


**Let's review the general method**
* Given a collection $\{x_i, y_i\}_i$ of evaluated points
* ...We train a surrogate-model $\hat{f}$ for $f$

**Then we proceed as follows:**
* We optimize an acquisition function $a_\hat{f}(x)$ to find a value $x^\prime$
* We evaluate $y^\prime = f(x^\prime)$
* If $y^\prime$ is better than the current optimum $f(x^*)$:
  - Then we replace $x^*$ with $x^\prime$
* We expand our collection of measurements to include $(x^\prime, y^\prime)$
* We retrain $\hat{f}$
* We repeat until a termination condition is reached

## SBO for Threshold Calibration in OUR USE CASE
**We will use SBO to tackle our policy definition problem**

$$\begin{align}
\mathop{\text{argmin}}_{\varepsilon} & \sum_{k \in K} \mathit{cost}(f(x_k\, \theta^*), 1/2) \\
\text{ s.t.: } & \theta^* = \mathop{\text{argmin}}_\theta L(f(x_k, \lambda), \mathbb{1}_{y_k \geq \varepsilon})
\end{align}$$

Here's our plan:

* We need to optimize over $\varepsilon$
* Our goal is minimizing the cost
* Computing the cost requires to re-define the classes
* ...And therefore to repeat training (e' per questo che e' un'operazione assurdamente costosa)


**As a first step, we need to define our black box function**(Che praticamente mi calcola il TARGET da un valore $\epsilon$ di CUT che specifico in params, fa poi training e ritorna il COSTO)
We will use a function class (in the `util` module) with this structure:

```python
class ClassifierCost:
    def __init__(self, machines, X, y, cost_model, init_epochs=20, inc_epochs=3):
        ...

    def __call__(self, params):
        ...
```

* In the constructor, we provide parameters that are fixed during optimization
* In the `__call__` method, we retrain the model and evaluate the cost
* The `__call__` method is executed when we try to invoke an object of this class
* ...Meaning that we can treat an object of this class as a normal function

**It is worth having a deeper look at the `__call__` method**

```python
def __call__(self, params):
    theta = params[0] # There is only one parameter to optimize
    lbl = (self.y >= theta) # Redefine classes
    # Determine the number of epochs and retrain
    epochs = self.init_epochs if not self.is_init else self.inc_epochs
    self.is_init = True
    train_nn_model(self.nn, self.X, lbl, loss='binary_crossentropy', epochs=epochs,
            verbose=0, patience=10, batch_size=32, validation_split=0.2)
    ...
```

* At each execution we redefine the classes
* We use warm starting to make the process faster
* Each training attempt after the first uses only a few epochs

```python
def __call__(self, params):
    ...
    self.stored_weights[theta] = self.nn.get_weights() # Store weights
    # Evaluate cost
    pred = np.round(self.nn.predict(self.X, verbose=0).ravel())
    cost, fails, slack = self.cost_model.cost(self.machines, pred, 0.5, return_margin=True)
    return cost
```

* We store the weights in a dictionary for later retrieval
* We need this to rebuild the optimal network once optimization is over
* Finally, we evaluate the cost
* The actual code in `util` also prints some information

```
ccf = util.ClassifierCost(machines=tr['machine'], X=tr_s[dt_in], y=tr['rul'], cost_model=cmodel)
```

```python
pbounds = {'eps': (1, 20)} # Box constraints
optimizer = BayesianOptimization(f=ccf, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=3, n_iter=10)
```

|   iter    |  target   |    eps    |
| 1         | 1.868e+04 | 8.116     |
| 2         | 1.63e+04  | 19.06     |
| 3         | 1.665e+04 | 14.91     |
| 4         | -4.407e+0 | 3.215     |
| 5         | 1.777e+04 | 9.325     |
| 6         | 1.782e+04 | 12.29     |
| 7         | 1.678e+04 | 17.06     |
| 8         | 1.628e+04 | 20.0      |
| 9         | 1.828e+04 | 11.0      |
| 10        | 1.747e+04 | 7.044     |
| 11        | 1.874e+04 | 7.725     |
| 12        | 1.767e+04 | 13.48     |
| 13        | 1.91e+04  | 7.887     |
**We can access the best $\varepsilon$ value from a result data structure**
```
print(optimizer.max)
best_eps = optimizer.max['params']['eps']
nn = keras.models.clone_model(ccf.nn)
nn.set_weights(ccf.stored_weights[best_eps])
```

**Finally, we can evaluate our classifier** e le performances sono migliori del regression approach.
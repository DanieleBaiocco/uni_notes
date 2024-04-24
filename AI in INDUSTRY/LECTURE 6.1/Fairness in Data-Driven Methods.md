RICORDA: ML MODELS sono FLESSIBILISSIMI, FLESSIBILISSIMI, FLESSIBILISSIMI.
Serve ESPERIENZA in Machine Learning, non ho KNOWN TASKS e KNOWN WAYS of solving the tasks.

**Say we want to estimate the risk of violent crimes in given population**
* This is obviously a very _ethically sensitive (and questionable) task_
* ...Since our model may easily end up _discriminating some social groups_

**This makes it a good test case to discuss _fairness in data-driven methods_**
**Fairness in data-driven methods is _very actual topic_**

* As data-driven systems become more pervasive
* They have the potential to significantly affect social groups
**This is so critical that the topic is about _starting to be regulated_**

* The EU has drafted [Ethics Guidelines for Trustworthy AI](https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai)
* ...And is in the process of releasing a big [AI act](https://www.europarl.europa.eu/news/en/headlines/society/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence)
* In some fields, models that do not comply with some rules cannot be deployed

# Loading and Preparing the Dataset

**We will run an experiment on the ["crime" UCI dataset](https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime)**

We will use a pre-processed version made available by our support module:
Ho sto dataset che ha per ogni entry UNA COMUNITA' con informazioni come la 'race', 'popDensity', 'pctUsePubTrans',... e voglio PREDICTARE 'violentPerPop' che sarebbe il numero di VIOLENTI ogni 100k persone dentro quella comunita'.

Per adesso vedo il problema come REGRESSION.
- **NOTA**: The only categorical input is "race" (0 = primarily white, 1 = primarily black) , and this is also the attribute that we will use to check for discrimination

**STANDARDIZZO LE FEATURES NUMERICHE** come prima cosa:
```python
attributes, taet = data.columns[3:-1], data.columns[-1]
nf = [a for a in attributes if a != 'race'] + [target]

tr_frac = 0.8 # 80% data for training
tr_sep = int(len(data) * tr_frac)
tmp = data.iloc[:tr_sep]

sdata = data.copy()
sdata[nf] = (sdata[nf] - tmp[nf].mean()) / (tmp[nf].std())

sdata[attributes] = sdata[attributes].astype(np.float32)
sdata[target] = sdata[target].astype(np.float32)
```

Traino un linear REGRESSOR (senza hidden layers)
NOTA non standardizzo la race perche' e' un CATEGORICAL ATTRIBUTE (prende either 0 or 1)
Calcolo dopo il training le metrice R2 e MAE su train e test
Risultati:
*R2 score: 0.67 (training), 0.61 (test)
MAE: 0.39 (training), 0.45 (test)*

**Linear Regression is an interpretable ML model**
* In particular, we can have evaluate the _importance of each input attribute_
* This can be done in LR by _inspecting the weights_
We could try this approach to check for discrimination
HO BISOGNO DI UN MODELLO CHE SIA IN QUALCHE SENSO INTERPRETABILE, PER VEDERE SE C'E' DISCRIMINAZIONE (obbligatorio).
Se l'absolute value del peso della feature e' alto allora quella feature e' importante, altrimenti NON e' importante. Il SEGNO, se POSITIVO, mi dice che c'e' una POSITIVE CORRELATION tra quella feature e il target, altrimenti NEGATIVE. **NOTA**, non basta checkare la grandezza del PESO, PRIMA DEVI STANDARDIZZARE/NORMALIZZARE perche' se gli alttributi hanno scale diverse allora un PESO GRANDE potrebbe esser ERRONEAMENTE attribuito a un attributo che CONTA (quando magari quell'attributo non e' molto informativo per la predizione), SOLO perche' i suoi valori sono grandi.

**quindi PLOTTO I WEIGHTS**. 
```python
lr_weights = np.abs(nn.get_weights()[0].ravel())
util.plot_lr_weights(lr_weights, attributes, figsize=figsize)
```
![[discriminationweights.png]]
**Nota** che NON HO UN CUTOFF CHIARO da poter fare, per dire OLTRE QUESTO CUT gli weights NON  sono cosi' importanti, di conseguenza gli attributi a essi levati sono ininfluenti.

# Lasso REGRESSION
e' un modo per avere UN PIU' CHIARO CUTOFF, e' un modo per avere piu' interpretability. Sopra ho un dense weight matrix, vorrei una cosa PIU' SPARSA. Quindi applico UNA L1-REGULARIZATION, questo mi da' LASSO.
PRATICAMENTE aggiunge alla LOSS una PENALITA' che e' LA SOMMA dei PESI. Quindi il modello CERCHERA' di generare PESI PIU' PICCINI DI BASE. Voglio un modello che massimizzi la likelihood, MA DEVE ANCHE RISPETTARE ANCHE DELLE PROPRIETA', in questo caso PESI VICINI A 0. $\alpha$ mi dice quanto VOGLIO ENFATIZZARE la prior WRT the POSTERIOR (che e' la loss). 
La formula e'

$$
L(y, f(x, \theta)) = \|\theta^T x - y\|_2^2 + \alpha \|\theta\|_1
$$

Ci saranno attributi per cui il fattore della regularity SARA' SOVRASTANTE rispetto alresto.
Se $\alpha$ e' abbastanza alto VINCE LEI.
**Lasso is available in scikit-learn, and can be implemented in Keras/Tensorflow**

We just need to add L1 regularization over the output neuron:

```python
...
model_out = layers.Dense(output_shape, activation=output_activation,
        kernel_regularizer=regularizers.l1(l1=1e-3))(x)
...
```

Nel mio caso:
```python
nn2 = util.build_nn_model(input_shape=len(attributes), output_shape=1, hidden=[], output_activation='linear',
                          kernel_regularizers=[regularizers.l1(l1=1e-2)])
history = util.train_nn_model(nn2, tr[attributes], tr[target], loss='mse', batch_size=32, epochs=400, verbose=0, validation_split=0.2)
util.plot_training_history(history, figsize=figsize)
```
E' uno di QUEI CASI IN CUI NON POSSO FARE HYPERPARAM TUNING CON UN VALIDATION SET per l'L1
Perche' questo regola il LIVELLO DI SPARSIFICATION e non c'e' UN MIGLIORE LIVELLO DI SPARSIFICATION. Semplicemente ci gioco e quando sono FELICE BELLA LI'.

L1 inoltre e' anche MOLTO utile per preventare OVERFITTING, essendo una regularization tecnique.
I risultati sono ora:
*R2 score: 0.66 (training), 0.61 (test)
MAE: 0.39 (training), 0.45 (test)*
Sono gli stessi
ORA HO COME ABS VALUES DEI PESI:

![[discriminationweights2.png]]
e ora IL CUTOFF E' MOTLO PIU' FACILE DA TROVARE.
Quindi prendo I primi  15 attributi E 'race' NON e' presente
![[regressionattributeimportance.png]]
**The attribute "race" is nowhere to be seen!**
* This is _looks_ reassuring for our potential discrimination concerns
* ...But in fact _it is not_ (and we will proceed to check it)

# FAIRNESS METRICS
Per misurare la fairness pero' ho bisogno di METRICHE migliori, non posso dire di essere TRANQUILLO solo perche' nei primi quindici NON ho 'race'. 
Sono stati creati degli INDICATORS. In questo approccio quando COSTRUISCO il modello devo decidere a PRIORI il LIVELLO di FAIRNESS che voglio introdurre nel mio modello. Questi INDICATORS mi permettono di misurare la FAIRNESS.

## Disparate Treatment
* We will check whether different groups
* ...As defined by the value of a protected attribute ("race" for us)
* Are associated to different predictions
PLOTTO la distribuzione dei valori di 'violentPerPop' predictati in 3 modi diversi:
1. considerando TUTTE le predizioni
2. considerando SOLO le predizioni in cui la 'race' e' a  0 (i bianchi)
3. considerando SOLO le predizioni in cui la 'race' e' a 1 (i neri)
Ho che c'e' un HUGE GAP tra SOLO bianchi e SOLO neri:
![[boxplotraces.png]]
**This would happen _even if removed the "race" attribute_**
* The model would still be able to access race information
* ...By simply using correlates
Questo PUNTO QUA SOPRA E' importante, banalmente anche levassimo la 'race' il modello imparerebbe questa associazione ricostruendola dagli altri attributi.
QUINDI se banalmente NASCONDO/LEVARE QUESTO "SENSITIVE ATTRIBUTE" 'race' allora LA SITUA NON CAMBIERA'.

## DIDI
**Therefore, checking the important attributes is not enough**
* We need to _measure disparate treatment_ for the trained model
* ...And as we mentioned there are alternative metrics to do that
**We will use the one from [this AAAI paper](https://www.aaai.org/ojs/index.php/AAAI/article/download/3943/3821)**
* Given a set of categorical _protected attribute (indexes) $J_p$_
* ...The Disparate Impact Discrimination Index (for regression) is given by:
$$
\text{DIDI}_r = \sum_{j \in J_p} \sum_{v \in D_{j}} \left|\frac{1}{m} \sum_{i=1}^m y_i - \frac{1}{|I_{j,v}|} \sum_{i \in I_{j,v}} y_{i}\right|
$$
* Where $D_j$ is the domain of attribute $j$
* ...And $I_{j,v}$ is the set of example such that attribute $j$ has value $v$
Banalmente dai plot di sopra sarebbe fare *la differenza tra il gap tra race=0 e all* e tra *rance=1 e all* e sommarle.

**Let's make some intuitive sense of the $\text{DIDI}_r$ formula**
$$
\sum_{j \in J_p} \sum_{v \in D_{j}} \left|\frac{1}{m} \sum_{i=1}^m y_i - \frac{1}{|I_{j,v}|} \sum_{i \in I_{j,v}} y_{i}\right|
$$
* $\frac{1}{m} \sum_{i=1}^m y_i$ is just the average predicted value
* ...For examples where the protected attribute takes specific values
* $\frac{1}{|I_{j,v}|} \sum_{i \in I_{j,v}} y_{i}$ is the average prediction for a social group
**We penalize the group predictions for _deviating from the global average_**
* Obviously this is not necessarily the best definition, but it is something
* In general, different tasks will call for different discrimination indexes
...And don't forget the whole "can we actually measure ethics" issue ;-)

**We can compute the DIDI via the following function**
```python
def DIDI_r(data, pred, protected):
    res, avg = 0, np.mean(pred)
    for aname, dom in protected.items():
        for val in dom:
            mask = (data[aname] == val)
            res += abs(avg - np.mean(pred[mask]))
    return res
```
* `protected` contains the protected attribute names with their domain
**For our original Linear Regression model, we get**
```python
tr_DIDI = util.DIDI_r(tr, tr_pred, protected)
ts_DIDI = util.DIDI_r(ts, ts_pred, protected)
print(f'DIDI: {tr_DIDI:.2f} (training), {ts_DIDI:.2f} (test)')
```
RISULTATI: *DIDI: 2.07 (training), 2.16 (test)* 
Voglio migliorare questi risultati...
## Improving the DIDI
**We will try to _improve over this baseline_**
This is not a trivial task:
* Discrimination arises from a form of bias in the training set
* ...And bias is not necessarily bad
**In fact, ML works _because of bias_**
* Improving fairness requires to _get rid of part_ of this bias
* ...Which will lead to some _loss of accuracy_ (hopefully not too much)
Voglio quindi abbassare DIDI senza perdere TROPPA accuracy.
**We will see _one method_ to achieve this result**

### Constrained ML via Lagrangians
**Let's recap our goals:**

We want to train an accurate regressor ($L = \operatorname{MSE}$):
$$
\operatorname{argmin}_{\theta} \mathbb{E}_{x, y \sim P(X, Y)}\left[ L(y, f(x, \theta)) \right]
$$
We want to measure fairness via the DIDI:
$$
\operatorname{DIDI}(\hat{y}) = \sum_{j \in J_p} \sum_{v \in D_{j}} \left|\frac{1}{m} \sum_{i=1}^m \hat{y}_i - \frac{1}{|I_{j,v}|} \sum_{i \in I_{j,v}} \hat{y}_{i}\right|
$$
With $\hat{y}$ predictions of the model.
...And we want the DIDI to be low, e.g.:
$$
\operatorname{DIDI}(f(x, \theta)) \leq \varepsilon
$$

**We can use this information to re-state the training problem**
$$
\operatorname{argmin}_{\theta} \left\{ \mathbb{E}\left[ L(y, f(x, \theta)) \right] \mid \operatorname{DIDI}(f(x, \theta)) \leq \varepsilon \right\}
$$
* Training is now a **_constrained optimization_** problem. E' il mio primo CONSTRAINED OPTIMIZATION problem che incontro. QUINI non ho PURE OPTIMIZATION.
**We are requiring constraint satisfaction _on the training set_**
...Meaning that we'll have _no satisfaction guarantee on unseen examples_ (QUINDI STO CONSTRAIN VALE per il TRAIN ma non e' detto che valga anche nel TEST. Quindi NON e' detto che non avro' DISCRIMINATION nel test set)
* This is suboptimal, but doing better is very difficult
* ...Since our constraint is defined (conceptually) _on the whole distribution_
**We'll trust the model to generalize well enough**

### How to account for the constraint at training time?
There's more then one method: we'll see the most famous one in ML
POSSO usare l'DIDI come PENALITA' nella LOSS FUNCTION.
NOTA che il LASSO di prima era un SOFT CONSTRAINT (puo' essere VIOLATO). In sto caso voglio un HARD CONSTRAINT (NON deve essere violato).
**Let's consider a ML problem with _constrained output_**
In particular, let's focus on **problems in the FORM**:
$$
\text{argmin}_{\theta} \left\{ L(\hat{y}) \mid g(\hat{y}) \leq 0 \right\} \quad\text{ with: } \hat{y} = f(x, \theta)
$$
Where:
* $L$ is the loss (the notation omits ground truth label for sake of simplicity)
* $x$ is the training input
* $\hat{y}$ is the ML model output, i.e. $f(x, \theta)$
* $\theta$ is the parameter vector (we assume a parameterized model)
* $g$ is a constraint function

### Overview of constraints
**Example 1: logical _rules_**
E.g. hiearchies in multi-class classification ("A dog is also an animal"):
$$
\hat{y}_{i,dog} \leq \hat{y}_{i,animal}
$$
* This constraint is defined over _individual examples_
Magari ho appunto una multiclass classification e VOGLIO come constraint che se la probabilita' che quell'EXAMPLE sia *dog* ALLORA anche la probabilita' di *animal* DEVE essere grande (maggiore o uguale in sto caso).

**Example 2: _shape_ constraints**
E.g. input $x_j$ cannot cause the output to decrease (monotonicity)
$$
\hat{y}_{i} \leq \hat{y}_{k} \quad \forall i, k : x_{i,j} \leq x_{k,j} \wedge x_{i,h} = x_{k,h} \forall h \neq j
$$
* This is a _relational constraint_, i.e. defined over multiple examples
Se magari SO che AUMENTANDO un valore di un ATTRIBUTO nell'input PORTERA' SICURAMENTE UN AUMENTO ANCHE NELLA PREDICTION, allora voglio che questo CONSTRAINT venga rispettato.
La formula di sopra mi dice che se ho due training samples in cui l'unica differenza e' che uno ha il valore di un attributo che e' minore uguale rispetto all'altro valore allora il target sara' minore uguale rispetto all'altro target (tutti gli altri attributi RIMANGONO UGUALI, cambia solo il valore di UN ATTRIBUTO tra i due esempi). 
Questo per dire che CONSTRAINED OPTIMIZATION e' importante QUANDO VOGLIO CHE CERTE PROPRIETA' VENGANO soddisfatte al 100% ALMENO nel training set.

QUELLO CHE ANDRO' A FARE E' utilizzare il CONSTRAINT come PENALTY TERM
### Lagrangian Methods for Constrained ML
**One way to deal with this problem is to rely on a _Lagrangian Relaxation_**
Main idea: we _turn the constraints into penalty terms_:
* From the original constrained problem:
$$
\text{argmin}_{\theta} \left\{ L(\hat{y}) \mid g(\hat{y}) \leq 0 \right\} \quad\text{ with: } \hat{y} = f(x, \theta)
$$
* We obtain the following _unconstrained_ problem:
$$
\text{argmin}_{\theta} L(\hat{y}) + \lambda^T \max(0, g(\hat{y})) \quad\text{ with: } \hat{y} = f(x, \theta)
$$
* The new loss function is known as a _Lagrangian_ (in penalty form)
* $\max(0, g(\hat{y}))$ is sometimes known as _penalizer_ (or Lagrangian term). **NOTA** Se non avessi il max(.) sarebbe un errore perche' potrebbe capitare che _g_ e' negativo  (quindi il CONSTRAINT e' mega rispettato), ma in sto modo HO che _g_, senza max(.), ha la funzione di UN REWARD TERM, quando invece dovrebbe lasciare la LOSS invariata. 
* ...And the $\lambda$ is a vector of _multipliers_ . **NOTA** e' un vettore di MULTIPLIERS perche' solitamente in _g_ ho piu' di un CONSTRAINT, quindi ho un valore di $\lambda_i$ per ogni constraint _i_.

BASICALLY:
* When the constraint is _satisfied_ ($g(\hat{y}) \leq 0$), the penalizer is 0
* When the constraint is _violated_ (so $g(\hat{y}) > 0$), the penalizer is > 0
* Hence, in the _feasible area_, we still have the _original loss_
* ...In the _infeasible area_, we incur a penalty that can be controlled using $\lambda$
**Therefore:**
* If we choose $\lambda$ large enough, under **quite general conditions**
* ...We can guarantee that a feasible or  asymptotically feasible solution is found (non permetto al modello di entrare in REGIONI PROIBITE, in cui ho un risultato DISCRIMINATORIO).
This is the basis of the classical [penalty method](https://en.wikipedia.org/wiki/Penalty_method#:~:text=Penalty%20methods%20are%20a%20certain,of%20the%20original%20constrained%20problem.)

**NOTA** For some _specific cases_, the $\max(\cdot)$ operator is not necessary
* The Lagrangian term is instead just $\lambda^T g(\hat{y})$
* When this is true, we say that _strong duality_ holds
* When use use **non-linear model**, come nel nostro caso, strong duality typically does not hold. 
* Di conseguenza e' necessario aggiungere il _max($\cdot$)_ 
**NOTA** Quando uso _Equality constraints_ (i.e. $g(\hat{y}) = 0)$ can be modeled in these ways
*  $\lambda^T |g(\hat{y})|$
* Using a quadratic term, i.e. $g(\hat{y})^2$ is also possible
Senno' posso farlo usando due INEQUALITIES  e utilizzandole insieme dentro _g_  (posso fare che _g_ ha dentro 2 constraints).

In generale le *feasibility guaranteees* non sono sempre PRESENTI. Per averle devo avere che $\lambda$ sia grande abbastanza, che esista una feasible solution, che si rispettino tutti i discorsi fatti sopra sul max($\cdot$), ecc... 
Inoltre devo risolvere il problema PER OPTIMALITY, NON posso usare HEURISTICS. Nota che usando GD, essendo questo un processo randomico NON risolve per OPTIMALITY ma usa un approccio euristico, quindi RISCHIO nell'usarlo (di certo non ho la certezza di convergere a una ottima feasibility).
**NOTA** visto che uso GD, devo avere che il CONSTRAINT SIA DIFFERENTIABLE, altrimenti NON POSSO.

### Back to Our Fairness Constraint
**Ideally, we wish to train an ML model by solving**
$$
\operatorname{argmin}_{\theta} \left\{ \mathbb{E}\left[ L(y, f(x, \theta)) \right] \mid \operatorname{DIDI}(f(x, \theta)) \leq \varepsilon \right\}
$$
First, we obtain _a Lagrangian term for our constraint_:
$$
\lambda \max\left(0, \operatorname{DIDI}(f(x, \theta)) - \varepsilon \right)
$$
* We just have one constraint, so $\lambda$ is a scalar
* The threshold (i.e. $\varepsilon$) has been incorporated in the term
* The DIDI formula is differentiable, so we can use a NN for $f$
* ...Otherwise, we would have needed to use a differentiable approximation

**With the Lagrangian term, we can modify the loss function:**
$$
\operatorname{argmin}_{\theta} \mathbb{E}\left[ L(y, f(x, \theta))] +\lambda \max\left(0, \operatorname{DIDI}(f(x, \theta)) - \varepsilon \right) \right]
$$
* So, in principle we can implement the approach with _a custom loss function_
* In practice, things are trickier due to how the DIDI works:
$$
\operatorname{DIDI}(y) = \sum_{j \in J_p} \sum_{v \in D_{j}} \left|\frac{1}{m} \sum_{i=1}^m y_i - \frac{1}{|I_{j,v}|} \sum_{i \in I_{j,v}} y_{i}\right|
$$
* The computation requires information about the protected attribute (cioe' IO non ce l'ho nella colonna dei target QUANTI race=1 e quanti race=0 ci sono capisci? devo specificarlo o salvarmelo in qualche modo comunque. In tensorflow una CUSTOM LOSS FUNCTION prende solo y e ypred. Non anche l'info sul protected attribute. Allora creo un CUSTOM MODEL qui.)
* ...Which is not part of the ground truth (at least not by default)
This makes things more complicated...

**...To the point that is easier to use a _custom Keras model_**
```python
class CstDIDIRegressor(keras.Model):
    def __init__(self, base_pred, attributes, protected, alpha, thr): ...
		super(CstDIDIModel, self).__init__()
		self.base_pred = base_pred # Wrapped predictor
		self.alpha = alpha # This is the penalizer weight (i.e. lambda)
		self.thr = thr # This is the DIDI threshold (i.e. epsilon)
		self.protected = {list(attributes).index(k): dom for k, dom in protected.items()}
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')
		
    def call(self, data): 
		return self.base_pred(data)
		
    def train_step(self, data): 
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self.base_pred(x, training=True)
            mse = self.compiled_loss(y_true, y_pred)
            # Compute the constraint regularization term
            ymean = tf.math.reduce_mean(y_pred)
            didi = 0
            for aidx, dom in self.protected.items():
                for val in dom:
                    mask = (x[:, aidx] == val)
                    didi += tf.math.abs(ymean - tf.math.reduce_mean(y_pred[mask]))
            cst = tf.math.maximum(0.0, didi - self.thr)
            loss = mse + self.alpha * cst

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}
    @property
    def metrics(self): 
		return [self.ls_tracker,
		self.mse_tracker,
		self.cst_tracker]
```
* In the `__init__` method we pass all the additional information we need
* The `call` method is called when evaluating the model
* The `train_step` method is called by Keras while training

**Without a clear clue for choosing the Lagrangian multipliers** $\lambda$
...We picked $5$ as a guess
* Choosing a good weight is obviously an important issue
* We'll how to deal with that later

VOGLIO PROVARE A DIMEZZARE  IL  DIDI DEL MODELLO CHE HO COME BASELINE
* Since for our baseline we have $\operatorname{DIDI}(y) \simeq 2$
* ...allora seleziono come threshold  $\varepsilon = 1$ (cosi avro' il DIDI sotto 1 => lo dimezzo)

## Training the Constrained Model
**We can train the constrained model as usual** usando un linear regressor come prima
* Since the constraint is for all the population, we have `batch_size=len(tr)`
* We could use mini-batches, but that would result in some noise
```python
base_pred = util.build_nn_model(input_shape=len(attributes), output_shape=1, hidden=[])
nn = util.CstDIDIModel(base_pred, attributes, protected, alpha=5, thr=didi_thr)
history = util.train_nn_model(nn, tr[attributes], tr[target], loss='mse', validation_split=0., epochs=2000, batch_size=len(tr))
util.plot_training_history(history, figsize=figsize)
```
IMPORTANTISSIMO che `batch_size=len(tr)` (me lo aspettavo sai, mi stavo chiedendo sta cosa difatti)![[trainingconstraintviolation.png]]
Il training va cosi'. Ho tre losses. NOTA che la MSE scende molto bene. Pero' ogni tanto SI NOTANO dei PUNTINI BLU che concidono a quando PRATICAMENTE IL CONSTRAINT VIENE VIOLATO. In quei punti infatti il constraint E' VIOLATO e quindi LA LOSS (che e' la sum di mse e constraint) AUMENTA risultando in questi PUNTINI BLU che si differenziano rispetto alla mse. E nota che dopo uno SPIKE dovuto al CONSTRAINT VIOLATION ho un BUMP nell'mse (che quindi diventa PEGGIORE (nota le onde)). Questo per farti capire che c'e' sto tradeoff, in cui si perde un po' di accuracy causa constraint violations.
RISULTATI:
*R2 score: 0.54 (training), 0.47 (test)
DIDI: 0.95 (training), 1.06 (test)*
Ho perso MOLTA accuracy ma ho comunque CONSTRAINT SATISFACTION (anche nel test dai).
POSSO FARE MEGLIO?
**The constraint is satisfied (and the accuracy reduced, as expected)**
...But _why is there some slack_ in terms of constraint satisfaction? (NOTA: ho detto al modello di stare sotto a $\epsilon$ = 1 e mi e' arrivato a 0.9 (HO UNO SLACK))
Ho quindi che la mia $\lambda$ e' TROPPO GRANDE e dice al modello di andare TROPPO nella direzione opposta, appena questo scende di loss tramite mse. Cio' e' dovuto al fatto che usando GD, quidni un approccio euristico (e non un optimal solver), appena CROSSO la boundary ottimizzando per mse e vado in una INFEASIBLE REGION, il modello mi SBATTE FORTISSIMO FUORI DA QUELLA REGIONE a causa di un alto valore di $\lambda$. Se pero' utilizzo una $\lambda$ troppo piccola allora NON OTTENGO RISULTATI NELLA FEASIBLE AREA.
Posso ottimizzare questa $\lambda$ o con grid search o CON UN METODO MOOOLTO MIGLIORE.

## Choosing Multiplier Values
**We are currently solving this problem**
$$
\operatorname{argmin}_{\theta} \mathbb{E}\left[ L(y, \hat{y}) +\lambda \max\left(0, g(\hat{y}) \right) \right] \quad\text{ with: } \hat{y} = f(x, \theta)
$$
...By using (Stochastic) _Gradient Descent_
**This is an important detail**
* A large $\lambda$ may be fine theoretically
* ...But it may cause the gradient to be _unstable_
**Therefore:**
* With a convex model, we should still reach convergence, but _slowly_
* With a non-convex model, we may end up in a poor local optimum (come nel caso di sopra)

### PENALTY METHOD
questo e' il penalty method:
**We can think of increasing $\lambda$ gradually**
...Which leads to the classical _penalty method_
* $\lambda^{(0)} = 1$
* $\theta^{(0)} = \operatorname{argmin}_{\theta} \left\{ L(y, \hat{y}) + \lambda^{(0)T} \max(0, g(\hat{y})) \right\} \text{ with: } \hat{y} = f(x, \theta)$
* For $k = 1..n$
  - If $g(y) \leq 0$, stop
  - Otherwise $\lambda^{(k)} = r\lambda^{(k)}$, with $r \in (1, \infty)$
  - $\theta^{(k)} = \operatorname{argmin}_{\theta} \left\{ L(y, \hat{y}) + \lambda^{(k)T} \max(0, g(\hat{y})) \right\} \text{ with: } \hat{y} = f(x, \theta)$

**This can work, but there are a few issues**
* $\lambda$ grows quickly and may still become problematically large (quindi non trovo la giusta $\lambda$ perche' la incremento troppo)
* Early and late stages in SGD may call for _different values of $\lambda$_ (questo e' cruciale, cioe' magari una $\lambda$  va bene per la epoca 40 e mi genera $g(y) \leq 0$ tranquillamente ma magari la stessa a un'epoca piu' avanzata con $\theta$ imparati diversi, potrebbe NON dare $g(y) \leq 0$ ).

**A gentler approach consists in using _gradient ascent for the multipliers_**
Let's consider our modified loss:
$$
\mathcal{L}(\theta, \lambda) = L(y, f(x, \theta)) +\lambda^T \max\left(0, g(f(x, \theta)) \right)
$$
* This is actually differentiable in $\lambda$
**The gradient is also a very simple expression:**
$$
\nabla_{\lambda} \mathcal{L}(\theta, \lambda) = \max\left(0, g(f(x, \theta))\right)
$$
* For satisfied constraints, the partial derivative is 0
* For violated constraints, it is equal to the violation (se ho una grande violation SIGNIFICA CHE DEVO AUMENTARE LAMBDA INFATTI)
### Lagrangian Dual Approach

**Therefore, we can solve our constrained ML problem**
...By alternating _gradient descent and ascent_:
* $\lambda^{(0)} = 0$
* $\theta^{(0)} = \operatorname{argmin}_\theta \mathcal{L}(\lambda^{(0)}, \theta)$
* For $k = 1..n$ (or until convergence):
  - Obtain $\lambda^{(k)}$ via an ascent step with $\nabla_{\lambda} \mathcal{L}(\lambda, \theta^{(k-1)})$ (questo mi permette di FAR AUMENTARE LA PENALTY. Questo E' POSSIBILE perche' aumentare LA PENALTY NON viola niente, LA FEASIBLE REAGION rimane intoccata)
  - Obtain $\theta^{(k)}$ via a descent step with $\nabla_{\theta} \mathcal{L}(\lambda^{(k)}, \theta)$
  
**Technically, we are working with sub-gradients here** (non ottimizzo mai ENTRAMBI INSIEME.)
* When we optimize $\lambda$ (outer optimization loop), we keep $\theta$ fixed
* ...Meaning we are going to under-estimate the gradient
Still, this is often good enough!

**We might still reach impractical values for $\lambda$**
...But the gentle updates will keep the gradient more stable
* At the beginning, SGD will be free to prioritize accuracy
* After some iterations, both $\theta$ and $\lambda$ will be nearly (locally) optimal

### Implementing the Lagrangian Dual Approach

**We will implement the Lagrangian dual approach via another custom model**

```python
class LagDualDIDIRegressor(MLPRegressor):
    def __init__(self, base_pred, attributes, protected, thr):
        super(LagDualDIDIRegressor, self).__init__()
        # alpha e' definita come trainable variable settata a 0
        self.alpha = tf.Variable(0., name='alpha')
        ...

    def __custom_loss(self, x, y_true, sign=1): ...

    def train_step(self, data): ...
        
    def metrics(self): ...
```

* We no longer pass a fixed `alpha` weight/multiplier
* Instead we use a _trainable variable_

**We move the loss function computation in a dedicated method (`__custom_loss`)**

```python
def __custom_loss(self, x, y_true, sign=1):
    y_pred = self.base_pred(x, training=True) # obtain the predictions
    mse = self.compiled_loss(y_true, y_pred) # main loss
    ymean = tf.math.reduce_mean(y_pred) # average prediction
    didi = 0 # DIDI computation
    for aidx, dom in self.protected.items():
        for val in dom:
            mask = (x[:, aidx] == val)
            didi += tf.math.abs(ymean - tf.math.reduce_mean(y_pred[mask]))
    cst = tf.math.maximum(0.0, didi - self.thr) # regularizer
    loss = mse + self.alpha * cst
    return sign*loss, mse, cst
```

* The code is the same as before
* ...Except that we can flip the loss sign via a function argument (i.e. `sign`)


**In the training method, we make _two distinct gradient steps:_**
```python
    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape: # first loss (minimization)
            loss, mse, cst = self.__custom_loss(x, y_true, sign=1)
        grads = tape.gradient(loss, self.trainable_variables)
        grads[-1] = 0 * grads[-1] # null multiplier gradient
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        with tf.GradientTape() as tape: # second loss (maximization)
            loss, mse, cst = self.__custom_loss(x, y_true, sign=-1)
        grads = tape.gradient(loss, self.trainable_variables)
        for i in range(len(grads)-1): # null weight gradient
            grads[i] = 0 * grads[i]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        ...
```
* In principle, we could even have used two distinct optimizers![[trainingconstraintviolation2.png]]
* Il risultato converge motlo meglio senza OSCILLAZIONI pazze.
* **The new approach leads _fewer oscillations at training time_**
* La LOSS giustamente AUMENTA perche' aggiungo la penalty sempre di piu
* ORA ho CHE cst AUMENTA, significa che le VIOLATIONS AUMENTANO. Questo perche' $\lambda$ ha un valore che NON E' GRANDE ABBASTANZA PER portare le PREDIZIONI nella FEASIBLE REAGION. Alla fine il livello di violation (la loss della violation) raggiunge un plateau.Questo e' il punto in cui gradient ascent ha reso lambda grande abbastanza da stoppare il modello NEL FARE PEGGIO. A quel punto SCENDO SCENDO SCENDO finche' non porto le predizioni nella FEASIBLE REGION. A STO PUNTO HO linea un po' seghettata: qui ho che lambda sta sulla BOUNDARY tra feasible e unfeasible. e cio' e' perfeto.
Ho con evaluation infatti
*R2 score: 0.64 (training), 0.56 (test)
DIDI: 0.98 (training), 1.06 (test)*
Il chie e' GRANDIOSO, ho perso solo 0.2 di R2 dal PRIMO ESPERIMENTO FATTO CHE ERA BIASED e ho guadagnato in DIDI (ora e' molto piu' basso, alla meta' del suo valore originale).
*  The DIDI has the desired value (ANCHE SE on the test set this is only roughly true: 1.05)
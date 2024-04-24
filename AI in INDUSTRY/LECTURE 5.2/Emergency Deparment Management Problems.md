Ho questo dataset di gente che arriva all'ospedale Maggiore di Bologna
* Each row refers to a single patient
* `Triage` e' quando avviene LA PRIMA constatazione clinica (quella VELOCE che fanno sommariamente) (sarebbe quando parli COL BRODER praticamente, quello seduto)
* `TKCharge` is the time when a patient starts the first visit
* `Code` refers to the estimated priority (white < green < yellow < red)
* `Outcome` discriminates some special conditions (people quitting, fast tracks)

VOGLIO PREDICTARE GLI ARRIVI, quindi quanta gente arrivera' 
NOTA: il _Triage_ e' un buon PROXY del tempo di arrivo di un paziente (non avendo il secondo disponibile, utilizzeremo il primo).

Calcolo I DELTAS, ogni delta e' il tempo che passa TRA UN ARRIVO E UN ALTRO.
Nella maggior parte dei casi ho un intervallo di tempo di 2-4 minuti. Purtroppo essendo il Triage diverso dall'effettuale tempo di arrivo di un paziente, questo risultato non e' molto affidabile (magari sono arrivato nello stesso momento di un altro ma siamo stati CHECKATI sommariamente in tempi diversi e a distanza di tempo CAUSA FILA).

SE PLOTTO LA DISTRIBUZIONE DEL TEMPO DI ATTESA (computata facendo TKCharge-Triage) ho una HEAVY-TAILED DISTRIBUTION, questo perche' in PRONTO SOCCORSO SI ASPETTA TANTOOOO.

SE PLOTTO LA CODE DISTRIBUTION, HO TANTI GREEN, MENO GIALLI, POCHI WHITE, ANCORA MENO I RED.
- Green code (low severity) form the majority of arrivals
- Yellow and red codes (mid and high severity) are in smaller numbers
- White codes (lowest priority) are also not very frequent

SE PLOTTO GLI OUTCOME, SONO PER LA MAGGIOR PARTE ADMITTED, A SEGUIRE ABANDONED E INFINE FAST TRACK (che e'?)


SE PLOTTO LA DISTRIBUZIONE SUI MESI E' UNIFORME ( A PARTE NOV E DEC MA LI' E' PERCHE' i dati sono smessi di esser collezionati a OCT del 2018)
SE plotto LA DIST. PER IL GIORNO DELLA SETTIMANA UNIFORME (forse un po' piu gente il LUN)
SE PLOTTO PER L'ORA HO UN CLEAR PATTERN:
![[hourArrivalDistMAGGIORE.png]]
# DATA PREPARATION - BINNING

**In our considered problem:**
* We are not going to revise our decisions continuosly
* We are not interested in predicting the next arrival
Rather:
* We will take decisions _at fixed intervals_
* We care about the expected arrivals _in a given horizon_
**Overall, we need to choose _a meaningful time unit_**
In other words, we need to perform some kind of binning
* We used binning to downsample high-frequency data
* Here we will use binning to _aggregate events with a variable frequency_

PRIMA DI TUTTO 
### One Hot encode the CODES
```python
codes = pd.get_dummies(data['Code'])
codes.set_index(data['Triage'], inplace=True)
codes.columns = codes.columns.to_list()
codes
```
_get_dummies_ praticamente applica un one-hot encoding a valori categorici. Quindi se magari un entry era 'Green' adesso sara' True, False, False, False, (il true si riferisce a Green, i False agli altri tipi di codici).

### Resampling the CODES
```python
codes_b = codes.resample('H').sum()
```
Praticamente sto metodo prende bins di UN'ORA e somma i valori all'interno di quel bin (tipo se ho in un'ora per il Green Code l'array \[True, False, True, True] mi dara' come risultato 3 per la GREEN feature).
### Computing the TOTAL number of arrivals in a hour
```python
cols = ['white', 'green', 'yellow', 'red']
codes_b['total'] = codes_b[cols].sum(axis=1)
codes_b
```

### Aggiungo TIME INFORMATION
Voglio aggiungere a sto dataset l'informazione piu' importante che sara' poi il mio INPUT ovvero l'informazione sul TEMPO (che sembrerebbe essere l'unica info a mia disposizione per fare questa PREDICTION del numero di pazienti).

```python
codes_bt = codes_b.copy()
codes_bt['month'] = codes_bt.index.month
codes_bt['weekday'] = codes_bt.index.weekday
codes_bt['hour'] = codes_bt.index.hour
codes_bt
```
Nota che la TIME information la prendo DALL'INDEX, non ho bisogno del row Triage dell'inizio e di far calcoli strani.

### Con sti BINS ora si puo' accedere alla COUNT VARIABILITY
![[hourplot.png]]
Questo e' il risultato di fare
```python
codes_bt[['hour', 'total']].boxplot(by='hour', figsize=figsize);
```
Ora ho anche la STD il che non e' male. Dal plot noto che la STD e' PICCINA nelle ore notturne (quindi NON ho grande variabilita' li') mentre e' molto piu' grande in quelle diurne. Ora posso farlo perche' ho accesso al valore del TOTAL che e' il mio target. PRIMA era solo un HISTOGRAM basato su COUNTING.

# Arrival PREDICTION
Uso come target IL TOTAL e come INPUT la time information di base.
Potrei usare come target una delle colonne dei CODICI volendo eh, in modo da predictare quella invece del TOTAL number of arrivals.
E' UN PROBLEMA DI REGRESSION SICURAMENTE.
NOTA, dal Boxplot sull'ora GIA' POTREI DARE UNA RISPOSTA AL MIO PROBLEMA. Li' letteralmente c'e' scritto qual'e' la mean del numero di arrivi e la STD porcamiseriaccia. Pero' chiaramente e' una predizione che NON tiene presente del mese e del giorno della settimana. Vorrei che la predizione CAMBIASSE in base a quelle altre due informazioni.

## Analyzing the Conditional Arrival Distribution
Voglio vedere la distribuzione seguita dai dati nel caso in cui FISSO UNA DETERMINATA ORA.
```python
tmp = codes_b[codes_b.index.hour == 6]['total']
tmpv = tmp.value_counts(sort=False, normalize=True).sort_index()
util.plot_bars(tmpv, figsize=figsize)
```
![[conddist.png]]
E' come se avessi zoommato all'interno del boxplot corrispondente all'ora 6 e ne avessi visto la distribuzione dei valori.
**NOTA** non e' una normal distribution
**When we need to _count occurrences over time_...**

It's almost always worth checking the **_Poisson distribution_**, which models:

* The number of occurrences of a certain event in a given interval
* ...Assuming that these events are _independent_
* ...And they occur at a _constant rate_

**In our case:**

* The independence assumption is reasonable (arrivals do not affect each other)
* The constant rate is true _for the conditional probability_
* ...Assuming that we condition using the right features
* I.e. those that have an actual correlation with the arrivals
### Poisson Distribution

**The Poisson distribution is defined by a single parameter $\lambda$**

$\lambda$ is the rate of occurrence of the events

* The distribution has a _discrete support_
* The Probability Mass Function is:
$$
p(k, \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}
$$
* Both the _mean_ and the _standard deviation_ have the same value (i.e. $\lambda$)
* The distribution skewness is $\lambda^{-\frac{1}{2}}$
  - For low $\lambda$ values, there is a significant positive skew (to the left)
  - The distribution becomes less skewed for large $\lambda$

DIFATTI SE MOSTRO UNA POISSON DISTRIBUTION con MEAN CALCOLATA DAI dati a ORA FISSATA A 6.00 am allora ho un BUON MATCH tra le due distribuzioni.
```python
tmp = codes_b[codes_b.index.hour == 2]['total']
mu = tmp.mean()
dist = stats.poisson(mu)
x = np.arange(tmp.min(), tmp.max()+1)
util.plot_bars(tmpv, figsize=figsize, series=pd.Series(index=x, data=dist.pmf(x)))
```
![[poissondist.png]]
VOLENDO potrei andare su ogni ORA del giorno, per ogni GIORNO della settimana e per ogni MESE immagino e calcolare le mean $\lambda$. Pero' sembra una cosa molto dispendiosa. Questo e' un approccio chiamato LOOK UP TABLE come e' scritto qui sotto:
**We could build a table**
For example, we could compute average arrivals for every hour of the day
* These correspond to $\lambda$ for that hour, so we target the correct distribution
* ...But the approach has trouble scaling to multiple features
**We could train a regressor as usual**
For example a Linear Regressor or a Neural Network, with the classical MSE loss
* If we do this, it's easy to include multiple input features
* ...But we would be targeting the wrong type of distribution! (cioe' non avremmo la possibilita' di tenere into account la POISSON DISTRIBUTION, perche' con regression si usa la MSE che e' legata a NORMAL DISTRIBUTION).
## Neuro-Probabilistic Models
**In practice there is an alternative**
Let's start by build a _probabilistic model_ of our phenomenon:
$$
y \sim \text{Pois}(\lambda(x))
$$
* The number arrivals in a 1-hour bin (i.e. $y$)
* ...Is _drawn from a Poisson distribution_ (parameterized with a rate)
* ...But _the rate is a function_ of known input, i.e. $\lambda(x)$
**Then we can approximate lambda using an estimator**, leading to:
$$
y \sim \text{Pois}(\lambda(x, \theta))
$$
* $\lambda(x, \theta)$ can be any model, with parameter vector $\lambda$
This is a _hybrid_ approach, combining statistics and ML

**How do we train this kind of model?**
Just as usual, i.e. for (empirical) maximum log likelihood:
$$
\mathop{\text{argmin}}_\theta - \sum_{i=1}^m \log f(\hat{y}_i, \lambda(\hat{x}_i, \theta))
$$
* Where $f(\hat{y}_i, \lambda)$ is the probability of value $\hat{y}_i$ according to the distribution
* ...And $\lambda(\hat{x}_i, \theta)$ is the estimate rate for the input $\hat{x}_i$
**In detail, in our case we have:**

$$
\mathop{\text{argmin}}_\theta - \sum_{i=1}^m \log \frac{\lambda(\hat{x}_i, \theta)^{\hat{y}_i} e^{-\lambda(\hat{x}_i, \theta)}}{\hat{y}_i!}
$$
...Which is differentiable and _can be solved via gradient descent_!

## Building a Neuro-Probabilistic Model
**We can build this class of models by using custom loss functions**
...But it's easier to use a library such as [TensorFlow Probability](https://www.tensorflow.org/probability)
* TFP provides a layer the abstracts [a generic probability distribution](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DistributionLambda):
```python
tfp.layers.DistributionLambda(distribution_function, ...)
```
* And function (classes) to model [many statistical distributions](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions), e.g.:
```python
tfp.distributions.Poisson(log_rate=None, ...)
```
**About the `DistributionLambda` layer**
* Its input is a symbolic tensor (like for any other layer)
* Its output is tensor of probability distribution _objects_
* ...Rather than a tensor of numbers

**The `util` module contains code to build our neuro-probabilistic model**
```python
def build_nn_poisson_model(input_shape, hidden, rate_guess=1):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
    log_rate = layers.Dense(1, activation='linear')(x)
    lf = lambda t: tfp.distributions.Poisson(rate=rate_guess * tf.math.exp(t))
    model_out = tfp.layers.DistributionLambda(lf)(log_rate)
    model = keras.Model(model_in, model_out)
    return model
```
* An MLP architecture computes the `log_rate` tensor (corresponding to $\log \lambda(x)$)
* Using a log, we make sure the rate is _strictly positive_
* A `DistributionLambda` yield the output (a distribution object)
Vorrei FARTI CAPIRE COSA SUCCEDE QUA:
praticamente sta rete, dato un INPUT mi calcola UN RATE (messo a log per far in modo che il valore non sia negativo). Questo rate viene poi utilizzato per CALCOLARE IL VALORE RITORNATO DALLA POISSON. Quindi praticamente il modello impara negli weights degli hidden a mappare gli inputs al rate $\lambda$ legato a quegli specifici input temporali. E' molto meglio che fare lookup table.

### Initialization issue
**We need to be careful about _initial parameter estimates_**
```python
def build_nn_poisson_model(input_shape, hidden, rate_guess=1):
    ...
    lf = lambda t: tfp.distributions.Poisson(rate=rate_guess * tf.math.exp(t))
    ...
```
* Assuming standardized/normalized input, under default weight initialization
* ...The `log_rate` tensor will be initially close to 0
* Meaning out rate $\lambda$ would be initially close to $e^0 = 1$
**We need to make sure that this guess is _meaningful for our target_**
* In the code, this is achieve by scaling the rate
* ...With a guess that must be passed at model construction time\
Quindi scalandolo con _rate_guess_ lo metto da subito a valori che magari producono UNA MIGLIORE INIZIALIZZAZIONE del _rate_.

## Training a Neuro-Probabilistic Model
**Training the model requires to specify the loss function**
...Which in our case is the _negative log-likelihood_
* So, it turns out we do need a custom loss functions
* ...But with TFP this is easy to compute
**In particular, as loss function we _always_ use:**
```python
negloglikelihood = lambda y_true, dist: -dist.log_prob(y_true)
```
* The first parameter is the observed value (e.g. actual number of arrivals)
* The second is the distribution computed by the `DistributonLambda` layer
* ...Which provides the method `log_prob`

**IMPORTANTE** TRASFORMO L'ORA IN ONEHOTENCODING in modo da AIUTARE il mio NEUROPROBABILISTIC MODEL.
```python
np_data = pd.get_dummies(codes_bt, columns=['hour'], dtype='int32')
np_data.iloc[:2]
```
In sto modo infatti AIUTO IL MODELLO A IMPARARE UNA LOOKUP TABLE (che e' di base quello che deve fare). In sto modo infatti viene usato SOLO il weight legato all'input 1 per calcolare il risulato sull'hidden layer (per ogni hidden layer).

POI:
**The input data need to be standardized/normalized as usual**
In our case, we do this only for weekday (the hours are already $\in \{0, 1\}$)

```python
np_tr_in['weekday'] = np_tr_in['weekday'] / 6
```
IMMAGINO IO DEBBA STANDARDIZZARE ANCHE IL MONTH (cosa non fatta dal prof lol)(NON LO FA PERCHE' NON E' STATA AGGIUNTA COME INFO ci sta non e' cosi informativo).
As a rate guess we use the average over the training set (ha senso perche' praticamente sara' il mio primo valore per la MU della POISSON e sto rate non e' altro che una mean (NON su tutto il dataset ma solo su specifiche ORE ma e' comunuque un buon proxy e una buona inizializzazione)).
* This is easy to compute
* ...And will provide a better starting point for gradient descent
```python
negloglikelihood = lambda y_true, dist: -dist.log_prob(y_true)
nnp = util.build_nn_poisson_model(input_shape=len(in_cols), hidden=[], rate_guess=np_tr_out.mean())
history = util.train_nn_model(nnp, np_tr_in, np_tr_out, loss=negloglikelihood, validation_split=0.0, batch_size=32, epochs=30)
util.plot_training_history(history, figsize=figsize)
```
NOTA: gli hidden layers SONO MESSI A 0, in questo caso IL ONEHOT sull'ora ha ancora piu' senso perche si va direttamente sul Dense(1, activation="linear") che quindi prende SOLO il peso legato a quell'ora (che sara' NIENTE POPO DI MENO CHE DICIAMO IL LAMBDA CHE ERA IMPARABILE solo con uno studio statistico sui dati (facendo la mean degli arrivi su quell'ora)) e il PESO del giorno della settimana praticamente, che potrebbe INFLUIRE nel cambiare un po sto $\lambda$ legato a quell'ora.

## Predictions
**When we call the `predict` method on the model we obtain _samples_**
This means that the result of `predict` is _stochastic_ (ci sta, sto praticamente samplando da una poisson)
* Then we can call _methods over the distribution objects_
* ...To obtain means, standard deviations, and any other relevant statistics
## Evaluation
```python
tr_pred = nnp(np_tr_in.values).mean().numpy().ravel()
util.plot_pred_scatter(np_tr_out, tr_pred, figsize=figsize)
```
RISULTATI SU TRAIN:
*R2: 0.60
MAE: 1.93*
* This is a _stochastic_ process, making this $R^2$ value _very good_
* When the stochasticity is too high, using the $R^2$ _might not even be viable_
SUL TEST SET:
```
ts_pred = nnp(np_ts_in.values).mean().numpy().ravel()
util.plot_pred_scatter(np_ts_out, ts_pred, figsize=figsize)
```
Risultati:
R2: 0.60
MAE: 1.94
BUONO...niente overfitting
## Confidence Intervals

**Since our output is a distribution, we have access to _all sort of statistics_**

Here we will simply show the mean and stdev over one week of data:

```python
ts_pred_std = nnp(np_ts_in.values).stddev().numpy().ravel()
util.plot_series(pd.Series(index=np_ts_in.index[:24*7], data=ts_pred[:24*7]), std=pd.Series(index=np_ts_in.index[:24*7], data=ts_pred_std[:24*7]), figsize=figsize)
plt.scatter(np_ts_in.index[:24*7], np_ts_out[:24*7], marker='x');
```
![[confinterval.png]]
E' un risultato buono, ovviamente gli arrivi sono MOOOLTO NOISY pero' per quello che era ho fatto un buon risultato.


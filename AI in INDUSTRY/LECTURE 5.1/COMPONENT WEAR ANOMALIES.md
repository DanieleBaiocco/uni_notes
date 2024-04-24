**Let's consider the Vega skinwrapper family of packaging machines by [OCME](https://www.ocme.com/en?_ptc=4)**
* They work by wrapping products (bottles) in a _plastic film_
* ...Which is _cut and heated_, so that the film shrinks and stabilizes the content

**A public dataset for a skinwrapper is [available from Kaggle](https://www.kaggle.com/inIT-OWL/one-year-industrial-component-degradation)**
* The dataset contains _a single run-to-failure experiment_
* I.e. the machine was left running until its blade became unusable

**This is an example of anomaly _due to component wear_**
* It's a common type of anomaly
* ...And run-to-failure experiments are a typical way to investigate them

**All problems in this class share a few _properties_**
* There is a critical anomaly _at the end of the experiment_
* The behavior becomes _more and more distant_ from normal over time
...Meaning that they are good fit for many of the techniques we have studied

* The data refers to disjoint measurement windows
* Each segment contains data sampled _every 4ms_

![[lecture5columns.png]]
This is the data plotted after standardizing them, as feature map. La COLONNA 2 e' sospetta perche' e' tutto bianco. Cio' succede quando O IL DATO E' SEMPRE UGUALE O VARIA TROPPO. In sto caso e' perche' varia troppo (lo scopro qua sotto).

After having had a look at the data, what I found is that:
1. No missing value in the dataset
2. The data is neither normalized nor standardized
3. There are 519 segments overall each with 2048 samples
4. The mode is _a controlled parameter_ and does not change in the **middle** of a **segment**. Intuitively, the mode _has an impact_ on the machine behavior
5. **Column 8 is also a fixed over long periods of time***. This is _likely_ a controlled parameter. Ideally, we would speak with the customer (but in this excercise we can't)
6. **Column 2 peaks repeatedly over _short time periods_**.  There is nothing really wrong with this and it explains the mostly white row in our previous plot![[column2.png]]
7. **Column 3 contains an odd, localized, deviation**: This is likely the result of manual adjustment.We'd better keep this column off![[column3.png]]
8. **the same holds for column 5**. Anche in questo caso e' meglio dropparla. (NOTA CHE ENTRAMBE LE DEVIAZIONI ERANO NOTABILI GIA' DALLA HEATMAP della colonna 3 e della 5).

### BINNING APPROACH - PREPROCESSING
**This dataset contain _high-frequency data_ (4ms sampling period)**
* In this situation, feeding the raw data to a model does not usually make sense
* So we will use **SUBSAMPLING**
**A binning approach typically works as follows:**
We apply a sliding window, but so that its consecutive applications _do not overlap_
* Each window application is called a **_bin_**
* ...From which we extract one or more _features_
* ...By applying different **_aggregation functions_** (posso per esempio prendere LA MEAN, L'STD, non so, dei valori dentro QUEL BIN e il risultato saranno nuove features legate a quel dato, PIU' GRANULARI ma di numero maggiore).
**The result is series that contains a _smaller number of samples_**
...But typically a _larger number of features_


**We will apply binning to all columns not related to time** except for the two we chose to discard.
```
feat_in_r = data2.columns[[0, 1, 2, 4, 6, 7, 8]]
print(list(feat_in_r))
```
**First, we define which aggregation function to apply to each field**
```
aggmap = {a: ['mean', 'std', 'skew'] for a in feat_in_r}
aggmap['mode'] = 'first'
aggmap['pSpintor::VAX_speed'] = 'first'
```
For the features that are fixed over a segment, we pick the first value (ovvero mode e pSpintor::VAX_speed)
This is the code for BINNING
```python
binsize = 512 # 2 seconds of measurements
bins = []
for sname, sdata in data.groupby('segment'):
    sdata['bin'] = sdata.index // binsize # Build the bin numbers
    tmp = sdata.groupby('bin').agg(aggmap) # Apply the aggregation functions
    bins.append(tmp)
data_b = pd.concat(bins)
```
* We process each segment individually
* ...So that we are sure that no bin overlaps two segments

**We chose our bin size based on:**
* Having enough data to compensate noise
* Capture regular patterns (e.g. our spiking signal)

 Questo e' il RISULTATO
 ![[BINNING result.png]]
 We have much fewer rows, and more columns

NOTA in questo caso d'uso NON posso usare l'approccio di prima di REGRESSION/CLASSIFICATION per stimare la RUL, perche' HO SOLO UNA RUN (nell'altro caso avevo MOLTISSIME RUNS quindi la mia rete poteva imparare bene, in sto esperimento SOLO UNA VOLTA ho che la macchina si guasta, quindi imparare dai valori dei dati significa imparare SOLO dai dati di quando si sta per rompere la macchina (IN UNA SOLA OCCASIONE BRO e' poco)).

Immagino di poter usare una strategia di Density Estimation come una Gaussian Mixture FITTATA su dati SANI (di quando la macchina opera bene). Posso poi creare un SIGNAL per ogni punto e valori "strani" mai visti comporterrebbero un ALTO SIGNAL. => a quel punto posso magari dire "LA MACCHINA SI STA ROMPENDO".

# AUTOENCODER - BASELINE
**We'll try to detect the component state by learning an autoencoder**
* We'll train a model on the earlier data
* ...And then use the reconstruction error as a proxy for component wear

**We start as usual by splitting the training and test set** and then by standardizing the data (ricordati che AUTOENCODERS usano GD).
Dopo aver fittato sul train set (in cui ci sono solo valori SANI), ho come risultato dei segnali BRUTTI: degli SPIKES dove mi sarei aspettato un SIGNAL molto basso, anche verso L'INIZIO proprio.
![[risultatoSpikes.png]]
Questo e' il reconstruction error, come possiamo vedere, l'errore sta soprattutto in due colonne: la 2 e la 15.
![[reconstructionError.png]]
Gli spikes potrebbero essere dovuti a samples che NON ASSOMIGLIANO per niente agli altri dati del trianing set (anche samples nel training set stesso, da cui il modello non ha imparato niente perche' POCHI e quindi mezzi ignorati). Samples MOLTO RARI. Nel DATASET ci sono comunque dei CONTROLLED PARAMETERS come la MODE. Se per esempio nel training data c'e' UN SACCO LA MODE 1  e nel TEST DATA invece c'e' un'altra tipo di MODE e' ovvio che il risultato generalizzera' male nel test data. DEVO ASSOLUTAMENTE ACCERTARMI CHE LA DISTRIBUZIONE DELLE MODES SIA EQUA (spoiler NON LO E')
#  Altering the Training Distribution
Questa e' la distribuzione delle MODES su tutto il dataset:
*1        916
2        484
3        280
5        240
6         72
4         48
8         24
7         12*
Se vado INVECE A VEDERE LA DISTRIBUZIONE SUL TRAINING E QUELLA SUL TEST ho, come GIA' DETTO  E TEORIZZATO PRIMA una differenza di distribuzione:
![[distribution_discrepancy.png]]
POSSO ALTERARE LA PROBABILITY DISTRIBUTION DEL MIO TRAINING DATA. Posso fare due cose
* Usare un SELECTION BASED ENSEMBLE in cui traino PER OGNI MODE un modello (CONS: ho pochi dati in sto modo)
* Fare UPSAMPLING per modes meno frequenti e DOWNSAMPLING per mode piu' frequenti (FARO' questo)
PER CAPIRE COSA NON VA ECCOLO SPIEGATO:
**We are _training for maximum likelihood_**
...Ideally we would like to solve:

$$
\mathop{\text{argmax}}_{\theta} \mathbb{E}_{x, y \sim P}\left[ \prod_{i=1}^m f_\theta(y_i \mid x_i) \right]
$$

* $P$ represents the real (joint) distribution
* $f_\theta(\cdot \mid \cdot)$ is our estimated probability, with parameter vector $\theta$
* I.e. an estimator for a conditional distribution
* We distinguish $x$ (input) and $y$ (output) to cover generic supervised learning
* ...Even if for an autoencoder they are the same
MA IN PRATICA NON ho accesso alla FULL DISTRIBUTION, quindi USO IL TRAINING SET sperando sia RAPPRESENTATIVO DELLA FULL DISTRIBUTION, che e' una  Monte-Carlo approximation:

$$
\mathop{\text{argmax}}_{\theta} \prod_{i=1}^m f_\theta(y_i \mid x_i)
$$

* Typically, we consider _a single sample_ $x, y$ (i.e. the _training set_)
* The resulting objective (i.e. the big product) is sometimes called _empirical risk_

**Problems arise when _our sample is biased_.** E.g. because:
* We can collect data only under certain circumstances
* The dataset is the result of a selection process
* ...Or perhaps due to pure sampling noise

**So, Our issue is that the training sample is biased (it is _not representative_ of the true distribution)
> **How can we deal with this problem?**

* A possible solution would be to _alter the training distribution_
* ...So that it _matches more closely_ the test distribution
**...And this is actually something we can do!**
* E.g. we can use data augmentation, or subsampling
* ...Or we can use _sample weights_


### Importance Sampling (sample weights)
IL PROCEDIMENTO E' IL SEGUENTE:
**Let our training set consist of $\{(x_1, y_1), (x_2, y_2)\}$**
The corresponding optimization problem would be:
$$
\mathop{\text{argmax}}_{\theta} f_\theta(y_1 \mid x_1) f_\theta(y_2 \mid x_2)
$$
If sample \#2 occurred twice in the training data, we would have
$$
\mathop{\text{argmax}}_{\theta} f_\theta(y_1 \mid x_1) f_\theta(y_2 \mid x_2)^2
$$
Normalizing over the number of samples does not change the minima:

$$
\mathop{\text{argmax}}_{\theta} f_\theta(y_1 \mid x_1)^{\frac{1}{3}} f_\theta(y_2 \mid x_2)^{\frac{2}{3}}
$$
E questo qua sopra e' solo una considerazione sugli esponenti dell'optimization function di un training set in cui un'istanza e' ripetuta due volte.

A general training problem based on **Empirical Risk Minimization** (che e' quello che HO IO, il mio dataset) is the form:
$$
\mathop{\text{argmax}}_{\theta} \prod_{i=1}^m f_\theta(y_i \mid x_i)
$$
We can virtually _alter the training distribution_ via exponents:

$$
\mathop{\text{argmax}}_{\theta} \prod_{i=1}^m f_\theta(y_i \mid x_i)^{w_i}
$$
* We can do this to make the training distribution _more representative_
* E.g. when we expect a discrepancy between the training and test distribution

**When we switch to log-likelihood minimization**
...The exponents become _sample weights_

$$
\mathop{\text{argmin}}_{\theta} - \sum_{i=1}^m w_i \log f_\theta(y_i \mid x_i)
$$
We can _always_ view the weights as the ratio of two probabilities:
$$
w_i = \frac{p^*_i}{p_i}
$$
* $p_i$ is the **sampling bias** that we want to _cancel_
* $p^*_i$ is the **distribution we wish to _emulate_**

**This approach is known as _IMPORTANCE SAMPLING_**

IN PRACTICE:
**Let's apply the approach to our skinwrapper example**

We know there's an _unwanted sampling bias_ for some modes of operation

* Let $m(x_i)$ be the mode of operation for the $i$-th sample
* Then we can estimate $p_i$ as a frequency of occurrence:

$$
p_i = \frac{1}{n} \left|\{k : m(x_k) = m(x_i), k = 1..n\}\right|
$$
NOTA, IL CALCOLO E' FATTO SOLO SUL TRAINING SET (non posso guardare il test set per computare $p_i$).
We _don't want_ out anomaly detector to be sensitive to the mode
* So we can assumption a uniform distribution for $p^*_i$:

$$
p^*_i = \frac{1}{n}
$$
I remember that 
$$
w_i = \frac{p^*_i}{p_i}
$$
 **By combining the two we get:**

$$
w_i = \frac{1}{\left|\{k : m(x_k) = m(x_i), k = 1..n\}\right|}
$$

* I.e. the weight is just the inverse of the corresponding mode count

**We can compute the weigths by first obtaining inverse counts for all modes**
In code we have that :
```python
vcounts = data_b_tr['mode', 'first'].value_counts()
mode_weight = 1 / vcounts
```
Associo poi ogni peso a ogni SAMPLE nel mio dataset in base alla MODE di appartenenza
```python
sample_weight = mode_weight[data_b_tr['mode', 'first']]
```
Il training con sample weights si fa cosi': 
```python
nn2 = util.build_nn_model(input_shape=len(data_b.columns), output_shape=len(data_b.columns), hidden=[len(data_b.columns)//2])
history = util.train_nn_model(nn2, data_b_s_tr, data_b_s_tr, loss='mse', validation_split=0.0, batch_size=32, epochs=400, sample_weight=sample_weight)
util.plot_training_history(history, figsize=figsize)
```

Ho che DOPO IL TRAINING 
* Suspected anomalies in the middle sequence have almost disappeared
* ...And there is a much clearer plateau at the end of the signal

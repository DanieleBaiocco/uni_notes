Considero un database in cui ho RISTORANTI con 'avg_rating', 'num_reviews', 'D' (che indica quanto costa) e come target ho se il cliente HA CLICCATO O MENO su quel ristorante.
In sto modo voglio comprendere le **caratteristiche** che rendono un ristorante piu' CLICCABILE.
1. 'avg_rating' e' tra 0 e 5
2. 'num_reviews' mi dicono quante recensioni
3. 'dollar_rating' mi dice QUANTO spendero'
4. 'clicked' e' il target e mi dice se UN UTENTE ha cliccato o meno (questa NON e' una vera classe)
'clicked' **NON e' una vera classe** perche' e' STOCASTICA (HO eventi): posso presentare lo stesso ristorante a due UTENTI differenti e magari UNO lo clicca e l'altro NO.
Dal punto di vista del training NON FA DIFFERENZA. La differenza sta NELL'INTERPRETAZIONE del risultato. IN QUESTO CASO NON posso trasformare la probabilita' in OUTPUT in 0 o 1. Deve rimanere una PROBABILITA' quella predictata (non ho una classe).
STO IN QUESTO USECASE in un CONTROLLED SETTING, quindi HO ACCESSO ALLA **TARGET FUNCTION**.
E cio' che esce fuori e':
![[clickthrough.png]]
Noto che valori piu' alti di 'num_reviews' e 'avg_rating' portano a una PROBABILITA' PIU' ALTA DI CLICCARE. Noto anche che ristoranti con avg pricing (DD) tendono a essere piu' cliccati.

Se guardo la DATA DISTRIBUTION del training set ho che ho:
![[clickthroughdist.png]]
Nel VAL ho:
![[clickthroughdistval.png]]
Nel TEST ho:
![[clickthroughdisttest.png]]
**NOTA** nel TEST SET ho una distribuzione completamente diversa. IN train e val ho UNIFORM DIST. sul numero di reviews. Quando guardo il test noto che ho UN SACCO DI RISTORANTI con 0 o comunque con un basso numero di reviews. Ho poi pochi ristoranti con numero di reviews alto. CIO' E' CIO' CHE ACCADE NELLA REALTA'.

HO UNA DISCREPANCY perche' il train e val set li ho creati da TripAdvisor CHE CONSIGLIA ristoranti che SONO GIA' ALTI NEL NUMERO DELLE REVIEWS e nell'AVG RATING (consiglia BUONI RISTORANTI). Questa cosa si chiama **SELECTION BIAS**: l'app usa delle euristiche hand-crafted che mi mostrano solo ristoranti con alte reviews. Quindi gli utenti interagiscono SOLO con ristoranti perlopiu' buoni (con molte recensioni), NON interagendo invece con ristoranti MID (con poche reviews e con basso avg rating).
NON CI SONO QUINDI nel mio training/val set dati riferiti a ristoranti meno conosciuti (e' un problema, il mio training set non e' un buon proxy della REALE DISTRIBUZIONE).

**NOTA** la distribuzione del TEST SET solitamente NON e' ottenibile (cioe' la TRUE DISTRIBUTION non  e' disponibile) perche' NON ho TEST SET banalmente. Quindi sto cheattando, solo perche' e' un CONTROLLED EXPERIMENT. Nella vita reale DEVO CAPIRLO da solo che c'e' un selection bias parlando col CLIENTE.

L'approccio del passato a questa cosa e' usare **sample weights**. Ma TRASFORMARE la distribuzione del train delle num_reviews in quella del TEST delle num_reviews e' difficile (non ho come target distribution una UNIFORM, ho una distribuzione particolare).

**CIO' CHE POSSO FARE E' QUESTO PIUTTOSTO**: parlando col CLIENTE se trovo qualche REGOLA EMPIRICA CHE E' VERA sempre, allora posso INTRODURLA nel modello SOTTO FORMA DI CONSTRAINT, usando per esempio Lagrangian Approach. 
Quindi ho dopo un  data driven model che usa anche constraints, quindi e' piu' ROBUST.

Da sto model HO COME EMPIRICAL RULES che
1. se aumento 'avg_rating' ALLORA LA click proba andra' SU
2. se aumento il 'num_reviews', ALLORA LA click proba andra'SU
Voglio introdurre questi due constraints nel mio modello
MA PRIMA:
# Baseline Approach

1. Standardizzo gli attributi numerici
2. Rendo onehot encoding il 'dollar_rating'
Traino un MLP con una sigmoid alla fine usando binary_crossentropy come loss.
## Evaluating Baseline Approach
Non essendo questo un CLASSIFICATION PROBLEM non ha senso che utilizzo ACCURACY come metrica. 
Mi serve altro, uso la ROC CURVE.
A Receiver Operating Characteristic curve is a type of plot
* We consider multiple threshold values
	- Each threshold is meant to be used for discriminating between classes
	  - The usual rounding approach is equivalent to a 0.5 threshold
* On the $x$ axis, we report the false positive rate for each threshold
* On the $y$ axis, we report the true positive rate for each threshold

* The large the Area Under Curve (AUC), the better the performance
* The AUC value is guaranteed to be in the $[0, 1]$ interval
Ho come risultati nel mio caso 
*AUC score: 0.81 (training), 0.80 (validation), 0.76 (test)*
**NOTA** QUANDO voglio VALUTARE PROBABILITA' invece di CLASSI allora ROC CURVE e' una buona metrica

Il modello funziona bene su training set, ma un po' meno bene su test set come mi immaginavo.
![[clickthroughMLPpred.png]]
Queste sono le predictions come erano mostrate prima. Se guardi le differenze col ground truth
Nei risultati HO che a VOLTE aumentando 'average_rating',tenendo fisso 'num_reviews', la probabilita' di CLICK DIMINUISCE (localmente, per piccoli tratti per poi riaumentare). Ma gia' che diminuisca NON ha senso. So che c'e' una relazione di MONOTONIA tra avg_rating e click.

NON ci sono molti ML approaches che prendono in conto MONOTONICITY CONSTRAINTS. In KERAS linear regression SUPPORTA questo constraint aggiungendo al modello il constraint che i pesi NON SIANO NEGATIVI (in questo modo ho MONOTONICITY perche' per gli attributi a cui applico questo constraint ho che quando aumentano aumenta anche  il target). Nei DECISION TREES anche posso, perche' li' a ogni step devo decidere quale split fare con quale feature. Posso fare in modo che scarto tutti i possibili splits che non rispettano MONOTONICITY.
L'alternativa migliroe pero' forse e' usare 
## LATTICE MODELS
Suppongo che ho due variabli x e y. Una LATTICE e' una PIECEWISE LINEAR SURFACE che e' definita nello SPAZIO di queste due variabili.
E' una piecewise linear function, cio' significa che DEFINIISCO IL VALORE DELLA FUNZIONE **SOLO** per pochi punti chiamati KNOTS e per ogni altro punto, calcolo il valore della funzione tramite LINEAR INTERPOLATION dei KNOTS vicini (questa cosa si vede dall'immagine dai)
![[lattice.png]]
Il numero di QUESTI nodi e' specificabile per ogni asse. Ad esempio nell'immagine sembrerebbe che siano 3 per l'asse x e 4 per l'y ma non vorrei sbagliarmi. A seconda delle posizioni POI si computa la CARTISIAN PRODUCT che mi crea una GRID per cui TROVO I VALORI DELLA FUNZIONE.
Questo e' un modello perche' se ho il valore della funzione a livello del KNOT, allora ho il valore PER OGNI ALTRO PUNTO del modello.
Sta LATTICE puo' rappresentare una funzione arbitraria (non linear).
Questa LATTICE e' OTTIMIZZABILE tramite GRADIENT DESCENT.
Questo modello e' **INTERPRETABILE**.
**NOTA** nel caso della LINEAR REGRESSION e' COSI' semplice ASSICURARSI che ci sia MONOTONICITY perche' il modello E' INTERPRETABILE. So come le weights influiscono sull'output e cosa significano, e quindi e' facilissimo introdurre il constraint. Uguale qui. 
Qui posso mettere dei CONSTRAINTS sui KNOTS. Tipo nella figura di sopra se considero i KNOTS che stanno lungo x=2.5. Basta che dico al modello che i VALORI dei KNOTS devono essere uno MINORE UGUALE DEL SUCCESSIVO, in sto modo ho monotonicity (all'aumentare di y mi aumenta z). COME faccio in NN bro dai. Che glie dico ai weights di NN? eh.
LATTICE MODEL non viene utilizzato per SCALABILITY ISSUES, se ho tante input features LA SIZE DELLA GRID dei KNOTS aumenta ESPONENZIALMENTE. Se ho 3 FEATURES e 2 KNOTS per ogni asse ho $2^3$. 

## COSTRUIRE UN LATTICE MODEL
Prima di tutto ne costruisco uno SENZA dei constraints.
Costruisco la LATTICE SIZE, ovvero quanti KNOTS usare per ogni asse (ho 6 features ti ricordo: avg_rating, num_reviews, dollar_rating onehotencodato)
```
lattice_sizes = [4] * 2 + [2] * 4
```
Risultato: *\[4, 4, 2, 2, 2, 2]*
Uso quindi 4 KNOTS per i NUMERIC INPUTS E 2 KNOTS per i BOOLEAN INPUTS
Devo poi fare questo: 
```python
tr_ls = [tr_sc[c] * (s-1) for c, s in zip(dt_in_c, lattice_sizes)]
val_ls = [val_sc[c] * (s-1) for c, s in zip(dt_in_c, lattice_sizes)]
ts_ls = [ts_sc[c] * (s-1) for c, s in zip(dt_in_c, lattice_sizes)]
```
Ricordo che ho i valori numerici che erano standardizzati. Ad ogni modo per ogni attributo i valori al suo interno devono essere moltiplicati per il numero di KNOTS legato a quell'attributo. In sto modo ogni input e' scalato nel range \[0, $n_knots -1$]. Cio' e' richiesto dalla libreria che usero'. 
Inoltre devo avere che l'input deve essere diviso in MULTIPLE INPUT COLUMNS (ASSURDO chi ha scritto sta libreria e' un cane):
```python
mdl_inputs = []
for cname in dt_in_c:
    cname_in = layers.Input(shape=[1], name=cname)
    mdl_inputs.append(cname_in)
```
Creo la LATTICE:
```python
import tensorflow_lattice as tfl

mdl_out = tfl.layers.Lattice(lattice_sizes=lattice_sizes,
    output_min=0, output_max=1, name='lattice',
)(mdl_inputs)

lm = keras.Model(mdl_inputs, mdl_out)
```
Da un punto implementativo in TENSORFLOW il LATTICE e' un tf Layer.
Lo traino con loss a binarycrossentropy.
Lo valuto. Il risultato e':
*AUC score: 0.82 (training), 0.79 (validation), 0.76 (test)*
Ho piu' o meno gli stessi risultati di prima. Con predizione nel TEST un po' bruttina come prima. Se vado a vedere i risultati plottati come all'inizio ho![[latticemodel.png]]
Nota che ci sono punti piu definiti che sono praticamente i GRID POINTS. Da questi poi si scende o si sale perche' il modello fa linear interpolation tra knots vicini per fare uina prediction per tutti gli altri valori.
![[calibration.png]]
* The expected monotonicity constraints are _still violated_
* There are still many _mistakes for less represented areas_ of the input space
La differenza tra LATTICE  e una NN e' che la NN puo' spostare la posizione dei KNOTS, cambiarne il numero, creando grid con maggior numero di valori.
Anche qui la monotonicity non e' rispettata, e' ANCHE PEGGIO DI PRIMA.

## CALIBRATION
Ho un grosso problema con il LATTICE MODEL: la scalability. Praticamente dal numero di KNOTS si genera un NUMERO troppo alto di GRID POINTS. Il numero di GRID POINTS e' 
$$
n = \prod_{i=1}^m n_i
$$
con $n_i$ NUMERO DI KNOTS per ogni dimensione da 1 a m.
Per rendere la LATTICE scalabile faccio questo:
CREO UNA 1-D LATTICE PER OGNI attributo  NUMERICO, in modo tale che HO TANTI GRID POINTS quanti sono I KNOTS. Faccio poi linear interpolation per ottenere i valori della funzione per gli altri punti.

Per CATEGORICAL ATTRIBUTES posso specificare un KNOT per ogni CATEGORIA e semplicemente ho un valore predictato per quella categoria. NON HO UNA LINEAR INTERPOLATION perche' NON ho valori tra categoria. Quindi creo praticamente una semplice LOOKUP TABLE.
POSSO ora magari specificare anche 5 KNOTS per ogni NUMERIC ATTRIBUTE e poi usare magari SOLO 2 knots per l'OUTPUT di questa lattice per creare un'altra LATTICE che stavolta e' pero' molto piu' semplice.
Quindi SE VOGLIO 5 GRID VALUES PER OGNI ATTRIBUTO, e ho 2 ATTRIBUTI:
1. Nel primo caso(quello della sezione precedente) ottengo 5x5=25 GRID POINTS
2. Nel secondo caso(quello di adesso) ottengo 5x2 (il numero di knots fionddamentalmente) + 2x2 = 14
Cio' aumenta il bias e riduce la variance.
INOLTRE, per quanto detto su come la Calibration funziona sui CATEGORICAL ATTRIBUTES, calibration permette di usare CATEGORICAL INPUTS senza ONEHOTENCODING. Questo permette ulteriormente di migliorare sotto il punto di vista dei PARAMETRI (ho ancora meno GRID POINTS).
1. Preparo la lattice size del lattice DOPO la calibration. Questa sara' fatta da due KNOTS per feature, e le features saranno 3 perche' NON AVRO' PIU' ONEHOTENCODING
```lattice_sizes2 = [2] * 3```
2. Rimpiazzo la category data  del 'dollar_rating' con i CODICI 
```python
tr_sc2 = tr_s.copy()
tr_sc2['dollar_rating'].astype('category').cat.codes

val_sc2 = val_s.copy()
val_sc2['dollar_rating'].astype('category').cat.codes

ts_sc2 = ts_s.copy()
ts_sc2['dollar_rating'].astype('category').cat.codes
```
3. Faccio Piecewise Linear Calibration (LATTICE 1D ) sui numerical attributes
```python
avg_rating = layers.Input(shape=[1], name='avg_rating')
avg_rating_cal = tfl.layers.PWLCalibration(
    input_keypoints=np.quantile(tr_sc2['avg_rating'], np.linspace(0, 1, num=20)),
    output_min=0.0, output_max=lattice_sizes2[0] - 1.0, name='avg_rating_cal'
)(avg_rating)
```
```python
num_reviews = layers.Input(shape=[1], name='num_reviews')
num_reviews_cal = tfl.layers.PWLCalibration(
    input_keypoints=np.quantile(tr_sc['num_reviews'], np.linspace(0, 1, num=20)),
    output_min=0.0, output_max=lattice_sizes2[1] - 1.0, name='num_reviews_cal'
)(num_reviews)
```
Scelgo come POSIZIONE dei KNOTS il valore dei QUANTILES. Ricorda che la loro posizione e' FISSA ma il valore e' LEARNABLE ovviamente.

4. Per il categorical
```python
dollar_rating = layers.Input(shape=[1], name='dollar_rating')
dollar_rating_cal = tfl.layers.CategoricalCalibration(
    num_buckets=4,
    output_min=0.0, output_max=lattice_sizes2[2] - 1.0,
    name='dollar_rating_cal'
)(dollar_rating)
```
5. Creo il mdoello finale
```python
lt_inputs2 = [avg_rating_cal, num_reviews_cal, dollar_rating_cal]

mdl_out2 = tfl.layers.Lattice(
    lattice_sizes=lattice_sizes2,
    output_min=0, output_max=1, name='lattice',
)(lt_inputs2)

mdl_inputs2 = [avg_rating, num_reviews, dollar_rating]
lm2 = keras.Model(mdl_inputs2, mdl_out2)
```
Nota la differenza tra i parametri
*#Parameters in the original lattice: 256
\#Parameters in the new lattice: 52*
COME RISULTATO DELL'EVALUATION HO 
*AUC score: 0.80 (training), 0.80 (validation), 0.80 (test)*
HO GIA' evitato l'overfitting di prima SENZA mettere nessun tipo di constraint all'interno ricordo.
COME HO FATTO?
![[latticemodelcalibrated.png]]
Come si puo' vedere da questo risultato ho che ho imparato dei GRID POINT in maniera indipendente tra gli assi (cio' e' dato dalle linee orizzontali e verticali che vedi). In sto modo quidni ho solo fatto interpolation tra i gridpoints imparati in 1D semplicemente. E' difficile OVERFITTARE in sto caso. Questo model ha PIU' BIAS (ho perso expressivity) ma ha meno VARIANCE (e' piu' robusto).
INOLTRE avendo introdotto INDIVIDUAL CALIBRATION LAYERS, questi sono molto PIU' FACILI DA INTERPRETARE RISPETTO ALLA 2DLATTICE di prima infatti ho:

![[calibrationVlues.png]]
Questi so i valori imparati dalla mia rete per ogni attributo.
Guarda qua, da qui vedo che la MONOTONICITY non e' rispettata in avg_rating e num_reviews.
IN deep network NON posso fare in modo che un CONSTRAINT DI STO TIPO sia rispettato. In sto caso invece e' BANALE. se rispetto monotony nel calibration layer allora la avro anche nella lattice risultante finale.

## Shape Constraints
**Shape constraints translate into _constraints on the lattice parameters_**
* Let $\theta_{i, k, \bar{i}, \bar{k}}$ be the parameter for the $k$-th knot of input $i$...
* ...While all the remaining attributes and knots (i.e. $\overline{i}$ and $\overline{k}$) are fixed
**Then (increasing) _monotonicity_ translates to:**
$$
\theta_{i,k,\bar{i},\bar{k}} \leq \theta_{i,k+1,\bar{i},\bar{k}}
$$
* I.e. all else being equal, the lattice value at the grid points must be increasing
* Decreasing monotonicity is just the inverse
**Then _convexity_ translates to:**
$$
\left(\theta_{i,k+1,\bar{i},\bar{k}} - \theta_{i,k,\bar{i},\bar{k}}\right) \leq \left(\theta_{i,k+2,\bar{i},\bar{k}} - \theta_{i,k+1,\bar{i},\bar{k}}\right)
$$
* I.e. all else being equal, the adjacent parameter differences should increase
Il codice per mettere motonicity in "avg_rating" e' questo
```python
avg_rating2 = layers.Input(shape=[1], name='avg_rating')
avg_rating_cal2 = tfl.layers.PWLCalibration(
    input_keypoints=np.quantile(tr_s['avg_rating'], np.linspace(0, 1, num=20)),
    output_min=0.0, output_max=lattice_sizes2[0] - 1.0,
    monotonicity='increasing',
    kernel_regularizer=('hessian', 0, 1),
    name='avg_rating_cal'
)(avg_rating2)
```
basta che metto 'increasing' . Internamente nel training si FOCUSSA interamente per minimizzare la loss e poi SI FOCUSSA INTERAMENTE nel rendere il modello feasible secondo il constraint di motonicity. Quindi e' un po' diverso da quello che abbiamo fatto nella lezione 6.1 ma comunque simile nel concetto.
INOLTRE aggiungo anche kernel_regularizer = ('hessian', 0, 1)
Posso infatti APPLICARE una regolarizzazione per avere un cambio della derivativa che NON SIA TROPPO DRASTICO
* This is a regularization term that penalizes the **second derivative**
* ...Thus making the calibrator more linear
Per le "num_reviews" invece ho:
```python
num_reviews2 = layers.Input(shape=[1], name='num_reviews')
num_reviews_cal2 = tfl.layers.PWLCalibration(
    input_keypoints=np.quantile(tr_s['num_reviews'], np.linspace(0, 1, num=20)),
    output_min=0.0, output_max=lattice_sizes2[1] - 1.0,
    monotonicity='increasing',
    convexity='concave',
    kernel_regularizer=('wrinkle', 0, 1),
    name='num_reviews_cal'
)(num_reviews2)
```
Qua ho che visto che "se ho un buon numero di reviews allora la probabilita' di click e' alta, MA SE HO un ancora piu' alto numero di reviews allora LA PROBABILITA' DI CLICK non aumentera' cosi tanto" allora metto insieme il constraint della monotonicity e quello della *concave* convexity (cosi non mi aumenta eccessivamente nella parte finale).
Uso anche una wrinkle regularizer che penalizza la terza derivativa.
Posso specificare monotonicity anche per categorical attribute. Pero' NOTA che non ho in questo caso UN TOTAL ORDER, quindi un ordine tra TUTTI i valori di 'dollar_rating'. Ho dei PARTIAL ORDER  che sono:
1. ristoranti CHEAP <= ristoranti A PREZZO MEDIO ( <= meno cliccati di)
2. ristoranti A PREZZO ALTISSIMO <= ristoranti a PREZZO MEDIO
Queste sono regole che METTIAMO ho scoperto parlando col cliente
QUESTO E' il codice
```python
dollar_rating2 = layers.Input(shape=[1], name='dollar_rating')
dollar_rating_cal2 = tfl.layers.CategoricalCalibration(
    num_buckets=4,
    output_min=0.0, output_max=lattice_sizes2[2] - 1.0,
    monotonicities=[(0, 1), (3, 1)],
    name='dollar_rating_cal'
)(dollar_rating2)
```
IMPORTANTE quando creo la LATTICE FINALE, devo comunque SCRIVERE ANCHE QUI che ci sia monotonicity:
```python
lt_inputs3 = [avg_rating_cal2, num_reviews_cal2, dollar_rating_cal2]

mdl_out3 = tfl.layers.Lattice(
    lattice_sizes=lattice_sizes2,
    output_min=0, output_max=1,
    monotonicities=['increasing'] * 3, name='lattice',
)(lt_inputs3)

mdl_inputs3 = [avg_rating2, num_reviews2, dollar_rating2]
lm3 = keras.Model(mdl_inputs3, mdl_out3)
```
in modo che venga preservata
IL RISULTATO DEL TRAINING E'
*AUC score: 0.80 (training), 0.80 (validation), 0.80 (test)*
non ho perso niente
Ora la RESPONSE SURFACE e' cosi:
![[final_result.png]]
![[final_monotonicity.png]]
Qua la monotonocita' e' RISPETTATA INFATTI. NOTA la concativita' nel 'num_reviews' che avevo introdotto io.

## CONSIDERAZIONI FINALI

LATTICE MODELS non sono conosciuti. 
Sono interpretabili,  e l'introduzione del CALIBRATION LAYER li rende SCALABILI. Inoltre permette di introdurre CONSTRAINTS che sono stati appurati col cliente.
**NOTA** posso addirittura fare calibration e poi usare linear regression on top (senza usare la lattice on top). Devo anche li' pero' ensurare che i pesi siano NON NEGATIVI per la monotonicity.

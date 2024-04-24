# Data Visualization

When you have to load a dataset, look at the raw data with a text editor and try to understand if:

1. if it is a `csv` file or other
2. for `csv`, what is the _separator_ character (`,`,`;`,`\t`, ...) (it is possible to specify it via the sep attribute and by setting a different char for sep)
3. for `csv`, is there a _header_? it is a first row containing column names
4. if there is no header, look for reasonable names, e.g. for _UCI_ a `.names` file
5. if there is no header, look at the documentation of `read_csv` to see how to specify column names (there is  a practice way to do it)
6. try to understand if the dataset is supervised, and what is the _target class_

Cool things I found: 
1. *df.columns* is used to print the column names of df
2. *df.drop(column_name, axis = 1)* is used to drop a column, column_name must be a string
3. *pd.read_csv(url, names = column_names)* is used when in the csv there aren't the column names so it is needed to insert them manually. If there is an *header* in the csv file, then *read_csv* will automatically use the header as the column names. If you want to override the header by specifying the *names*, it is mandatory to specify *header=0*
4. *sns.pairplot(df, hue = target_label, height = height_num)* it produces pairplots for all the features inside df, except the *target_label* feature that is used to set the meaning of the color in the plot of the points of the dataset. Note: *target_label* **must** be a column of *df*. The one specified as hue will not be part of the features for which the pairplots are computed.
5. *df = pd.read_csv(url, sep = ';')* If the *header* contains column names separated by a ; instead of a , (the default one), you **must** specify the sep ';'.
6. *df.hist()* to make an histogram for each feature in df
7. *df\[target_column\].hist()* to make histogram of a specific column
8. *sns.pairplot(df, hue = 'quality', diag_kind = 'kde')* instead of showing histograms in the diagonal, it shows kernel density estimate plots (a bit different from discrete histograms).
9. *correlation_matrix = df.corr()* computes the correlation matrix. It is a symmetric matrix
10. *sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")* 
11. *sns.boxplot(data=df)* It generates a unique boxplot for each feature in the DataFrame `df`. However, a potential issue arises when the features have vastly different scales. If one feature has significantly higher values compared to others, it can overshadow the boxplots of features with smaller values. This discrepancy makes it challenging to clearly visualize and compare the distributions of all features. ![[RL/Cattura.png]]
the following code solves the issue ```
	fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
	axes = axes.flatten()
	for i, column in enumerate(df.columns):
	    sns.boxplot(x=df[column], ax=axes[i])
	plt.tight_layout()
	plt.show()
	```
	![[Cattura1 1.png]]
1.  If we want to make a subplot for each feature but by making a subplot for each target label bounded to that feature we can do 
 ```
 fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 20))

axes = axes.flatten()

for i, predictor in enumerate(predictors):

    sns.boxplot(x=df[target_column], y=df[predictor], ax=axes[i])

plt.plot()
```
so we just add y=df\[predictor]
![[Cattura3.png]]



```
from sklearn.preprocessing import OneHotEncoder

one = OneHotEncoder()

df0 = df.copy() # df0 is used to make modification to the table.

column_to_transform = 'Sex'

enc_data = one.fit_transform(df0[[column_to_transform]]).toarray()

  

l = list(one.categories_[0])

  

enc_df = pd.DataFrame(one.fit_transform(df0[[column_to_transform]]).toarray(), columns=l)

df = df.join(enc_df)

df0 = df.drop(columns='Sex')

df.head()
```
*df.sort_values(by = column)* sorts values of the pd.DataFrame
*min_points = 2 * X.shape\[1]*  as a rule of tumb remember that in DBSCAN the min_points must be set to 2 * num_features.


```
   if n_clusters > 1:

        X_cl = X[y_db!=-1,:]

        y_db_cl = y_db[y_db!=-1]

        silhouette = silhouette_score(X_cl,y_db_cl)
```

*df.dropna()* e droppa tutti gli items con valore Nan 
*df.fillna(0)* e filla gli items con valore Nan pari al valore specificato, in sto caso 0

*df1 = df1.groupby(level=0, axis=1).sum()*
```
basket = (df

            .groupby(['InvoiceNo', 'Description'])['Quantity'].sum()

            .unstack().reset_index()

            .fillna(0)

            .set_index('InvoiceNo')

)

basket
```
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
questi sono gli import per fare Association Rules

*min_supports = np.arange(0.20, 0.01, step=-0.01)* mi da' un range di values che decresce di uno step size di 0.01 da 0.20 fino a 0.01
se avessi fatto np.linspace(start, end, num_values) non avrei potuto controllare la step size che sarebbe stata INFERTA
con np.arange invece posso controllare la step size, ma non il numero totale di valori che conterra' questo array.
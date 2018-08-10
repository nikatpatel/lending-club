---
title: Modeling and Predictions
notebook:
nav_include: 3
---

```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import statsmodels.api as sm
from matplotlib import cm
from statsmodels.api import OLS
from pandas.tools.plotting import scatter_matrix
from pandas import scatter_matrix
import scipy as sci

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

%matplotlib inline
```

### Load, Visualize, and Clean Data


```python
target_col = "sub_grade"
full_loan_stats = pd.read_csv('git_repo/s109/Lending Club Project/LoanStats3d.csv', skiprows=1, skipfooter=2, engine='python')
loan_df = full_loan_stats.copy()
```


```python
loan_df.groupby(['loan_status']).agg({'loan_status': np.size})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_status</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>68293</td>
    </tr>
    <tr>
      <th>Current</th>
      <td>115887</td>
    </tr>
    <tr>
      <th>Default</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>229963</td>
    </tr>
    <tr>
      <th>In Grace Period</th>
      <td>2668</td>
    </tr>
    <tr>
      <th>Late (16-30 days)</th>
      <td>809</td>
    </tr>
    <tr>
      <th>Late (31-120 days)</th>
      <td>3467</td>
    </tr>
  </tbody>
</table>
</div>



We see that there are categories of the loan_status have very few observations. Since we are really interested in the loan getting to good term, instead of trying to predict the status, we turn it in a binary category, indicating if it is current (non risky) or falls into any of the other categories, which we'll call risky.


```python
# We create a new feature that will inform of what we'll be considering a default, which we'll use as an outcome that 
# we'll want to avoid.

loan_df['risk'] = int(0)
badLoan = ['Charged Off','Late (31-120 days)',
       'Late (16-30 days)', 'In Grace Period']

loan_df.loc[(loan_df['delinq_amnt'] > 0) | (loan_df['pct_tl_nvr_dlq'] != 100) | 
              (loan_df['total_rec_late_fee'] > 1) | (loan_df['delinq_2yrs'] > 0) | 
              (loan_df['pub_rec_bankruptcies'] > 0) | (loan_df['debt_settlement_flag'] != 'N') |
              loan_df['loan_status'].isin(badLoan),'risk'] = 1

predDefault = loan_df[['risk','grade','loan_amnt','annual_inc','term','int_rate','emp_length','tax_liens','total_acc',
                       'total_cu_tl','hardship_loan_status','num_sats','open_rv_24m','pub_rec','tax_liens',
                       'tot_coll_amt','tot_hi_cred_lim','total_bal_ex_mort','total_cu_tl']]

predDefault = pd.get_dummies(predDefault, columns=['grade', 'term'], drop_first = False)

f, ax = plt.subplots(figsize=(10, 10))

corr = predDefault.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(10, 200, as_cmap=True, center = 'light'),
            square=True, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11455fda0>




![png](output_5_1.png)


We see some correlations that may be interesting to explore further between features that indicate potential risk (as by the new feature added to the dataset) and others. We see that the variables by which Lending Club seems to grade loans do indeed have a potential effect on risk (such as term of loan or loan ammount) but we also see others that they don't seem to take in so much consideration as having tax lien or derogatory public records.


It's interesting to point out that better grades, as assigned by Lending Club, don't necessarily correspond with less risk of default, as seen by "charged off" having a negative correlation with Grade G and positive with better levels.

In favor of Lending Club's grading system we see that there seems to be an intrinsic higher risk on higher interest paying loans, at least through this rough preliminary analysis.

##### After all that visualization! Lets Clean the Data!


```python
loans_df = loan_df.copy()

# Drop columns that have been included in imputing the value of the 'risk' new feature
loans_df = loans_df.drop(['delinq_amnt','pct_tl_nvr_dlq','total_rec_late_fee','delinq_2yrs','pub_rec_bankruptcies','loan_status'], axis=1)

#Include all continuous features and outcome 'risk' class
loans_df = loans_df.select_dtypes(include=['float64']).join(loans_df['risk'])

# Include discrete variables that have shown a potential relation with the outcome.
loans_df = loans_df.join(loan_df[['term','grade','emp_length','home_ownership']])
```


```python
def get_columns_to_drop(df):
    """Returns a list of columns from df that is all NaN"""
    columns_to_drop = []
    for col in loans_df.columns:
        unique_rows = loans_df[col].unique()
        if (unique_rows.size == 1 and not isinstance(unique_rows[0], str) and np.isnan(unique_rows[0])):
            columns_to_drop.append(col)
    return columns_to_drop
```


```python
# drop columns that contains all NaN values
loans_df_columns_to_drop = get_columns_to_drop(loans_df)
loans_df = loans_df.drop(loans_df_columns_to_drop, axis=1)
```

We now process NaN on a column by column basis to impute the appropriate value in each case


```python
# After studying each column, we see that the best way to remove nan for these cases is to impute 0
for col in loans_df.columns:
    loans_df[col].fillna(0, inplace=True)
```

After cleaning the dataset of `loans_df`, we were able to reduce from 145 categories to 55 categories.


```python
loans_df.shape
```




    (421095, 57)



### Dimensionality Reduction


```python
# In order to be able to apply PCA we need to hot-one-code all the categorical variables in the dataset
loans_df = pd.get_dummies(loans_df, columns=['term','grade','emp_length','home_ownership'], drop_first=True, prefix=['term_','grade_','emp_','home_'])
```


```python
loans_df.shape
```




    (421095, 74)




```python
# Split dataset between train and test. We stratify on 'risk' to make sure that we have representation of both categories in the datasets.
loans_train, loans_test = train_test_split(loans_df, test_size=0.2, random_state=8, stratify=loans_df['risk'])

x_train = loans_train.drop(['risk'], axis=1)
y_train = loans_train['risk']

x_test = loans_test.drop(['risk'], axis=1)
y_test = loans_test['risk']

# We scale continuous variables before applying PCA
scale_transformer = StandardScaler(copy=True).fit(x_train)
x_train_scaled = scale_transformer.transform(x_train)
x_test_scaled = scale_transformer.transform(x_test)


# Generate 10 PCA to study the amount of variance that each of them is able to explain
pca_transformer = PCA(10).fit(x_train_scaled)
x_train_10 = pca_transformer.transform(x_train_scaled)
x_test_10 =  pca_transformer.transform(x_test_scaled)
```


```python
# Plot first two components to explore if they are able to discriminate among classes.
colors = ['g','r']
label_text = ["NonRisk", "Risk"]
markers = ["o","x"]

for risk in [0,1]:
    cur_df = x_train_10[y_train==risk]
    plt.scatter(cur_df[:,0], cur_df[:,1], c = colors[risk], label=label_text[risk], alpha = .7, marker=markers[risk])
    
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimention 2")
plt.title("PCA")
plt.legend();
```


![png](output_19_0.png)



```python
# Create a dataframe with the variables and the absolute values of their coeficients
PCA_vectors = pd.DataFrame(abs(pca_transformer.components_), columns=x_train.columns)

# Sort variables by their weight in the coefficients of the first PCA vectors
PCA_vectors.iloc[0]
```




    funded_amnt_inv                               0.003194
    installment                                   0.003579
    annual_inc                                    0.004728
    dti                                           0.012508
    mths_since_last_delinq                        0.006737
    mths_since_last_record                        0.004169
    out_prncp                                     0.020477
    out_prncp_inv                                 0.020477
    total_pymnt                                   0.015342
    total_pymnt_inv                               0.015348
    total_rec_prncp                               0.016374
    total_rec_int                                 0.004767
    recoveries                                    0.001999
    collection_recovery_fee                       0.002007
    last_pymnt_amnt                               0.004883
    mths_since_last_major_derog                   0.006799
    annual_inc_joint                              0.018639
    dti_joint                                     0.019457
    open_acc_6m                                   0.281309
    open_act_il                                   0.273195
    open_il_12m                                   0.271417
    open_il_24m                                   0.297812
    mths_since_rcnt_il                            0.151085
    total_bal_il                                  0.266801
    il_util                                       0.320248
    open_rv_12m                                   0.266434
    open_rv_24m                                   0.283477
    max_bal_bc                                    0.239433
    all_util                                      0.320085
    inq_fi                                        0.236266
                                                    ...   
    hardship_amount                               0.012522
    hardship_length                               0.011415
    hardship_dpd                                  0.010243
    orig_projected_additional_accrued_interest    0.011508
    hardship_payoff_balance_amount                0.012883
    hardship_last_payment_amount                  0.009883
    settlement_amount                             0.002405
    settlement_percentage                         0.001925
    settlement_term                               0.002988
    term__ 60 months                              0.000836
    grade__B                                      0.000131
    grade__C                                      0.003026
    grade__D                                      0.001994
    grade__E                                      0.000265
    grade__F                                      0.002576
    grade__G                                      0.001080
    emp__1 year                                   0.000686
    emp__10+ years                                0.001875
    emp__2 years                                  0.000843
    emp__3 years                                  0.000399
    emp__4 years                                  0.000410
    emp__5 years                                  0.001830
    emp__6 years                                  0.000905
    emp__7 years                                  0.003613
    emp__8 years                                  0.000045
    emp__9 years                                  0.001497
    emp__< 1 year                                 0.004138
    home__MORTGAGE                                0.001176
    home__OWN                                     0.000031
    home__RENT                                    0.001180
    Name: 0, Length: 73, dtype: float64



### Random Forest


```python
rf = RandomForestClassifier(n_estimators=25, max_depth=5).fit(x_train,y_train)
rf.score(x_test,y_test)
```




    0.9463066528930526



A Random Forest Classifier was fitted on the training data set and resulted in a 94% accuracy using these features when evaluating the model's performance on the test sit. Repeated runs of each model would result in minor changes to the accuracy scores.


```python
rf_feature_importance = rf.feature_importances_
```


```python
plt.figure(figsize=(10,15))
plt.barh(np.arange(len(x_train.columns)), rf_feature_importance)
plt.title("Random Forest Feature Importance")
plt.yticks(np.arange(len(x_train.columns)), x_train.columns)
plt.ylabel("Features")
# plt.axvline(x=0.03, color = 'k', linestyle = '--')
```




    Text(0,0.5,'Features')




![png](output_25_1.png)


From the random forest model, it generated a forest of trees of depth of 5. From these multiple generated trees, the importance of each feature is plotted above. 


```python
def get_features_of_rf(feature_names, rf_features, level=0.01):
    f_name = []
    f_val = []
    for name, val in zip(feature_names, rf_features):
        if val > level:
            f_name.append(name)
            f_val.append(val)
    return f_name, f_val
imp_feature_names, imp_feature_vals = get_features_of_rf(x_train.columns, rf_feature_importance)
```


```python
plt.figure(figsize=(10,10))
plt.barh(np.arange(len(imp_feature_names)), imp_feature_vals)
plt.title("Random Forest Feature Importance")
plt.yticks(np.arange(len(imp_feature_names)), imp_feature_names)
plt.ylabel("Features")
```




    Text(0,0.5,'Features')




![png](output_28_1.png)


The following features of the plot are the features the random forest model considered to be the most importance in all of the generated trees. 

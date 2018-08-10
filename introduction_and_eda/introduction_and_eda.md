---
title: Introduction and EDA
notebook:
nav_include: 2
---

### Libraries for EDA:


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

%matplotlib inline
```

### Load, Visualize, and Clean Data

#### Load Data


After reviewing all the available datasets we notice that the features change from 2015 onward and appear to be more informative than on previous years.

Each row corresponds to a snapshot of a loan at the end of the period and has information on the loan itself and the borrowers. Since LendingClub does not provide a unique id for loans or borrowers **it's not posible to join together several periods** to increase the amount of data because we'd be repeating information on loans at differente times, which would distort the outcome of the study.


```python
# Remove % sign from 'int_rate' column and cast it to a float
for i in range(len(loan_df['int_rate'].values)):
    loan_df['int_rate'].values[i] = str(loan_df['int_rate'].values[i])[:-1]
    if (loan_df['int_rate'].values[i] == 'na'):
        loan_df['int_rate'].values[i] = '0'
    else:
        loan_df['int_rate'].values[i] = loan_df['int_rate'].values[i]
loan_df['int_rate'] = pd.to_numeric(loan_df['int_rate'])
```

After loading and cleaning the data we start by making simple visualizations, grouping and descriptive statistics of the dataset by different features to have a first glance at the data. We understand that Lending Club grades loans by their risk which translates in higher risk loans paying higher interests and vice versa. Understanding this and considering the goal of the analysis we decide to work on an initial hypothesis:

- If we can understand how Lending Club grades a loan, we should be able to improve on their grading criteria, "regrade" loans and invest in higher return loans that are less risky than LC graded to be.

To achieve this we will work on three strategies:

- Build a model that accurately grades loans by Lending Club's standards
- Build a model that accurately predicts the likelyhood of default
- Combine Lending Club's data with macro economic indicators that can give us exogenous confounding variables that would potentially increase the predicting accuracy of both models and thus, our competitive advantage  


```python
loan_df.groupby(['loan_status','grade']).agg({ 
    'loan_status' : np.size,
    'loan_amnt' : np.mean,
    'int_rate' : np.mean
})
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
      <th></th>
      <th>loan_status</th>
      <th>loan_amnt</th>
      <th>int_rate</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th>grade</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">Charged Off</th>
      <th>A</th>
      <td>3612</td>
      <td>14094.282946</td>
      <td>7.213441</td>
    </tr>
    <tr>
      <th>B</th>
      <td>12426</td>
      <td>13868.322469</td>
      <td>10.259063</td>
    </tr>
    <tr>
      <th>C</th>
      <td>21014</td>
      <td>14355.960312</td>
      <td>13.389042</td>
    </tr>
    <tr>
      <th>D</th>
      <td>15623</td>
      <td>15779.000512</td>
      <td>16.795443</td>
    </tr>
    <tr>
      <th>E</th>
      <td>10792</td>
      <td>18200.984526</td>
      <td>19.370108</td>
    </tr>
    <tr>
      <th>F</th>
      <td>3865</td>
      <td>20046.390686</td>
      <td>23.731457</td>
    </tr>
    <tr>
      <th>G</th>
      <td>961</td>
      <td>20155.515088</td>
      <td>26.891779</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Current</th>
      <th>A</th>
      <td>18127</td>
      <td>15301.736360</td>
      <td>6.872574</td>
    </tr>
    <tr>
      <th>B</th>
      <td>34785</td>
      <td>15612.177663</td>
      <td>9.960306</td>
    </tr>
    <tr>
      <th>C</th>
      <td>34442</td>
      <td>16564.476511</td>
      <td>13.259895</td>
    </tr>
    <tr>
      <th>D</th>
      <td>16589</td>
      <td>17525.979565</td>
      <td>16.710302</td>
    </tr>
    <tr>
      <th>E</th>
      <td>9376</td>
      <td>19247.586924</td>
      <td>19.226877</td>
    </tr>
    <tr>
      <th>F</th>
      <td>2204</td>
      <td>20154.764065</td>
      <td>23.459170</td>
    </tr>
    <tr>
      <th>G</th>
      <td>364</td>
      <td>20015.178571</td>
      <td>26.868984</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Default</th>
      <th>B</th>
      <td>2</td>
      <td>15900.000000</td>
      <td>10.760000</td>
    </tr>
    <tr>
      <th>C</th>
      <td>4</td>
      <td>14600.000000</td>
      <td>13.585000</td>
    </tr>
    <tr>
      <th>D</th>
      <td>1</td>
      <td>11200.000000</td>
      <td>15.610000</td>
    </tr>
    <tr>
      <th>E</th>
      <td>1</td>
      <td>14250.000000</td>
      <td>18.550000</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Fully Paid</th>
      <th>A</th>
      <td>51215</td>
      <td>14514.148687</td>
      <td>6.945670</td>
    </tr>
    <tr>
      <th>B</th>
      <td>69021</td>
      <td>13615.635821</td>
      <td>10.046774</td>
    </tr>
    <tr>
      <th>C</th>
      <td>62729</td>
      <td>13815.288383</td>
      <td>13.286663</td>
    </tr>
    <tr>
      <th>D</th>
      <td>29006</td>
      <td>15047.968524</td>
      <td>16.689827</td>
    </tr>
    <tr>
      <th>E</th>
      <td>13758</td>
      <td>18058.189780</td>
      <td>19.269988</td>
    </tr>
    <tr>
      <th>F</th>
      <td>3454</td>
      <td>20102.569485</td>
      <td>23.580599</td>
    </tr>
    <tr>
      <th>G</th>
      <td>780</td>
      <td>20803.717949</td>
      <td>26.720795</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">In Grace Period</th>
      <th>A</th>
      <td>134</td>
      <td>16099.626866</td>
      <td>7.029254</td>
    </tr>
    <tr>
      <th>B</th>
      <td>542</td>
      <td>15741.835793</td>
      <td>10.141974</td>
    </tr>
    <tr>
      <th>C</th>
      <td>939</td>
      <td>17085.250266</td>
      <td>13.363024</td>
    </tr>
    <tr>
      <th>D</th>
      <td>560</td>
      <td>18384.196429</td>
      <td>16.803232</td>
    </tr>
    <tr>
      <th>E</th>
      <td>359</td>
      <td>19329.038997</td>
      <td>19.310474</td>
    </tr>
    <tr>
      <th>F</th>
      <td>111</td>
      <td>19999.774775</td>
      <td>23.522973</td>
    </tr>
    <tr>
      <th>G</th>
      <td>23</td>
      <td>24117.391304</td>
      <td>27.102609</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Late (16-30 days)</th>
      <th>A</th>
      <td>38</td>
      <td>15505.263158</td>
      <td>7.100526</td>
    </tr>
    <tr>
      <th>B</th>
      <td>148</td>
      <td>14544.932432</td>
      <td>9.980338</td>
    </tr>
    <tr>
      <th>C</th>
      <td>269</td>
      <td>16383.085502</td>
      <td>13.311822</td>
    </tr>
    <tr>
      <th>D</th>
      <td>180</td>
      <td>18393.472222</td>
      <td>16.685778</td>
    </tr>
    <tr>
      <th>E</th>
      <td>131</td>
      <td>20111.832061</td>
      <td>19.212137</td>
    </tr>
    <tr>
      <th>F</th>
      <td>35</td>
      <td>21058.571429</td>
      <td>23.507429</td>
    </tr>
    <tr>
      <th>G</th>
      <td>8</td>
      <td>22684.375000</td>
      <td>27.493750</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Late (31-120 days)</th>
      <th>A</th>
      <td>210</td>
      <td>14635.952381</td>
      <td>7.143857</td>
    </tr>
    <tr>
      <th>B</th>
      <td>682</td>
      <td>14984.237537</td>
      <td>10.092053</td>
    </tr>
    <tr>
      <th>C</th>
      <td>1170</td>
      <td>15824.145299</td>
      <td>13.341427</td>
    </tr>
    <tr>
      <th>D</th>
      <td>695</td>
      <td>16982.014388</td>
      <td>16.796734</td>
    </tr>
    <tr>
      <th>E</th>
      <td>531</td>
      <td>20078.813559</td>
      <td>19.303258</td>
    </tr>
    <tr>
      <th>F</th>
      <td>148</td>
      <td>19827.871622</td>
      <td>23.651284</td>
    </tr>
    <tr>
      <th>G</th>
      <td>31</td>
      <td>20937.903226</td>
      <td>27.193871</td>
    </tr>
  </tbody>
</table>
</div>



A first view at the distribution of loans by their status shows us that there is no evident logic as to how a loan will come to term just by looking at their grade, amount or interest rate.


```python
# Try to find the variables that LC considers to assign their grade.
gradingMat = loan_df[['grade','loan_amnt','annual_inc','term','int_rate','delinq_2yrs','mths_since_last_delinq',
                       'emp_length','home_ownership','pub_rec_bankruptcies','tax_liens']]

gradingMatDumm = pd.get_dummies(gradingMat, columns=['grade', 'term','emp_length','home_ownership'])

fig, ax = plt.subplots(figsize=(10, 10))

corr = gradingMatDumm.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(10, 200, as_cmap=True, center = 'light'),
            square=True, ax=ax);
ax.set_title('Lending Club Grading Criteria', fontsize=18)
```




    Text(0.5,1,'Lending Club Grading Criteria')




![png](output_10_1.png)


We can see that grade is related to loan amount and term of loan. Grade A seems to consider more variables, such as deliquency in the past 2 years and bankrupcies.

We'll explore now if there are signs of features being potential good predictors of a loan resulting in a default, this being determined by loans with `hardship flag` != N, for which late recovery fees have been collected, loans with percentage of never delinquent != 100, loans with deliquent amounts > 0 or loan status being "charged off".


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




    <matplotlib.axes._subplots.AxesSubplot at 0x1c54840eb8>




![png](output_12_1.png)


We see some correlations that may be interesting to explore further between features that indicate potential risk (as by the new feature added to the dataset) and others. We see that the variables by which Lending Club seems to grade loans do indeed have a potential effect on risk (such as term of loan or loan ammount) but we also see others that they don't seem to take in so much consideration as having tax lien or derogatory public records.


It's interesting to point out that better grades, as assigned by Lending Club, don't necessarily correspond with less risk of default, as seen by "charged off" having a negative correlation with Grade G and positive with better levels.

In favor of Lending Club's grading system we see that there seems to be an intrinsic higher risk on higher interest paying loans, at least through this rough preliminary analysis.

##### After all that visualization! Lets Clean the Data!


```python
loans_df = full_loan_stats.copy()
loans_df = loans_df.select_dtypes(include=['float64']).join(loans_df[target_col])
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


```python
loans_df.shape
```




    (421095, 55)



After cleaning the dataset of `loans_df`, we were able to reduce from 145 categories to 55 categories.


```python
loans_df.head()
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
      <th>funded_amnt_inv</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>mths_since_last_delinq</th>
      <th>mths_since_last_record</th>
      <th>out_prncp</th>
      <th>out_prncp_inv</th>
      <th>total_pymnt</th>
      <th>total_pymnt_inv</th>
      <th>...</th>
      <th>hardship_amount</th>
      <th>hardship_length</th>
      <th>hardship_dpd</th>
      <th>orig_projected_additional_accrued_interest</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>settlement_amount</th>
      <th>settlement_percentage</th>
      <th>settlement_term</th>
      <th>sub_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3600.0</td>
      <td>123.03</td>
      <td>55000.0</td>
      <td>5.91</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>822.09</td>
      <td>822.09</td>
      <td>3560.87</td>
      <td>3560.87</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>C4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27300.0</td>
      <td>977.00</td>
      <td>65000.0</td>
      <td>25.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>8727.52</td>
      <td>8727.52</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>D3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000.0</td>
      <td>672.73</td>
      <td>145000.0</td>
      <td>12.28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4513.01</td>
      <td>4513.01</td>
      <td>19473.39</td>
      <td>19473.39</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>C2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>253.78</td>
      <td>55000.0</td>
      <td>35.70</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5558.20</td>
      <td>5558.20</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25000.0</td>
      <td>581.58</td>
      <td>79000.0</td>
      <td>34.53</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>14490.92</td>
      <td>14490.92</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>C4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>




```python
loans_df.describe()
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
      <th>funded_amnt_inv</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>mths_since_last_delinq</th>
      <th>mths_since_last_record</th>
      <th>out_prncp</th>
      <th>out_prncp_inv</th>
      <th>total_pymnt</th>
      <th>total_pymnt_inv</th>
      <th>...</th>
      <th>deferral_term</th>
      <th>hardship_amount</th>
      <th>hardship_length</th>
      <th>hardship_dpd</th>
      <th>orig_projected_additional_accrued_interest</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>settlement_amount</th>
      <th>settlement_percentage</th>
      <th>settlement_term</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>421095.000000</td>
      <td>421095.000000</td>
      <td>4.210950e+05</td>
      <td>421093.000000</td>
      <td>217133.000000</td>
      <td>74415.000000</td>
      <td>421095.000000</td>
      <td>421095.000000</td>
      <td>421095.000000</td>
      <td>421095.000000</td>
      <td>...</td>
      <td>2225.0</td>
      <td>2225.000000</td>
      <td>2225.0</td>
      <td>2225.000000</td>
      <td>1823.000000</td>
      <td>2225.000000</td>
      <td>2225.000000</td>
      <td>8848.000000</td>
      <td>8848.000000</td>
      <td>8848.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>15234.156426</td>
      <td>441.846219</td>
      <td>7.696561e+04</td>
      <td>19.148367</td>
      <td>34.023391</td>
      <td>66.592609</td>
      <td>1732.089410</td>
      <td>1731.268075</td>
      <td>14748.917175</td>
      <td>14743.200629</td>
      <td>...</td>
      <td>3.0</td>
      <td>116.966769</td>
      <td>3.0</td>
      <td>13.935281</td>
      <td>344.569715</td>
      <td>9255.759685</td>
      <td>180.003506</td>
      <td>5052.308930</td>
      <td>47.616197</td>
      <td>12.406420</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8567.935757</td>
      <td>244.846944</td>
      <td>7.394996e+04</td>
      <td>8.885104</td>
      <td>21.990270</td>
      <td>25.577899</td>
      <td>3935.594539</td>
      <td>3933.582273</td>
      <td>9341.303762</td>
      <td>9337.937536</td>
      <td>...</td>
      <td>0.0</td>
      <td>99.994608</td>
      <td>0.0</td>
      <td>9.907728</td>
      <td>293.692525</td>
      <td>6329.988185</td>
      <td>184.856483</td>
      <td>3642.461692</td>
      <td>6.384622</td>
      <td>7.760753</td>
    </tr>
    <tr>
      <th>min</th>
      <td>900.000000</td>
      <td>14.010000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>0.640000</td>
      <td>3.0</td>
      <td>0.000000</td>
      <td>1.920000</td>
      <td>55.730000</td>
      <td>0.020000</td>
      <td>152.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8500.000000</td>
      <td>263.930000</td>
      <td>4.600000e+04</td>
      <td>12.590000</td>
      <td>15.000000</td>
      <td>50.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7721.267930</td>
      <td>7718.905000</td>
      <td>...</td>
      <td>3.0</td>
      <td>38.700000</td>
      <td>3.0</td>
      <td>5.000000</td>
      <td>115.290000</td>
      <td>3931.600000</td>
      <td>37.860000</td>
      <td>2193.000000</td>
      <td>45.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>14000.000000</td>
      <td>385.410000</td>
      <td>6.500000e+04</td>
      <td>18.600000</td>
      <td>31.000000</td>
      <td>67.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12610.810000</td>
      <td>12605.850000</td>
      <td>...</td>
      <td>3.0</td>
      <td>86.580000</td>
      <td>3.0</td>
      <td>15.000000</td>
      <td>252.060000</td>
      <td>8005.090000</td>
      <td>116.390000</td>
      <td>4285.825000</td>
      <td>45.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>20000.000000</td>
      <td>578.790000</td>
      <td>9.169050e+04</td>
      <td>25.340000</td>
      <td>50.000000</td>
      <td>82.000000</td>
      <td>666.310000</td>
      <td>666.020000</td>
      <td>19995.681232</td>
      <td>19989.640000</td>
      <td>...</td>
      <td>3.0</td>
      <td>167.310000</td>
      <td>3.0</td>
      <td>23.000000</td>
      <td>495.870000</td>
      <td>13356.340000</td>
      <td>273.130000</td>
      <td>7007.715000</td>
      <td>50.000000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>35000.000000</td>
      <td>1445.460000</td>
      <td>9.500000e+06</td>
      <td>999.000000</td>
      <td>176.000000</td>
      <td>120.000000</td>
      <td>28405.440000</td>
      <td>28405.440000</td>
      <td>57769.155762</td>
      <td>57769.160000</td>
      <td>...</td>
      <td>3.0</td>
      <td>638.130000</td>
      <td>3.0</td>
      <td>30.000000</td>
      <td>1914.390000</td>
      <td>29401.040000</td>
      <td>1247.480000</td>
      <td>30000.000000</td>
      <td>166.670000</td>
      <td>65.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 54 columns</p>
</div>



### Regression Testing

#### OLS


```python
# Split into test and train
loans_df = loans_df.fillna(0)
train_loan_df, test_loan_df = train_test_split(loans_df, test_size=0.2)
y_loan_train = train_loan_df[target_col]
X_loan_train = train_loan_df.drop(target_col, axis=1)
y_loan_test = test_loan_df[target_col]
X_loan_test = test_loan_df.drop(target_col, axis=1)
```


```python
# For a character letter pair from A1 to G5 returns it's order in the list
# E.g. A1 = 1, A2 = 2, B1 = 6, G5 = 35
def grade_to_int(grade):
    if len(grade) != 2:
        return 0
    let = ord(grade[0]) - ord('A')
    dig = int(grade[1])
    return let * 5 + dig

def num_to_grade(num):
    let = chr(int(np.floor(num/5)) + ord('A'))
    dig = int(np.round(num)) % 5
    # 1 indexed grades
    if dig == 0:
        let = chr(ord(let) - 1)
        dig = 5
    return let + str(dig)
```


```python
OLSModel = OLS(y_loan_train.map(grade_to_int), sm.add_constant(X_loan_train)).fit()
```


```python
r2_score(OLSModel.predict(sm.add_constant(X_loan_test)), y_loan_test.map(grade_to_int))
```

    0.3303237488371052


The OLS model resulted in a $R^2$ value of 0.3303.


```python
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(OLSModel.predict(sm.add_constant(X_loan_test)), y_loan_test.map(grade_to_int), alpha=0.1)
ax.set_xlabel('$R^2$ Values', fontsize=18)
ax.set_ylabel(target_col, fontsize=18)
y_values = dict((i, num_to_grade(i)) for i in list(range(1, 36)))
ax.set_yticks(list(y_values.keys()))
ax.set_yticklabels(list(y_values.values()))
ax.set_title('OLS Model $R^2$ Values', fontsize=18)
```




    Text(0.5,1,'OLS Model $R^2$ Values')




![png](output_29_1.png)


#### OLS with RFE


```python
model = LinearRegression()
selector = RFE(model).fit(X_loan_train, y_loan_train.map(grade_to_int))
```


```python
r2_score(selector.predict(X_loan_test), y_loan_test.map(grade_to_int))
```




    -0.48433446033411376



That's worse than before. Let's try again and only remove 1/4th of the features to see the $R^2$ improves.


```python
selector = RFE(model, 77).fit(X_loan_train, y_loan_train.map(grade_to_int))
```


```python
r2_score(selector.predict(X_loan_test), y_loan_test.map(grade_to_int))
```




    0.34345254884079535



Much better! With more trials we can probably find the optimum number of features to remove


```python
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(OLSModel.predict(sm.add_constant(X_loan_test)), y_loan_test.map(grade_to_int), alpha=0.1)
ax.set_xlabel('$R^2$ Values', fontsize=18)
ax.set_ylabel(target_col, fontsize=18)
y_values = dict((i, num_to_grade(i)) for i in list(range(1, 36)))
ax.set_yticks(list(y_values.keys()))
ax.set_yticklabels(list(y_values.values()))
ax.set_title('OLS RFE Model $R^2$ Values', fontsize=18)
```




    Text(0.5,1,'OLS RFE Model $R^2$ Values')




![png](output_37_1.png)


#### Logistic Regression

Let's try our Logistic Regressions on our dataset. Future plans are to include cross validations with Ridge/Lasso Regularization to prevent overfitting. 

_Due to the limited computational resources, the following executations can take a while to process._


```python
model = LogisticRegression()
model.fit(X_loan_train, y_loan_train.map(grade_to_int))
```


```python
model.score
```


```python
r2_score(model.predict(X_loan_test), y_loan_test.map(grade_to_int))
```


```python
plt.scatter(model.predict(X_loan_test), y_loan_test.map(grade_to_int), alpha=0.1)
plt.show()
```

#### Logistic Regression with RFE


```python
model = LogisticRegression()
selector = RFE(model).fit(X_loan_train, y_loan_train.map(grade_to_int))
```


```python
r2_score(selector.predict(X_loan_test), y_loan_test.map(grade_to_int))
```


```python
selector = RFE(model, 77).fit(X_loan_train), y_loan_train.map(grade_to_int), step=5)
r2_score(selector.predict(X_loan_test), y_loan_test.map(grade_to_int))
```


```python
plt.scatter(selector.predict(X_loan_test), y_loan_test.map(grade_to_int), alpha=0.1)
```

### Feature Engineering

Let's try to use feature engineering to improve the model. We will be importing the average adjusted gross income based on the applicant's zip code and determining if the demographic area is significant. This dataset was available publicly on www.irs.gov/statistics.

Since the dataset from `LendingTree` only contains the first 3 digits of the applicant's zip code, we will use the average of the first 3 digits of the zip code for the demographic adjusted gross income. Future plans are to include cross validations with Ridge/Lasso Regularization to prevent overfitting. 

Time for some data loading and cleaning to see the results!

#### Load and Clean Data


```python
# Adding `zip_code` from `full_loans_df`, since it was removed earlier
loans_df['zip_code'] = full_loan_stats['zip_code'].str[:3].astype(np.int64)
```


```python
# IRS specifies NI as Number of Returns and A00100 as Total Adjusted Gross Income
full_agi = pd.read_csv('15zpallnoagi.csv')
agi_df = full_agi[['ZIPCODE', 'N1', 'A00100']].copy()
agi_df['adj_gross_income'] = round((agi_df['A00100']/agi_df['N1'])*1000, 2)
agi_df['zip_code'] = agi_df['ZIPCODE'].astype(str).str[:3].astype(np.int64)
```


```python
# Group the adjusted gross income by the first three digits of the zip code
agi_df = agi_df.groupby(['zip_code'], as_index=False)['adj_gross_income'].mean()
agi_df = agi_df.round({'adj_gross_income': 2})
```


```python
# Use a left join to join `agi_df` onto `loans_df`
loans_df = pd.merge(loans_df, agi_df, how='left', on=['zip_code'])
loans_df = loans_df.fillna(0)
```

#### OLS


```python
train_loan_df, test_loan_df = train_test_split(loans_df, test_size=0.2)
y_loan_train = train_loan_df[target_col]
X_loan_train = train_loan_df.drop(target_col, axis=1)
y_loan_test = test_loan_df[target_col]
X_loan_test = test_loan_df.drop(target_col, axis=1)
```


```python
OLSModel = OLS(y_loan_train.map(grade_to_int), sm.add_constant(X_loan_train)).fit()
```


```python
r2_score(OLSModel.predict(sm.add_constant(X_loan_test)), y_loan_test.map(grade_to_int))
```

    0.34136288693706146


The OLS model with adjusted gross income resulted in a $R^2$ value of 0.3414. <br />
The OLS model with no income resulted in a $R^2$ value of 0.3303. <br />
The adjusted gross income did improve our model, but was not signicantly.


```python
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(OLSModel.predict(sm.add_constant(X_loan_test)), y_loan_test.map(grade_to_int), alpha=0.1)
ax.set_xlabel('$R^2$ Values', fontsize=18)
ax.set_ylabel(target_col, fontsize=18)
y_values = dict((i, num_to_grade(i)) for i in list(range(1, 36)))
ax.set_yticks(list(y_values.keys()))
ax.set_yticklabels(list(y_values.values()))
ax.set_title('OLS Adjusted Gross Income $R^2$ Values', fontsize=18)
```




    Text(0.5,1,'OLS Adjusted Gross Income $R^2$ Values')




![png](output_60_1.png)


#### OLS with RFE


```python
model = LinearRegression()
selector = RFE(model, 55).fit(X_loan_train, y_loan_train.map(grade_to_int))
r2_score(selector.predict(X_loan_test), y_loan_test.map(grade_to_int))
```




    0.32201252789623513



When using RFE, it lowers our $R^2$ value, but not significantly.


```python
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(OLSModel.predict(sm.add_constant(X_loan_test)), y_loan_test.map(grade_to_int), alpha=0.1)
ax.set_xlabel('$R^2$ Values', fontsize=18)
ax.set_ylabel(target_col, fontsize=18)
y_values = dict((i, num_to_grade(i)) for i in list(range(1, 36)))
ax.set_yticks(list(y_values.keys()))
ax.set_yticklabels(list(y_values.values()))
ax.set_title('OLS RFE Adjusted Gross Income $R^2$ Values', fontsize=18)
```




    Text(0.5,1,'OLS RFE Adjusted Gross Income $R^2$ Values')




![png](output_64_1.png)


#### Conclusion

At this stage of the analysis we still can't confirm our discard our initial hypothesis (that we can build a model that assesses risk better than that of LC) but we feel confident that we can. 

LC strongly bases their grading system on FICO, loan amount and term of the loan, which in turn does not make it too different from the traditional banking system. We would like to build a model based on more complex predictors that can give an oportunity to a wider group of population by identifiying "good borrowers" that may not have a perfect FICO score but will not default.

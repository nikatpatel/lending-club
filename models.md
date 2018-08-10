---
title: Modeling and Predictions
notebook:
nav_include: 4
---

```
loan_df.groupby(['loan_status']).agg({'loan_status': np.size})
```
!(https://github.com/nikatpatel/lending-club/blob/master/Images/loan_status.png)

---
title: Results and Conclusion
notebook:
nav_include: 4
---

## Conclusion
Our preliminary EDA revealed that Lending Club grades loans based on only a small set of variables from the borrower and the prospective loan. There appeared to be a very strong correlation between grade and a combination of FICO, amount of the loan, term, borrower's income and income to debt ratio. We were able to prove these intuitions to a certain extent by building models that, had we had more time to work on, would be able to predict the grade assigned by LC to a prospective loan.

From that point we turned our study to analyzing the correlation between grades and how a loan would come to term, that is, if it would be fully paid without late payments or defaults and if there were other predictors of this outcome, that would improve on just using grade to determine the quality of a potential investment. This study indicated that there were in fact some aditional variables that would serve better than pure grades as predictors and that there would be a way to "re-grade" loans based on these features to indicate what would be a good investment.

From the first models fit on Random Forest, based on results from PCA, this intuition is more strongly observed and we think that, had the study continued with more comlex models or meta-models, this initial hypothesis would have in fact been confirmed. 

## Future Work 
Our study gives strong indications toward confirming our intuition and that of many previous studies that with the available historical data from the Lending Club one can build a predictive model that would increase the average return on the investment and reduce the risk at the same time.

It would make sense to continue this study by applying ensemble models and neural networks on the base of the findings above. These type of models would be able to render better results than the ones explored in this study.

We would also like analyze other datasets, if available, that are similar to the data available through Lending Club and analyze if the results are similar to the ones we stated above. One hypothesis we had was to compare interest rates with grades and/or FICO scores and see if those observations would confirm our intuition. There are many variations to our curiosity and it would be hard to condense them into one overall project question. For this curiosity is what makes us intridged and what will make us great Data Scientists and Data Engineers.

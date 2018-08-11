---
title: Results and Conclusion
notebook:
nav_include: 4
---

## Conclusion

Our preliminary EDA revealed that Lending Club grades loans based on only a small set of variables from the borrower and the prospective loan. There appeared to be a very strong correlation between grade and a combination of FICO, amount of the loan, term, borrower’s income and income to debt ratio. We were able to prove these intuitions to a certain extent by building models that, had we had more time to work on, would be able to predict the grade assigned by LC to a prospective loan.

From that point we turned our study to analyzing the correlation between grades and how a loan would come to term, that is, if it would be fully paid without late payments or defaults and if there were other predictors of this outcome, that would improve on just using grade to determine the quality of a potential investment. This study indicated that there were in fact some additional variables that would serve better than pure grades as predictors and that there would be a way to “re-grade” loans based on these features to indicate what would be a good investment.

From these findings we decided to create a new feature (risk) that we’d want to predict to avoid potentially risky investments. For this we included as risky = True any loans that had late payments, defaults, had enter in settlement negotiations... With this in consideration, we created a subset of our dataset where we removed any variable that would be correlated to these as well as any column with information that would not be known by a potential investor at the time of deciding.

By subsequent filtering and processing of features we selected the 30 most relevant features from PCA and we were very impressed by the results from fitting Random Forest to such a small subset, from which we obtained a score of almost 90% on our test set with a model that trained in only a few seconds. We believe that by implementing more complex meta-models, ensembles or fitting Neural Networks we should be able to create a strong model that would render better results.

A way to test these results would be to select a set of loans through these models and run a simulation on what would have been the returns on these investments and compare them with the average returns of investors in that period for similar investments.

## Future Work 
Our study gives strong indications toward confirming our intuition and that of many previous studies that with the available historical data from the Lending Club one can build a predictive model that would increase the average return on the investment and reduce the risk at the same time.

It would make sense to continue this study by applying ensemble models and neural networks on the base of the findings above. These type of models would be able to render better results than the ones explored in this study.

We would also like analyze other datasets, if available, that are similar to the data available through Lending Club and analyze if the results are similar to the ones we stated above. One hypothesis we had was to compare interest rates with grades and/or FICO scores and see if those observations would confirm our intuition. There are many variations to our curiosity and it would be hard to condense them into one overall project question. For this curiosity is what makes us intridged and what will make us great Data Scientists and Data Engineers.

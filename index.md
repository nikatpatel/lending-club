---
title: Introduction
notebook:

---

**Statement and Background**<br/>
LendingClub is a platform which is set out to transform the banking industry by connecting borrowers to investors and providing transparent information about various investment options. Investors are provided alternative investment options rather than the traditional options provided by the banking industry. Borrowers are given alternative means of funding for their projects and businesses. 

**Goals**<br/>
To build an investment diversification recommender that aims to maximize returns with optimized feature selection for predicting the probability of loan defaults and avoid risky investments.

**Available Resources/Data**<br/>

LendingClub provides over ten years of data in issued loans and declined loans. These data files contain 145 fields of information per loan that was issued or declined. The information available consists of details from the start of the loan to the most recent status of the loan. With all these fields of information, not all of these will be useful for our model. We will be removing any fields that would cause unfairness to the resulting model. Categorical field such as home_ownership and loan_status will be modified to binary values. For our purposes we have decided to use only the year of 2015 as our dataset.

We will also use open datasets with macroeconomic indicators, such as population adjusted gross income, to explore the potential influence that these may have in the activity at LendingClub.

Credit Card Default :  Executive Summary

Overview

"To default or not to default..."

How do we predict credit card default? And how do we know if a customer will default next month? To attempt at answering this question, we analysed data of 30,000 customers collected by a Taiwanese Bank between Apr and Sep 2005. The objective was to spot trends and try to predict if a particular customer would default or not. 

Method

The dataset was obtained from the UCI directory - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
We divided the target column (default_payment_next_month) into two - 1 being default and 0 being no default and conducted some EDA based on demographics to understand our data better. Following this, we explored two models - logistic regression and random forest - as classification techniques to help understand which classes our data belonged to. We chose our key metric as Recall or Sensitivity which shows the number of true positives out of the total number of positives. Why recall matters to us is because of the False Negatives component i.e. the chance of knowing if someone has defaulted if they have actually defaulted. Not knowing this number will be a heavy cost to the bank and hence this is our key metric. Since the data was heavily skewed towards number of non defaulters, we did a random sampling with SMOTE choosing over sampling as our strategy. 

We ran the following classifiers :

1. Logistic Regression
2. PCA + Logistic Regression
3. Support Vector Machine
4. Random Forest

Findings

The best model was Logistic, it yielded a recall of 0.61 and an f1 score of 0.55. Overall none of the models were found to be very good predictors and we concluded that we would need to revisit the classifiers with more features and samples such as :
- income, monthly expenses
- related accounts such as savings, loans etc.
- longer payment history (12 or 24 month)


Reference docs 

1. Presentation - credit_card_default
2. Master jupyter notebook - creditcard.ipynb
3. Refactoring jupyter notebook - functions.py



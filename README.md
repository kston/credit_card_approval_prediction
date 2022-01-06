
# Credit Card Approval Prediction

![CREDIT CARD APPROVAL](static/img/creditcard.jpg)

# Introduction:

## The Problem
A Company wants to automate the Credit Card eligibility process based on customer detail provided while filling online application.

They have given a problem to identify the customers segments which are eligible for Credit Card approval, so that they can specifically target these customers. It uses personal information and data submitted by credit card applicants to predict the probability of future defaults and credit card borrowings. The bank is able to decide whether to issue a credit card to the applicant.

## The Solution

I set out to find ways to use data to help the company by identifying those who should have or not their credit card approvaled.

The goal of this project is to model the historical submission for credit card from the company to predict whether a constumer request will ultimately be approved or declined. The client classified as a risk client should have the submited declined.

## Directory:

[Project Notebook](ml/Credit_Card_Approval_Prediction.ipynb)

[Data](credit_card_approval_prediction/ml/data/)

# Model Development

We use the SearchGrid method from Sklearn in the models above to find the best model to classify clients as risky_clients or not. 

The models:

## Logistic Regression  
Logistic Regression is a Machine Learning algorithm which is used for the classification problems, it is a predictive analysis algorithm and based on the concept of probability. It works by transforming a Linear Regression into a classification model through the use of the logistic function.

## K-Nearest Neighbors  
K Nearest Neighbour is a simple algorithm that stores all the available cases and classifies the new data or case based on a similarity measure. It is mostly used to classifies a data point based on how its neighbours are classified. It’s commonly used to solve problems in various industries because its ease of use, application to classification and regression problems, and the ease of interpretability of the results it generates.

## Support Vector Machines - SVM   
SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm finds a line (or hyperplane in dimensions greater than 2) in between different classes of data such that the distance on either side of that line or hyperplane to the next-closest data points is maximized.

## Decision Tree 
Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome. There are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches. The decisions or the test are performed on the basis of features of the given dataset. It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions. It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure. In order to build a tree, we use the CART algorithm, which stands for Classification and Regression Tree algorithm.

## Random Forest  
Random forest is the most simple and widely used algorithm. Used for both classification and regression. It is an ensemble of randomized decision trees. Each decision tree gives a vote for the prediction of target variable. Random forest choses the prediction that gets the most vote.

Random Forest is a ensemble bagging algorithm to achieve low prediction error. It reduces the variance of the individual decision trees by randomly selecting trees and then either average them or picking the class that gets the most vote.
## XGBoost  
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. It was initially developed by Tianqi Chen in 2014 and much faster until gradient boost, so it is a preferred boosting method. Thanks to the hyperparameters it contains, many adjustments can be made such as regularization hyperparameter prevent overfitting.

## LightGBM  
LGBM (stands for Light Gradient Boosting Machine) was initially released by Microsoft in 2017 and is another Gradient Boosting method preferred by users and is a decision tree based. The key difference from other methods is that it splits the tree based on leaves, that is, it can detect and stop the units needed by point shooting (remember: the others are in depth-based or level-based).

As LGBM is leaf-based, as seen in the figure above, LGBM is a very effective method in terms of reducing the error and therefore increasing the accuracy and speed. You can split categorical data with the special algorithm, but an integer value such as index must be entered instead of the string name of the column.

# Conclusion

The goal of this project was create a model that predict if the client who applied for a credit card in a company was a risky client or not. The client who are classified as risky can't have the credit card approved. After evaluate several models, the LightGBM was the one with most accuracy both in the train dataset and test dataset with score greater than 98%. The roc_curve shown AUC with 76% witch is also a good score. When we analyze the feature importance, we could see that the features witch shown the number of years employed(yers_employed), the clients imcome( amt_income_total), the age and the total number of months that the client is with the company(total_months), witch are all numerical categories, are the most important ones to predict a good/bad clients. The categorical feature with most importance is the name_income_type_working, witch shown that the clients income comes from working. Although the data was quite unbalanced, we worked around it by using the Imbalanced-learn (imported as imblearn) library that is an open source, MIT-licensed library relying on scikit-learn (imported as sklearn) and provides tools when dealing with classification with imbalanced classes.

# Deployment

App was put in production by using:

> Javascript / CSS / Bootstrap / HTML

> Flask

> Heroku

The deployed web app is live at https://creditcardapproval2022.herokuapp.com/

<a href="https://picasion.com/"><img src="https://i.picasion.com/pic91/2e1ff0d2360e23605bb104fd0d2f877c.gif" width="400" height="358" border="0" alt="https://picasion.com/" /></a><br /><a href="https://picasion.com/"></a>


# References

https://www.kaggle.com/rikdifos/credit-card-approval-prediction-using-ml

https://scialert.net/fulltext/?doi=jas.2010.1841.1858

https://towardsdatascience.com/k-nearest-neighbours-explained-7c49853633b6

https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989

https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d

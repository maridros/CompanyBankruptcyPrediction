# CompanyBankruptcyPrediction
Training of different classification algorithms with the aim of classifying companies into the ones that will fail and the ones that will succeed.
## Requirements
- Python
- Skicit-learn
- Numpy
- Pandas
- Keras
## Code process and results
### Problem
This code trains classification algorithms in order to predict whether a company will go bankrupt or not, based on 11 specific performance indicators of the corresponding company.
For this purpose an excel file is given (see InputData folder), which includes the corresponding data from some companies and the information on whether they eventually went bankrupt.

Specifically the data are the following:
1. 8 first columns (A to H): performance indicators of companies
2. Next 3 columns (I to K): binary activity indicators
3. 12th column (L): result (1 if they did not go bankrupt and 2 if they went bankrupt)
4. 13th column (M): year to which the above information relates

Finally, the model that will be created is considered successful if it meets two conditions:
- Finds with a success rate of at least 62% the companies that will go bankrupt.
- Finds with a success rate of at least 70% the companies that will not go bankrupt.
### Implemented solution
The code follows and compares two different approaches. In both approaches the classifiers that are trained and tested are the following:
- LDA (Linear Discriminant Analysis)
- Logistic Regression
- Decision Tree
- K-NN (K-Nearest Neighboors)
- Naive Bayes
- SVM (Support Vector Machines)
- ANN (Artificial Neural Network)

The only difference between the first and the second approach is that in the second approach some of the training data were ignored in order to have 3 healthy / 1 bankrupt ratio in the training dataset, which is much better than in the first case in which the ratio was approximately 49/1, as the healthy companies are the 98% of the hole dataset. Also in both approaches there were used some parameteres of the algorithms which can also solve the problem of the imbalanced dataset, like custom weights for each class. KNN was the only algorithm in which it was not used any method to balance the data, except from the ignoring of an amount of them in the second approach. In this algorithm there was a big improvement in the recall of the smaller class (bankrupt companies) in the second approach. 
### Results
Here you can see the comparison of the results of each approach (Part I and Part II) for each algorithm:
![bankrupt01](https://user-images.githubusercontent.com/89779679/132727988-ddff21f2-d3c8-4817-8604-b2c8fa2f5b5c.jpg)

And here you can see the comparison with more metrics:
![bankrupt02](https://user-images.githubusercontent.com/89779679/132728386-270d7ef5-66a6-48fa-ad31-bc516fca2562.jpg)

In conclusion, based on the above two diagrams (and especially the first one that concerns the two main conditions), all the classifiers, except K-NN, achieved the two goals (>=62% recall of class 1 and >=70% recall of class 0). The classifier that seems to have the best performance is the Logistic regression classifier, but other classifiers, such as SVM, have very close performance. Also in case of KNN we can see the imporovement which has been achived after removing data in order to make the training dataset more balanced (Part II). However this method has the drawback of losing importan information. So, for further improvement a more representative sample could be found. The best would be a larger sample with a sample ratio of about 1/1 of each class.

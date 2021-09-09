# Drosou Maria
# Department of Informatics And Computer Engineering, University of West Attica
# e-mail: cs151046@uniwa.gr
# A.M.: 151046

# import libraries
import pandas as pd  # excel reading
import sklearn  # we need this for the classifier
import numpy as np  # mathematical operations
import keras  # artificial neural networks

# import functions for creating models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Linear Discriminant Analysis
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.tree import DecisionTreeClassifier  # Decision Trees
from sklearn.neighbors import KNeighborsClassifier  # k-Nearest Neighbors
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.svm import SVC  # Support Vector Machine

# import secondary functions for metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix  # used for calculating TP, TN, FP, FN

# import secondary functions for splitting and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# import function for resampling (we need it at Part 1 II for downsampling the majority class)
from sklearn.utils import resample


# =====================================================
#                       Part 1 I
# =====================================================

# read the data
dataFile = 'InputData/Dataset2Use_Assignment2.xlsx'
sheetName = 'Total'

try:
    # Confirm file exists
    sheetValues = pd.read_excel(dataFile, sheetName)
    print(' .. successful parsing of file:', dataFile)
    print("Column headings")
    print(sheetValues.columns)
except FileNotFoundError:
    print(FileNotFoundError)


# create the input array
# input data include all columns except from the last two columns (output and year)
inputData = sheetValues[sheetValues.columns[:-2]].values

# now convert the categorical values to unique class id and save the name-to-id match
outputData = sheetValues[sheetValues.columns[-2]]
outputData, levels = pd.factorize(outputData)

# check if everything has been completed successfully by doing some printing
print(' .. we have', inputData.shape[0], 'available paradigms.')
print(' .. each paradigm has', inputData.shape[1], 'features')

print(' ... the distribution for the available class labels is:')
for classIdx in range(0, len(np.unique(outputData))):
    tmpCount = sum(outputData == classIdx)
    tmpPercentage = tmpCount/len(outputData)
    print(' .. class', str(classIdx), 'has', str(tmpCount), 'instances', '(', '{:.2f}'.format(tmpPercentage), '%)')


# now create train and test sets
X_train, X_test, y_train, y_test = train_test_split(inputData, outputData, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# calculate the number of train and test samples, as well as how many of them are positive (bankrupt)
X_train_quantity = X_train.shape[0]
P_train = sum(y_train)
X_test_quantity = X_test.shape[0]
P_test = sum(y_test)
print(' ... Train samples:')
print(' .. Total:', X_train_quantity)
print(' .. Bankrupt only:', P_train)
print(' ... Test samples:')
print(' .. Total:', X_test_quantity)
print(' .. Bankrupt only:', P_test)

# Create a dataframe for the results
resultsPart1Df = pd.DataFrame(columns=['Classifier Name', 'Training or test set',
                                       'Number of training samples', 'Number of non-healthy companies in sample',
                                       'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'F1 score', 'Accuracy'])

# Create dataframe for storing recall results for both classes
recallResults = pd.DataFrame(columns=['Classifier Name', 'Recall train set class 0', 'Recall test set class 0',
                                      'Recall train set class 1', 'Recall test set class 1'])


# Create a function for calculating the scores of each classifier and informing the results dataframe
def build_scores_df(data, model, output_train, output_pred_train, output_test, output_pred_test):
    # calculate the scores
    acc_train = accuracy_score(output_train, output_pred_train)
    acc_test = accuracy_score(output_test, output_pred_test)
    pre_train = precision_score(output_train, output_pred_train, average='macro')
    pre_test = precision_score(output_test, output_pred_test, average='macro')
    rec_train = recall_score(output_train, output_pred_train, average='macro')
    rec_test = recall_score(output_test, output_pred_test, average='macro')
    f1_train = f1_score(output_train, output_pred_train, average='macro')
    f1_test = f1_score(output_test, output_pred_test, average='macro')
    # print the scores
    print('Accuracy scores of', model, 'are:',
          'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
    print('Precision scores of', model, 'are:',
          'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
    print('Recall scores of', model, 'are:',
          'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
    print('F1 scores of', model, 'are:',
          'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))

    # calculate TN, FP, FN, TP and print them
    train_tn, train_fp, train_fn, train_tp = confusion_matrix(output_train, output_pred_train).ravel()
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(output_test, output_pred_test).ravel()
    print(model, 'TN: train:', train_tn, 'and test:', test_tn)
    print(model, 'FP: train:', train_fp, 'and test:', test_fp)
    print(model, 'FN: train:', train_fn, 'and test:', test_fn)
    print(model, 'TP: train:', train_tp, 'and test:', test_tp)

    # calculate the recall of each class and print them
    rec_train_separated = recall_score(output_train, output_pred_train, average=None)
    rec_test_separated = recall_score(output_test, output_pred_test, average=None)
    print('Class 0 (not bankrupt) recall score of', model, ':',
          'train: {:.2f}'.format(rec_train_separated[0]), 'and test: {:.2f}.'.format(rec_test_separated[0]))
    print('Class 1 (bankrupt) recall score of', model, ':',
          'train: {:.2f}'.format(rec_train_separated[1]), 'and test: {:.2f}.'.format(rec_test_separated[1]))

    # import the recall results to the recall results dataframe
    global recallResults
    recallResults = recallResults.append({'Classifier Name': model,
                                          'Recall train set class 0': rec_train_separated[0],
                                          'Recall test set class 0': rec_test_separated[0],
                                          'Recall train set class 1': rec_train_separated[1],
                                          'Recall test set class 1': rec_test_separated[1]}, ignore_index=True)

    # import the scores to the results dataframe
    data = data.append({'Classifier Name': model, 'Training or test set': 'train set',
                        'Number of training samples': X_train_quantity,
                        'Number of non-healthy companies in sample': P_train,
                        'TP': train_tp, 'TN': train_tn, 'FP': train_fp, 'FN': train_fn, 'Precision': pre_train,
                        'Recall': rec_train, 'F1 score': f1_train, 'Accuracy': acc_train}, ignore_index=True)
    data = data.append({'Classifier Name': model, 'Training or test set': 'test set',
                        'Number of training samples': X_test_quantity,
                        'Number of non-healthy companies in sample': P_test,
                        'TP': test_tp, 'TN': test_tn, 'FP': test_fp, 'FN': test_fn, 'Precision': pre_test,
                        'Recall': rec_test, 'F1 score': f1_test, 'Accuracy': acc_test}, ignore_index=True)
    return data


# now the models

print('=============================================================')
print('                 Linear Discriminant Analysis                ')
print('=============================================================')

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[.55, .45])
lda.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = lda.predict(X_train)
y_pred_test = lda.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart1Df = build_scores_df(resultsPart1Df, 'LDA', y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                     Logistic Regression                     ')
print('=============================================================')

# Logistic Regression
logreg = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear')
logreg.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart1Df = build_scores_df(resultsPart1Df, 'Logistic regression classifier',
                                 y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                        Decision Trees                       ')
print('=============================================================')

# Decision Trees
dtc = DecisionTreeClassifier(class_weight='balanced', max_depth=5)
dtc.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = dtc.predict(X_train)
y_pred_test = dtc.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart1Df = build_scores_df(resultsPart1Df, 'Decision Tree classifier', y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                     k-Nearest Neighbors                     ')
print('=============================================================')

# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart1Df = build_scores_df(resultsPart1Df, 'K-NN classifier', y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                         Naive Bayes                         ')
print('=============================================================')

# Naive Bayes
gnb = GaussianNB(priors=[.55, .45])
gnb.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart1Df = build_scores_df(resultsPart1Df, 'GNB classifier', y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                   Support Vector Machine                    ')
print('=============================================================')

# Support Vector Machine
svm = SVC(gamma='scale', class_weight='balanced')
svm.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart1Df = build_scores_df(resultsPart1Df, 'SVM classifier', y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                 Feed Forward Neural Network                 ')
print('=============================================================')

# Feed Forward Neural Network
CustomModel = keras.models.Sequential()
CustomModel.add(keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'))
CustomModel.add(keras.layers.Dense(2, activation='softmax'))
# display the architecture
CustomModel.summary()
# compile model using accuracy to measure model performance
CustomModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# calculate the weight for each class (useful for imbalanced data)
class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weight = dict(enumerate(class_weight))
# train the model
CustomModel.fit(X_train, keras.utils.np_utils.to_categorical(y_train),
                epochs=100, verbose=False, class_weight=class_weight)
# now check for both train and test data, how well the model learned the patterns
y_pred_train = CustomModel.predict_classes(X_train)
y_pred_test = CustomModel.predict_classes(X_test)
# calculate scores and import them to the results data frame
resultsPart1Df = build_scores_df(resultsPart1Df, 'ANN classifier', y_train, y_pred_train, y_test, y_pred_test)

# -----------------------------------------------------

# export the results of all models to excel file
writer = pd.ExcelWriter('OutputData/Results.xlsx')
resultsPart1Df.to_excel(writer, 'Part_I', index=False)
writer.save()

# =====================================================
#                       Part 1 II
# =====================================================

# read the data
dataFile = 'InputData/Dataset2Use_Assignment2.xlsx'
sheetName = 'Total'

try:
    # Confirm file exists
    sheetValues = pd.read_excel(dataFile, sheetName)
    print(' .. successful parsing of file:', dataFile)
    print("Column headings")
    print(sheetValues.columns)
except FileNotFoundError:
    print(FileNotFoundError)


# create the input array
# input data include all columns except from the last two columns (output and year)
inputData = sheetValues[sheetValues.columns[:-2]].values

# now convert the categorical values to unique class id and save the name-to-id match
outputData = sheetValues[sheetValues.columns[-2]]
outputData, levels = pd.factorize(outputData)

# check if everything has been completed successfully by doing some printing
print(' .. we have', inputData.shape[0], 'available paradigms.')
print(' .. each paradigm has', inputData.shape[1], 'features')

print(' ... the distribution for the available class labels is:')
for classIdx in range(0, len(np.unique(outputData))):
    tmpCount = sum(outputData == classIdx)
    tmpPercentage = tmpCount/len(outputData)
    print(' .. class', str(classIdx), 'has', str(tmpCount), 'instances', '(', '{:.2f}'.format(tmpPercentage), '%)')


# now create train and test sets
X_train, X_test, y_train, y_test = train_test_split(inputData, outputData, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# downsample majority class

P_train_inputs = X_train[(y_train == 1)]  # keep only the inputs of class 1 (bankrupt)
N_train_inputs = X_train[(y_train == 0)]  # keep only the inputs of class 0 (not bankrupt)

minorClassCount = len(P_train_inputs)  # calculate the number of class 1 elements (minority class)
print('Minority class quantity =', minorClassCount)  # print the result

# calculate the number of the desirable class 0 elements (three times the quantity of class 1 elements)
targetQuantity = minorClassCount*3

# downsample the inputs of class 0
N_train_inputs_downsampled = resample(N_train_inputs, replace=False, n_samples=targetQuantity, random_state=123)

# print previous and new quantity of the minority class
print('Majority class quantity at first:', len(N_train_inputs))
print('Majority class quantity downsampled:', len(N_train_inputs_downsampled))

# generate the new outputs for class 0 and class 1 according to their quantity of elements
N_train_outputs_downsampled = np.zeros(len(N_train_inputs_downsampled), dtype=int)
P_train_outputs = np.ones(len(P_train_inputs), dtype=int)

# combine both classes inputs and outputs in two new arrays
X_train_balanced = np.concatenate((P_train_inputs, N_train_inputs_downsampled))
Y_train = np.concatenate((P_train_outputs, N_train_outputs_downsampled))


# calculate the number of train and test samples, as well as how many of them are positive (bankrupt)
X_train_quantity = X_train_balanced.shape[0]
P_train = sum(Y_train)
X_test_quantity = X_test.shape[0]
P_test = sum(y_test)
print(' ... Train samples:')
print(' .. Total:', X_train_quantity)
print(' .. Bankrupt only:', P_train)
print(' ... Test samples:')
print(' .. Total:', X_test_quantity)
print(' .. Bankrupt only:', P_test)

# Create a dataframe for the results
resultsPart2Df = pd.DataFrame(columns=['Classifier Name', 'Training or test set',
                                       'Number of training samples', 'Number of non-healthy companies in sample',
                                       'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'F1 score', 'Accuracy'])

# now the models

print('=============================================================')
print('                 Linear Discriminant Analysis                ')
print('=============================================================')

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[.55, .45])
lda.fit(X_train_balanced, Y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = lda.predict(X_train_balanced)
y_pred_test = lda.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart2Df = build_scores_df(resultsPart2Df, 'LDA', Y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                     Logistic Regression                     ')
print('=============================================================')

# Logistic Regression
logreg = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear')
logreg.fit(X_train_balanced, Y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = logreg.predict(X_train_balanced)
y_pred_test = logreg.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart2Df = build_scores_df(resultsPart2Df, 'Logistic regression classifier',
                                 Y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                        Decision Trees                       ')
print('=============================================================')

# Decision Trees
dtc = DecisionTreeClassifier(class_weight='balanced', max_depth=5)
dtc.fit(X_train_balanced, Y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = dtc.predict(X_train_balanced)
y_pred_test = dtc.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart2Df = build_scores_df(resultsPart2Df, 'Decision Tree classifier', Y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                     k-Nearest Neighbors                     ')
print('=============================================================')

# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn.fit(X_train_balanced, Y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = knn.predict(X_train_balanced)
y_pred_test = knn.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart2Df = build_scores_df(resultsPart2Df, 'K-NN classifier', Y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                         Naive Bayes                         ')
print('=============================================================')

# Naive Bayes
gnb = GaussianNB(priors=[.55, .45])
gnb.fit(X_train_balanced, Y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train_balanced)
y_pred_test = gnb.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart2Df = build_scores_df(resultsPart2Df, 'GNB classifier', Y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                   Support Vector Machine                    ')
print('=============================================================')

# Support Vector Machine
svm = SVC(gamma='scale', class_weight='balanced')
svm.fit(X_train_balanced, Y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = svm.predict(X_train_balanced)
y_pred_test = svm.predict(X_test)
# calculate scores and import them to the results data frame
resultsPart2Df = build_scores_df(resultsPart2Df, 'SVM classifier', Y_train, y_pred_train, y_test, y_pred_test)


print('=============================================================')
print('                 Feed Forward Neural Network                 ')
print('=============================================================')

# Feed Forward Neural Network
CustomModel = keras.models.Sequential()
CustomModel.add(keras.layers.Dense(64, input_dim=X_train_balanced.shape[1], activation='relu'))
CustomModel.add(keras.layers.Dense(2, activation='softmax'))
# display the architecture
CustomModel.summary()
# compile model using accuracy to measure model performance
CustomModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# define the weight of each class (useful for imbalanced data)
class_weight = {0: 1., 1: 3.}
# train the model
CustomModel.fit(X_train_balanced, keras.utils.np_utils.to_categorical(Y_train),
                epochs=100, verbose=False, class_weight=class_weight)
# now check for both train and test data, how well the model learned the patterns
y_pred_train = CustomModel.predict_classes(X_train_balanced)
y_pred_test = CustomModel.predict_classes(X_test)
# calculate scores and import them to the results data frame
resultsPart2Df = build_scores_df(resultsPart2Df, 'ANN classifier', Y_train, y_pred_train, y_test, y_pred_test)

# -----------------------------------------------------

# export the new results of all models to the same excel file to a new sheet
resultsPart2Df.to_excel(writer, 'Part_II', index=False)
writer.save()
writer.close()

# export the recall results of all models to excel file
recall_writer = pd.ExcelWriter('OutputData/RecallResults.xlsx')
recallResults.to_excel(recall_writer, 'All', index=False)
recall_writer.save()
recall_writer.close()

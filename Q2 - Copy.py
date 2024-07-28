import pandas as pd
import numpy as np
#importing for visualizing data and plotting 
import matplotlib.pyplot as plt
import seaborn as sns
#Reading csv file into a dataframe
df = pd.read_csv('DataB.csv')
#Get the last column in another variable
gndColumn = df['gnd']
#Removing unnecessary columns
df = df.loc[:, ~df.columns.isin(['Unnamed: 0', 'gnd'])]
#Visualizing top rows of data file after dropping columns
df.head()

#Normalize by subtracting mean
df = df - df.mean()

#Visualizing top rows of data file after dropping columns
df.head()

#Get the data in a matrix
A = np.matrix(df)
#Creating covariance matrix by using transpose matrix
covarianceMatrix = np.cov(A.transpose())
#Calculate the eigenvectors and eigenvalues
eigenValue, eigenVector = np.linalg.eig(covarianceMatrix)
#Considering first 2 columns of the eigenvector
eigenVectors12 = np.take(eigenVector, [0, 1], axis=1)
#Input is projected on the first 2 PCs and taking transpose:
projection12 = np.matmul(eigenVectors12.T,A.T).transpose()
#Create a dataframe PrincipalComponent12 with 3 columns consisting of the first 2 PCs and gnd column
PrincipalComponent12 = pd.DataFrame(projection12, columns=['PCA Principal Component 1', 'PCA Principal Component 2']).assign(gndColumn=gndColumn.values)
#Visuallize the dataframe PrincipalComponent12
PrincipalComponent12.head(10)


#Setting the plot size and axes
fig = plt.figure(figsize=(10, 8))

#Scatterplot with predefined set of colours
sns.scatterplot(x = "PCA Principal Component 1", y = "PCA Principal Component 2", data = PrincipalComponent12, palette = 'Set1', hue = "gndColumn")

#Considering the fifth and sixth columns of the eigenvector
eigenVectors56 = np.take(eigenVector, [4, 5], axis=1)
#Input is projected on the first 2 PCs and taking transpose:
projection56 = np.matmul(eigenVectors56.T,A.T).transpose()
#Create a dataframe PrincipalComponent56 with 3 columns consisting of the 5th and 6th PCs and gnd column
PrincipalComponent56 = pd.DataFrame(projection56, columns=['PCA Principal Component 5', 'PCA Principal Component 6']).assign(gndColumn=gndColumn.values)
#Visuallize the dataframe PrincipalComponent56
PrincipalComponent56.head(10)


#Setting the plot size and axes
fig = plt.figure(figsize=(10, 8))

#Scatterplot with predefined set of colours
sns.scatterplot(x = "PCA Principal Component 5", y = "PCA Principal Component 6", data = PrincipalComponent56, palette = 'Set1', hue = "gndColumn")


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

retainedVarList = []
testErrorList = []
trainErrorList = []
principalComponentsN = [2,4,10,30,60,200,500,784]

for i in principalComponentsN:
        #Considering first i PCs of the eigenvector
        eigenVectorsN = np.take(eigenVector, range(i), axis=1)

        #Input is projected on the first i PCs, taking transpose and get the dataframe
        PrincipalComponentN = pd.DataFrame(np.matmul(eigenVectorsN.T,A.T).transpose())

        #sum(eigenValue) gives the total variance and eigenValue[:i] gives variance because of first i components
        #The division gives the amount of variance retained when we consider first i components
        varRetained = np.divide(sum(np.take(eigenValue,range(i))),sum(eigenValue))

        #Create training and testing data
        X_train, X_test, y_train, y_test = train_test_split(PrincipalComponentN,gndColumn,random_state=39,test_size=0.2) 

        #Training data
        naiveBayesClassifier = GaussianNB().fit(X_train,y_train)

        #Get predictions for training and testing data
        y_train_pred,y_test_pred = naiveBayesClassifier.predict(X_train),naiveBayesClassifier.predict(X_test)

        #Get classification errors for training and testing data
        trainingError = (1 - accuracy_score(y_train, y_train_pred))
        testingError = (1 - accuracy_score(y_test, y_test_pred))
        retainedVarList.append(varRetained) 
        testErrorList.append(testingError)
        trainErrorList.append(trainingError)

#Expressing variance which is retained against training and testing errors in a line plot 
fig, ax = plt.subplots(figsize=(10, 8))
#Training error is indicated by red line and testing error by the green line
ax.plot(retainedVarList,trainErrorList,label='Training error',color='red')
ax.plot(retainedVarList,testErrorList,label='Testing error',color='green')
ax.legend(loc='upper left')
#Name the axes in the plot
plt.setp(ax, xlabel="Retained Variance", ylabel="Error")
#Display the plot
plt.show()


#Importing LDA library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#Training the input and fitting it to 2 dimensional space by using LDA for dimensionality reduction 
LDAComponent12 = LinearDiscriminantAnalysis(n_components=2).fit_transform(df, gndColumn)
#Considering 2 LDA components
LDADataframe = pd.DataFrame(LDAComponent12,columns=['LDA Component 1','LDA Component 2']).assign(gndColumn=gndColumn.values)
LDADataframe.head(10)

#Setting the plot size and axes 
fig = plt.figure(figsize=(10, 8))

#Scatterplot with predefined set of colours
sns.scatterplot(x = "LDA Component 1", y = "LDA Component 2", data = LDADataframe, palette = 'Set1', hue = "gndColumn")

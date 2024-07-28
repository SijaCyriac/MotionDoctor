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

'''
Step 1 of PCA is to normalize the sample matrix
In our case, the sample matrix is n x d where n=2066 and d=784.
To normalize we use, a sample 
Xij = Xij - mean(Xj) where i ranges from 1 to n and j ranges from 1 to d
'''

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


'''
We can see that using PCA and first and second Principal components which contain the maximum variance helps to distinguish between the various classes. 
The classes have been set to different colours to show the variation. 
We were successfully able to linearly transform the data to 2 dimensions. We also note that classes corresponding to red, blue and orange are more separated. 
Classes represented by purple and green are much more similar to each other than the other classes because of which they appear more superimposed on each other.
'''

#Considering the fifth and sixth columns of the eigenvector
eigenVectors56 = np.take(eigenVector, [4, 5], axis=1)
#Input is projected on the first 2 PCs and taking transpose:
projection56 = np.matmul(eigenVectors56.T,A.T).transpose()
#Create a dataframe PrincipalComponent56 with 3 columns consisting of the 5th and 6th PCs and gnd column
PrincipalComponent56 = pd.DataFrame(projection56, columns=['PCA Principal Component 5', 'PCA Principal Component 6']).assign(gndColumn=gndColumn.values)
#Visuallize the dataframe PrincipalComponent56
PrincipalComponent56.head(10)


# In[378]:


#Setting the plot size and axes
fig = plt.figure(figsize=(10, 8))

#Scatterplot with predefined set of colours
sns.scatterplot(x = "PCA Principal Component 5", y = "PCA Principal Component 6", data = PrincipalComponent56, palette = 'Set1', hue = "gndColumn")


# 3. Here, when we plot for the 5th and 6th principal component we see that the classes are not as distinguishable from each other as we had seen with principal components 1 and 2. This is because, most of the features or information of the data is stored in the initial principal components. The plot shows the different classes but they are very close and more superimposed or mixed with each other. This shows that we get more variance(distinguishing classes) from the initial principal components. To retain the largest part of variance we must focus on just the initial principal components.

# The retained variance is the ratio of variance on using m components over total variance of d components.
# 
# $$
# RV = \left( \sum_{i=1}^m \lambda_i \right) / \left( \sum_{i=1}^d \lambda_i \right)
# $$
# 
# The retained variance in the new dimension will be higher when higher number of components are considered and will be closer to 1 or 100%

# In[373]:


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


# In[374]:


#Importing LDA library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#Training the input and fitting it to 2 dimensional space by using LDA for dimensionality reduction 
LDAComponent12 = LinearDiscriminantAnalysis(n_components=2).fit_transform(df, gndColumn)
#Considering 2 LDA components
LDADataframe = pd.DataFrame(LDAComponent12,columns=['LDA Component 1','LDA Component 2']).assign(gndColumn=gndColumn.values)
LDADataframe.head(10)


# In[379]:


#Setting the plot size and axes
fig = plt.figure(figsize=(10, 8))

#Scatterplot with predefined set of colours
sns.scatterplot(x = "LDA Component 1", y = "LDA Component 2", data = LDADataframe, palette = 'Set1', hue = "gndColumn")


# 5. We can see that LDA is able to clearly distinguish the classes. The colours which are used to represent the different classes are separated from each other more vividly as compared to that seen in PCA with first 2 components. It is able to differentiate between the different classes except for classes that are represented with purple and green colours which are more similar to each other than other classes. LDA plot shows the classes clustered with very little overlap, showing that it obtains the variance between the classes wheras PCA obtains the variance within the input data.

# 6. We want to prove that PCA is the best linear method for transformation (with orthonormal bases).
# 
#     PCA uses orthonormal bases i.e. eigenvectors which are independent of each other and normalized. PCA does not require labels for linearly transforming data i.e. it is unsupervised, whereas LDA requires labels i.e. it is supervised. Also, PCA doesn't assume that the data needs to be normally distributed but linear methods like LDA needs data to be normally distributed. So, we can say that PCA is the best method as it doesn't depend on labels or data to be normally distributed. Also, PCA obtains the maximum variance within data helping to get the maximum amount of information about the data whereas other methods like LDA focus on separating the classes by obtaining variance between the classes and may lose information because of bias. PCA minimizes the mean square error which is optimal. 
# 

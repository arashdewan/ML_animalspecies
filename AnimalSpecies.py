#!/usr/bin/env python
# coding: utf-8

# ### Animal Species

# In[1]:


import warnings
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np
from numpy import arange
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RepeatedKFold
import seaborn as sns
import statistics as st


# In[2]:


url = "https://philchodrow.github.io/PIC16A/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)


# #### Creating train (X) and test (Y) data from dataset:

# In[ ]:


# Sets predictor data (X) to include all variables except Species and target data (y) to only Species (our target variable)
X = penguins.drop(labels = "Species", axis = 1)
y = penguins[["Species"]]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)


# In[ ]:


# Checks that data sets were created correctly
X.shape, y.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape


# #### Cleaning and checking cleanup for X_train and X_test:

# In[ ]:


def X_cleanup(X):
  """
  X_cleanup: A function that takes in a dataframe X and cleans it so it is ready for analysis
  Paramaters: 
    X: The dataframe to be cleaned
  Returns:
    X: The cleaned data frame
  """
  # Drops unnessecary colums from X
  X = X.drop(["Region", "studyName", "Individual ID", "Stage", "Comments", 
            "Clutch Completion", "Date Egg", "Sample Number"], axis = 1)
  
  # Drops NaN and Na values from X
  X = X.dropna()

  # Removes rows in X where "Sex" is coded as "."
  X = X[X["Sex"] != "."]

  # Recodes Sex values of MALE and FEMALE to 1 and 2 respectively
  X["Sex"] = X["Sex"].map({"MALE": 1, "FEMALE": 2})

  # Recodes Island values of Torgersen, Dream, and Biscoe to 1, 2 and 3 respectively
  X["Island"] = X["Island"].map({"Torgersen": 1, "Dream": 2, "Biscoe": 3})
  
  return X


# In[ ]:


# Cleaning X_train, X_test and X
X_train = X_cleanup(X_train)
X_test = X_cleanup(X_test)
X = X_cleanup(X)


# In[ ]:


# Checking that X_train, X_test, and X were cleaned correctly
X_train, X_test, X


# #### Cleaning and checking cleanup for y_train and y_test:

# In[ ]:


def y_cleanup(y, X):
  """
  y_cleanup: A function that takes in a data frame y and a data frame X and cleans y so it is ready for analysis
  Paramaters:
    y: The data frame to be cleaned
    X: Another data frame that is used to ensure the same indicies are used for y and X
  Returns:
    y: The cleaned data frame
  """
  # Ensures that indicies of y are the same as the indicies of X
  y = y.loc[X.index.to_numpy()]

  # Recodes the values of species from Adelie, Chinstrap, and Gentoo to 1, 2, and 3 respectively
  y["Species"] = y["Species"].map({"Adelie Penguin (Pygoscelis adeliae)": 1,
                                 "Chinstrap penguin (Pygoscelis antarctica)": 2,
                                 "Gentoo penguin (Pygoscelis papua)": 3})
  return y


# In[ ]:


# Cleaning y_train, y_test, and y
y_train = y_cleanup(y_train, X_train)
y_test = y_cleanup(y_test, X_test)
y = y_cleanup(y, X)


# In[ ]:


# Checks that y_train, y_test, and y were cleaned properly
y_train, y_test, y


# #### Observing the relationships of variables

# In[ ]:


# Creates a temporary data frame for visual analysis. Executes the same functions as X_cleanup but doesn't recode Island variable so X_cleanup was not used
temp_df = penguins
temp_df = temp_df.drop(["Region", "studyName", "Individual ID", "Stage", "Comments", 
            "Clutch Completion", "Date Egg", "Sample Number"], axis = 1)
temp_df = temp_df.dropna()
temp_df = temp_df[temp_df["Sex"] != "."]
temp_df["Sex"] = temp_df["Sex"].map({"MALE": 1, "FEMALE": 2})

# Creates a table comparing the mean culmen length, flipper length, body mass, and Delta 15 N for each species, island, and sex
temp_df.groupby(["Species", "Island", "Sex"])["Culmen Length (mm)", "Flipper Length (mm)", "Body Mass (g)", "Delta 15 N (o/oo)"].mean()


# In[ ]:


sns.set_theme(style="ticks")
sns.pairplot(temp_df[["Culmen Length (mm)", "Culmen Depth (mm)", "Body Mass (g)","Species"]], 
             hue = "Species", diag_kind = 'hist', ).fig.suptitle("Pairplot of Culmen Length, Culmen Depth, and Body Mass", y = 1.08)


# In[ ]:


fig, ax = plt.subplots(2, figsize = (10,8), sharey = True)
#creating a function that we can call to take in the various types of data we need
def hist(df, column, alpha, index):
  ax[index].hist(df[column], alpha = alpha)
  ax[index].set(ylabel = "Frequency",
         xlabel = column
         )
temp_df.groupby("Species", as_index = False).apply(hist, "Delta 15 N (o/oo)", 0.3, 0)
temp_df.groupby("Species", as_index = False).apply(hist, "Flipper Length (mm)", 0.3, 1)
ax[0].legend(labels = ["Adelie", "Chinstrap", "Gentoo"], title = "Species")
plt.tight_layout()


# In[ ]:


#scatter plot
fig, ax = plt.subplots(1, 2, figsize = (15,10), sharey = True)

#a list of unique species
unique = list(set(temp_df['Species']))

#labeling x and y values and titles of scatterplot
ax[0].set(xlabel = "Body Mass (g)",
          ylabel = "Delta 15 N (o/oo)",
          title = "Male")
ax[1].set(xlabel = "Body Mass (g)",
          title = "Female")

#scatter values of every unique species for each sex
for i in range(len(unique)):
  massM = temp_df['Body Mass (g)'][temp_df['Species'] == unique[i]][temp_df['Sex'] == 1]
  nitroM = temp_df['Delta 15 N (o/oo)'][temp_df['Species'] == unique[i]][temp_df['Sex'] == 1]
  massF = temp_df['Body Mass (g)'][temp_df['Species'] == unique[i]][temp_df['Sex'] == 2]
  nitroF = temp_df['Delta 15 N (o/oo)'][temp_df['Species'] == unique[i]][temp_df['Sex'] == 2]
  
  ax[0].scatter(massM, nitroM)
  ax[0].legend(unique)

  ax[1].scatter(massF, nitroF)
  ax[1].legend(unique)


# #### Feature Selection

# In[ ]:


# Sets potential_features to the different combinations of features found from the above plots
potential_features = [['Sex', 'Culmen Length (mm)', 'Culmen Depth (mm)'], ['Sex', 'Culmen Length (mm)', 'Flipper Length (mm)'], 
                      ['Island', 'Culmen Length (mm)', 'Body Mass (g)'], ['Island', 'Culmen Length (mm)', 'Flipper Length (mm)']]

def cv_score(cols, model):
  """
  cv_score: A function that takes in specified columns and a model and returns the mean of 10 cross-validation scores of the model using the specified cols
  Parameters:
    cols: A list of features in the data set to be scored
    model: The model to be used for cross-validation
  Returns: 
    cv_score prints the columns it is using and returns the mean of 10 cross_validation scores of the model using the specified columns
  """
  print("Using columns: {0}".format(cols))
  return cross_val_score(model, X_train[cols], np.array(y_train["Species"]), cv = 10).mean()


def test_data_score(cols, model):
  """
  test_data_score: A function that takes in specified columns and a model and returns the score of the fitted model against the test set
  Parameters:
    cols: A list of features in the data set to be scored
    model: The model to be used for cross-validation
  Returns:
    test_data_score fits the model using the specified columns and returns the score of the fitted model against the test set
  """
  return model.fit(X_test[cols], np.array(y_test["Species"])).score(X_test[cols], np.array(y_test["Species"]))

for features in potential_features:
  print("CV score: {0}".format(cv_score(features, LogisticRegression(max_iter = 500))))
  print("Test score: {0}".format(test_data_score(features, LogisticRegression(max_iter = 500))))


# In[ ]:


# Sets features to best features found from above functions
features = ['Sex', 'Culmen Length (mm)', 'Culmen Depth (mm)']

# Sets X, X_train, and X_test to only include features columns
X = X[features]
X_train = X_train[features]
X_test = X_test[features]


# #### Modeling
# 

# In[ ]:


def fit_and_score(model):
  """
  fit_and_score: A Function that takes in a model with specified paramaters and fits the model to the training data and scores it against the training and testing data
  Paramaters:
    model: A model with specified paramaters that should be fit
  Returns:
    fit_and_score prints the scores of the fitted model against the train data and the test data and returns the fitted model
  """
  print("Score on the training data:", model.fit(X_train, np.array(y_train["Species"])).score(X_train, np.array(y_train["Species"])))
  print("Score on the testing data:", model.fit(X_train, np.array(y_train["Species"])).score(X_test, np.array(y_test["Species"])))
  return model.fit(X_train, np.array(y_train["Species"]))


# In[ ]:


def ConfusionMatrix(model):
  """
  ConfusionMatrix: A function that takes in a fitted model and prints the accuracy score of its predictions as well as a confusion matrix for the fitted model
  Parameters:
    model: A fitted model
  Returns:
    ConfusionMatrix does not return any values and instead prints the accuracy score of the fitted model's predictions and a confusion matrix for the fitted model
  """
  # Evaluates the classifier on the test set
  y_pred = model.predict(X_test)
  print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
  # Creates a confusion matrix
  cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                    index = ['Actual ' + i for i in ["Adelie", "Chinstrap", "Gentoo"]],
                    columns = ['Predicted ' + i for i in ["Adelie", "Chinstrap", "Gentoo"]])
  print('Confusion Matrix:')
  display(cm)


# In[ ]:


def DecisionRegion(model, X_set = X, y_set = y, increment = 0.01):
  """
  DecisionRegion: A function that takes in a fitted model and two sets of data and makes a decision region plot based on the model and sets of data
  Parameters: 
    model: A fitted model
    X_set: A data frame of predictor variables. Can be an entire data set, the training set, or the testing set. Default is X (entire set of predictors)
    y_set: A data frame of the target variable. Can be the entire data set, the training set, or the testing set. Defauilt is y (entire set of targets)
    increment: Adjusts number of data points created for meshgrid. Default is 0.01. Adjusting increment can assist with the runtime for different model families
    (Note - X_set and Y_set must have the same number of rows, i.e if X_set = X_train then y_set must = y_train or if X_set = X then y_set must = y)
  Returns:
    DecisionRegion does not return any values and instead plots the decision region of the fitted model for X_set and y_set
  """

  # Checks if X_set and y_set are the same number of rows and raises a TypeError if not
  if X_set.shape[0] != y_set.shape[0]:
    raise TypeError("X_set and Y_set must have the same number of rows!")

  # Creates two dictionaries. One for labeling the Sex of each plot and the other for labeling the suptitle of the entire plot
  sex_dict = {1: "Male", 2: "Female"}
  title_dict = {X.shape: "Full Data Set", X_train.shape: "Training Set", X_test.shape: "Testing Set"}

  # Creates string for the model being used
  model_string = str(model).split("(")[0]

  # Assigns the unique qualitative variable values of X to unique_qual_values
  unique_qual_values = [int(i) for i in np.unique(X["Sex"])]

  # Creates a 2 side by side plots that share a y-axis
  fig, ax = plt.subplots(1, 2, figsize = (12, 7), sharey = True)

  # Creates a meshgrid of the dataset (uses all possible values (incrementing by 0.01) from min to max of each feature in X)
  f1_min, f1_max = X["Culmen Length (mm)"].min() - 1, X["Culmen Length (mm)"].max() + 1
  f2_min, f2_max = X["Culmen Depth (mm)"].min() - 1, X["Culmen Depth (mm)"].max() + 1
  f1, f2 = np.meshgrid(np.arange(f1_min, f1_max, increment), np.arange(f2_min, f2_max, increment))

  # For loop that iterates through each unique value of the qualitative feature of X
  for i in unique_qual_values:

    # Predicts the species of each point in the meshgrid
    Z = model.predict(np.c_[np.ones(f1.ravel().shape) * 1.0 * i, f1.ravel(), f2.ravel()])
    Z = Z.reshape(f1.shape)

    # Plots the test set samples as a scatter plot
    ax[i - 1].scatter(X_set[X_set["Sex"]==i]["Culmen Length (mm)"], 
                       X_set[X_set["Sex"]==i]["Culmen Depth (mm)"], 
                       c=y_set[X_set["Sex"]==i]["Species"], cmap='jet') 
  
    # Plots the decision regions
    ax[i - 1].contourf(f1, f2, Z, alpha=0.2, cmap='jet') 
    
    # Sets the xlabel of the plot to Culmen Length (mm) and the title to the respective qualitiative feature value
    ax[i - 1].set_xlabel('Culmen Length (mm)')
    ax[i - 1].set_title('Sex = ' + sex_dict[i])

  # Makes a legend for each species and their respective color
  legend0 = mpatches.Patch(color = 'red', label = 'Adelie', alpha = 0.2)
  legend1 = mpatches.Patch(color = 'green', label = 'Chinstrap', alpha = 0.2)
  legend2 = mpatches.Patch(color = 'blue', label = 'Gentoo', alpha = 0.2)
  fig.legend(handles = [legend0, legend1, legend2],loc = (0.85,0.8), fontsize = 'medium',framealpha = 1)

  # Sets the ylabel for the entire plot to Culmen Depth (mm) and the title for the entire plot to the type of data set used for X_set and y_set as well as the model and parameters used
  ax[0].set_ylabel('Culmen Depth (mm)')
  plt.suptitle('Decision Regions of the {0} Classifier: {1}'.format(model_string, title_dict[X_set.shape]))
  plt.tight_layout()
  plt.show()


# #### Logistic Regression Classification
# 

# In[ ]:


max_score = 0

# For loop that iterates from 0 to 49
for iter in range(50):

  # Ignores any warnings from LogisticRegression() (there were often warnings when the max_iter was small and it would make the output very hard to read)
  warnings.simplefilter("ignore")

  # Sets cv_score to the mean cross-validation score of the LogisticRegression model
  cv_score = cross_val_score(LogisticRegression(max_iter = iter*10), X_train, np.array(y_train["Species"]), cv = 10).mean()

  # Checks if the cv_score is larger than the current max and sets max_score to current cv_score and best_max_iter to iter*10 if yes
  if cv_score > max_score:
    max_score = cv_score
    best_max_iter = iter*10
    
print("The best value for max_iter is {0} with a cross-validation score of {1}".format(best_max_iter, max_score))


# In[ ]:


logistic_model = fit_and_score(LogisticRegression(max_iter = best_max_iter))


# In[ ]:


ConfusionMatrix(logistic_model)


# In[ ]:


DecisionRegion(logistic_model, X, y)


# #### Nearest Neighbors Classification

# In[ ]:


#defining a KNN model with that runs the test before using cross validation to find the best hyper parameters
model1 = KNeighborsClassifier(n_neighbors = 50)
model1.fit(X_train, y_train["Species"])
test_score = model1.score(X_test, y_test["Species"])
train_score = model1.score(X_train, y_train["Species"])
print('Testing Score Without CV: {:,.4f}'.format(test_score))
print('Training Score Without CV: {:,.4f}'.format(train_score))


# In[ ]:


warnings.simplefilter("ignore") #ignores warnings of KNieghborsClassifier do to poor parameters as it goes through the GridSearchCv
model2 = KNeighborsClassifier()
#creating a dictionary where that will be set in the GridSearchCV to run through a range of values for the hyper parameters of n_neighbors and leaf_size
param_grid = {'n_neighbors' : np.arange(1,10),
              'leaf_size' : np.arange(1,20)}
#with GridSearchCV, we are able to test multiple parameters within a range of values to find the best accuracy
model2_gscv = GridSearchCV(model2, param_grid, cv = 10)
model2_gscv.fit(X, y)
print('Best Parameters: {0}'.format(model2_gscv.best_params_))


# In[ ]:


warnings.simplefilter("ignore") #ignores warnings of KNieghborsClassifier do to poor parameters as it goes through the GridSearchCv 
#recreated a KNN model but instead set the best hyper parameters we got from the above section
model3 = KNeighborsClassifier(n_neighbors = 3, leaf_size = 1)
#creating a dictionary where with different 'weights' parameters so that we can see what the most accurate one is for our code
w_grid = dict()
w_grid['weights'] = ['distance', 'uniform']
model3_gscv = GridSearchCV(model3, w_grid, cv = 10)
model3_gscv.fit(X, y)
print('Best Parameter: {0}'.format(model3_gscv.best_params_))


# In[ ]:


KNN_model = fit_and_score(KNeighborsClassifier(n_neighbors = 3, leaf_size = 1, weights = 'uniform'))


# In[ ]:


ConfusionMatrix(KNN_model)


# In[ ]:


DecisionRegion(KNN_model, X, y, 0.1)


# #### Ridge Regression Classification

# In[ ]:


clf = RidgeClassifier()
orig = clf.fit(X_train, y_train)

print('Original Training Data: {0}'.format(orig.score(X_train, y_train)))
print('Original Testing Data: {0}'.format(orig.score(X_test, y_test)))


# In[ ]:


model = RidgeClassifier(alpha = 1)
cv = RepeatedKFold(n_splits = 5, n_repeats = 10, random_state = 1)

grid = dict()
grid['alpha'] = arange(0, 1, 0.01)
search = GridSearchCV(model, grid, cv = cv, n_jobs = -1)
results = search.fit(X_train, y_train)

print('Best Parameter: {0}'.format(results.best_params_))


# In[ ]:


modelRidge = fit_and_score(RidgeClassifier(alpha = 0))


# In[ ]:


ConfusionMatrix(modelRidge)


# In[ ]:


DecisionRegion(modelRidge, X, y)


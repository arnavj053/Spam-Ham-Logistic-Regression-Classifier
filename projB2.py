#!/usr/bin/env python
# coding: utf-8

# # Project B2: Spam/Ham Classification - Build Your Own Model
# 
# ## Feature Engineering, Classification, and Cross-Validation
# ## Due Date: Thursday, November 30th, 11:59 PM PDT
# You must submit this assignment to Gradescope by the on-time deadline, Thursday, November 30th, 11:59 PM. 
# Please read the syllabus for the grace period policy. 
# No late submissions beyond the grace period will be accepted. While course staff is happy to help you if you encounter difficulties with submission, we may not be able to respond to last-minute requests for assistance (TAs need to sleep, after all!). 
# **We strongly encourage you to plan to submit your work to Gradescope several hours before the stated deadline.** 
# This way, you will have ample time to reach out to staff for submission support.
# 
# ### Collaboration Policy
# 
# Data science is a collaborative activity. 
# While you may talk with others about this project, we ask that you **write your solutions individually**. 
# If you discuss the assignments with others, please **include their names** in the collaborators cell below.

# In[79]:


# Initialize Otter
import otter
grader = otter.Notebook("projB2.ipynb")


# **Collaborators**: *list collaborators here*

# ## Introduction
# In this project, you will build and improve on the concepts and functions you implemented in Project B1 to create your own classifier to distinguish spam emails from ham (non-spam) emails. We will evaluate your work based on your model's accuracy and written responses in this notebook.
# 
# After this assignment, you should feel comfortable with the following:
# 
# - Using `sklearn` libraries to process data and fit classification models,
# - Validating the performance of your model and minimizing overfitting, and
# - Generating and analyzing ROC curves.
# 
# ## Content Warning
# This is a **real-world** dataset– the emails you are trying to classify are actual spam and legitimate emails. As a result, some of the spam emails may be in poor taste or be considered inappropriate. We think the benefit of working with realistic data outweighs these inappropriate emails and wanted to warn you at the beginning of the project so that you are made aware.
# 
# If you feel uncomfortable with this topic, **please contact your GSI or the instructors, or reach out via the Fall 2023 [extenuating circumstances form](https://docs.google.com/forms/d/e/1FAIpQLSffIhNMwsxP9Pd5l_9dzx_V4VffsNIOOjOexLT9VgUtmzw4AA/viewform).**

# In[80]:


# Run this cell to suppress all FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ## Grading
# Grading is broken down into autograded answers and free responses. 
# 
# For autograded answers, the results of your code are compared to provided and/or hidden tests.
# 
# For free response questions, readers will evaluate how well you answered the question and/or fulfilled the requirements of the question.
# 
# Question | Manual | Points
# ----|----|----
# 1 | Yes | 6
# 2a | Yes | 4
# 2b | Yes | 2
# 3 | Yes | 3
# 4a | No | 5
# 4b | No | 10
# Total | 4 | 30

# In[81]:


import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)

from datetime import datetime
from IPython.display import display, HTML


# <br/>
# <hr style="border: 5px solid #003262;" />
# <hr style="border: 1px solid #fdb515;" />
# 
# # Setup and Recap
# 
# Here, we will provide a summary of Project B1 to remind you of how we cleaned the data, explored it, and implemented methods helpful in building your own model.

# ## Loading and Cleaning Data
# 
# Remember that in email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email. 
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). Your labeled training dataset contains 8,348 labeled examples, and the unlabeled test set contains 1,000 unlabeled examples.
# 
# Run the following cell to load the data into a `DataFrame`.
# 
# The `train` `DataFrame` contains labeled data that you will use to train your model. It contains four columns:
# 
# 1. `id`: An identifier for the training example.
# 1. `subject`: The subject of the email.
# 1. `email`: The text of the email.
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam).
# 
# The `test` `DataFrame` contains 1,000 unlabeled emails. You will predict labels for these emails and submit your predictions to the autograder for evaluation.

# In[82]:


import zipfile
with zipfile.ZipFile('spam_ham_data.zip') as item:
    item.extractall()


# In[83]:


original_training_data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Convert the emails to lowercase as the first step of text processing.
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()


# Feel free to explore the dataset above along with any specific spam and ham emails that interest you. Keep in mind that our data may contain missing values, which are handled in the following cell.

# In[84]:


# Fill any missing or NAN values.
print('Before imputation:')
print(original_training_data.isnull().sum())
original_training_data = original_training_data.fillna('')
print('------------')
print('After imputation:')
print(original_training_data.isnull().sum())


# ## Training/Validation Split
# 
# Recall that the training data we downloaded is all the data we have available for both training models and **validating** the models that we train. We, therefore, split the training data into separate training and validation datasets. You will need this **validation data** to assess the performance of your classifier once you are finished training. 
# 
# As in Project B1, we set the seed (`random_state`) to 42. **Do not modify this in the following questions, as our tests depend on this random seed.**

# In[85]:


# This creates a 90/10 train-validation split on our labeled data.
from sklearn.model_selection import train_test_split
train, val = train_test_split(original_training_data, test_size = 0.1, random_state = 42)

# We must do this in order to preserve the ordering of emails to labels for words_in_texts.
train = train.reset_index(drop = True)


# ## Feature Engineering
# 
# We need a numeric feature matrix $\mathbb{X}$ and a vector of corresponding binary labels $\mathbb{Y}$ to train a logistic regression model. In Project B1, we implemented the function `words_in_texts`, which creates numeric features derived from the email text and uses those features for logistic regression. 
# 
# For this project, we have provided you with an implemented version of `words_in_texts`. Remember that the function outputs a 2-dimensional `NumPy` array containing one row for each email text. The row should contain a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. 
# 
# Run the following cell to see how the function works on some text.

# In[86]:


from projB2_utils import words_in_texts

words_in_texts(['hello', 'bye', 'world'], pd.Series(['hello', 'hello worldhello']))


# ## EDA and Basic Classification
# 
# In Project B1, we proceeded to visualize the frequency of different words for both spam and ham emails and used `words_in_texts(words, train['email'])` to directly to train a classifier. We also provided a simple set of 5 words that might be useful as features to distinguish spam/ham emails. 
# 
# We then built a model using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier from `sklearn`.
# 
# Run the following cell to see the performance of a simple model using these words and the `train` `DataFrame`.

# In[87]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train['email'])
Y_train = np.array(train['spam'])

X_train[:5], Y_train[:5]


# In[88]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)

training_accuracy = model.score(X_train, Y_train)
print("Training Accuracy: ", training_accuracy)


# ## Evaluating Classifiers

# In our models, we are evaluating the accuracy of the training set, which may provide a misleading accuracy measure. In Project B1, we calculated various metrics to lead us to consider more ways of evaluating a classifier, in addition to overall accuracy. Below is a reference to those concepts.
# 
# Presumably, our classifier will be used for **filtering**, i.e., preventing messages labeled `spam` from reaching someone's inbox. There are two kinds of errors we can make:
# - **False positive (FP)**: A ham email gets flagged as spam and filtered out of the inbox.
# - **False negative (FN)**: A spam email gets mislabeled as ham and ends up in the inbox.
# 
# To be clear, we label spam emails as 1 and ham emails as 0. These definitions depend both on the true labels and the predicted labels. False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier in addition to overall accuracy:
# 
# **Precision**: Measures the proportion of emails flagged as spam that are actually spam. Mathematically, $\frac{\text{TP}}{\text{TP} + \text{FP}}$.
# 
# **Recall**: Measures the proportion  of spam emails that were correctly flagged as spam. Mathematically, $\frac{\text{TP}}{\text{TP} + \text{FN}}$.
# 
# **False positive rate**: Measures the proportion  of ham emails that were incorrectly flagged as spam. Mathematically, $\frac{\text{FP}}{\text{FP} + \text{TN}}$.
# 
# The below graphic (modified slightly from [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)) may help you understand precision and recall visually:<br />
# <center>
# <img alt="precision_recall" src="precision_recall.png" width="600px;" />
# </center>
# 
# Note that a True Positive (TP) is a spam email that is classified as spam, and a True Negative (TN) is a ham email that is classified as ham.

# <br/>
# <hr style="border: 1px solid #fdb515;" />
# 
# # Moving Forward - Building Your Own Model
# 
# With this in mind, it is now your task to make the spam filter more accurate. To get full credit on the accuracy part of this assignment, you must get at least **85%** accuracy on both the train and test set (see Q4 for the partial credit breakdown). To see your accuracy on the test set, you will use your classifier to predict every email in the `test` `DataFrame` and upload your predictions to Gradescope.
# 
# **Gradescope limits to 3 submissions per day to meet the threshold**. You will be able to see your accuracy on the entire test set when submitting to Gradescope. In the case that you are approved for an extension, you are granted 3 more submissions for each day the deadline has been extended.
# 
# Here are some ideas for improving your model:
# 
# 1. Finding better features based on the email text. Some example features are:
#     1. Number of characters in the subject/body
#     1. Number of words in the subject/body
#     1. Use of punctuation (e.g., how many '!'s were there?)
#     1. Number/percentage of capital letters 
#     1. Whether the email is a reply to an earlier email or a forwarded email
# 1. Finding better (and/or more) words to use as features. Which words are the best at distinguishing emails? This requires digging into the email text itself. 
# 1. Better data processing. For example, many emails contain HTML as well as text. You can consider extracting the text from the HTML to help you find better words. Or, you can match HTML tags themselves, or even some combination of the two.
# 1. Model selection. You can adjust the parameters of your model (e.g. the penalty type, the regularization parameter, or any arguments in `LogisticRegression`) to achieve higher accuracy. Recall that you should use cross-validation to do feature and model selection properly! Otherwise, you will likely overfit to your training data.
#     1. We have imported `GridSearchCV` for you. You may use sklearn's `GridSearchCV` ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)) class to perform cross-validation. You do not need to code your own cross-validation from scratch, though you are welcome to do so.
# 
# Here's an example of how to use `GridSearchCV`. Suppose we wanted to experiment with 3 different solvers (numerical methods for optimizing the mode) models for a `LogisticRegression` model `lr_model`. 
# 1. We could define a dictionary specifying the hyperparameters and the specific values we want to try out like so: `parameters = {'solver':[{'lbfgs', 'liblinear', 'newton-cg', 'saga']}`.
# 2. Running `grid = GridSearchCV(estimator=lr_model, param_grid=parameters)` would give us a model for each combination of hyperparameters we are testing - in this case, just 4 models.
# 3. We fit each model to some training data `X_train` and `Y_train` using `grid_result = grid.fit(X_train, Y_train)`.
# 4. Indexing into `grid_result.cv_results_` with a particular metric (in this case, `mean_test_score`), we get an array with the scores corresponding to each of the models. `grid_result.cv_results_['mean_test_score']`.
# Feel free to experiment with other hyperparameters and metrics as well, the documentation is your friend!     
#        
# You may use whatever method you prefer in order to create features, but **you may only use the packages we've imported for you in the cell below or earlier in this notebook**. In addition, **you are only allowed to train logistic regression models**. No decision trees, random forests, k-nearest-neighbors, neural nets, etc.
# 
# We have not provided any code to do this, so feel free to create as many cells as you need in order to tackle this task. However, answering questions 1, 2, and 3 should help guide you.
# 
# **Note:** You may want to use your **validation data** to evaluate your model and get a better sense of how it will perform on the test set. However, you may overfit your validation set if you try to optimize your validation accuracy too much. Alternatively, you can perform cross-validation on the entire training set.

# In[89]:


# import libraries
# you may use any of these to create your features
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import GridSearchCV
import re
from collections import Counter


# In[90]:


# Define your processing function, processed data, and model here. 
# You may find it helpful to look through the rest of the questions first!
def top_diff_words(train_data, text_column='email', target_column='spam', top_n=40):
    spam_emails = train_data[train_data[target_column] == 1][text_column]
    ham_emails = train_data[train_data[target_column] == 0][text_column]
    
    spam_words = Counter(" ".join(spam_emails).split())
    ham_words = Counter(" ".join(ham_emails).split())
    
    total_spam_words = sum(spam_words.values())
    total_ham_words = sum(ham_words.values())
    
    spam_df = pd.DataFrame(list(spam_words.items()), columns=['word', 'spam_count'])
    ham_df = pd.DataFrame(list(ham_words.items()), columns=['word', 'ham_count'])

    spam_df['spam_freq'] = spam_df['spam_count'] / spam_df['spam_count'].sum()
    ham_df['ham_freq'] = ham_df['ham_count'] / ham_df['ham_count'].sum()

    merged_df = pd.merge(spam_df[['word', 'spam_freq']], ham_df[['word', 'ham_freq']], on='word', how='outer').fillna(0)

    merged_df['freq_diff'] = abs(merged_df['spam_freq'] - merged_df['ham_freq'])

    # Sort by the difference and get top N words
    top_words = merged_df.sort_values(by='freq_diff', ascending=False).head(top_n)['word']
    print(top_words)
    return top_words.tolist()

significant_words = top_diff_words(train, 'email', 'spam', 40)

# Number of chracters in the body of email
X_train = words_in_texts(significant_words, train['email'])
Y_train = train['spam'].values
X_train = words_in_texts(significant_words, train['email'])


my_model = LogisticRegression()
my_model.fit(X_train, Y_train)
Y_predictions = my_model.predict(X_train)

training_accuracy = (Y_train == Y_predictions).sum() / len(Y_train)
print("Training Accuracy: ", training_accuracy)

X_train_val = words_in_texts(significant_words, val['email'])
Y_train_val = val['spam'].values
X_train_val = words_in_texts(significant_words, val['email'])
my_model_new = LogisticRegression()
my_model_new.fit(X_train_val, Y_train_val)
Y_predictions_new = my_model_new.predict(X_train_val)

validation_accuracy = (Y_train_val == Y_predictions_new).sum() / len(Y_train_val)
print("Validation Accuracy: ", validation_accuracy)


# <!-- BEGIN QUESTION -->
# 
# <br>
# 
# ---
# 
# ## Question 1
# 
# In the following cell, describe the process of improving your model. You should use at least 2-3 sentences each to address the following questions:
# 
# 1. How did you find better features for your model?
# 2. What did you try that worked or didn't work?
# 3. What was surprising in your search for good features?
# 

# Initially, I was just trying to features for my model by using logic in the sense of thinking about whether it makes sense to compare the content of spam vs ham emails by the character/word count, whether it was a reply email or not and I did those calculations and appended them to the training table that I then used to fit to my logistic regression model and made changes/additions depending on whether the associated trainig accuracy went up or down. Therefore, if a feature improved the training accuracy of the model, I decided to keep it and remove it otherwise. Then, initially I was just using the words that I had used from the last project to determine how to differentiate between spam and ham emails but then I realized that it could cause big problems with overfitting. Therefore, I decided to define a function that called top_diff_words that took all the labeled spam and ham data from the training set and then I joined all those emails together and split them by a space and then found the counts of all those words. After that, I took the sum of the total number of spam words and ham words and stored them in their respective variables. After that, I defined two dataframes that stored the words and their respective counts for spam and ham respectively which I then merged together into one dataframe. From there, I defined a column that calculated the difference in the frequency of these 2 columns and sorted the frequency difference order by descending order to get the most important words that had the highest difference in frequency for spam and ham emails. I used the top 40 words and this increased my accuracy so much that I didn't even end up using some of the other features I had initally such as character count and if an email was a reply or not. What was surprising in my search for some of these features was that some of these words were just symbols like >, =, or <!-- as it never came in my thought process when I was trying to think of words or things that could differentiate these emails. I realized that things like these and punctuation and structure of these emails has a big impact on your classification of whether an email is spam or not. 

# <!-- END QUESTION -->
# 
# <br/>
# <hr style="border: 1px solid #fdb515;" />
# 
# # Exploratory Data Analysis
# 
# In the cell below, show a visualization you used to select features for your model. 
# 
# Please include:
# 
# 1. A plot showing something meaningful about the data that helped you during feature selection, model selection, or both.
# 2. Two or three sentences describing what you plotted and its implications with respect to your features.
# 
# Feel free to create as many plots as you want in your feature selection process, but select only one for the response cell below.
# 
# **You should not just produce an identical visualization to Question 3 in Project B1.** For this section, we’d like you to go beyond the analysis you performed in Project B1. Choose some plot other than the 1-dimensional distribution of some quantity for spam and ham emails. In particular, do not produce a bar plot of proportions like you created in Question 3 of Project B1. Any other plot is acceptable, **as long as it comes with thoughtful commentary.** Here are some ideas:
# 
# 1. Consider the correlation between multiple features (look up correlation plots and `sns.heatmap` ([documentation](https://seaborn.pydata.org/generated/seaborn.heatmap.html)). 
# 1. Try to show redundancy in a group of features (e.g., `body` and `html` might co-occur relatively frequently, or you might be able to design a feature that captures all HTML tags and compares them to these). 
# 1. Visualize which words have high or low values for helpful statistics.
# 1. Visually depict whether spam emails tend to be wordier (in some sense) than ham emails.

# <!-- BEGIN QUESTION -->
# 
# <br>
# 
# ---
# 
# ## Question 2a
# 
# Generate your visualization in the cell below.
# 

# In[91]:


most_significant_words = top_diff_words(train, 'email', 'spam', 5)
X_train_spec = words_in_texts(most_significant_words, train['email'])
correlation_matrix = pd.DataFrame(X_train_spec, columns=most_significant_words).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Top Words in Spam Detection")
plt.show()


# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# <br>
# 
# ---
# 
# ## Question 2b
# 
# Write your commentary in the cell below.
# 

# This is a correlation heatmap of some of the top words that are used as features in detection of whether an email is spam or not. Here we can see the correlation between some of these words to determine if they are used in a specific type of email (exhibit positive correlation) or not (exhibit negative correlation). I could use these learnings and see that if some things are very highly correlated, then I can clean some of the features I am using by taking them out to avoid redundancy. However, by seeing this visualization, I see that there isn't a single one word here specifically that has a very correlation with another word which is why I decided to keep all these words as features for my classifier in building the logistic regression model.

# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# <br>
# 
# ---
# 
# ## Question 3: ROC Curve
# 
# In most cases, we won't be able to get 0 false positives and 0 false negatives, so we have to compromise. For example, in the case of cancer screenings, false negatives are comparatively worse than false positives — a false negative means that a patient might not discover that they have cancer until it's too late. In contrast, a patient can receive another screening for a false positive.
# 
# Recall that logistic regression calculates the probability that an example belongs to a particular class. To classify an example, we say that an email is spam if our classifier gives it $\ge 0.5$ probability of being spam. However, **we can adjust that cutoff threshold**: We can say that an email is spam only if our classifier gives it $\ge 0.7$ probability of being spam, for example. This is how we can trade off false positives and false negatives.
# 
# The Receiver Operating Characteristic (ROC) curve shows this trade-off for each possible cutoff probability. In the cell below, plot an ROC curve for your final classifier (the one you use to make predictions for Gradescope) on the training data. Refer to Lecture 23 to see how to plot an ROC curve.
# 
# **Hint**: You'll want to use the `.predict_proba` method for your classifier instead of `.predict` to get probabilities instead of binary predictions.
# 

# In[92]:


y_probabilities = my_model.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_train, y_probabilities)
plt.plot(fpr, tpr)
plt.xlim([0, 1])
plt.ylim([0,1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()


# <!-- END QUESTION -->
# 
# <br/>
# <hr style="border: 1px solid #fdb515;" />
# 
# # Question 4
# 
# ### Grading Scheme
# 
# Your grade for Question 4 will be based on your model’s accuracy when making predictions on the training set and your model’s accuracy when making predictions on the test set. The tables below provide scoring guidelines. If your accuracy lies in a particular range, you will receive the number of points associated with that range.
# **Important**: While your training accuracy can be checked at any time in this notebook, your test accuracy can only be checked by submitting your model’s predictions to Gradescope. **You may only submit to Gradescope 3 times per day to meet the threshold**. In the case that you are approved for an extension, you are granted 3 more submissions for each day the deadline has been extended. Plan ahead to make sure you have enough time to fine-tune your model! The thresholds are as follows:
# 
# Points | 5 | 3 | 1.5 | 0
# --- | --- | --- | --- | ---
# **Training** Accuracy | 85% and Above | \[80, 85) | \[70, 80) | Below 70%
# 
# Points | 10 | 6 | 3 | 0
# --- | --- | --- | --- | ---
# **Testing** Accuracy | 85% and Above | \[80, 85) | \[70, 80) | Below 70%
# 
# 
# <br><br>
# 
# ---
# 
# ## Question 4a: Train Predictions
# Assign your predictions for the class of each data point in the training set `train` to the variable `train_predictions`.

# In[93]:


train_predictions = Y_predictions

# print your training accuracy 
training_accuracy = np.mean(train_predictions == train["spam"])
training_accuracy


# In[94]:


grader.check("q4a")


# <br>
# 
# ---
# 
# ## Question 4b: Test Set Predictions
# 
# The following code will write your predictions on the test dataset to a CSV file. **You will need to submit this file to the "Project B2 Test Set Predictions" assignment on Gradescope to get credit for this question.**
# 
# Assign your predictions for the class of each data point in the test set `test` to a 1-dimensional array called `test_predictions`. **Please make sure you save your predictions to `test_predictions`, as this is how part of your score for this question will be determined.**
# 
# **Remember that if you've performed transformations or featurization on the training data, you must also perform the same transformations on the test data in order to make predictions.** For example, if you've created features for the words "drug" and "money" on the training data, you must also extract the same features in order to use scikit-learn's `.predict(...)` method.
# 
# **Gradescope limits you to 3 submissions per day to meet the threshold.** In the case that you are approved for an extension, you are granted 3 more submissions for each day the deadline has been extended.
# 
# The provided tests check that your predictions are in the correct format but are worth 0 points in the *Project B2 Coding assignment*. You must additionally submit to the *Project B2 Test Set Predictions* assignment to evaluate your classifier accuracy.

# In[95]:


X_test = words_in_texts(significant_words, test['email'])
test_predictions = my_model.predict(X_test)


# In[96]:


grader.check("q4b")


# The following cell generates a CSV file with your predictions. **You must submit this CSV file to the "Project B2 Test Predictions" assignment on Gradescope to get credit for this question.** You can only submit to Gradescope a maximum of 3 times per day, so please use your submissions wisely!

# In[97]:


# Assuming that your predictions on the test set are stored in a 1-dimensional array called
# test_predictions. Feel free to modify this cell as long you create a CSV in the right format.

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": test['id'], 
    "Class": test_predictions,
}, columns=['Id', 'Class'])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = "submission_{}.csv".format(timestamp)
submission_df.to_csv(filename, index=False)

print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
display(HTML("Download your test prediction <a href='" + filename + "' download>here</a>."))
print('You may now upload this CSV file to Gradescope for scoring.')#


# <br/><br/>
# <hr style="border: 5px solid #003262;" />
# <hr style="border: 1px solid #fdb515;" />
# 
# ## Mila congratulates you for finishing Project B2!
# 
# ![](mila.jpeg)
# 
# Below, you will see two cells. Running the first cell will automatically generate a PDF of all questions that need to be manually graded, and running the second cell will automatically generate a zip with your autograded answers. **You are responsible for both the coding portion (the zip from Project B2) and the written portion (the PDF with from Project B2) to their respective Gradescope portals and checking that they are the most recent copy or the copy you wish to submit (including plots and all written answers).** The coding proportion should be submitted to Project B2 Coding as a single zip file, and the written portion should be submitted to Project B2 Written as a single pdf file. When submitting the written portion, please ensure you select pages appropriately. In addition, you must submit your test prediction in Q4b to **Project B2 Test Set Predictions** for the corresponding points. 
# 
# If there are issues with automatically generating the PDF in the first cell, you can try downloading the notebook as a PDF by clicking on `File -> Save and Export Notebook As... -> PDF`. If that doesn't work either, you can manually take screenshots of your answers to the manually graded questions and submit those. Either way, **you are responsible for ensuring your submission follows our requirements, we will NOT be granting regrade requests for submissions that don't follow instructions.**

# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**
# 
# After you have run the cell below and generated the zip file, you can open the PDF <a href='projB2.pdf' download>here</a>.

# In[98]:


# Save your notebook first, then run this cell to export your submission.
grader.export(run_tests=True)


#  

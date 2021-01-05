# import libraries
import pandas as pd
import re
import numpy as np

from sqlalchemy import create_engine

# nltk library and modules 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
from nltk import pos_tag 

# sci-kit modules
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# pickle
import pickle

# loading NLTK modules if needed

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# connecting the engine to data
engine =  create_engine('sqlite:///crisis_messages.db').connect() 

# load data from database
df = pd.read_sql_table('crisis_messages', engine) 

X = df['message']
Y = df.iloc[:, 4:]


# ### 2. Write a tokenization function to process your text data

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 
stemmer = PorterStemmer() 

def tokenize(text):
    # excluding non-alphabetical and non-numeric characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenizing the text
    words = word_tokenize(text)
    
    # convert to lowercase 
    tokenized_words = [i.lower() for i in words]
    
    # lemmatizing and excluding stop words
    lemmatized_words = [lemmatizer.lemmatize(i) for i in tokenized_words if i not in stop_words]
    
    # tagging the words to find verbs
    tagged_words = pos_tag(lemmatized_words)
    
    # stemming verbs
    items_number = range(len(tagged_words))
    
    for i in items_number:

        tag = tagged_words[i][1]
        word = tagged_words[i][0]
        
        # finding verbs identified by pos_tage and 
        # finding words that end with "ing"
        if (tag == 'VB' or word.endswith('ing')):
            tagged_words[i] = stemmer.stem(word)
        # keeping rest of the the words as they are   
        else:
            tagged_words[i] = word
            
    tokenized_sentence = ' '.join(tagged_words)

    return tokenized_sentence


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

df['message_tokenized'] = df['message'].apply(tokenize)

X = df['message_tokenized']
y = df.iloc[:, 4:40]

pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('moc', MultiOutputClassifier(RandomForestClassifier(), n_jobs=None))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.2, random_state=42)

# train classifier
pipeline.fit(X_train, y_train)

# # predict on test data
y_pred = pipeline.predict(X_test)

# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

def display_results(y_test, y_pred):
    n = 0
    
    for label in y_test.columns:
        
        pred = np.transpose(y_pred)[n]
        test = y_test.iloc[:, n]
        
        print (label)
        print (classification_report(pred, test))
#         print (accuracy_score(test, pred))
        n += 1  

display_results(y_test, y_pred) 


# ### 6. Improve your model
# Use grid search to find better parameters. 

pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
#     ('clf', RandomForestClassifier()),
    # using multioutputclassifier to enables mapping to multiple outputs
    ('moc', MultiOutputClassifier(RandomForestClassifier(), n_jobs=None))
])

parameters = {'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2')
             }

cv =  GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

display_results(y_test, y_pred)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    ]

for classifier in classifiers:    
    enhanced_pipelines = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('moc', MultiOutputClassifier(classifier, n_jobs=None))
    ])

    # train classifier
    enhanced_pipelines.fit(X_train, y_train)

    # predict on test data
    y_pred = enhanced_pipelines.predict(X_test)

    display_results(y_test, y_pred)  


# ### 9. Export your model as a pickle file
# save the model to disk
filename = 'disaster_response_classifier.sav'
pickle.dump(enhanced_pipelines, open(filename, 'wb'))
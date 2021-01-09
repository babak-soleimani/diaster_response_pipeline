# import libraries
import pandas as pd
import re
import numpy as np
import pickle
from sqlalchemy import create_engine

# importing nltk library and modules 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
from nltk import pos_tag 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# sci-kit modules
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data (database_filepath='sqlite:///crisis_messages.db',table_name='crisis_messages',column_name='message'):
    """
        Load database and tables
        Args: 
            database_filepath (str): file path of sqlite database
            table_name (str): name of the database table
            column_name (str): column including messages
        Return:
            X (pandas dataframe): Features
            y (pandas dataframe): Targets/ Labels
    """

    # connecting the sql engine to data
    engine =  create_engine(database_filepath).connect() 

    # load data from database
    df = pd.read_sql_table(table_name, engine) 

    X = df[column_name]
    y = df.iloc[:, 4:]

    return X,y

def tokenize(text):
    """
    Returns the message text converted to its root form
    
    Args:
        text(string): message
    Returns:
        lemmatized (list): list of the message texts converted to their root form
    """

    # retreiving the list of stop words
    stop_words = stopwords.words('english')

    # defining lemmatizer and stemmer transformers 
    lemmatizer = WordNetLemmatizer() 
    stemmer = PorterStemmer() 

    # excluding non-alphabetical and non-numeric characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenizing the text
    tokens = word_tokenize(text)
    words = [w for w in tokens if w not in stop_words]
    
    # stemming words
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in stemmed]
    
    return lemmatized

def model_builder():
    """
    Returns the classification model
    
    Args:
        None
    Returns:
        cv: Grid search model object
    """
    # define the step of pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize,ngram_range=(1,2),max_df=0.75)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))])

    # define the parameters to fine tuning
    parameters = {'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2')
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def display_results(y_test, y_pred):
    """
    Displays the results of prediction using the model
    
    Args:
        y_test (pandas dataframe): actual values of the y dataset
        y_pred (pandas datafrane): predicted values using the model
    Returns:
        None
    """    
    n = 0
    for label in y_test.columns:
        pred = np.transpose(y_pred)[n]
        test = y_test.iloc[:, n]
        print(label, classification_report(pred, test))
        n += 1  


def model_trainer(filename = 'disaster_response_classifier.pkl'):
    """
    Trains the model and exports a pickle file
    
    Args:
        filename (str): name of the pickle file to export
    Returns:
        None
    """      
    # loading the data
    X, y = load_data()

    # splitting the data into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=0.3, random_state=42)

    # building the model
    model = model_builder()

    # fitting the model using the training data
    model.fit(X_train, y_train)

    # displaying results
    y_pred = model.predict(X_test)
    display_results(y_test, y_pred) 

    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))

model_trainer()
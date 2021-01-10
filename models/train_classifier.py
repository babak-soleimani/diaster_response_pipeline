# import libraries
import pandas as pd
import re
import numpy as np
import pickle
from sqlalchemy import create_engine
import sys

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

def load_data (database_filepath):
    """
        Load database and tables
        Args: 
            database_filepath (str): file path of sqlite database
            table_name (str): name of the database table
        Return:
            X (pandas dataframe): Features
            y (pandas dataframe): Targets/ Labels
    """

    # connecting the sql engine to data
    data_file = 'sqlite:///' +  database_filepath + '.db'
    engine =  create_engine(data_file).connect() 

    # load data from database
    df = pd.read_sql_table(database_filepath, engine) 

    X = df['message']
    y = df.iloc[:, 4:]

    return X,y, y.columns

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

def build_model():
    """
    Returns the classification model
    
    Args:
        None
    Returns:
        cv: Grid search model object
    """
    # define the step of pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))])

    # define the parameters to fine tuning
    parameters = {'vect__max_df': (0.5, 0.75, 1.0)
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2')
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model (model, X_test, Y_test, category_names):
    """
    Displays the results of prediction using the model
    
    Args:
        y_test (pandas dataframe): actual values of the y dataset
        y_pred (pandas datafrane): predicted values using the model
    Returns:
        None
    """  
    y_pred = model.predict(X_test)
    
    n = 0
    for label in category_names:
        pred = np.transpose(y_pred)[n]
        test = Y_test.iloc[:, n]
        print(label, classification_report(pred, test))
        n += 1  


def save_model(model, model_filepath):
    """
    Takes the model and the export file path and exports a pickle file
    
    Args:
        model: classifier model
        model_filepath (str): file path to save the pickle file
    Returns:
        None
    """      
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Performs model building, training, evaluating, and saving
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        load_data(database_filepath)
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
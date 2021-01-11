# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath='disaster_messages.csv', categories_filepath='disaster_categories.csv'):
    """ 
    load, read and merge distaster messages and categories
    
    Args:
        message_filepath(string): the file path of messages.csv
        categories_filepath(string): the file path of categories.csv
    
    Return:
        df(pandas Dataframe): merged dataframe created from the two input files
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, left_on = 'id', right_on = 'id')

    return df

def clean_data (df):
    """
    cleans up the data by organizing categories, removing duplicates and missing data
    Args:
        df (pandas Dataframe): dataframe resulted from merging database
    
    Return:
        df (pandas Dataframe): dataframe prepared for classification

    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    cleaner = lambda a : a [:-2]
    category_colnames = [cleaner(i) for i in row]

    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast="integer")
        
        # converting to binary values
        categories[column] = categories[column].replace(2,1)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace= True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    df[df.duplicated() == True].shape[0]

    # drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename='crisis_messages.db',table_name='crisis_messages'):
    """
    Writing the cleaned dataframe to a SQL database
    
    Args:
        df(Dataframe): the dataframe ready to be exported for modelling
        database_filename(string): the file path to save file .db
    Return:
        None
    """

    # remove the file if it already exists
    dbpath = 'sqlite:///'+ database_filename
    table = table_name
    engine = create_engine(dbpath)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    command = "DROP TABLE IF EXISTS {};".format(table)
    cursor.execute(command)
    connection.commit()
    cursor.close() 
    
    # create the table in the sql file
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

main()

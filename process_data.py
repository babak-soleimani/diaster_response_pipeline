# import libraries
import pandas as pd
from sqlalchemy import create_engine

# prompting the user to enter file paths
dataset_path = input("Enter your file path of the datasets: ") 

# prompting the user to enter database path
dataset_path = input("Enter your file path of the datasets: ") 

# load messages dataset
messages = pd.read_csv(dataset_path + 'messages.csv')

# load categories dataset
categories = pd.read_csv(dataset_path + 'categories.csv')

# merge datasets
df = messages.merge(categories, left_on = 'id', right_on = 'id')

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

# drop the original categories column from `df`
df.drop(columns=['categories'], inplace= True)

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)

# check number of duplicates
df[df.duplicated() == True].shape[0]

# drop duplicates
df = df.drop_duplicates()

# writing the results to a database
engine = create_engine('sqlite:///crisis_messages.db')
df.to_sql(dataset_path + 'crisis_messages', engine, index=False)


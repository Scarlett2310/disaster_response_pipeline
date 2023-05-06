import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge the messages and categories data"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """Clean and transform the merged data"""
    # split the 'categories' column
    categories = df['categories'].str.split(pat=';', expand=True)
    # New name of categories column
    category_colnames = categories.iloc[0].apply(lambda x: x[:-2]).tolist()
    # rename aboved columns
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from str to int
        categories[column] = categories[column].astype(int)
    # drop the original 'categories' column from 'df'
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # drop 'child_alone' as redundant
    df = df.drop('child_alone', axis=1)
    # replace 'related' values of 2 into 1
    df['related'] = df['related'].replace(2, 1)
    return df

def save_data(df, database_filename):
    """Save the cleaned data to an SQLite database"""
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Response_disaster_table', engine, index=False, if_exists='replace')
 
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
              'Response_disaster.db')


if __name__ == '__main__':
    main()
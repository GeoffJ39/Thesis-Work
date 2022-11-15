from asyncore import read
import numpy as np
import pandas as pd
import os
import sys

def read_c(filepath, total_pandas):
    df = pd.read_csv(filepath)
    uniq = df.name.unique()
    total_pandas = pd.concat([total_pandas, df], axis=0, ignore_index=True)
    for col in df:
        if col not in ['start time', 'end time', 'participant id']:
            print(col)
            #print(df[col].unique())
    return uniq, df

def split_by_game(df, unique):

    #create a data frame dictionary to store data frames
    DataFrameDict = {elem : pd.DataFrame() for elem in unique}

    for key in DataFrameDict.keys():
        DataFrameDict[key] = df[:][df.name == key]
    
    return DataFrameDict

if __name__ == "__main__":
    directory = 'C:\\Users\geoff\Documents\GitHub\Thesis-Work\Re__introduction_and_request_for_data'

    org_stdout = sys.stdout
    # iterate over files in that directory
    unique_games = []
    total_pandas = pd.DataFrame()
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if f[-4:] == '.csv':
                with open(f + '.txt', 'w') as w:
                    sys.stdout = w
                    uniq, df_read = read_c(f, total_pandas)
                    total_pandas = pd.concat([total_pandas, df_read], axis=0, ignore_index=True)
                    unique_games.append(uniq)
                    sys.stdout = org_stdout

    unique_games = np.unique(np.concatenate(unique_games, axis = 0))
    Data_dict = split_by_game(total_pandas, unique_games)

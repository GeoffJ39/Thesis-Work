from asyncore import read
import numpy
import pandas as pd
import os
import sys

def read_c(filepath):
    df = pd.read_csv(filepath)
    for col in df:
        print(col)
        print(df[col].unique())
    
directory = 'C:\\Users\geoff\Documents\GitHub\Thesis-Work\Re__introduction_and_request_for_data'

org_stdout = sys.stdout
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f + '.txt', 'w') as w:
            sys.stdout = w
            read_c(f)
            sys.stdout = org_stdout

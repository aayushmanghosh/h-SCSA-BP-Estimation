'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.0
''' 

import pandas as pd
import os
import FileBrowser

# Choose params.txt containing the column title for each of the collected feature 
# (N = 21) including sbp and dbp values
param_file = FileBrowser.uigetfile()             

with open(param_file) as params:
    columns = params.read()
    columns = columns.split()
    index = list(range(len(columns)))
    col_names = dict(zip(index, columns))

# Choose the data files, which are collected as an output of the respective MATLAB code
files = FileBrowser.uigetfile()                  
absolute_path = os.path.dirname(os.path.abspath('__file__'))
relative_path = r'\FeatureFiles'
folder = os.path.join(absolute_path, relative_path) # Subject to change based on the type of experiment we are conducting 
df_list = []

for file in files:
    filename = os.path.join(folder, file)
    df = pd.read_csv(filename, index_col=False, header=None)
    df_list.append(df)

# General rule for naming convention: (Dataset_name)-(Number-of-features)-(Type, eg. Noisy/Normal/PCA/ICA/SCSA)
expt = r'\MIMIC-II-21-Normal.csv' # Subject to change depending on the experiment (user specific)    
big_df = pd.concat(df_list)
big_df = big_df[big_df[0]!=0]
big_df.rename(columns=col_names, inplace=True)
csvfile = os.path.join(absolute_path, expt)
big_df.to_csv(csvfile, index=False)
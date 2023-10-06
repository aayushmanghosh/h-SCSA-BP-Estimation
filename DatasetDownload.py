'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.0
''' 

import wfdb
import pandas as pd
import FileBrowser

# Choose records.txt containing the title for each of the collected dataset (corresponding to each patient) 
# Depending on the number of records present in the record.txt (the code should iterate that many time)
record_file = FileBrowser.uigetfile()  

with open(record_file) as record:
    index = record.readlines()

record.close()
l = []
for i in range(len(index)):
    l.append(index[i][:-2])

temp = []
temp.append('')  
var = list(range(1,51))
for i in var:
    if i <= 9:
        temp.append('_000'+str(i))
    else:
        temp.append('_00' + str(i))

main_sig = ['II','PLETH','ABP']
final_list = []
flag = 4000 # Change according to the number of files you want to download and the index you want to download
        
for k in l[4001:5001]: # Change according to the number of files you want to download and the index you want to download
    flag += 1
    print('index of file under iteration:',flag)
    for i in temp:
        file_no = k+i
        print('iteration no {}'.format(file_no))
        try:
            signal,fields = wfdb.rdsamp('{}'.format(file_no),sampfrom=25000,sampto = 125000, pn_dir='mimic3wdb/34/{}'.format(k))
        except:
            continue
        
        if set(fields['sig_name']).intersection(set(main_sig)) == set(main_sig):
            print('Successful {}'.format(file_no))
            II_index = fields['sig_name'].index('II')
            PLETH_index = fields['sig_name'].index('PLETH')
            ABP_index = fields['sig_name'].index('ABP')
            df = pd.DataFrame(signal)
            df = df[[II_index,PLETH_index,ABP_index]]
            df.to_csv('D:\\Physionet_Dataset\\34_sayan_3\\{}.csv'.format(file_no)) # Change according to the directory you want to store
            final_list.append(file_no) 

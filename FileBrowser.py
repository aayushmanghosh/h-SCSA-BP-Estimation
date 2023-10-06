'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.0
''' 

# Importing the necessary libraries and modules. This is a simple GUI to load files that are to be used by the RL Algorithms
from tkinter import Tk, Button, Label, StringVar
from tkinter import filedialog
import os

def uigetfile():
    def getfile():
        files = filedialog.askopenfilenames(initialdir = os.getcwd(),
                                              title = "Select a File",
                                              filetypes = (("csv files",
                                                            "*.csv"),
                                                            ("Text files",
                                                             "*.txt*")))
        label_file_explorer.configure(text="No. of Files Selected: "+str(len(files)))
        myStrVar.set(list(files))

    window = Tk()
    window.title('File Selection')
    window.geometry("350x150")
    window.config(background = "white")
    label_file_explorer = Label(window,
                                text = "File Explorer",
                                width = 50, height = 4,
                                fg = "blue")
    
    button_explore = Button(window,
                            text = "Browse Files",
                            command = getfile)
      
    button_exit = Button(window,
                         text = "Use Selected Files",
                         command = window.destroy)
    
    label_file_explorer.grid(column = 4, row = 1)
    
    global myStrVar
    myStrVar = StringVar()
    button_explore.grid(column = 4, row = 3) 
    button_exit.grid(column = 4,row = 5)
    window.mainloop()
    filenames = myStrVar.get()
    file_nums = len(filenames[1:-1].split(','))
    last_elem = filenames[1:-1].split(',')[-1]
    
    if last_elem == '':
        file_list = filenames[1:-1].split(',')[0].strip().strip('\'')
    else:
        file_list = []
        
        for i in range(file_nums):
            file_list.append(filenames[1:-1].split(',')[i].strip().strip('\''))
        
    return file_list
import numpy as np

class C100Dataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = None  # X (data) of training set.
    tr_y = None  # Y (label) of training set.
    ts_x = None # X (data) of test set.
    ts_y = None # Y (label) of test set.

    def __init__(self, filename):
        ## read the csv for dataset (cifar100.csv, cifar100_lt.csv or cifar100_nl.csv), 
        # 
        # Format:
        #   image file path,classname
        
        ### TODO: Read the csv file and make the training and testing set
        ### TODO: assign each dataset
        datax, datay = np.loadtxt(filename, unpack = True, max_rows=28723, delimiter=',', dtype='str')
        self.tr_x = np.array([i for i in datax if '/train/' in i])
        self.tr_y = np.array([datay[i] for i,x in enumerate(datax) if '/train/' in x])
        
        
        datax = np.loadtxt(filename, unpack = True, skiprows=28723, delimiter=',', dtype='str')
        self.ts_x = np.array([i for i in datax if '/test/' in i])

    def getDataset(self):
        return [self.tr_x, self.tr_y, self.ts_x]
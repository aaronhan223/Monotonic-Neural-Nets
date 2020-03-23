import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_f1(n=1000,
                sigma=0.3):
    X = np.linspace(0.1,4,num=n)
    epsilon = np.random.normal(loc=0.0,scale=sigma,size=X.shape)
    return X, .5 * np.log(X) + epsilon
   
def generate_f2(n=1000,
                sigma=0.3):
    X = np.linspace(-2,2,num=n)
    epsilon = np.random.normal(loc=0.0,scale=sigma,size=X.shape)
    return X, X**2 + epsilon

def generate_f3(n=1000,
                sigma=0.3):
    X = np.linspace(-2,2,num=n)
    epsilon = np.random.normal(loc=0.0,scale=sigma,size=X.shape)
    return X, X*(X-1.5)*(X+1.5) + epsilon

def generate_f4(n=1000,
                sigma=0.3):
    X = np.linspace(-2,2,num=n)
    epsilon = np.random.normal(loc=0.0,scale=sigma,size=X.shape)
    return X, 2*np.sin(np.pi * X) + np.pi * X / 2 + epsilon

def load_data(fn=0):
    functions = [generate_f1,generate_f2,generate_f3,generate_f4]
    X,y = functions[fn]()
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    out_df = pd.DataFrame(
        {'X':X, 'y':y} 
    )
    
    n_train = int(out_df.shape[0] * 0.8)

    output = {
        'data_train': out_df[:n_train].values,
        'data_test':  out_df.values, # we want to see across for viz
        'monotonicity': [1],
        'X_cols': [0],
        'Y_col': 1
    }
    
    return output

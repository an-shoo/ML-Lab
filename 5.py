import pandas as pd
import numpy as np

# 1. Load the playtennis.csv dataset
data = pd.read_csv('playtennis.csv')
concepts = np.array(data.iloc[:,:-1])
target = np.array(data.iloc[:,-1])

def learn(concepts,target):
    specific_h = concepts[0].copy()
    general_h = [['?' for i in range(len(specific_h))] for i in range(len(specific_h))]
    for i,h in enumerate(concepts):
        if target[i] == 'yes':
            print("Positive example")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        else:
            print("Negative example")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        print("Step {}".format(i+1))
        print(specific_h)
        print(general_h)
    
    general_h = [h for h in general_h if h != ['?' for _ in range(len(specific_h))]]
    return specific_h, general_h
final_s, final_g = learn(concept,target)
print(final_s)
print(final_g)

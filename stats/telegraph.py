import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('out.dat',sep='\t')
plt.hist(df['x2'],bins=10)
plt.show()

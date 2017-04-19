import matplotlib.pyplot as plt
import pandas as pd
import sys
import matplotlib
import os

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)

# The inital set of baby names and bith rates
names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
births = [968, 155, 77, 578, 973]

BabyDataSet = list(zip(names, births))

path = 'births1880.csv'

df = pd.DataFrame(data=BabyDataSet,
                  columns=['Names', 'Births'])

df.to_csv(path,
          index=False,
          header=False)

df = pd.read_csv(path, names=['Names', 'Births'])

os.remove(path)

Sorted = df.sort_values(['Births'], ascending=False)
print("MAximum of births: " + str(df['Births'].max()))

# Create graph
df['Births'].plot()

# Maximum value in the data set
MaxValue = df['Births'].max()
# Name associated with the maximum value
MaxName = df['Names'][df['Births'] == df['Births'].max()].values
# Text to display on graph
Text = str(MaxValue) + " - " + MaxName
# Add text to graph
plt.annotate(Text, xy=(1, MaxValue), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

plt.show()

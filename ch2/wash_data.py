import pandas as pd




df = pd.read_csv('C:/Users/danie/Documents/spyder/creditcard.csv')

df_fraud = df.iloc[:, 30]
df_amount = df.iloc[:, 29] * df.iloc[:, 30]

print(df_amount.mean())
print(sum(df_amount))
print(sum(df_fraud))
for i in df:
    print(df[i].mean())

for i in df:
    print(df[i].std())
    
    
for i in df:
    print(df[i].skew())

for i in df:
    print(df[i].min())

for i in df:
    print(df[i].max())




from src import DataFactor
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

DF = DataFactor(db_path='db')
df = DF.getriskdb()


df['Totalcost'] = df['Totalcost'].fillna(0)
df['M_ClaimCount'] = df['M_ClaimCount'].astype(float)
df['sev'] =  np.where(df['M_ClaimCount'] > 0, df['Totalcost']/df['M_ClaimCount'], 0)
totalcost = list(np.log(df[(df['Totalcost'] > 0)]['Totalcost']))
severity = list(np.log(df[(df['sev'] > 0)]['sev']))

dist_total_cost = sns.displot(totalcost, kde=True, rug=False)
plt.savefig('dist_total_cost.png')


dist_severity =  sns.displot(severity, kde=True, rug=False)
plt.savefig('dist_severity.png')
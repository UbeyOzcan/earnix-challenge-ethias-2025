from src import Factory
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:

    df = DF.replace_empty_cost(dataframe=df)

    df = DF.replace_mode_mean(dataframe=df)
    df.to_csv('db/clean_train.csv', sep=";", index=False)
    return df


def sev_dist(df: pd.DataFrame, name:str) -> None:

    df['M_ClaimCount'] = df['M_ClaimCount'].astype(float)
    df['ln_cost'] = np.where(df['Totalcost'] > 0, np.log(df['Totalcost']), 0)
    df = df[df['ln_cost'] > 0]
    dist_severity =  sns.displot(df, x='ln_cost', kde=True, rug=False)
    plt.savefig(f'dist_ln_cost_{name}.png')
    return None

def prediction_dist(df:pd.DataFrame) -> None:
    df = df[df['Actual'] > 0]
    actual = [np.where(df['Actual'] > 0, np.log(df['Actual']), 0)]
    pred = [np.where(df['Pred'] > 0, np.log(df['Pred']), 0)]

    actual_pred  = [a/b for a,b in zip(actual,pred)]

    sns.displot(actual, kde=True, rug=False)
    plt.savefig(f'dist_actual.png')
    sns.displot(pred, kde=True, rug=False)
    plt.savefig(f'dist_predicted.png')
    sns.displot(actual_pred, kde=True, rug=False)
    plt.savefig(f'dist_actual_predicted.png')
    return None

do_clean = False
do_dist = False
do_dist_pred = False
calc_mse = True
if __name__ == '__main__':
    DF = Factory(db_path='db')
    if calc_mse:
        df = DF.getriskdb()
        df_dict = DF.dist_separator(dataframe=df, sep=600)
        mse = DF.mse_calc(df=df_dict['dist_1'])
        print(mse)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class Factory:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def getriskdb(self) -> pd.DataFrame:
        return pd.read_csv(f'{self.db_path}/Train Data.csv', sep=";", decimal=',')

    def getriskdb_clean(self) -> pd.DataFrame:
        return pd.read_csv(f'{self.db_path}/clean_train.csv', sep=";", decimal=',')

    def getgeodb(self) -> pd.DataFrame:
        return pd.read_csv(f'{self.db_path}/Geo Enrichment.csv')

    def remove_cols(self, dataframe: pd.DataFrame, colname: str) -> pd.DataFrame:
        return dataframe[dataframe.columns.drop(list(dataframe.filter(like=colname)))]

    def replace_empty_cost(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['Totalcost'] =  dataframe['Totalcost'].fillna(0)
        return dataframe

    def replace_mode_mean(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for i in dataframe.columns:
            if dataframe[i].dtype == object or dataframe[i].dtype == int:
                dataframe[i] = dataframe[i].fillna(dataframe[i].mode()[0])
            else :
                dataframe[i] = dataframe[i].fillna(dataframe[i].mean())

        return dataframe

    def dist_separator(self, dataframe:pd.DataFrame, sep:int) -> dict:

        dataframe['M_ClaimCount'] = dataframe['M_ClaimCount'].astype(float)
        dataframe = dataframe[dataframe['M_ClaimCount'] < 2]
        n_claims = dataframe['M_ClaimCount'].sum()
        df_dist_1 = dataframe.copy()
        df_dist_2 = dataframe.copy()
        del dataframe

        df_dist_1['Totalcost'] = np.where(df_dist_1['Totalcost'] <= sep, df_dist_1['Totalcost'] , 0)
        df_dist_1['M_ClaimCount'] = np.where((df_dist_1['Totalcost'] <= sep) & (df_dist_1['Totalcost'] > 0), 1, 0)

        df_dist_2['Totalcost'] = np.where(df_dist_2['Totalcost'] > sep, df_dist_2['Totalcost'], 0)
        df_dist_2['M_ClaimCount'] = np.where(df_dist_2['Totalcost'] > sep, 1, 0)
        print(df_dist_1['M_ClaimCount'].sum() + df_dist_2['M_ClaimCount'].sum() == int(n_claims))
        #df_dist_1.to_csv(f'{self.db_path}/Train_Lower.csv', sep = ";", decimal = ",", index=False)
        #df_dist_2.to_csv(f'{self.db_path}/Train_Upper.csv', sep = ";", decimal = ",", index=False)

        df_dict = {'dist_1': df_dist_1, 'dist_2': df_dist_2}
        return df_dict

    def get_pred_obs(self) -> pd.DataFrame:
        return pd.read_csv(f'{self.db_path}/DataPredictions_GBM_Tweedie_full.csv', sep=",", decimal='.')

    def mse_calc(self, df: pd.DataFrame) -> list:

        df.insert(len(df.columns), "Pred", 278.071)
        df.insert(len(df.columns), "Pred_1", 0)
        actual = [df['Totalcost']]
        pred = [df['Pred']]
        pred_1 = [df['Pred_1']]
        mse = mean_squared_error(actual, pred)
        mse_1 = mean_squared_error(actual, pred_1)
        return [mse, mse_1]
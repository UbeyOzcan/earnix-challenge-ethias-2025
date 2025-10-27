import pandas as pd
import numpy as np


class DataFactor:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def getriskdb(self) -> pd.DataFrame:
        return pd.read_csv(f'{self.db_path}/Train Data.csv', sep=";", decimal=',')

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
import pandas as pd

class DataFactor:
    def __init__(self, db_path:str):
        self.db_path = db_path

    def getriskdb(self):
        return pd.read_csv(f'{self.db_path}/Train Data.csv', sep=";", decimal=',')

    def getgeodb(self):
        return pd.read_csv(f'{self.db_path}/Geo Enrichment.csv')


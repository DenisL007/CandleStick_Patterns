import datetime as dt

import pandas_datareader.data as web


class ImportDataFrame():
    # evaluating time range of our data
    start = dt.datetime(2000, 1, 1)
    year = None
    end = dt.datetime.today()

    #TODO - add index column to dataframe

    def __init__(self, stock_id):
        self.stock_name = stock_id
        # create a DATA FRAME  with values from google finance  from year 2000 till today
        self.il = web.DataReader(self.stock_name, 'google', self.start, self.end)

    # this function returns data_frame with data from last year only
    def get_last_year(self):
        print(self.il.last_valid_index())

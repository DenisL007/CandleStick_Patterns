import datetime as dt

import pandas_datareader.data as web


class ImportDataFrame():
    # evaluating time range of our data
    start = dt.datetime( 2015, 1, 1)
    year = None
    end = dt.datetime.today()

    def __init__(self, stock_id):
        self.stock_name = stock_id
        # create a DATA FRAME  with values from google finance  from year 2000 till today
        self.il = web.DataReader(self.stock_name, 'google', self.start, self.end)
        # adding Index column evaluated bu row number , starting from 0
        #self.il = self.il.assign( Index= [i for i in  range(len(self.il))])
        #self.il.set_index('Index', inplace=True)
     #   self.il = self.il[['Index', 'Open', 'High', 'Low', 'Close',  'Volume']]
    # this function returns data_frame with data from last year only
    def get_last_year(self):
        print(self.il.last_valid_index())

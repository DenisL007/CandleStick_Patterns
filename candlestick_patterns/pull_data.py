import datetime as dt
import pandas_datareader.data as web
import pandas_datareader as pdr

class ImportDataFrameGoogle():
    # evaluating time range of our data
    start = dt.datetime(2000, 1, 1)
    year = None
    end = dt.datetime.today()

    def __init__(self, stock_id):
        self.stock_name = stock_id
        # create a DATA FRAME  with values from google finance  from year 2000 till today
        self.il = web.DataReader(self.stock_name, 'google', self.start, self.end)
        # adding Index column evaluated bu row number , starting from 0
        #self.il = self.il.assign( Index= [i for i in  range(len(self.il))])
        #self.il.set_index('Index', inplace=True)
        #self.il = self.il[['Index', 'Open', 'High', 'Low', 'Close',  'Volume']]
        #this function returns data_frame with data from last year only

    def importDataFrameGoogle_Test(NONE):
        dframe = ImportDataFrameGoogle("INTC")
        print(dframe.il.head())

#uncomment to run test ImportDataFrameGoogle("AAPL").importDataFrameGoogle_Test()

class ImportDataFrameYahoo():

    def __init__(self, stock__id):
        self.stock_name = stock__id
        self.il = pdr.get_data_yahoo( stock__id )

    def importDataFrameYahoo_Test(NONE):
        dframe = ImportDataFrameYahoo("INTC")
        print(dframe.il.head())
#uncomment to run test ImportDataFrameYahoo("AAPL").importDataFrameYahoo_Test()
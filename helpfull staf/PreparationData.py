import datetime as dt
import pandas_datareader.data as web
from matplotlib import style
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
import plotly.plotly as py
import plotly.graph_objs as go

style.use('ggplot')

# evaluating time range of our data
start = dt.datetime(2000, 1, 1)
end = dt.datetime.today()
# get user's input /// ---- here need to add use input verification
# stock_name = input('Enter a valid stock name to work with \n')
# this line with defoult value is for testing only
stock_name = 'INTC'
# create a DATA FRAME  with values from google/finance of APPLE inc. from year 2000 till today
df = web.DataReader(stock_name, 'google', start, end)

#-------------- UNCOMENT THIS SECTION TO WATCH  HOW EVERITHIN WORK'S -----------------------------------------------
# creates CSV file with all imported data from /google.com
df.to_csv('C:/Users/dleshko/Desktop/CandleStick_Patterns/intc.csv')
# we can read the CSV file with our preferences
# here is documentation for this  function
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
#< example > data = pd.read_csv('C:/TestProgram/data.csv' ,parse_dates = True ,index_col = 0 )
#now we can read CSV file into DATA_FRAME variable
dframe = pd.read_csv('C:/Users/dleshko/Desktop/CandleStick_Patterns/intc.csv' )
# not necessary create csv file, with dframe = df we can work with data directly
# NOW WE CAN USE VARIABLE 'DFRAME' TO MANIPULATE THE STOCK DATA
#< example > we can add and evaluate aditional column named 'Average' to data frame
dframe ['Average'] = 0
# print the first  five tuples or last ten
print(dframe.head())
print(dframe.tail(10))
#-------------- THIS SECTION WAS COMMENTED FOR FASTER COMPILATION OF THE PROGRAM -----------------------------------

# <editor-fold desc="Here some various graphs that we can draw">
#once we have CSV file read it is faster than download all data from web
dframe = pd.read_csv('C:/Users/dleshko/Desktop/CandleStick_Patterns/intc.csv')

#--------------------- REMOVE COMENTS TO VIEW PLOT EXAMPLES---------------------------------------------------------
# this is how pandas is creating a graph
dframe.plot()
plt.show()
# plot single or multiplie element(colums) of data frame
print ( dframe[['High','Low','Close']].tail(100) )
dframe[['High','Low','Close']].tail(100).plot()
plt.show()
# </editor-fold>
#--------------------- REMOVE COMENTS TO VIEW PLOT EXAMPLES---------------------------------------------------------

# <editor-fold desc="some manipulations that we can make on the data">
# add new column Mooving average of 100 elements by 'Close' column
dframe['100 M.Avg by Close'] = dframe['Close'].rolling(window = 100).mean()
#the is some 'not anvaibale' data first 100 lines bocouse how the MA calculated ,so just drop them if we need
dframe.dropna(inplace = True)
# or we just can use this line of code intead ( " min_periods = 0" added )
#dframe['100 M.Avg by Close'] = dframe['Close'].rolling(window = 100 , min_periods = 0).mean()
print (dframe.tail(5))
# </editor-fold>

#"----- DENIS OT TAKOI KRASOTI MOI KOMP HEREET I ZAVISAET NA PARU MINUT CHTOB VISRAT' ETOT GRAFIK -------------------
# or we can grab values from pandas and work with them with pure MatPlotLib library
ax1 = plt.subplot2grid((6,1) , (0,0) , rowspan=5 , colspan=1)
ax2= plt.subplot2grid((6,1) , (5,0) , rowspan=1 , colspan=1 , sharex=ax1)
ax1.plot(dframe.index, dframe['Close'])
ax1.plot(dframe.index, dframe['100 M.Avg by Close'])
ax2.bar(dframe.index, dframe['Volume'])
plt.show()
#--------------------- REMOVE COMENTS TO VIEW PLOT EXAMPLES---------------------------------------------------------




#----- CANDLE GRPH'S ---- P_O_K_A V R_A_R_O_B_O_T_K_E-----------------------------------------------------------------------
# candlestick graph needs a dates colums as dates to work with them so index is no needed -- >>>
df = pd.read_csv('C:/Users/dleshko/Desktop/CandleStick_Patterns/intc.csv' ,parse_dates = True ,index_col = 0 )
df_ohlc = df['Close'].resample('10D' ).ohlc()
df_volume = df['Volume'].resample('10D').sum()

print(df_ohlc)

ax1 = plt.subplot2grid((6,1) , (0,0) , rowspan=5 , colspan=1)
ax2= plt.subplot2grid((6,1) , (5,0) , rowspan=1 , colspan=1 , sharex=ax1)


plt.show()
#--------------------- REMOVE COMENTS TO VIEW PLOT EXAMPLES---------------------------------------------------------
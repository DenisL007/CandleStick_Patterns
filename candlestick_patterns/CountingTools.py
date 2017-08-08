#!/usr/bin/python3
from pandas import Series, read_csv, Series, DataFrame, value_counts
from scipy import stats
from numpy import arange,linspace
import matplotlib.pyplot as plt


class Tools_cl(object):


    def __init__(self,data_frame,ma_period=1,lr_period=2):
        self.ma_period = ma_period
        self.lr_period = lr_period
        self.hammer_trend_slope = 0.087
        self.df = data_frame
        #self.moving_average(200)
        self.one_ma_cross_trend_detection()
       #self.candle_size_analysis()
        #self.define_long_candlestick()
        #self.define_short_candlestick()
        #print(self.issuer_list)
        # self.ma_linear_regression()
        # self.umbrella_candle()
        # self.hammer()
        # self.hanging_man()
        #self.issuer_list.to_csv('issuer_list.csv')

    def one_ma_cross_trend_detection(self,period=200):
#        ma_name = 'MA_{period}'.format(period=period)
#        if(ma_name in self.df):
#            pass
#        else:
#            self.moving_average(period)
        self.df['One_MA'] = Series.rolling(self.df['Close'], window=period, min_periods=period).mean()
        self.df['O_ma_CTD'] = 0
        for index in self.df.index:
            if self.df.Close[index] > self.df.One_MA[index]:
                self.df.O_ma_CTD = 1
            if self.df.Close[index] < self.df.One_MA[index]:
                self.df.O_ma_CTD = -1











    def price_level_analysis(self):#how much cs at each price level
        pass

    def candle_size_analysis(self):
        self.df['Size'] = abs((self.df.Open - self.df.Close).round(decimals=2))  #count candle size (need to round - pandas have a bug)
        minSize = min(self.df.Size)
        maxSize = max(self.df.Size)
        totalSize = len(self.df.Size)
        hist_data = self.df.Size.value_counts()  # prepare histogram data
        hist_data = hist_data.sort_index()
        df = hist_data.to_frame().reset_index()  # pass Series to DF
        df.columns=['Size','Count']
        df['% of total']= ((df.Count/totalSize)*float(100)).round(decimals=3)  # calculate % for each size from total
        return df

    def define_long_candlestick(self,long_cs_period = 5):
        self.df['Long_CS'] = ''
        avrSize = Series.rolling(self.df.Size, window=long_cs_period, min_periods=long_cs_period).mean()  # count average SIZE for period
        for index in range(len(avrSize)):       # if current cs_size greater that avr Size add "LCS" to dataframe
            if(self.df.Size[index] > 1.3*avrSize[index]):
                self.df.set_value(index,'Long_CS','LCS')

    def define_short_candlestick(self,short_cs_period = 15):
        self.df['Short_CS'] = ''
        avrSize = Series.rolling(self.df.Size, window=short_cs_period, min_periods=short_cs_period).mean()  # count average SIZE for period
        for index in range(len(avrSize)):
            if(self.df.Size[index] < 0.51*avrSize[index]):# if current cs_size less that avr Size add "SCS" to dataframe
                self.df.set_value(index,'Short_CS','SCS')

    def moving_average(self,period=1):
        name = 'MA_{period}'.format(period=period)
        self.df[name] = Series.rolling(self.df['Close'], window=period, min_periods=period).mean()

    def ma_linear_regression(self):
        y = arange(float(self.lr_period))
        for index in range((self.ma_period - 1), len(self.df) - self.lr_period + 1,
                           1):  # from end of issuer_list
            for index2 in range(self.lr_period): #prepare array len(lr_period)
                y[index2] = self.df['MA'][index+index2]
            slope, intercept, r_value, p_value, std_err = stats.linregress(arange(self.lr_period), y)
            self.df.set_value([index + self.lr_period - 1], 'LR', slope)

    def umbrella_candle (self):
        u_shadow_size_parameter = 0.2
        body_size_parameter = 0.1
        for index in range(len(self.df)):
            Open = self.df['Open'][index]
            High = self.df['High'][index]
            Low = self.df['Low'][index]
            Close = self.df['Close'][index]
            if (Close > Open): #white candle
                Candle = High - Low
                Body = Close - Open
                U_Shadow = High - Close
                L_Shadow = Open - Low
                if ((Body > 0) & (L_Shadow >= float(2) * Body) & (U_Shadow <= u_shadow_size_parameter * Candle) & (Body >= body_size_parameter * Candle)): # parameters of umbrella candle
                    self.df.set_value([index], 'Umbrella', 'True')
                else:
                    self.df.set_value([index], 'Umbrella', 'False')
            else: # same for black candle
                Candle = High - Low
                Body = Open - Close
                U_Shadow = High - Open
                L_Shadow = Close - Low
                if ((Body > 0) & (L_Shadow >= float(2) * Body) & (U_Shadow <= u_shadow_size_parameter * Candle) & (Body >= body_size_parameter * Candle)):
                    self.df.set_value([index], 'Umbrella', 'True')
                else:
                    self.df.set_value([index], 'Umbrella', 'False')

    def average_slope(self):
        pass

    def local_min(self,price='Close',period=5):

        pass

    def local_max(self):
        pass

    def gap_finder(selfself):
        pass

    def hammer(self):
        self.df['Hammer'] = 'False'
        umbrella_index = self.df.loc[(self.df.Umbrella == 'True')].index.tolist()
        for index in umbrella_index:
            if (index >= self.ma_period + self.lr_period - 2):
                if((self.df.Low[index-1] >= self.df.Low[index]) & (self.df.Low[index+1] >= self.df.Low[index])):
                    if (self.df.LR[index] < 0):
                        self.df.set_value([index+1], 'Hammer', 'HA')
                        self.df.set_value([index], 'Hammer', 'H')
                        self.df.set_value([index-1], 'Hammer', 'HB')

    def hanging_man(self):
        self.df['Hanging Man'] = 'False'
        umbrella_index = self.df.loc[(self.df.Umbrella == 'True')].index.tolist()
        for index in umbrella_index:
            if (index >= self.ma_period + self.lr_period - 2):
                if ((self.df.High[index - 1] <= self.df.High[index]) & (
                    self.df.High[index + 1] <= self.df.High[index])):
                    if (self.df.LR[index] > 0):
                        self.df.set_value([index + 1], 'Hanging Man', 'HMA')
                        self.df.set_value([index], 'Hanging Man', 'HM')
                        self.df.set_value([index - 1], 'Hanging Man', 'HMB')

    def support_resistance_level_analysis(self):
        pass

    def min_price_analysis(self):
        pass

    def max_price_analysis(self):
        pass

    def p_ma_d_trend_analysis(self):
        pass

    def lr_trend_analysis(self):
        pass

    def ma_d_trend_analysis(self):
        pass


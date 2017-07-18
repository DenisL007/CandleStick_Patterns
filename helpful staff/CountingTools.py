#!/usr/bin/python3
from pandas import Series, read_csv, Series, DataFrame, value_counts
from scipy import stats
from numpy import arange,linspace
import matplotlib.pyplot as plt


class Tools_cl(object):


    def __init__(self,file,ma_period=1,lr_period=2):
        self.file = file
        self.ma_period = ma_period
        self.lr_period = lr_period
        self.hammer_trend_slope = 0.087
        self.issuer_list = read_csv(self.file)
        print(self.issuer_list)
        self.candle_size_analysis()
        self.define_long_candlestick()
        self.define_short_candlestick()
        #print(self.issuer_list)
        # self.ma_linear_regression()
        # self.umbrella_candle()
        # self.hammer()
        # self.hanging_man()
        #self.issuer_list.to_csv('issuer_list.csv')

#how much cs at each price level
    def price_level_analysis(self):
        pass

    def candle_size_analysis(self):
        self.issuer_list['Size'] = abs((self.issuer_list.Open - self.issuer_list.Close).round(decimals=2))  #count candle size (need to round - pandas have a bug)
        minSize = min(self.issuer_list.Size)
        maxSize = max(self.issuer_list.Size)
        totalSize = len(self.issuer_list.Size)
        hist_data = self.issuer_list.Size.value_counts()  # prepare histogram data
        hist_data = hist_data.sort_index()
        df = hist_data.to_frame().reset_index()  # pass Series to DF
        df.columns=['Size','Count']
        df['% of total']= ((df.Count/totalSize)*float(100)).round(decimals=3)  # calculate % for each size from total
        return df

    def define_long_candlestick(self,long_cs_period = 5):
        self.issuer_list['Long_CS'] = ''
        avrSize = Series.rolling(self.issuer_list.Size, window=long_cs_period, min_periods=long_cs_period).mean()  # count average SIZE for period
        for index in range(len(avrSize)):       # if current cs_size greater that avr Size add "LCS" to dataframe
            if(self.issuer_list.Size[index] > 1.3*avrSize[index]):
                self.issuer_list.set_value(index,'Long_CS','LCS')
        #print(self.issuer_list.Long_CS.value_counts())

    def define_short_candlestick(self,short_cs_period = 15):
        self.issuer_list['Short_CS'] = ''
        avrSize = Series.rolling(self.issuer_list.Size, window=short_cs_period, min_periods=short_cs_period).mean()  # count average SIZE for period
        for index in range(len(avrSize)):
            if(self.issuer_list.Size[index] < 0.51*avrSize[index]):# if current cs_size less that avr Size add "SCS" to dataframe
                self.issuer_list.set_value(index,'Short_CS','SCS')

    def moving_average(self):
        self.issuer_list['MA'] = Series.rolling(self.issuer_list['Close'], window=self.ma_period, min_periods=self.ma_period).mean()

    def ma_linear_regression(self):
        y = arange(float(self.lr_period))
        for index in range((self.ma_period - 1), len(self.issuer_list) - self.lr_period + 1,
                           1):  # from end of issuer_list
            for index2 in range(self.lr_period): #prepare array len(lr_period)
                y[index2] = self.issuer_list['MA'][index+index2]
            slope, intercept, r_value, p_value, std_err = stats.linregress(arange(self.lr_period), y)
            self.issuer_list.set_value([index + self.lr_period - 1], 'LR', slope)

    def umbrella_candle (self):
        u_shadow_size_parameter = 0.2
        body_size_parameter = 0.1
        for index in range(len(self.issuer_list)):
            Open = self.issuer_list['Open'][index]
            High = self.issuer_list['High'][index]
            Low = self.issuer_list['Low'][index]
            Close = self.issuer_list['Close'][index]
            if (Close > Open): #white candle
                Candle = High - Low
                Body = Close - Open
                U_Shadow = High - Close
                L_Shadow = Open - Low
                if ((Body > 0) & (L_Shadow >= float(2) * Body) & (U_Shadow <= u_shadow_size_parameter * Candle) & (Body >= body_size_parameter * Candle)): # parameters of umbrella candle
                    self.issuer_list.set_value([index], 'Umbrella', 'True')
                else:
                    self.issuer_list.set_value([index], 'Umbrella', 'False')
            else: # same for black candle
                Candle = High - Low
                Body = Open - Close
                U_Shadow = High - Open
                L_Shadow = Close - Low
                if ((Body > 0) & (L_Shadow >= float(2) * Body) & (U_Shadow <= u_shadow_size_parameter * Candle) & (Body >= body_size_parameter * Candle)):
                    self.issuer_list.set_value([index], 'Umbrella', 'True')
                else:
                    self.issuer_list.set_value([index], 'Umbrella', 'False')

    def average_slope(self):
        pass

    def local_min(self,price='Close',period=5):

        pass

    def local_max(self):
        pass

    def gap_finder(selfself):
        pass

    def hammer(self):
        self.issuer_list['Hammer'] = 'False'
        umbrella_index = self.issuer_list.loc[(self.issuer_list.Umbrella == 'True')].index.tolist()
        for index in umbrella_index:
            if (index >= self.ma_period + self.lr_period - 2):
                if((self.issuer_list.Low[index-1] >= self.issuer_list.Low[index]) & (self.issuer_list.Low[index+1] >= self.issuer_list.Low[index])):
                    if (self.issuer_list.LR[index] < 0):
                        self.issuer_list.set_value([index+1], 'Hammer', 'HA')
                        self.issuer_list.set_value([index], 'Hammer', 'H')
                        self.issuer_list.set_value([index-1], 'Hammer', 'HB')

    def hanging_man(self):
        self.issuer_list['Hanging Man'] = 'False'
        umbrella_index = self.issuer_list.loc[(self.issuer_list.Umbrella == 'True')].index.tolist()
        for index in umbrella_index:
            if (index >= self.ma_period + self.lr_period - 2):
                if ((self.issuer_list.High[index - 1] <= self.issuer_list.High[index]) & (
                    self.issuer_list.High[index + 1] <= self.issuer_list.High[index])):
                    if (self.issuer_list.LR[index] > 0):
                        self.issuer_list.set_value([index + 1], 'Hanging Man', 'HMA')
                        self.issuer_list.set_value([index], 'Hanging Man', 'HM')
                        self.issuer_list.set_value([index - 1], 'Hanging Man', 'HMB')

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


def main():

    a = Tools_cl('intc.csv')
    a.ma_period = 10


if __name__ == "__main__": main()

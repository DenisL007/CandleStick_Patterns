#!/usr/bin/python3
from pandas import Series, read_csv, Series, DataFrame, value_counts
from scipy import stats
from numpy import arange,linspace, where,diff,concatenate,savetxt,array,all,zeros
import matplotlib.pyplot as plt
from math import pi
from bokeh.plotting import figure, show, output_file

class Tools_cl(object):


    def __init__(self,data_frame,ma_period=1,lr_period=2):
        self.ma_period = ma_period
        self.lr_period = lr_period
        self.hammer_trend_slope = 0.087
        self.df = data_frame
        self.df_plot()
        #self.moving_average(200)
        #self.one_ma_cross_trend_detection()
        #self.tree_ma_cross_trend_detection()
        #self.candle_size_analysis()
        #self.define_long_candlestick()
        #self.define_short_candlestick()
        #print(self.issuer_list)
        # self.ma_linear_regression()
        # self.umbrella_candle()
        # self.hammer()
        # self.hanging_man()
        print(self.df)
        self.df.to_csv('df.csv')

    def df_plot(self):
        mids = (self.df.Open + self.df.Close) / 2
        spans = abs(self.df.Close - self.df.Open)

        inc = self.df.Close > self.df.Open
        dec = self.df.Open > self.df.Close
        w = 12 * 60 * 60 * 1000  # half day in ms
        output_file("candlestick.html", title="candlestick.py example")

        TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

        p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1800, plot_height=1100, toolbar_location="right")

        p.segment(self.df.index, self.df.High, self.df.index, self.df.Low, color="black")
        p.rect(self.df.index[inc], mids[inc], w, spans[inc], fill_color="#D5E1DD", line_color="black")
        p.rect(self.df.index[dec], mids[dec], w, spans[dec], fill_color="#F2583E", line_color="black")

        p.title.text = "Intel Candlestick"
        #p.xaxis.major_label_orientation = pi /4
        p.grid.grid_line_alpha = 0.7

        show(p)  # open a browser

    def one_ma_cross_trend_detection(self,source='Close',period=200,min_days_trend=2):#omcrtd  / min_days_trend - minimum days with specific trend direction to count as trend change
        self.df['omcrtd_ma'] = Series.rolling(self.df[source], window=period, min_periods=period).mean() # can be changed to EMA
        #self.df['omcrtd_ma'] = self.df['Close'].ewm(span=200,min_periods=200).mean()  # can be changed to EMA
        self.df['omcrtd_trend'] = where(self.df[source] > self.df['omcrtd_ma'], 1, 0)
        self.df['omcrtd_trend'] = where(self.df[source] < self.df['omcrtd_ma'], -1, self.df['omcrtd_trend'])
        #find trend change point
        omcrtd_trend = self.df['omcrtd_trend']
        bounds = (diff(omcrtd_trend) != 0) & (omcrtd_trend[1:] != 0)
        bounds = concatenate(([omcrtd_trend[0] != 0], bounds))
        self.df['omcrtd_bounds'] = bounds
        bounds_idx = where(bounds)[0]
        relevant_bounds_idx = array([idx for idx in bounds_idx if all(omcrtd_trend[idx] == omcrtd_trend[idx:idx + min_days_trend])])
        # Make sure start and end are included
        if relevant_bounds_idx[0] != 0:
            relevant_bounds_idx = concatenate(([0], relevant_bounds_idx))
        if relevant_bounds_idx[-1] != len(omcrtd_trend) - 1:
            relevant_bounds_idx = concatenate((relevant_bounds_idx, [len(omcrtd_trend) - 1]))
        # write data to dataframe
        temp_array = zeros(len(self.df),dtype=bool)
        for index in range (len(relevant_bounds_idx)):
            temp_array[relevant_bounds_idx[index]] = 'True'
        self.df['omcrtd_relevant_bounds'] = temp_array
        # Iterate segments + plot
        #for start_idx, end_idx in zip(relevant_bounds_idx[:-1], relevant_bounds_idx[1:]):
        #    # Slice segment
        #    segment = market_data.iloc[start_idx:end_idx + 1, :]
        #    x = np.array(mdates.date2num(segment.index.to_pydatetime()))
        #    # Plot data
        #    data_color = 'green' if signal[start_idx] > 0 else 'red'
        #    plt.plot(segment.index, segment['Close'], color=data_color)
        #    # Plot fit
        #    coef, intercept = np.polyfit(x, segment['Close'], 1)
        #    fit_val = coef * x + intercept
        #    fit_color = 'yellow' if coef > 0 else 'blue'
        #    plt.plot(segment.index, fit_val, color=fit_color)

    def tree_ma_cross_trend_detection(self,source='Close',p_s=20,p_m=50,p_l=200,min_days_trend=2): #min_days_trend - minimum days with specific trend direction to count as trend change
        self.df['tmcrtd_ma_l'] = Series.rolling(self.df[source], window=p_l, min_periods=p_l).mean() # can be changed to EMA
        self.df['tmcrtd_ma_m'] = Series.rolling(self.df[source], window=p_m, min_periods=p_m).mean() # can be changed to EMA
        self.df['tmcrtd_ma_s'] = Series.rolling(self.df[source], window=p_s, min_periods=p_s).mean() # can be changed to EMA
        self.df['tmcrtd_trend'] = where( ((self.df['tmcrtd_ma_s'] > self.df['tmcrtd_ma_m']) & (self.df['tmcrtd_ma_m'] > self.df['tmcrtd_ma_l'])),1,0)
        self.df['tmcrtd_trend'] = where( ((self.df['tmcrtd_ma_s'] < self.df['tmcrtd_ma_m']) & (self.df['tmcrtd_ma_m'] < self.df['tmcrtd_ma_l'])),-1,self.df['tmcrtd_trend'])
        tmcrtd_trend = self.df['tmcrtd_trend']
        #find trend change point
        bounds = (diff(tmcrtd_trend) != 0) & (tmcrtd_trend[1:] != 0)
        bounds = concatenate(([tmcrtd_trend[0] != 0], bounds))
        self.df['tmcrtd_bounds'] = bounds
        bounds_idx = where(bounds)[0]
        relevant_bounds_idx = array(
            [idx for idx in bounds_idx if all(tmcrtd_trend[idx] == tmcrtd_trend[idx:idx + min_days_trend])])
        # Make sure start and end are included
        if relevant_bounds_idx[0] != 0:
            relevant_bounds_idx = concatenate(([0], relevant_bounds_idx))
        if relevant_bounds_idx[-1] != len(tmcrtd_trend) - 1:
            relevant_bounds_idx = concatenate((relevant_bounds_idx, [len(tmcrtd_trend) - 1]))
        # write data to dataframe
        temp_array = zeros(len(self.df), dtype=bool)
        for index in range(len(relevant_bounds_idx)):
            temp_array[relevant_bounds_idx[index]] = 'True'
        self.df['tmcrtd_relevant_bounds'] = temp_array

    def zigzig_trend_detection(self,source='Close'):
        pass



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


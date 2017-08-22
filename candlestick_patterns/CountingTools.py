#!/usr/bin/python3
from pandas import Series, read_csv, Series, DataFrame, value_counts,concat
from scipy import stats
from numpy import arange,linspace, where,diff,concatenate,savetxt,array,all,zeros,column_stack,hstack,asarray,isfinite,interp,nan
import matplotlib.pyplot as plt


class Tools_cl(object):


    def __init__(self,data_frame,ma_period=1,lr_period=2):
        self.ma_period = ma_period
        self.lr_period = lr_period
        self.hammer_trend_slope = 0.087
        self.df = data_frame
        self.moving_average(200)
        self.one_ma_cross_trend_detection(period=20)
        self.df.omcrtd_plot=self.df['omcrtd_plot'].interpolate()
        self.plot_candles(pricing=self.df,title='Intc',technicals=[self.df.omcrtd_plot])
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


    def plot_candles(sefl,pricing, title=None, volume_bars=False, color_function=None, technicals=None):
        """
        Args:
          pricing: A pandas dataframe with columns ['open_price', 'close_price', 'high', 'low', 'volume']
          title: An optional title for the chart
          volume_bars: If True, plots volume bars
          color_function: A function which, given a row index and price series, returns a candle color.
          technicals: A list of additional data series to add to the chart.  Must be the same length as pricing.

          technicals:
          SMA = talib.SMA(last_hour['close_price'].as_matrix())
          plot_candles(last_hour, title='1 minute candles + SMA', technicals=[SMA])

          color_function:
          def highlight_dojis(index, open_price, close_price, low, high):
          return 'm' if dojis[index] else 'k'
          plot_candles(last_hour,title='Doji patterns highlighted',color_function=highlight_dojis)
        """

        def default_color(index, Open, Close, Low, High):
            return 'r' if Open[index] > Close[index] else 'g'

        color_function = color_function or default_color
        technicals = technicals or []
        open_price = pricing['Open']
        close_price = pricing['Close']
        low = pricing['Low']
        high = pricing['High']
        oc_min = concat([open_price, close_price], axis=1).min(axis=1)
        oc_max = concat([open_price, close_price], axis=1).max(axis=1)

        if volume_bars:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1)
        if title:
            ax1.set_title(title)
        x = arange(len(pricing))
        candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
        candles = ax1.bar(x, oc_max - oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
        lines = ax1.vlines(x + 0.0, low, high, color=candle_colors, linewidth=1)
        ax1.xaxis.grid(False)
        ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
        # Assume minute frequency if first two bars are in the same day.
        frequency = 'minute' if (pricing.index[1] - pricing.index[0]).days == 0 else 'day'
        time_format = '%d-%m-%Y'
        if frequency == 'minute':
            time_format = '%H:%M'
        # Set X axis tick labels.
        plt.xticks(x, [date.strftime(time_format) for date in pricing.index], rotation='vertical')
        for indicator in technicals:
            ax1.plot(x, indicator)



        if volume_bars:
            volume = pricing['Volume']
            volume_scale = None
            scaled_volume = volume
            if volume.max() > 1000000:
                volume_scale = 'M'
                scaled_volume = volume / 1000000
            elif volume.max() > 1000:
                volume_scale = 'K'
                scaled_volume = volume / 1000
            ax2.bar(x, scaled_volume, color=candle_colors)
            volume_title = 'Volume'
            if volume_scale:
                volume_title = 'Volume (%s)' % volume_scale
            ax2.set_title(volume_title)
            ax2.xaxis.grid(False)
        plt.show()

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
        #self.df['omcrtd_plot'] = nan
        self.df.loc[self.df['omcrtd_relevant_bounds'] == True , 'omcrtd_plot'] = self.df.Close
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


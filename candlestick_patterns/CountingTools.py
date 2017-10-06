#!/usr/bin/python3
from pandas import Series, read_csv, Series, DataFrame, value_counts,concat
from scipy import stats
from numpy import arange,linspace, where,diff,concatenate,savetxt,array,all,zeros,column_stack,hstack,asarray,isfinite,interp,nan
import matplotlib.pyplot as plt


class Tools_cl(object):


    def __init__(self,data_frame):
        self.df = data_frame

 #  calculate and plot trend by ma
#        self.one_ma_cross_trend_detection(period=50)
#        self.tree_ma_cross_trend_detection()
#        self.one_ma_cross_volume_trend_detection()
#        self.df['int_omcrtd_plot']=self.df['omcrtd_plot'].interpolate()
#        self.plot_candles(pricing=self.df,title='Intc',o=True)#,technicals=[self.df.int_omcrtd_plot])

 # calculate ZZ
        #self.zigzig_trend_detection(deviation=3,backstep=5,depth=5)
        #self.plot_candles(pricing=self.df, title='Intc', technicals=[self.df['zz_3%'].interpolate()])

        #self.hammer()
        #self.hanging_man()
        self.candle_spec()
        self.umbrella_candle_new()
        self.umbrella_candle()
        #self.bullish_engulfing()
        #self.bearish_engulfing()
        #self.candle_size_analysis()
        #self.define_long_candlestick()
        #self.define_short_candlestick()
        #print(self.issuer_list)
        # self.ma_linear_regression()
        # self.hammer()
        # self.hanging_man()
        print(self.df)
        self.df.to_csv('df.csv')


    def plot_candles(sefl,pricing, title=None, volume_bars=False,o=False,t=False, ot=False,color_function=None, technicals=None):
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

        if (ot):
            fig, (ax1, ax3, ax4) = plt.subplots(3,1, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        elif (o | t):
            fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        elif volume_bars:
            fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
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

        if ot:
            omcrtd = pricing['omcrtd_trend']
            tmcrtd = pricing['tmcrtd_trend']
            ax3.bar(x,omcrtd)
            omcrtd_title = 'omcrtd'
            ax3.set_title(omcrtd_title)
            ax3.xaxis.grid(False)
            ax4.bar(x,tmcrtd)
            tmcrtd_title = 'tmcrtd'
            ax4.set_title(tmcrtd_title)
            ax4.xaxis.grid(False)

        if o:
            omcrtd = pricing['omcrtd_trend']
            ax3.bar(x,omcrtd)
            omcrtd_title = 'omcrtd'
            ax3.set_title(omcrtd_title)
            ax3.xaxis.grid(False)

        if t:
            tmcrtd = pricing['tmcrtd_trend']
            ax3.bar(x,tmcrtd)
            tmcrtd_title = 'tmcrtd'
            ax3.set_title(tmcrtd_title)
            ax3.xaxis.grid(False)

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

    def one_ma_cross_volume_trend_detection(self,source='Volume',period=20,min_days_trend=0):
        self.df['omcvrtd_ma'] = Series.rolling(self.df[source], window=period, min_periods=period).mean() # can be changed to EMA
        #self.df['omcvrtd_ma'] = self.df['Close'].ewm(span=200,min_periods=200).mean()  # can be changed to EMA
        self.df['omcvrtd_trend'] = where(self.df[source] > self.df['omcvrtd_ma'], 1, 0)
        self.df['omcvrtd_trend'] = where(self.df[source] < self.df['omcvrtd_ma'], -1, self.df['omcvrtd_trend'])
        #find trend change point
        omcvrtd_trend = self.df['omcvrtd_trend']
        bounds = (diff(omcvrtd_trend) != 0) & (omcvrtd_trend[1:] != 0)
        bounds = concatenate(([omcvrtd_trend[0] != 0], bounds))
        self.df['omcvrtd_bounds'] = bounds
        bounds_idx = where(bounds)[0]
        relevant_bounds_idx = array([idx for idx in bounds_idx if all(omcvrtd_trend[idx] == omcvrtd_trend[idx:idx + min_days_trend])])
        # Make sure start and end are included
        if relevant_bounds_idx[0] != 0:
            relevant_bounds_idx = concatenate(([0], relevant_bounds_idx))
        if relevant_bounds_idx[-1] != len(omcvrtd_trend) - 1:
            relevant_bounds_idx = concatenate((relevant_bounds_idx, [len(omcvrtd_trend) - 1]))
        # write data to dataframe
        temp_array = zeros(len(self.df),dtype=bool)
        for index in range (len(relevant_bounds_idx)):
            temp_array[relevant_bounds_idx[index]] = 'True'
        self.df['omcvrtd_relevant_bounds'] = temp_array
        #self.df['omcrtd_plot'] = nan
        self.df.loc[self.df['omcvrtd_relevant_bounds'] == True , 'omcvrtd_plot'] = self.df.Volume
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
        # omcrtd  / min_days_trend - minimum days with specific trend direction to count as trend change

    def zigzig(self,deviation=5,backstep=1,depth=1):
        zz_name ='zz_{deviation}%'.format(deviation=deviation)
        temp_array = zeros(len(self.df))
        pl=self.df.Low
        ph=self.df.High
        h,hp,l,lp=ph[0],0,pl[0],0
        el,eh,elp,ehp = pl[0],ph[0],0,0
        ehf,elf=1,1
        dev=deviation/100
        for i in range(len(self.df)):
            if(((ph[i]>=h) | (~ehf & elf)) & ( i-elp>= backstep) & ( i-ehp>= depth)):
                h,hp=ph[i],i
            if(((pl[i]) <= l) | (ehf & ~elf) & (i-ehp>= backstep) & ( i-elp>= depth)):
                l,lp=pl[i],i
            if (((1-dev) * h > pl[i]) & ehf & ( i-hp>= backstep) & ( i-ehp>= depth)):
                ehf,elf=0,1
                eh = h
                ehp=hp
                h, hp, l, lp = ph[i], i,pl[i], i
                temp_array[ehp]=1
            if (((1+dev) * l < ph[i]) & elf & ( i-lp>= backstep) & ( i-elp>= depth)):
                ehf, elf = 1, 0
                el = l
                elp=lp
                h, hp, l, lp = ph[i], i,pl[i], i
                temp_array[elp]=-1
        self.df.loc[temp_array == 1 , zz_name] = self.df.High
        self.df.loc[temp_array == -1, zz_name] = self.df.Low
        self.df.loc[temp_array == 1 , 'zz_extremum'] = 'High'
        self.df.loc[temp_array == -1, 'zz_extremum'] = 'Low'

    def zz_trend_detection(self):
        pass

    def moving_average(self,period=1):
        name = 'MA_{period}'.format(period=period)
        self.df[name] = Series.rolling(self.df['Close'], window=period, min_periods=period).mean()

    def umbrella_candle(self,lower_shadow_greater_body=2.0,upper_shadow_size=0.2):
        self.df.loc[((self.df.Candle_trend != 'Doji') & (self.df.Lower_shadow >= lower_shadow_greater_body * self.df.Body) & (self.df.Upper_shadow <= upper_shadow_size * self.df.Candle)), 'Umbrella'] = 'True'

    def ma_linear_regression(self,lr_period=5,ma_period=20):
        self.moving_average(ma_period)
        ma_name = 'MA_{period1}'.format(period1=ma_period)
        lr_name = 'LR_{period2}'.format(period2=lr_period)
        slopes = zeros(len(self.df),dtype=float)
        y = arange(float(lr_period))
        for index in range((ma_period - 1), len(self.df)-lr_period+1 , 1):  # from end of issuer_list
            for index2 in range(lr_period): #prepare array len(lr_period)
                y[index2] = self.df[ma_name][index+index2]
            slope, intercept, r_value, p_value, std_err = stats.linregress(arange(lr_period), y)
            slopes[index+lr_period-1]=slope
        self.df[lr_name]=slopes

    def candle_spec(self):
        self.df['Body'] = abs((self.df.Close - self.df.Open).round(decimals=2))
        self.df['Candle'] = abs((self.df.High - self.df.Low).round(decimals=2))
        self.df['Candle_trend'] = where(self.df.Open < self.df.Close, 'Bull', 'Bear')
        self.df['Candle_trend'] = where(abs((self.df.Open - self.df.Close).round(decimals=2)) < 0.03, 'Doji', self.df['Candle_trend'])
        self.df['Upper_shadow'] = where(self.df.Open >= self.df.Close,(self.df.High - self.df.Open).round(decimals=2),(self.df.High - self.df.Close).round(decimals=2))
        self.df['Lower_shadow'] = where(self.df.Open >= self.df.Close,(self.df.Close - self.df.Low).round(decimals=2),(self.df.Open - self.df.Low).round(decimals=2))

    def hammer(self,u_u=0.2,lrp=5,lrma=20,confirmation_candle = False):
        self.umbrella_candle(upper_shadow_size=u_u)
        self.ma_linear_regression(lr_period=lrp, ma_period=lrma)
        lr_name = 'LR_{period2}'.format(period2=lrp)
        umbrella_index = self.df.loc[(self.df.Umbrella == 'True')].index.tolist()
        self.df['Hammer_LR'] = None
        for i in umbrella_index:
            local_index=self.df.index.get_loc(i)
            if(local_index > lrp+lrma-2):
                if((self.df.Low[local_index] <= self.df.Low[local_index-1]) ):
                    if(confirmation_candle):
                        if(self.df.Low[local_index] <= self.df.Low[local_index+1]):
                            if(self.df[lr_name][local_index] < 0.0):
                                self.df.set_value(self.df.index[local_index], 'Hammer_LR', 'True')
                    else:
                        if(self.df[lr_name][local_index] < 0.0):
                            self.df.set_value(self.df.index[local_index], 'Hammer_LR','True')

    def hanging_man(self,u_u=0.2,lrp=5,lrma=20,confirmation_candle = False):
        self.umbrella_candle(upper_shadow_size=u_u)
        self.ma_linear_regression(lr_period=lrp, ma_period=lrma)
        lr_name = 'LR_{period2}'.format(period2=lrp)
        umbrella_index = self.df.loc[(self.df.Umbrella == 'True')].index.tolist()
        self.df['Hanging_Man_LR'] = None
        for i in umbrella_index:
            local_index=self.df.index.get_loc(i)
            if(local_index > lrp+lrma-2):
                if((self.df.Open[local_index] > self.df.Open[local_index-1]) & (self.df.Open[local_index] > self.df.Close[local_index-1])):
                    if(confirmation_candle):
                        if((self.df.Close[local_index+1] < self.df.Open[local_index]) & (self.df.Close[local_index+1] < self.df.Close[local_index])):
                            if (self.df[lr_name][local_index] > 0.0):
                                self.df.set_value(self.df.index[local_index], 'Hanging_Man_LR', 'True')
                    else:
                        if(self.df[lr_name][local_index] > 0.0):
                            self.df.set_value(self.df.index[local_index], 'Hanging_Man_LR','True')


###########################################################################################

    def bullish_engulfing(self,lrp=5,lrma=20):
        self.candle_spec()
        self.ma_linear_regression(lr_period=lrp, ma_period=lrma)
        lr_name = 'LR_{period2}'.format(period2=lrp)
        self.df['Bullish_engulfing'] = None
        Bull_indexes = self.df.loc[(self.df.Candle_trend == 'Bull')].index.tolist()
        for bull_index in Bull_indexes:
            i =self.df.index.get_loc(bull_index)
            if(i > lrp+lrma-2):
                if(self.df.Candle_trend[i-1] != 'Bull'):
                    if((self.df.Open[i] <= self.df.Open[i-1]) & (self.df.Open[i] <= self.df.Close[i-1])):
                        if ((self.df.Close[i] > self.df.Open[i - 1]) & (self.df.Close[i] > self.df.Close[i - 1])):
                            if(self.df[lr_name][i-1] < 0.0):
                                self.df.set_value(self.df.index[i-1],'Bullish_engulfing', 'True')
                                self.df.set_value(self.df.index[i],'Bullish_engulfing', 'True')

    def bearish_engulfing(self, lrp=5, lrma=20):
        self.candle_spec()
        self.ma_linear_regression(lr_period=lrp, ma_period=lrma)
        lr_name = 'LR_{period2}'.format(period2=lrp)
        self.df['Bearish_engulfing'] = None
        Bear_indexes = self.df.loc[(self.df.Candle_trend == 'Bear')].index.tolist()
        for bear_index in Bear_indexes:
            i = self.df.index.get_loc(bear_index)
            if (i > lrp + lrma - 2):
                if (self.df.Candle_trend[i - 1] != 'Bear'):
                    if ((self.df.Open[i] >= self.df.Open[i - 1]) & (self.df.Open[i] >= self.df.Close[i - 1])):
                        if ((self.df.Close[i] < self.df.Open[i - 1]) & (self.df.Close[i] < self.df.Close[i - 1])):
                            if (self.df[lr_name][i - 1] > 0.0):
                                self.df.set_value(self.df.index[i - 1], 'Bearish_engulfing', 'True')
                                self.df.set_value(self.df.index[i], 'Bearish_engulfing', 'True')


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

    def average_slope(self):
        pass

    def local_min(self,price='Close',period=5):
        pass

    def local_max(self):
        pass

    def gap_finder(selfself):
        pass

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


'''    def umbrella_candle (self,upper_shadow_size_parameter=0.2,body_size_parameter = 0.1):
        temp_array = zeros(len(self.df))
        for index in range(len(self.df)):
            Open,High,Low,Close = self.df['Open'][index],self.df['High'][index],self.df['Low'][index],self.df['Close'][index]
            if (Close >= Open): #white candle
                Candle = High - Low
                Body = Close - Open
                U_Shadow = High - Close
                L_Shadow = Open - Low
                if ((Body > 0) & (L_Shadow >= float(2) * Body) & (U_Shadow <= upper_shadow_size_parameter * Candle) & (Body >= body_size_parameter * Candle)): # parameters of umbrella candle
                    temp_array[index] = 1
            else: # same for black candle
                Candle = High - Low
                Body = Open - Close
                U_Shadow = High - Open
                L_Shadow = Close - Low
                if ((Body > 0) & (L_Shadow >= float(2) * Body) & (U_Shadow <= upper_shadow_size_parameter * Candle) & (Body >= body_size_parameter * Candle)):
                    temp_array[index] = 1
            self.df.loc[temp_array == 1 , 'Umbrella'] = 'True'
'''
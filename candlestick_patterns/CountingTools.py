#!/usr/bin/python3
from pandas import Series, read_csv, Series, DataFrame, value_counts,concat
from scipy import stats
from numpy import arange,linspace, where,diff,concatenate,savetxt,array,all,zeros,column_stack,hstack,asarray,isfinite,interp,nan
import matplotlib.pyplot as plt
import timeit

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
        #self.moving_average(period=20)
        #self.ma_linear_regression()
        self.pin_bar()
        self.candle_spec()
        #self.doji_type()
        #self.gap_finder()
        #self.candle_size_analysis()
        #self.dark_cloud_cover()
        #self.piercing_pattern()
        #self.bullish_engulfing()
        #self.bearish_engulfing()
        #self.on_neck()
        #self.in_neck()
        #self.thrusting_pattern()
        #self.doji_morning_star()
        #self.doji_evening_star()
        #self.abandoned_baby_on_uptrend()
        #self.abandoned_baby_on_downtrend()
        #self.tweezers_bottom()
        #self.tweezers_top()
        #self.bearish_belt_hold()
        #self.bullish_belt_hold()
        #self.candle_size_analysis()
        #self.define_long_candlestick()
        #self.define_short_candlestick()
        #print(self.issuer_list)
        #self.ma_linear_regression()
        # self.hammer()
        # self.hanging_man()
        #self.bullish_belt_hold()
        print(self.df)
        self.df.to_csv('df.csv')




    def check_column_exist(self,ma_p=0,ma_s='None',candle_spec=False,lr_p=0,lr_ma_p=0,lr_ma_s='Close',doji=False,volume_p=0):
        if(ma_p != 0):
            ma_name='MA_{source}_{period1}'.format(period1=ma_p,source=ma_s)
            if ma_name not in self.df:
                self.moving_average(period=ma_p,source=ma_s)
        if(lr_p != 0):
            lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
            if lr_name not in self.df:
                self.ma_linear_regression(lr_period=lr_p,ma_period=lr_ma_p,ma_source=lr_ma_s)
        if(candle_spec):
            if (('Body' not in self.df) | ('Candle' not in self.df) | ('Candle_direction' not in self.df) | ('Lower_shadow' not in self.df) | ('Upper_shadow' not in self.df) | ('Long_Short' not in self.df) | ('Marubozu' not in self.df)):
                self.candle_spec()
        if(doji):
            if('Doji_type' not in self.df):
                self.doji_type()
        if(volume_p != 0):
            v_name='Vma{p}'.format(p=volume_p)
            if v_name not in self.df:
                self.moving_avarage(period=volume_p,source='Volume')

    def moving_average(self,period=1,source='Close'):
        name = 'MA_{source}_{period}'.format(period=period,source=source)
        self.df[name] = Series.rolling(self.df[source], window=period, min_periods=period).mean()

    def pin_bar(self,lower_shadow_size=0.5,body_size=0.26,volume=0,small_body=False):
        self.check_column_exist(candle_spec='True', volume_p=volume)
        pin_name = 'Pin_{lss}'.format(lss=lower_shadow_size)
        self.df.loc[(self.df.Lower_shadow >= lower_shadow_size * self.df.Candle) & (self.df.Body <= body_size * self.df.Candle) & (~small_body | (self.df.Long_Short_B == 'Short')), pin_name] = 'True'
        -- выходить за рамки предыдущей свечи!!!!!

    def inverted_pin_bar(self,upper_shadow_greater_body=2.0,lower_shadow_greater_body=1.0):
        self.check_column_exist(candle_spec='True')
        name = 'Inverted_pin_{lsgb}_{uss}'.format(usgb=upper_shadow_greater_body, lss=lower_shadow_size)
        self.df.loc[((self.df.Candle_direction != 'Doji') & (self.df.Upper_shadow >= upper_shadow_greater_body * self.df.Body) & (self.df.Lower_shadow <= lower_shadow_greater_body * self.df.Body)), name] = 'True'

    def ma_linear_regression(self,lr_period=5,ma_period=20,ma_source='Close'):
        ma_name = 'MA_{source}_{period1}'.format(period1=ma_period,source=ma_source)
        self.check_column_exist(ma_p=ma_period,ma_s=ma_source)
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_period,p2=ma_source,p3=ma_period)
        slopes = zeros(len(self.df),dtype=float)
        y = arange(float(lr_period))
        for index in range((ma_period - 1), len(self.df)-lr_period+1 , 1):  # from end of issuer_list
            for index2 in range(lr_period): #prepare array len(lr_period)
                y[index2] = self.df[ma_name][index+index2]
            slope, intercept, r_value, p_value, std_err = stats.linregress(arange(lr_period), y)
            slopes[index+lr_period-1]=slope
        self.df[lr_name]=slopes

    def candle_spec(self,l_s_b=1.3,s_s_b=0.5,b_avr_p=7,l_s_c=1.3,s_s_c=0.5,c_avr_p=7,debug=True):
        self.df['Body'] = abs((self.df.Close - self.df.Open).round(decimals=2))
        self.df['Candle'] = abs((self.df.High - self.df.Low).round(decimals=2))
        self.df['Candle_direction'] = where(self.df.Open <= self.df.Close, 'Bull', 'Bear')
        self.df['Upper_shadow'] = where(self.df.Open >= self.df.Close,(self.df.High - self.df.Open).round(decimals=2),(self.df.High - self.df.Close).round(decimals=2))
        self.df['Lower_shadow'] = where(self.df.Open >= self.df.Close,(self.df.Close - self.df.Low).round(decimals=2),(self.df.Open - self.df.Low).round(decimals=2))
        self.df['Shadow_lenght'] = (self.df.Upper_shadow + self.df.Lower_shadow).round(decimals=2)
        self.df['Body_avr'] = Series.rolling(self.df['Body'], window=b_avr_p, min_periods=b_avr_p).mean()
        self.df['Long_Short_B'] = where(self.df.Body >= l_s_b * self.df.Body_avr, 'Long', '')
        self.df['Long_Short_B'] = where(self.df.Body <= s_s_b * self.df.Body_avr, 'Short', self.df['Long_Short_B'])
        self.df['Candle_avr'] = Series.rolling(self.df['Candle'], window=c_avr_p, min_periods=c_avr_p).mean()
        self.df['Long_Short_C'] = where(self.df.Candle >= l_s_c * self.df.Candle_avr, 'Long', '')
        self.df['Long_Short_C'] = where(self.df.Candle <= s_s_c * self.df.Candle_avr, 'Short', self.df['Long_Short_C'])
        self.df['Marubozu'] = where((self.df.Lower_shadow == 0.0) & (self.df.Upper_shadow == 0.0),'Marubozu','')
        if(debug):
            self.df['Low_S_to_Cand'] = (self.df.Lower_shadow / self.df.Candle).round(decimals=2)
            self.df['Upp_s_to_Cand'] = (self.df.Upper_shadow / self.df.Candle).round(decimals=2)
            self.df['Bod_to_Cand']   = (self.df.Body / self.df.Candle).round(decimals=2)


    def doji_type(self,doji_size= 0.1, shadow_lenght= 0.1):
        self.check_column_exist(candle_spec=True)
        self.df['Doji_type'] = where(abs((self.df.Open - self.df.Close).round(decimals=2)) <= doji_size * (self.df.High - self.df.Low).round(decimals=2), 'Doji', '')
        indexes = self.df.loc[(self.df.Doji_type == 'Doji')].index.tolist()
        for index in indexes:
            i =self.df.index.get_loc(index)
            if(self.df.High[i] == self.df.Low[i]):
                self.df.set_value(self.df.index[i], 'Doji_type', 'Flat')
            elif ((self.df.Upper_shadow[i] < shadow_lenght * self.df.Candle[i]) & (self.df.Lower_shadow[i] >= 0.5 * self.df.Candle[i])):
                self.df.set_value(self.df.index[i], 'Doji_type', 'Dragonfly')
            elif ((self.df.Lower_shadow[i] < shadow_lenght * self.df.Candle[i]) & (self.df.Upper_shadow[i] >= 0.5 * self.df.Candle[i])):
                self.df.set_value(self.df.index[i], 'Doji_type', 'Gravestone')
            elif ((self.df.Lower_shadow[i] > 0.35 * self.df.Candle[i]) & (self.df.Upper_shadow[i] > 0.35 * self.df.Candle[i])):
                self.df.set_value(self.df.index[i], 'Doji_type', 'Long_legged')

    def gap_finder(self):
        self.check_column_exist(candle_spec=True)
        self.df['Gap_body'] = None
        self.df['Gap_candle'] = None
        for i in range(len(self.df)-1):
            if(self.df.Candle_direction[i] == 'Bull'):
                if(((self.df.Close[i] < self.df.Open[i+1]) & (self.df.Close[i] < self.df.Close[i+1])) | ((self.df.Open[i] > self.df.Open[i+1]) & (self.df.Open[i] > self.df.Close[i+1]))):
                    self.df.set_value(self.df.index[i], 'Gap_body', 'GAP')
                    self.df.set_value(self.df.index[i+1], 'Gap_body', 'GAP')
                if((self.df.High[i] < self.df.Low[i+1]) | (self.df.Low[i] > self.df.High[i+1])):
                    self.df.set_value(self.df.index[i], 'Gap_candle', 'GAP')
                    self.df.set_value(self.df.index[i + 1], 'Gap_candle', 'GAP')
            elif(self.df.Candle_direction[i] == 'Bear'):
                if(((self.df.Close[i] > self.df.Open[i+1]) & (self.df.Close[i] > self.df.Open[i+1])) | ((self.df.Open[i] < self.df.Open[i+1]) & (self.df.Open[i] < self.df.Close[i+1]))):
                    self.df.set_value(self.df.index[i], 'Gap_body', 'GAP')
                    self.df.set_value(self.df.index[i + 1], 'Gap_body', 'GAP')
                if((self.df.High[i] < self.df.Low[i+1]) | (self.df.Low[i] > self.df.High[i+1])):
                    self.df.set_value(self.df.index[i], 'Gap_candle', 'GAP')
                    self.df.set_value(self.df.index[i + 1], 'Gap_candle', 'GAP')




#continuational



#reversal


    def black_spinning_top(self):
        indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Black_spinning_top'] = None
        for index in indexes:
            i =self.df.index.get_loc(index)
            if((self.df.Lower_shadow > 0.0) | (self.df.Upper_shadow > 0.0)):
                if((self.df.Body < self.df.Upper_shadow) | (self.df.Body < self.df.Lower_shadow)):
                    self.df.set_value(self.df.index[i], 'Black_spinning_top', 'True')

    def white_spinning_top(self):
        indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['White_spinning_top'] = None
        for index in indexes:
            i =self.df.index.get_loc(index)
            if((self.df.Lower_shadow > 0.0) | (self.df.Upper_shadow > 0.0)):
                if((self.df.Body < self.df.Upper_shadow) | (self.df.Body < self.df.Lower_shadow)):
                    self.df.set_value(self.df.index[i], 'White_spinning_top', 'True')

    def bearish_belt_hold(self, ls=0.3, us=0.03 ):
        self.check_column_exist(candle_spec='True')
        self.df['Bearish_belt_hold'] = None
        indexes = self.df.loc[(self.df.Candle_direction == 'Bear') & (self.df.Long_Short_B == 'Long') & (self.df.Long_Short_C == 'Long')].index.tolist()
        for index in indexes:
            i = self.df.index.get_loc(index)
            print(bear)
            if(self.df.Upper_shadow[i] <= us * self.df.Candle[i]):
                print(self.df.Upper_shadow[i])
                print(self.df.Candle[i])
                print(us * self.df.Candle[i])
                if(self.df.Lower_shadow[i] <= ls * self.df.Candle[i]):
                    print(self.df.Lower_shadow[i])
                    self.df.set_value(self.df.index[i], 'Bearish_belt_hold', 'True')

    def bullish_belt_hold(self, ls=0.03, us=0.3 ):
        self.check_column_exist(candle_spec=True)
        self.df['Bullish_belt_hold'] = None
        indexes = self.df.loc[(self.df.Candle_direction == 'Bull') & (self.df.Long_Short_B == 'Long') & (self.df.Long_Short_C == 'Long')].index.tolist()
        for index in indexes:
            i = self.df.index.get_loc(index)
            if(self.df.Upper_shadow[i] <= us * self.df.Candle[i]):
                if(self.df.Lower_shadow[i] <= ls * self.df.Candle[i]):
                    self.df.set_value(self.df.index[i], 'Bearish_belt_hold', 'True')

    def gapping_down_doji(self):
        self.check_column_exist(doji='True')
        self.df['Gapping_down_doji'] = None
        indexes= self.df.loc[(self.df.Candle_direction == 'Doji')].index.tolist()
        for index in indexes:
            i = self.df.index.get_loc(index)
            if(i > 0):
                if((self.df.High[i] < self.df.Low[i-1]) & (self.df.Doji_type[i-1] != 'Flat')):
                    self.df.set_value(self.df.index[i], 'Gapping_down_doji', 'True')

    def gapping_up_doji(self):
        self.check_column_exist(doji='True')
        self.df['Gapping_up_doji'] = None
        indexes= self.df.loc[(self.df.Candle_direction == 'Doji')].index.tolist()
        for index in indexes:
            i = self.df.index.get_loc(index)
            if(i > 0):
                if((self.df.Low[i] > self.df.High[i-1]) & (self.df.Doji_type[i-1] != 'Flat')):
                    self.df.set_value(self.df.index[i], 'Gapping_up_doji', 'True')

    def hammer(self,l_u=2.0,u_u=1.0,lr_p=5,lr_ma_p=20,lr_ma_s='Close',gap = False):
        pin_name = 'Pin_{lsgb}_{uss}'.format(lsgb=l_u,uss=u_u)
        if pin_name not in self.df:
            self.pin_bar(lower_shadow_greater_body=l_u,upper_shadow_greater_body=u_u)
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p,p2=lr_ma_s,p3=lr_ma_p)
        self.check_column_exist(lr_p=lr_p,lr_ma_p=lr_ma_p,lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df[pin_name] == 'True')].index.tolist()
        self.df['Hammer'] = None
        for index in indexes:
            i=self.df.index.get_loc(index)
            if(i > lr_p+lr_ma_p-2):
                if(self.df[lr_name] < 0.0):
                        self.df.set_value(self.df.index[i], 'Hammer', 'True')

    def hanging_man(self,l_u=2.0,u_u=0.2,lr_p=5,lr_ma_p=20,lr_ma_s='Close',gap = False): #body above trendline???
        pin_name = 'Pin_{lsgb}_{uss}'.format(lsgb=l_u,uss=u_u)
        if pin_name not in self.df:
            self.pin_bar(lower_shadow_greater_body=l_u,upper_shadow_greater_body=u_u)
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p,p2=lr_ma_s,p3=lr_ma_p)
        self.check_column_exist(lr_p=lr_p,lr_ma_p=lr_ma_p,lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df[pin_name] == 'True')].index.tolist()
        self.df['Hanging_Man'] = None
        for index in indexes:
            i=self.df.index.get_loc(index)
            if(i > lr_p+lr_ma_p-2):
                if (self.df[lr_name] > 0.0):
                    self.df.set_value(self.df.index[i], 'Hanging_Man', 'True')

    def shooting_star(self,u_u=2.0,l_u=0.2,lr_p=5,lr_ma_p=20,lr_ma_s='Close',gap = False):
        inverted_pin_name = 'Inverted_pin_{usgb}_{lss}'.format(usgb=u_u,lss=l_u)
        if inverted_pin_name not in self.df:
            self.inverted_pin_bar(upper_shadow_greater_body=u_u,lower_shadow_greater_body=l_u)
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(lr_p=lr_p,lr_ma_p=lr_ma_p,lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df[inverted_pin_name] == 'True')].index.tolist()
        self.df['Shooting_star'] = None
        for index in indexes:
            i=self.df.index.get_loc(index)
            if(i > lr_p+lr_ma_p-2):
                if(self.df[lr_name][i] > 0.0):
                    self.df.set_value(self.df.index[i], 'Shooting_star', 'True')

    def inverted_hammer (self,l_u=2.0,u_u=0.2,lr_p=5,lr_ma_p=20,lr_ma_s='Close',gap = False):
        inverted_pin_name = 'Inverted_pin_{usgb}_{lss}'.format(usgb=u_u, lss=l_u)
        if inverted_pin_name not in self.df:
            self.inverted_pin_bar(upper_shadow_greater_body=u_u, lower_shadow_greater_body=l_u)
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df[inverted_pin_name] == 'True')].index.tolist()
        self.df['Inverted_hammer'] = None
        for index in indexes:
            i=self.df.index.get_loc(index)
            if(i > lr_p+lr_ma_p-2):
                if(self.df[lr_name][i] < 0.0):
                    self.df.set_value(self.df.index[i], 'Inverted_hammer', 'True')

    def bullish_engulfing(self,lr_p=5,lr_ma_p=20,lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p,p2=lr_ma_s,p3=lr_ma_p)
        self.check_column_exist(candle_spec='True',lr_p=lr_p,lr_ma_p=lr_ma_p,lr_ma_s=lr_ma_s)
        self.df['Bullish_engulfing'] = None
        Bull_indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        for bull_index in Bull_indexes:
            i =self.df.index.get_loc(bull_index)
            if(i > lr_p+lr_ma_p-2):
                if(self.df.Candle_direction[i-1] != 'Bull'):
                    if((self.df.Open[i] <= self.df.Open[i-1]) & (self.df.Open[i] <= self.df.Close[i-1])):
                        if ((self.df.Close[i] > self.df.Open[i - 1]) & (self.df.Close[i] > self.df.Close[i - 1])):
                            if(self.df[lr_name][i-1] < 0.0):
                                self.df.set_value(self.df.index[i-1],'Bullish_engulfing', 'BE1')
                                self.df.set_value(self.df.index[i],'Bullish_engulfing', 'BE2')

    def bearish_engulfing(self, lr_p=5, lr_ma_p=20,lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p,p2=lr_ma_s,p3=lr_ma_p)
        self.check_column_exist(candle_spec='True',lr_p=lr_p,lr_ma_p=lr_ma_p,lr_ma_s=lr_ma_s)
        self.df['Bearish_engulfing'] = None
        Bear_indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        for bear_index in Bear_indexes:
            i = self.df.index.get_loc(bear_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df.Candle_direction[i - 1] != 'Bear'):
                    if ((self.df.Open[i] >= self.df.Open[i - 1]) & (self.df.Open[i] >= self.df.Close[i - 1])):
                        if ((self.df.Close[i] < self.df.Open[i - 1]) & (self.df.Close[i] < self.df.Close[i - 1])):
                            if (self.df[lr_name][i - 1] > 0.0):
                                self.df.set_value(self.df.index[i - 1], 'Bearish_engulfing', 'BE1')
                                self.df.set_value(self.df.index[i], 'Bearish_engulfing', 'BE2')

    def dark_cloud_cover(self, lr_p=5, lr_ma_p=20,lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p,p2=lr_ma_s,p3=lr_ma_p)
        self.check_column_exist(candle_spec='True',lr_p=lr_p,lr_ma_p=lr_ma_p,lr_ma_s=lr_ma_s)
        Bear_indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Dark_cloud_cover'] = None
        for bear_index in Bear_indexes:
            i = self.df.index.get_loc(bear_index)
            if (i > lr_p + lr_ma_p - 2):
                 if (self.df.Candle_direction[i - 1] == 'Bull'):
                     if(self.df.Open[i] > self.df.Close[i-1]):
                         if((self.df.Close[i] < (self.df.Open[i-1]+(self.df.Body[i-1]/2))) & (self.df.Close[i] >= self.df.Open[i-1])):
                             if (self.df[lr_name][i-1] > 0.0):
                                 self.df.set_value(self.df.index[i-1], 'Dark_cloud_cover', 'DC1')
                                 self.df.set_value(self.df.index[i], 'Dark_cloud_cover', 'DC2')

    def piercing_pattern (self, lr_p=5, lr_ma_p=20,lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p,p2=lr_ma_s,p3=lr_ma_p)
        self.check_column_exist(candle_spec='True',lr_p=lr_p,lr_ma_p=lr_ma_p,lr_ma_s=lr_ma_s)
        Bull_indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Piercing_pattern'] = None
        for bull_index in Bull_indexes:
            i = self.df.index.get_loc(bull_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df.Candle_direction[i - 1] == 'Bear'):
                    if(self.df.Open[i] < self.df.Close[i-1]):
                        if((self.df.Close[i] > (self.df.Close[i-1] + self.df.Body[i-1]/2)) & (self.df.Close[i] < self.df.Open[i-1])):
                            if(self.df[lr_name][i-1] < 0.0):
                                self.df.set_value(self.df.index[i - 1], 'Piercing_pattern', 'PP1')
                                self.df.set_value(self.df.index[i], 'Piercing_pattern', 'PP2')

    def on_neck(self, lr_p=5, lr_ma_p=20,lr_ma_s='Close',close_delta=0.2):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p,p2=lr_ma_s,p3=lr_ma_p)
        self.check_column_exist(candle_spec='True',lr_p=lr_p,lr_ma_p=lr_ma_p,lr_ma_s=lr_ma_s)
        Bull_indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['On_Neck'] = None
        for bull_index in Bull_indexes:
            i = self.df.index.get_loc(bull_index)
            if (i > lr_p + lr_ma_p - 2):
                if(self.df.Shadow_lenght[i] <= 2.0*self.df.Body[i]):
                    if (self.df.Candle_direction[i-1] == 'Bear'):
                        if (self.df[lr_name][i] < 0.0):
                            if(self.df.Open[i] < self.df.Low[i-1]):
                                p_delta = ((close_delta * self.df.Lower_shadow[i-1])+self.df.Low[i-1]).round(decimals=2);
                                m_delta = (self.df.Low[i-1] - (close_delta * self.df.Lower_shadow[i-1])).round(decimals=2);
                                if(m_delta <= self.df.Close[i] <= p_delta):
                                    self.df.set_value(self.df.index[i - 1], 'On_Neck', 'ON1')
                                    self.df.set_value(self.df.index[i], 'On_Neck', 'ON2')

    def in_neck(self, lr_p=5, lr_ma_p=20, lr_ma_s='Close', close_delta=0.2):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        Bull_indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['In_Neck'] = None
        for bull_index in Bull_indexes:
            i = self.df.index.get_loc(bull_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df.Shadow_lenght[i] <= 2.0 * self.df.Body[i]):
                    if (self.df.Candle_direction[i-1] == 'Bear'):
                        if (self.df[lr_name][i] < 0.0):
                            if (self.df.Open[i] < self.df.Low[i - 1]):
                                m_delta = ((close_delta * self.df.Lower_shadow[i - 1]) + self.df.Low[i - 1]).round(decimals=2);
                                p_delta = (self.df.Close[i - 1] + (close_delta * self.df.Body[i - 1])).round(decimals=2);
                                if (m_delta <= self.df.Close[i] <= p_delta):
                                    self.df.set_value(self.df.index[i - 1], 'In_Neck', 'IN1')
                                    self.df.set_value(self.df.index[i], 'In_Neck', 'IN2')

    def thrusting_pattern(self, lr_p=5, lr_ma_p=20, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        Bull_indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Thrusting_pattern'] = None
        for bull_index in Bull_indexes:
            i = self.df.index.get_loc(bull_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df.Candle_direction[i-1] == 'Bear'):
                     if (self.df[lr_name][i] < 0.0):
                         if (self.df.Open[i] < self.df.Low[i - 1]):
                             p_delta = (self.df.Close[i-1] + (0.5 * self.df.Body[i-1])).round(decimals=2);
                             m_delta = self.df.Close[i-1]
                             if (m_delta < self.df.Close[i] < p_delta):
                                 self.df.set_value(self.df.index[i-1], 'Thrusting_pattern', 'TP1')
                                 self.df.set_value(self.df.index[i], 'Thrusting_pattern', 'TP2')

    def morning_star(self,lr_p=5, lr_ma_p=5, lr_ma_s='Close', second_gap=False):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        Bull_indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Morning_star'] = None
        for bull_index in Bull_indexes:
            i = self.df.index.get_loc(bull_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i-2] < 0.0):
                    if(self.df.Candle_direction[i-2] == 'Bear'):
                        if ((self.df.Candle_direction[i-1] != 'Doji') & (self.df.Long_Short_B[i-1] == 'Short')):
                            if((self.df.Close[i-2] > self.df.Open[i-1]) & (self.df.Close[i-2] > self.df.Close[i-1])):
                                if(self.df.Close[i] >= (self.df.Close[i-2]+(self.df.Body[i-2]/2)).round(decimals=2)):
                                    if(second_gap):
                                        if((self.df.Open[i] > self.df.Open[i-1]) & (self.df.Open[i] > self.df.Close[i-1])):
                                            self.df.set_value(self.df.index[i-2], 'Morning_star', 'MS1g')
                                            self.df.set_value(self.df.index[i-1], 'Morning_star', 'MS2g')
                                            self.df.set_value(self.df.index[i], 'Morning_star', 'MS3g')
                                    else:
                                        if(self.df.Open[i] >= self.df.Low[i-1]):
                                            self.df.set_value(self.df.index[i-2], 'Morning_star', 'MS1')
                                            self.df.set_value(self.df.index[i-1], 'Morning_star', 'MS2')
                                            self.df.set_value(self.df.index[i], 'Morning_star', 'MS3')

    def evening_star(self,lr_p=5, lr_ma_p=5, lr_ma_s='Close', second_gap=False):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        Bear_indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Evening_star'] = None
        for bear_index in Bear_indexes:
            i = self.df.index.get_loc(bear_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i-2] > 0.0):
                    if(self.df.Candle_direction[i-2] == 'Bull'):
                        if ((self.df.Candle_direction[i-1] != 'Doji') & (self.df.Long_Short_B[i-1] == 'Short')):
                            if((self.df.Close[i-2] < self.df.Open[i-1]) & (self.df.Close[i-2] < self.df.Close[i-1])):
                                if(self.df.Close[i] <= (self.df.Close[i-2]-(self.df.Body[i-2]/2)).round(decimals=2)):
                                    if(second_gap):
                                        if((self.df.Open[i] < self.df.Open[i-1]) & (self.df.Open[i] < self.df.Close[i-1])):
                                            self.df.set_value(self.df.index[i-2], 'Evening_star', 'ES1g')
                                            self.df.set_value(self.df.index[i-1], 'Evening_star', 'ES2g')
                                            self.df.set_value(self.df.index[i], 'Evening_star', 'ES3g')
                                    else:
                                        if(self.df.Open[i] <= self.df.High[i-1]):
                                            self.df.set_value(self.df.index[i-2], 'Evening_star', 'ES1')
                                            self.df.set_value(self.df.index[i-1], 'Evening_star', 'ES2')
                                            self.df.set_value(self.df.index[i], 'Evening_star', 'ES3')

    def doji_morning_star(self,lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Doji_morning_star'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 2] < 0.0):
                    if(self.df.Candle_direction[i-2] == 'Bear'):
                        if (self.df.Candle_direction[i-1] == 'Doji'):
                            if((self.df.Close[i-2] > self.df.Open[i-1])&(self.df.Close[i-2] > self.df.Close[i-1])):
                                if(self.df.High[i-1] > self.df.Low[i-2]):
                                    if((self.df.Open[i] > self.df.Open[i-1]) & (self.df.Open[i] > self.df.Close[i-1])):
                                        if(self.df.Close[i] >= (self.df.Close[i-2]+(self.df.Body[i-2]/2)).round(decimals=2)):
                                            self.df.set_value(self.df.index[i - 2], 'Doji_morning_star', 'DMS1')
                                            self.df.set_value(self.df.index[i - 1], 'Doji_morning_star', 'DMS2')
                                            self.df.set_value(self.df.index[i], 'Doji_morning_star', 'DMS3')

    def doji_evening_star(self,lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        Bear_indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Doji_evening_star'] = None
        for bear_index in Bear_indexes:
            i = self.df.index.get_loc(bear_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i-2] > 0.0):
                    if(self.df.Candle_direction[i-2] == 'Bull'):
                        if (self.df.Candle_direction[i-1] == 'Doji'):
                            if((self.df.Close[i-2] < self.df.Open[i-1]) & (self.df.Close[i-2] < self.df.Close[i-1])):
                                if(self.df.Low[i-1] < self.df.High[i-2]):
                                    if((self.df.Open[i] < self.df.Open[i-1]) & (self.df.Open[i] < self.df.Close[i-1])):
                                        if(self.df.Close[i] <= (self.df.Close[i-2]-(self.df.Body[i-2]/2)).round(decimals=2)):
                                            self.df.set_value(self.df.index[i-2], 'Doji_evening_star', 'DES1')
                                            self.df.set_value(self.df.index[i-1], 'Doji_evening_star', 'DES2')
                                            self.df.set_value(self.df.index[i], 'Doji_evening_star', 'DES3')

    def bearish_doji_star(self,lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Doji')].index.tolist()
        self.df['Bearish_doji_star'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i-1] > 0.0):
                    if(self.df.Candle_direction[i-1] == 'Bull'):
                        if((self.df.Open[i] > self.df.Close[i-1]) & (self.df.Close[i] > self.df.Close[i-1])):
                            self.df.set_value(self.df.index[i-1], 'Bearish_doji_star', 'BDS1')
                            self.df.set_value(self.df.index[i], 'Bearish_doji_star', 'BDS2')

    def bullish_doji_star(self,lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Doji')].index.tolist()
        self.df['Bullish_doji_star'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i-1] < 0.0):
                    if(self.df.Candle_direction[i-1] == 'Bear'):
                        if((self.df.Open[i] < self.df.Close[i-1]) & (self.df.Close[i] < self.df.Close[i-1])):
                            self.df.set_value(self.df.index[i-1], 'Bullish_doji_star', 'BDS1')
                            self.df.set_value(self.df.index[i], 'Bullish_doji_star', 'BDS2')

    def abandoned_baby_on_uptrend(self,lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        Bear_indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Abandoned_baby_on_uptrend'] = None
        for bear_index in Bear_indexes:
            i = self.df.index.get_loc(bear_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i-2] > 0.0):
                    if(self.df.Candle_direction[i-2] == 'Bull'):
                        if (self.df.Candle_direction[i-1] == 'Doji'):
                            if((self.df.High[i-2] < self.df.Low[i-1]) & (self.df.High[i] < self.df.Low[i-1])):
                                self.df.set_value(self.df.index[i - 2], 'Abandoned_baby_on_uptrend', 'ABOU1')
                                self.df.set_value(self.df.index[i - 1], 'Abandoned_baby_on_uptrend', 'ABOU2')
                                self.df.set_value(self.df.index[i], 'Abandoned_baby_on_uptrend', 'ABOU3')

    def abandoned_baby_on_downtrend(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        Bull_indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Abandoned_baby_on_downtrend'] = None
        for bull_index in Bull_indexes:
            i = self.df.index.get_loc(bull_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 2] < 0.0):
                    if(self.df.Candle_direction[i-2] == 'Bear'):
                        if (self.df.Candle_direction[i-1] == 'Doji'):
                            if((self.df.Low[i-2] > self.df.High[i-1]) & (self.df.Low[i] > self.df.High[i-1])):
                                self.df.set_value(self.df.index[i - 2], 'Abandoned_baby_on_downtrend', 'ABOD1')
                                self.df.set_value(self.df.index[i - 1], 'Abandoned_baby_on_downtrend', 'ABOD2')
                                self.df.set_value(self.df.index[i], 'Abandoned_baby_on_downtrend', 'ABOD3')

    def bullish_harami(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Bullish_harami'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 1] < 0.0):
                    if(self.df.Candle_direction[i-1] == 'Bear'):
                        if(self.df.Close[i-1] < self.df.Open[i]):
                            self.df.set_value(self.df.index[i-1], 'Bullish_harami', 'BH1')
                            self.df.set_value(self.df.index[i], 'Bullish_harami', 'BH2')

    def bearish_harami(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        Bear_indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Bearish_harami'] = None
        for bear_index in Bear_indexes:
            i = self.df.index.get_loc(bear_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 1] > 0.0):
                    if(self.df.Candle_direction[i-1] == 'Bull'):
                        if(self.df.Close[i-1] > self.df.Open[i]):
                            self.df.set_value(self.df.index[i-1], 'Bearish_harami', 'BH1')
                            self.df.set_value(self.df.index[i], 'Bearish_harami', 'BH2')

    def bullish_harami_cross(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        Doji_indexes = self.df.loc[(self.df.Candle_direction == 'Doji')].index.tolist()
        self.df['Bullish_harami_cross'] = None
        for doji_index in Doji_indexes:
            i = self.df.index.get_loc(doji_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 1] < 0.0):
                    if(self.df.Candle_direction[i-1] == 'Bull'):
                        if(self.df.Close[i-1] < self.df.Open[i]):
                            self.df.set_value(self.df.index[i], 'Bullish_harami_cross', 'BHC1')
                            self.df.set_value(self.df.index[i], 'Bullish_harami_cross', 'BHC2')

    def bearish_harami_cross(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        Doji_indexes = self.df.loc[(self.df.Candle_direction == 'Doji')].index.tolist()
        self.df['Bearish_harami_cross'] = None
        for doji_index in Doji_indexes:
            i = self.df.index.get_loc(doji_index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 1] > 0.0):
                    if (self.df.Candle_direction[i-1] == 'Bear'):
                        if (self.df.Close[i-1] > self.df.Open[i]):
                            self.df.set_value(self.df.index[i-1], 'Bearish_harami_cross', 'BHC1')
                            self.df.set_value(self.df.index[i], 'Bearish_harami_cross', 'BHC2')

    def tweezers_bottom(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        self.df['Tweezers_bottom'] = None
        for i in range(lr_p + lr_ma_p - 2,len(self.df),1):
            if (self.df[lr_name][i] < 0.0):
                if((self.df.Candle_direction[i] != 'Doji_flat') & (self.df.Candle_direction[i+1] != 'Doji_flat')):
                    if(self.df.Low[i] == self.df.Low[i+1]):
                        self.df.set_value(self.df.index[i], 'Tweezers_bottom', 'TB1')
                        self.df.set_value(self.df.index[i+1], 'Tweezers_bottom', 'TB2')

    def tweezers_top(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        self.df['Tweezers_top'] = None
        for i in range(lr_p + lr_ma_p - 2,len(self.df),1):
            if (self.df[lr_name][i] > 0.0):
                if((self.df.Candle_direction[i] != 'Doji_flat') & (self.df.Candle_direction[i+1] != 'Doji_flat')):
                    if(self.df.High[i] == self.df.Hugh[i+1]):
                        self.df.set_value(self.df.index[i], 'Tweezers_top', 'TT1')
                        self.df.set_value(self.df.index[i+1], 'Tweezers_top', 'TT2')

    def bearish_counterattack_lines(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Bearish_counterattack_lines'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 1] > 0.0):
                    if(self.df.Candle_direction[i-1] == 'Bull'):
                        if(self.df.Close[i] == self.df.Close[i-1]):
                            self.df.set_value(self.df.index[i - 1], 'Bearish_counterattack_lines', 'BCL1')
                            self.df.set_value(self.df.index[i], 'Bearish_counterattack_lines', 'BCL2')

    def bullish_counterattack_lines(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Bullish_counterattack_lines'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 1] < 0.0):
                    if(self.df.Candle_direction[i-1] == 'Bear'):
                        if(self.df.Close[i] == self.df.Close[i-1]):
                            self.df.set_value(self.df.index[i - 1], 'Bullish_counterattack_lines', 'BCL1')
                            self.df.set_value(self.df.index[i], 'Bullish_counterattack_lines', 'BCL2')

    def bearish_separating_lines(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Bearish_separating_lines'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 1] < 0.0):
                    if(self.df.Candle_direction[i-1] == 'Bull'):
                        if(self.df.Open[i] == self.df.Open[i-1]):
                            self.df.set_value(self.df.index[i - 1], 'Bearish_separating_lines', 'BSL1')
                            self.df.set_value(self.df.index[i], 'Bearish_separating_lines', 'BSL2')

    def bullish_separating_lines(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Bullish_separating_lines'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 1] > 0.0):
                    if(self.df.Candle_direction[i-1] == 'Bear'):
                        if(self.df.Open[i] == self.df.Open[i-1]):
                            self.df.set_value(self.df.index[i - 1], 'Bullish_separating_lines', 'BSL1')
                            self.df.set_value(self.df.index[i], 'Bullish_separating_lines', 'BSL2')

    def descending_hawk(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Descending_hawk'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 1] > 0.0):
                    if(self.df.Open[i] > self.df.Open[i-1]):
                        if (self.df.Close[i] < self.df.Close[i - 1]):
                            self.df.set_value(self.df.index[i - 1], 'Descending_hawk', 'DH1')
                            self.df.set_value(self.df.index[i], 'Descending_hawk', 'DH2')

    def homing_pigeon(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Homing_pigeon'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 1] < 0.0):
                    if(self.df.Open[i] < self.df.Open[i-1]):
                        if (self.df.Close[i] > self.df.Close[i - 1]):
                            self.df.set_value(self.df.index[i - 1], 'Homing_pigeon', 'HP1')
                            self.df.set_value(self.df.index[i], 'Homing_pigeon', 'HP2')

    def advance_block(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Advance_block'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if((self.df.Candle_direction[i-2] == 'Bull') & (self.df.Candle_direction[i-1] == 'Bull') & (self.df.Candle_direction[i] == 'Bull')):
                    if (self.df[lr_name][i - 2] > 0.0):
                        if((self.df.Open[i-1] > self.df.Open[i-2]) & (self.df.Open[i-1] < self.df.Close[i-2]) & (self.df.Close[i-1] > self.df.Close[i-2])):
                            if((self.df.Open[i] > self.df.Open[i-1]) & (self.df.Open[i] < self.df.Close[i-1]) & (self.df.Close[i] > self.df.Close[i-1])):
                                self.df.set_value(self.df.index[i-2], 'Advance_block', 'AB1')
                                self.df.set_value(self.df.index[i-1], 'Advance_block', 'AB2')
                                self.df.set_value(self.df.index[i], 'Advance_block', 'AB3')

    def high_wave(self, lengh = 3.0):
        self.check_column_exist(candle_spec='True')
        self.df['High_wave'] = None
        indexes = self.df.loc[(self.df.Long_Short_B == 'Short')].index.tolist()
        for index in indexes:
            i = self.df.index.get_loc(index)
            if((self.df.Upper_shadow[i] > lengh * self.df.Body) | (self.df.Lower_shadow[i] > lengh * self.df.Body)):
                self.df.set_value(self.df.index[i], 'High_wave', 'True')

    def downside_takusi_gap(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bull')].index.tolist()
        self.df['Downside_takusi_gap'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 2] < 0.0):
                    if((self.df.Candle_direction[i-2] == 'Bear') & (self.df.Candle_direction[i-1] == 'Bear')):
                        if(self.df.High[i-1] < self.df.Low[i-2]):
                            if((self.df.Open[i] > self.df.Close[i-1]) & (self.df.Open[i] < self.df.Open[i-1])):
                                if(self.df.Close[i] < self.df.Close[i-2]):
                                    self.df.set_value(self.df.index[i - 2], 'Downside_takusi_gap', 'DTG1')
                                    self.df.set_value(self.df.index[i - 1], 'Downside_takusi_gap', 'DTG2')
                                    self.df.set_value(self.df.index[i], 'Downside_takusi_gap', 'DTG3')

    def upside_takusi_gap(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Upside_takusi_gap'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 2] > 0.0):
                    if((self.df.Candle_direction[i-2] == 'Bull') & (self.df.Candle_direction[i-1] == 'Bull')):
                        if(self.df.Low[i-1] > self.df.High[i-2]):
                            if((self.df.Open[i] < self.df.Close[i-1]) & (self.df.Open[i] > self.df.Open[i-1])):
                                if(self.df.Close[i] > self.df.Close[i-2]):
                                    self.df.set_value(self.df.index[i - 2], 'Upside_takusi_gap', 'UTG1')
                                    self.df.set_value(self.df.index[i - 1], 'Upside_takusi_gap', 'UTG2')
                                    self.df.set_value(self.df.index[i], 'Upside_takusi_gap', 'UTG3')

    def upside_gap_two_crows(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        lr_name = 'LR_{p1}_{p2}_{p3}'.format(p1=lr_p, p2=lr_ma_s, p3=lr_ma_p)
        self.check_column_exist(candle_spec='True', lr_p=lr_p, lr_ma_p=lr_ma_p, lr_ma_s=lr_ma_s)
        indexes = self.df.loc[(self.df.Candle_direction == 'Bear')].index.tolist()
        self.df['Upside_gap_two_crows'] = None
        for index in indexes:
            i = self.df.index.get_loc(index)
            if (i > lr_p + lr_ma_p - 2):
                if (self.df[lr_name][i - 2] > 0.0):
                    if((self.df.Candle_direction[i-2] == 'Bull') & (self.df.Candle_direction[i-1] == 'Bear')):
                        if(self.df.Close[i-2] < self.df.Close[i-1]):
                            if((self.df.Open[i] > self.df.Open[i-1]) & (self.df.Close[i] < self.df.Close[i-1])):
                                self.df.set_value(self.df.index[i - 2], 'Upside_gap_two_crows', 'UGTC1')
                                self.df.set_value(self.df.index[i - 1], 'Upside_gap_two_crows', 'UGTC2')
                                self.df.set_value(self.df.index[i], 'Upside_gap_two_crows', 'UGTC3')





    def three_soldiers(self):
        pass

    def three_crows(self):
        pass

    def three_methods(self):
        pass

    def towers_top(self):
        pass

    def towers_bottom(self):
        pass

    def deliberation(self, lr_p=5, lr_ma_p=5, lr_ma_s='Close'):
        pass

    def gapping_play_up(self):
        pass

    def gapping_play_down(self):
        pass

    def ten_new_tops(self):
        pass

    def ten_new_lows(self):
        pass

    def dumping_top(self):
        pass

    def fly_pan_bottom(self):
        pass

    def mat_hold(self):
        pass

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

    def price_level_analysis(self):#how much cs at each price level
        pass

    def candle_size_analysis(self):
        hist_data = self.df.Body.value_counts().sort_index()  # prepare histogram data
        print(hist_data)

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


'''    def umbrella_candle_old (self,upper_shadow_size_parameter=0.2,body_size_parameter = 0.1):
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

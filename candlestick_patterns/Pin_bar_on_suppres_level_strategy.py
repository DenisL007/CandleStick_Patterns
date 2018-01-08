#!/usr/bin/python3


class Pin_Level_st(object):


    # Pin bar on up trend
    # 1. Find Pin bar
    # 2. Define if Pin bar local maximum for last X days
    # 3. Calculate borders for supres from Pin bar High price of 1%
    # 4. Calculate supres
    # 5. Open position in zz_trend opposite direction

# * better optimization for supres, add gaps
# * add check_column to local_min/max


    def __init__(self,stock):
        self.st = stock
        self.sell_on_pin_bar()
        # self.st.plot_candles(pricing=self.st.df, title='Intc', technicals=[self.st.df['zz_60_8%'].interpolate()])


    def sell_on_pin_bar (self):
        min_name = self.st.local_min(source='Low',period=5)
        max_name = self.st.local_max(source='High', period=5)
        zz_name = self.st.zigzig(deviation = 3, backstep = 1, depth = 3,last=True)
        supres = self.st.supres_3()
        pin_name = self.st.pin_bar()
        # self.st.df.to_csv('df.csv')
        pin_df = (self.st.df.loc[self.st.df[pin_name].notnull()])
#        min_max_pin_df = pin_df.ix[(pin_df[min_name] == pin_df['Low']) | (pin_df[max_name] == pin_df['High'])]
        max_pin_df = pin_df.ix[(pin_df[max_name] == pin_df['High'])]
        deviation_max_pin_df = max_pin_df.copy()
        deviation_max_pin_df['deviation_high'] = deviation_max_pin_df.High + (deviation_max_pin_df.High * 0.01)
        deviation_max_pin_df['deviation_low']  = deviation_max_pin_df.Low - (deviation_max_pin_df.High * 0.01)
        supres_deviation_max_pin_df = (deviation_max_pin_df.copy()).reset_index(drop=True)
        for pin in range(len(supres_deviation_max_pin_df)):
            for sp in range(len(supres)):
                if( (supres.iloc[sp,0]  >= supres_deviation_max_pin_df.loc[pin, 'deviation_low']) & (supres.iloc[sp,0]  <= supres_deviation_max_pin_df.loc[pin, 'deviation_high']) ):
                    supres_deviation_max_pin_df.set_value(supres_deviation_max_pin_df.index[pin], 'supres', True)
# checker - what price after 2/3/5/10 days, maximum profit , risk value , volume correlation, long-tern trend correlation.
        print(supres)

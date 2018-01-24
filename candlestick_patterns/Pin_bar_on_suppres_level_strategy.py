#!/usr/bin/python3
from numpy import zeros,ones
from pandas import DataFrame

class Pin_Level_st(object):


    # Pin_bar_on_up_trend_1
    # 1. Find Pin bar
    # 2. Define if Pin bar local maximum for last X days
    # 3. Calculate borders for supres from Pin bar High price of 1%
    # 4. Calculate supres
    # 5. Open position in zz_trend opposite direction

# * better optimization for supres, add gaps and 1.5 years periods
# * add check_column to local_min/max


    def __init__(self,stock):
        self.st = stock
        self.pin_bar_on_up_trend_1(max_last_x_days=5,percent_deviation=0.01,supres_periods=True)
        self.check_sell_direction(strategy_name='pin_bar_on_up_trend_1')
        self.check_buy_direction(strategy_name='pin_bar_on_up_trend_1')

    def pin_bar_on_up_trend_1 (self,max_last_x_days=3,percent_deviation=0.01,supres_periods=False):
        vol_name = self.st.moving_average(period=23,source='Volume')
        min_name = self.st.local_min(source='Low',period=max_last_x_days)
        max_name = self.st.local_max(source='High', period=max_last_x_days)
        zz_name = self.st.zigzig(deviation = 3, backstep = 1, depth = 3,last=True)
        zz_long_name = self.st.zigzig(deviation = 10, backstep = 1, depth = 5,last=True)
        if(supres_periods):
            supres = self.st.supres_one_and_half_year_periods()
        else:
            supres = self.st.supres_3()
        pin_name = self.st.pin_bar()
        pin_df = (self.st.df.loc[self.st.df[pin_name].notnull()])
        max_pin_df = pin_df.ix[(pin_df[max_name] == pin_df['High'])]
        deviation_max_pin_df = max_pin_df.copy()
        deviation_max_pin_df['deviation_high'] = deviation_max_pin_df.High + (deviation_max_pin_df.High * percent_deviation)
        deviation_max_pin_df['deviation_low']  = deviation_max_pin_df.Low - (deviation_max_pin_df.High * percent_deviation)
        supres_deviation_max_pin_df = (deviation_max_pin_df.copy()).reset_index()
        supres_deviation_max_pin_df = supres_deviation_max_pin_df.rename(columns={'index':'orig_ind'})
        for pin in range(len(supres_deviation_max_pin_df)):
            for sp in range(len(supres)):
                if( (supres.iloc[sp,0]  >= supres_deviation_max_pin_df.loc[pin, 'deviation_low']) & (supres.iloc[sp,0]  <= supres_deviation_max_pin_df.loc[pin, 'deviation_high']) ):
                    supres_deviation_max_pin_df.set_value(supres_deviation_max_pin_df.index[pin], 'supres', True)
        supres_deviation_max_pin_df = (supres_deviation_max_pin_df.loc[supres_deviation_max_pin_df['supres'].notnull()])
        orig_ind = supres_deviation_max_pin_df['orig_ind']
        self.st.df['pin_bar_on_up_trend_1'] = None
        for i in orig_ind:
            self.st.df.set_value(self.st.df.index[i], 'pin_bar_on_up_trend_1', 'bingo')
        self.st.save_df(name="pin_bar_on_up_trend_1.csv")
        return

# checker - what price after 2/3/5/10 days, maximum profit , risk value , volume correlation, long-tern trend correlation.
    def buy_diff_price(self,source="Close",st_ind=0,ind=0,spread = 0.0):
        return self.st.df[source][st_ind+ind] - self.st.df[source][st_ind] + 2*spread

    def sell_diff_price(self,source="Close",st_ind=0,ind=0,spread = 0.0):
        return self.st.df[source][st_ind] - self.st.df[source][st_ind+ind] - 2*spread

    def max_price(self,source="Close",st_ind=0,ind=0):
        return self.st.df[source][st_ind:st_ind + ind + 1].max()

    def idxmax_price(self,source="Close",st_ind=0,ind=0):
        return self.st.df[source][st_ind:st_ind + ind + 1].idxmax()

    def min_price(self, source="Close", st_ind=0, ind=0):
        return self.st.df[source][st_ind:st_ind + ind + 1].min()

    def idxmin_price(self, source="Close", st_ind=0, ind=0):
        return self.st.df[source][st_ind:st_ind + ind + 1].idxmin()

    def sell_diff_percent(self, source="Close", st_ind=0, ind=0,spread=0.0):
        diff = self.sell_diff_price(source=source, st_ind=st_ind, ind=ind,spread=spread)
        pr = diff/self.st.df[source][st_ind]
        if ( pr < -0.02):
            pr = -0.02
        return pr

    def buy_diff_percent(self, source="Close", st_ind=0, ind=0,spread=0.0):
        diff = self.buy_diff_price(source=source, st_ind=st_ind, ind=ind,spread=spread)
        pr = diff/self.st.df[source][st_ind]
        if (pr < -0.02):
            pr = -0.02
        return pr

    def check_sell_direction(self,strategy_name=None,spread = 0.08,close_period=50):
        ind = self.st.df.loc[self.st.df[strategy_name] == 'bingo'].index
        diff_percent,idxmin_price,idxmax_price,min_price,max_price,diff_price = zeros((len(ind),close_period), dtype=float),zeros((len(ind),close_period), dtype=float),zeros((len(ind),close_period), dtype=float),zeros((len(ind),close_period), dtype=float),zeros((len(ind),close_period), dtype=float),zeros((len(ind),close_period), dtype=float)
        for i in range(len(ind)):
            for j in range(1,close_period,1):
                diff_price[i][j]  = self.sell_diff_price(st_ind=ind[i],ind=j,spread = spread)
                max_price[i][j]  = self.max_price(st_ind=ind[i],ind=j)
                idxmax_price[i][j]  = self.idxmax_price(st_ind=ind[i],ind=j)
                min_price[i][j]  = self.min_price(st_ind=ind[i],ind=j)
                idxmin_price[i][j]  = self.idxmin_price(st_ind=ind[i],ind=j)
                diff_percent[i][j] = self.sell_diff_percent(st_ind=ind[i],ind=j,spread = spread)

        result_array = ones((len(ind)+1,close_period), dtype=float)
        for i in range(len(ind)):
            for j in range(1,close_period,1):
                result_array[i][j] = result_array[i][j] * (1 + diff_percent[i][j])
        for j in range(1,close_period,1):
            for i in range(len(ind)):
                result_array[len(ind)][j] = result_array[len(ind)][j] * result_array[i][j]
        result_df = DataFrame(result_array)
        result_df.to_csv("check_sell_direction.csv")

    def check_buy_direction(self,strategy_name=None,spread = 0.08,close_period=50):
        ind = self.st.df.loc[self.st.df[strategy_name] == 'bingo'].index
        diff_percent,idxmin_price,idxmax_price,min_price,max_price,diff_price = zeros((len(ind),close_period), dtype=float),zeros((len(ind),close_period), dtype=float),zeros((len(ind),close_period), dtype=float),zeros((len(ind),close_period), dtype=float),zeros((len(ind),close_period), dtype=float),zeros((len(ind),close_period), dtype=float)

        for i in range(len(ind)):
            for j in range(1,close_period,1):
                diff_price[i][j]  = self.buy_diff_price(st_ind=ind[i],ind=j,spread = spread)
                max_price[i][j]  = self.max_price(st_ind=ind[i],ind=j)
                idxmax_price[i][j]  = self.idxmax_price(st_ind=ind[i],ind=j)
                min_price[i][j]  = self.min_price(st_ind=ind[i],ind=j)
                idxmin_price[i][j]  = self.idxmin_price(st_ind=ind[i],ind=j)
                diff_percent[i][j] = self.buy_diff_percent(st_ind=ind[i],ind=j,spread = spread)

        result_array = ones((len(ind)+1,close_period), dtype=float)
        for i in range(len(ind)):
            for j in range(1,close_period,1):
                result_array[i][j] = result_array[i][j] * (1 + diff_percent[i][j])
        for j in range(1,close_period,1):
            for i in range(len(ind)):
                result_array[len(ind)][j] = result_array[len(ind)][j] * result_array[i][j]
        result_df = DataFrame(result_array)
        result_df.to_csv("check_buy_direction.csv")


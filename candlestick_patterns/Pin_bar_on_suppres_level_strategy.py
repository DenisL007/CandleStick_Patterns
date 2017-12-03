#!/usr/bin/python3



class Pin_Level_st(object):
#1.Determine Trend (Long/Short)
#use ZZ with big/small depth - calculate m from high/low
# (check ZZ vs O/T mcrtd)
#2.Find Support Resistance levels
#3.Find Pin bar/Inverted Pin bar


    def __init__(self,stock):
        self.st = stock
        self.run()
        print(self.st.df)
        self.st.plot_candles(pricing=self.st.df, title='Intc', technicals=[self.st.df['zz_10_3%'].interpolate(),self.st.df['zz_60_8%'].interpolate()])


    def run (self):
        self.st.zigzig(deviation = 8, backstep = 30, depth = 60)
        self.st.zigzig(deviation = 3, backstep = 1, depth = 10)

import os

# pay attention for how the  import need to be done
from candlestick_patterns.pull_data import ImportDataFrameGoogle
from candlestick_patterns.pull_data import ImportDataFrameYahoo

from candlestick_patterns.CountingTools import Tools_cl

#TODO - GUI for browser
#TODO - GUI(plot - bokeh) - for different trend definition "https://www.mql5.com/en/articles/136"

if __name__ == '__main__':
    # make sure that not all the names from data sheet file are working correctly
    stock_names = {1: ['MMM', '3M Co.'],
                   2: ['AAPL', 'ACE Limited'],
                   3: ['ABT', 'Abbott Laboratories'],
                   4: ['ANF', 'Abercrombie & Fitch Company'],
                   5: ['INTC', 'Intel Corp']}

    os.system("cls")
    #
    for key in stock_names:
        print(stock_names[key])
    print('\nWELCOME TO THE PROGRAM \n')

    list = "INTC"
    data_frame = ImportDataFrameGoogle(list)
    intc = Tools_cl(data_frame.il)


import os

# pay attention for how the  import need to be done
from candlestick_patterns.pull_data import ImportDataFrame
from candlestick_patterns.CountingTools import Tools_cl

#TODO - GUI for browser

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
    #kString = input(
    #print('Please select from the list below by enter the key value\n'))
    #k = int(kString)
    # a single key in a dictionary can refer to only a single value so create a list to get access for first value
    #list = stock_names[k]
    list = "INTC"
    data_frame = ImportDataFrame(list[0])
    intc = Tools_cl(data_frame.il)
    # thats how you access to data frame
    #print(data_frame.il)

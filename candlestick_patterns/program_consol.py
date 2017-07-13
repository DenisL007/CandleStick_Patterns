import os

from pull_data import ImportDataFrame

if __name__ == '__main__':
    # make sure that not all the names from data sheet file are working correctly
    stock_names = {1: ['MMM', '3M Co.'],
                   2: ['AAPL', 'ACE Limited'],
                   3: ['ABT', 'Abbott Laboratories'],
                   4: ['ANF', 'Abercrombie & Fitch Company']}

    os.system("cls")
    #
    for key in stock_names:
        print(stock_names[key])
    print('\nWELCOME TO THE PROGRAM \n')
    kString = input(
    print('Please select from the list below by enter the key value\n'))
    k = int(kString)
    # a single key in a dictionary can refer to only a single value so create a list to get access for first value
    list = stock_names[k]

    data_frame = ImportDataFrame(list[0])
    # thats how you access to data frame
    print(data_frame.df)

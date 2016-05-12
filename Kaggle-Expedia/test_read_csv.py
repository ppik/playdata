import os
import timeit
import sys

def time_read_csv(filename, numlines=10, numtries=10):
    s = 'pd.read_csv("{}", nrows={:d})'.format(filename, numlines)
    setup = 'import pandas as pd'
    return timeit.timeit(stmt=s, setup=setup, number=numtries)

if __name__=='__main__':
    filename = 'data/train.csv'
    numlines = 1000000
    numtries = 1

    filenames = [filename, filename+'.gz']
    times = []

    for filename in filenames:
        if os.path.exists(filename):
            time = time_read_csv(filename, numlines, numtries)
            print("{}: {:f} s".format(filename, time/numtries))
            times.append(time)
        else:
            times.append(float('nan'))

    print('Time increase: {:.0f}%'.format((times[-1]-times[0])/times[0]*100))

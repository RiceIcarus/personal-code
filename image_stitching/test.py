from pylab import *
import t1a1
import os
import pandas as pd


def similarity1(list0, list1):
    return mean(abs(list0 - list1))


def q1():
    imglist, imgnumlist = t1a1.readoriginimg('2')
    print(type(imglist[0]))
    print(similarity1(imglist[0][:, 0, 2], imglist[1][:, -1, 2]))


if __name__ == '__main__':
    # imglist, imgnumlist = t1a1.readoriginimg('4')
    # print(imgnumlist)
    a = array([[1, 2, 3], [3, 2, 1]])
    df1 = pd.DataFrame(a)
    df1.to_csv('movieinfo_fig.csv', index=False, encoding='gbk')

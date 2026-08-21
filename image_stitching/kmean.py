# k-mean
from pylab import *
import t1a1
import t1a4


# k均值聚类
def kmean2(data, i):
    minc, maxc = 0, 0.1
    t = 0
    while t < i:
        t += 1
        mind, maxd = [], []
        for each in data:
            if abs(each - minc) < abs(each - maxc):
                mind.append(each)
            else:
                maxd.append(each)
        minc, maxc = mean(mind), mean(maxd)
    return min(maxd), len(mind), len(maxd)


if __name__ == '__main__':
    imglist = t1a1.readoriginimg('4')
    imgnumlist = t1a1.readimgnum('4')
    prematchtab, simall = t1a4.presimmatch(imglist, imgnumlist)
    print(kmean2(simall, 8))

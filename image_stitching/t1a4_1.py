# 第四题   先拼行再拼列

from pylab import *
import t1a1
import t1a2


# 相似度1：两行每个像素两两对应相减的绝对值的平均值
def similarity1(list0, list1):
    return mean(abs(list0 - list1))


# # 相似度2：两行每个像素两两对应相减的平方的平均值
# def similarity1(list0, list1):
#     return mean([i ** 2 for i in list0 - list1])


# # 相似度3：每个像素三色中最大值两两对应相减的绝对值的平均值
# def similarity1(list0, list1):
#     list0max = array([max(i) for i in list0])
#     list1max = array([max(i) for i in list1])
#     return mean(abs(list0max - list1max))


def sizesort(imglist):
    sizel = [[[], [], []], [[], [], []], [[], [], []]]
    t = {320: 0, 400: 1, 480: 2}
    for i in range(len(imglist)):
        s = imglist[i].shape[:2]
        sizel[t[s[0]]][t[s[1]]].append(i)
    return sizel


def step1row(imglist, imgnumlist):
    imglist_row, imgnumlist_row = t1a2.bestsimmatch_joint_row(imglist, imgnumlist, 0.06)  # 设置相似度阈值
    t1a1.saveimg(imglist_row, 't1a4_1', 'step1row')
    return imgnumlist_row


def step1col(imgnumlist):
    imglist = t1a1.readimg('t1a4_1', 'step1row')
    imglist_col, imgnumlist_col = t1a2.bestsimmatch_joint_col(imglist, imgnumlist, 0.06)  # 设置相似度阈值
    t1a1.saveimg(imglist_col, 't1a4_1', 'step1col')
    return imgnumlist_col


def step2(imgnumlist):
    imglist = t1a1.readimg('t1a4_1', 'step1col')
    imglist_row, imgnumlist_row = t1a2.bestsimmatch_joint_row(imglist, imgnumlist, 0.08)  # 设置相似度阈值
    imglist_col, imgnumlist_col = t1a2.bestsimmatch_joint_col(imglist_row, imgnumlist_row, 0.08)
    t1a1.saveimg(imglist_col, 't1a4_1', 'step2')
    return imgnumlist_col


def step3(imgnumlist):
    imglist = t1a1.readimg('t1a4_1', 'step2')
    imglist_row1, imgnumlist_row1 = t1a2.bestsimmatch_joint_row(imglist, imgnumlist, 0.085)
    imglist_col1, imgnumlist_col1 = t1a2.bestsimmatch_joint_col(imglist_row1, imgnumlist_row1, 0.085)  # 设置相似度阈值
    imglist_row2, imgnumlist_row2 = t1a2.bestsimmatch_joint_row(imglist_col1, imgnumlist_col1, 0.085)
    imglist_col2, imgnumlist_col2 = t1a2.bestsimmatch_joint_col(imglist_row2, imgnumlist_row2, 0.085)
    t1a1.saveimg(imglist_col2, 't1a4_1', 'step3')
    return imgnumlist_col2


if __name__ == '__main__':
    imglist, imgnumlist = t1a1.readoriginimg('4')
    imgnumlist_row1 = step1row(imglist, imgnumlist)
    imgnumlist_col1 = step1col(imgnumlist_row1)
    imgnumlist2 = step2(imgnumlist_col1)
    imgnumlist3 = step3(imgnumlist2)

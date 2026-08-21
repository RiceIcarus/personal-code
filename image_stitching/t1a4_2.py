# 第四题

from pylab import *
import t1a1
import t1a2


# 相似度1：两行每个像素两两对应相减的绝对值的平均值
def similarity1(list0, list1):
    return mean(abs(list0 - list1))


# 预相似匹配 列
def presimmatch_col(imglist):
    prematchtab_col = zeros([len(imglist), 3, 2])  # 图片序号，每张图片的上、下最佳匹配图片序号
    for i in range(len(imglist)):
        bestmatch_up, bestmatch_down = [inf, -1], [inf, -1]  # 最佳匹配的相似度及图片序号
        for j in range(len(imglist)):
            if j != i and imglist[i][0, :, :].shape == imglist[j][-1, :, :].shape:  # 不是原图，列数相等
                sim_up1 = similarity1(imglist[i][0, :, :], imglist[j][-2, :, :])  # 计算两张图片相邻处的相似度
                sim_up2 = similarity1(imglist[i][1, :, :], imglist[j][-1, :, :])
                sim_up = mean([sim_up1, sim_up2])
                if sim_up < bestmatch_up[0]:  # 和当前最佳匹配的相似度比较，如果相似度更小
                    bestmatch_up = [sim_up, j]  # 则将它替换
                sim_down1 = similarity1(imglist[i][-1, :, :], imglist[j][1, :, :])
                sim_down2 = similarity1(imglist[i][-2, :, :], imglist[j][0, :, :])
                sim_down = mean([sim_down1, sim_down2])
                if sim_down < bestmatch_down[0]:
                    bestmatch_down = [sim_down, j]
        prematchtab_col[i] = [[i, 0], bestmatch_up, bestmatch_down]
        # [[图片序号, 是否已经拼接（0未拼，1已拼）], [上方最佳匹配图片相似度及序号], [下方最佳匹配图片相似度及序号]]
    return prematchtab_col


t1a2.presimmatch_col = presimmatch_col


# 预相似匹配 行
def presimmatch_row(imglist):
    prematchtab_row = zeros([len(imglist), 3, 2])  # 图片序号，每张图片的上、下最佳匹配图片序号
    for i in range(len(imglist)):
        bestmatch_left, bestmatch_right = [inf, -1], [inf, -1]  # 最佳匹配的相似度及图片序号
        for j in range(len(imglist)):
            if j != i and imglist[i][:, 0, :].shape == imglist[j][:, -1, :].shape:  # 不是原图，行数相等
                sim_left1 = similarity1(imglist[i][:, 0, :], imglist[j][:, -2, :])  # 计算两张图片相邻处的相似度
                sim_left2 = similarity1(imglist[i][:, 1, :], imglist[j][:, -1, :])
                sim_left = mean([sim_left1, sim_left2])
                if sim_left < bestmatch_left[0]:  # 和当前最佳匹配的相似度比较，如果相似度更小
                    bestmatch_left = [sim_left, j]  # 则将它替换
                sim_right1 = similarity1(imglist[i][:, -1, :], imglist[j][:, 1, :])
                sim_right2 = similarity1(imglist[i][:, -2, :], imglist[j][:, 0, :])
                sim_right = mean([sim_right1, sim_right2])
                if sim_right < bestmatch_right[0]:
                    bestmatch_right = [sim_right, j]
        prematchtab_row[i] = [[i, 0], bestmatch_left, bestmatch_right]
        # [[图片序号, 是否已经拼接（0未拼，1已拼）], [左方最佳匹配图片相似度及序号], [右方最佳匹配图片相似度及序号]]
    return prematchtab_row


t1a2.presimmatch_row = presimmatch_row


def step1row(imglist, imgnumlist):
    imglist_row, imgnumlist_row = t1a2.bestsimmatch_joint_row(imglist, imgnumlist, 0.06)  # 设置相似度阈值
    t1a1.saveimg(imglist_row, 't1a4', 'step1row')
    return imgnumlist_row


def step1col(imgnumlist):
    imglist = t1a1.readimg('t1a4', 'step1row')
    imglist_col, imgnumlist_col = t1a2.bestsimmatch_joint_col(imglist, imgnumlist, 0.06)  # 设置相似度阈值
    t1a1.saveimg(imglist_col, 't1a4', 'step1column')
    return imgnumlist_col


def step2(imgnumlist):
    imglist = t1a1.readimg('t1a4', 'step1column')
    imglist_row, imgnumlist_row = t1a2.bestsimmatch_joint_row(imglist, imgnumlist, 0.08)  # 设置相似度阈值
    imglist_col, imgnumlist_col = t1a2.bestsimmatch_joint_col(imglist_row, imgnumlist_row, 0.08)  # 设置相似度阈值
    t1a1.saveimg(imglist_col, 't1a4', 'step2')
    return imgnumlist_col


def step3(imgnumlist):
    imglist = t1a1.readimg('t1a4', 'step2')
    imglist_row, imgnumlist_row = t1a2.bestsimmatch_joint_row(imglist, imgnumlist, 0.085)  # 设置相似度阈值
    imglist_col, imgnumlist_col = t1a2.bestsimmatch_joint_col(imglist_row, imgnumlist_row, 0.085)  # 设置相似度阈值
    t1a1.saveimg(imglist_col, 't1a4', 'step3')
    return imgnumlist_col


def step4(imgnumlist):
    imglist = t1a1.readimg('t1a4', 'step3')
    imglist_row, imgnumlist_row = t1a2.bestsimmatch_joint_row(imglist, imgnumlist, 0.085)  # 设置相似度阈值
    imglist_col, imgnumlist_col = t1a2.bestsimmatch_joint_col(imglist_row, imgnumlist_row, 0.085)  # 设置相似度阈值
    t1a1.saveimg(imglist_col, 't1a4', 'step4')
    return imgnumlist_col


if __name__ == '__main__':
    imglist, imgnumlist = t1a1.readoriginimg('4')
    imgnumlist_row1 = step1row(imglist, imgnumlist)
    imgnumlist_col1 = step1col(imgnumlist_row1)
    # imgnumlist2 = step2(imgnumlist_col1)
    # imgnumlist3 = step3(imgnumlist2)
    # imgnumlist4 = step4(imgnumlist3)
    # print(imgnumlist2)

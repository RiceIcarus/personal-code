# 第四题

from pylab import *
import t1a1
import t1a2


# 相似度1：两行每个像素两两对应相减的绝对值的平均值
def similarity1(list0, list1):
    return mean(abs(list0 - list1))


# 预相似匹配（存最佳匹配）
def presimmatch(imglist):
    prematchtab = []  # 图片序号，每张图片的上、下、左、右最佳匹配图片序号
    for i in range(len(imglist)):
        bestmatch_up, bestmatch_down, bestmatch_left, bestmatch_right = [-1, inf], [-1, inf], [-1, inf], [-1, inf]
        for j in range(len(imglist)):
            if j != i:  # 不是原图
                if imglist[i].shape[1] == imglist[j].shape[1]:  # 列数相等
                    sim_up = similarity1(imglist[i][0, :, :], imglist[j][-1, :, :])  # 上边界相似度
                    if sim_up < bestmatch_up[1]:  # 相似度更低
                        bestmatch_up = [j, sim_up]  # 替换最佳匹配
                    sim_down = similarity1(imglist[i][-1, :, :], imglist[j][0, :, :])  # 下边界相似度
                    if sim_down < bestmatch_down[1]:
                        bestmatch_down = [j, sim_down]
                if imglist[i].shape[0] == imglist[j].shape[0]:  # 行数相等
                    sim_left = similarity1(imglist[i][:, 0, :], imglist[j][:, -1, :])  # 左边界相似度
                    if sim_left < bestmatch_left[1]:
                        bestmatch_left = [j, sim_left]
                    sim_right = similarity1(imglist[i][:, -1, :], imglist[j][:, 0, :])  # 右边界相似度
                    if sim_right < bestmatch_right[1]:
                        bestmatch_right = [j, sim_right]
        prematchtab.append([[i, 0], bestmatch_up, bestmatch_down, bestmatch_left, bestmatch_right])
        # [[图片编号, 是否已经拼接（0未拼，1已拼）], 上方最佳匹配图片序号及相似度, 下方, 左方, 右方]
    return prematchtab


# 预相似匹配（存前三匹配）
'''
def presimmatch(imglist, imgnumlist):
    prematchtab = []  # 图片序号，每张图片的上、下、左、右最佳匹配图片序号
    for i in range(len(imglist)):
        bestmatch_up_l, bestmatch_down_l, bestmatch_left_l, bestmatch_right_l = [], [], [], []  # 图片序号及相似度列表
        for j in range(len(imglist)):
            if j != i:
                if imglist[i][0, :, :].shape == imglist[j][-1, :, :].shape:  # 不是原图，列数相等
                    bestmatch_up_l.append([j, similarity1(imglist[i][0, :, :], imglist[j][-1, :, :])])  # [图片序号, 相似度]
                    bestmatch_down_l.append([j, similarity1(imglist[i][-1, :, :], imglist[j][0, :, :])])
                if imglist[i][:, 0, :].shape == imglist[j][:, -1, :].shape:  # 不是原图，行数相等
                    bestmatch_left_l.append([j, similarity1(imglist[i][:, 0, :], imglist[j][:, -1, :])])
                    bestmatch_right_l.append([j, similarity1(imglist[i][:, -1, :], imglist[j][:, 0, :])])
        bestmatch_up = sorted(bestmatch_up_l, key=lambda x: x[1])[:3]  # 前五最佳匹配（按照相似度排序）
        bestmatch_down = sorted(bestmatch_down_l, key=lambda x: x[1])[:3]
        bestmatch_left = sorted(bestmatch_left_l, key=lambda x: x[1])[:3]
        bestmatch_right = sorted(bestmatch_right_l, key=lambda x: x[1])[:3]
        prematchtab.append([[i, 0, imgnumlist[i]], bestmatch_up, bestmatch_down, bestmatch_left, bestmatch_right])
        # [[图片编号, 是否已经拼接（0未拼，1已拼）, 图片拼接序号表], [上方前三最佳匹配图片序号及相似度], [下], [左], [右]]
    return prematchtab
'''


# 预分类
def presort(prematchtab, simthr):
    rowlist, collist, unklist = [], [], []
    for img in prematchtab:
        if img[1][1] < simthr and img[2][1] < simthr and img[3][1] < simthr and img[4][1] < simthr:  # 四边全部小
            unklist.append(img)
        else:
            colsim = mean([img[1][1], img[2][1]])
            rowsim = mean([img[3][1], img[4][1]])
            if rowsim < colsim:  # 判断归于横拼组
                rowlist.append(img)
            else:  # 判断归于竖拼组
                collist.append(img)
    return rowlist, collist, unklist


# 最佳相似度匹配，预行拼接
def bestsimmatch_prejoint_row(prematchtab, rowlist, simthr):
    rowimgnolist = []
    for img in rowlist:  # 循环横拼列表的每一张图片
        if img[0][1] == 0:  # 如果当前图片尚未拼接
            img[0][1] = 1  # 设置为已拼接
            rowimgno = [img[0][0]]  # 设置拼接编号表
            img1 = img  # 设置当前拼接图片
            j = 1
            while j:  # 仍能向左拼接时循环
                img2 = prematchtab[img1[3][0]]  # 取出当前拼接图片向左拼接的最佳匹配图片
                if img1[3][0] != -1 and img2[0][1] == 0 and img2[4][0] == img1[0][0] and img1[3][1] < simthr:  # 互为最佳匹配
                    img2[0][1] = 1  # 设置为已拼接
                    rowimgno.insert(0, img2[0][0])  # 往左拼接
                    img1 = img2  # 更新当前拼接图片
                else:  # 不能拼时退出循环
                    j = 0
            img1 = img
            j = 1
            while j:  # 仍能向右拼接时循环
                img2 = prematchtab[img1[4][0]]  # 取出当前拼接图片向右拼接的最佳匹配图片
                if img1[4][0] != -1 and img2[0][1] == 0 and img2[3][0] == img1[0][0] and img1[4][1] < simthr:
                    img2[0][1] = 1
                    rowimgno.append(img2[0][0])  # 往右拼接
                    img1 = img2
                else:
                    j = 0
            rowimgnolist.append(rowimgno)
    return rowimgnolist


# 最终行拼接
def simmatch_joint_row(imglist, imgnumlist, prematchtab, rowimgnolist):
    rowimglist, rowimgnumlist = [], []
    for eachrow in rowimgnolist:
        if len(eachrow) == 5:  # 图片正好5张
            if sum([imglist[i].shape[1] for i in eachrow]) == 2000:  # 宽为2000，就拼起来
                rowimg = imglist[eachrow[0]]
                rowimgnum = imgnumlist[eachrow[0]]
                for k in eachrow[1:]:
                    rowimg = hstack((rowimg, imglist[k]))
                    rowimgnum = hstack((rowimgnum, imgnumlist[k]))
                rowimglist.append(rowimg)
                rowimgnumlist.append(rowimgnum)
            else:
                for k in eachrow:  # 归零
                    prematchtab[k][0][1] = 0
        elif len(eachrow) < 5:  # 图片小于5张
            for k in eachrow:  # 归零
                prematchtab[k][0][1] = 0
        elif len(eachrow) > 5:  # 图片大于5张
            t = 1
            for j in range(len(eachrow) - 5):
                if sum([imglist[i].shape[1] for i in eachrow[j:j + 5]]) == 2000:  # 找到宽为2000的5张图片，就拼起来
                    rowimg = imglist[eachrow[j]]
                    rowimgnum = imgnumlist[eachrow[j]]
                    for k in eachrow[j + 1:j + 5]:
                        rowimg = hstack((rowimg, imglist[k]))
                        rowimgnum = hstack((rowimgnum, imgnumlist[k]))
                    rowimglist.append(rowimg)
                    rowimgnumlist.append(rowimgnum)
                    del eachrow[j:j + 5]
                    for k in eachrow:
                        prematchtab[k][0][1] = 0
                    t = 0
                    break
            if t:
                for k in eachrow:  # 将其他的归零
                    prematchtab[k][0][1] = 0
    return rowimglist, rowimgnumlist


# 最佳相似度匹配，预列拼接
def bestsimmatch_prejoint_col(prematchtab, collist, simthr):
    colimgnolist = []
    for img in collist:  # 循环竖拼列表的每一张图片
        if img[0][1] == 0:  # 如果当前图片尚未拼接
            img[0][1] = 1  # 设置为已拼接
            colimgno = [img[0][0]]  # 设置拼接编号表
            img1 = img  # 设置当前拼接图片
            j = 1
            while j:  # 仍能向上拼接时循环
                img2 = prematchtab[img1[1][0]]  # 取出当前拼接图片向上拼接的最佳匹配图片
                if img1[1][0] != -1 and img2[0][1] == 0 and img2[2][0] == img1[0][0] and img1[1][1] < simthr:  # 互为最佳匹配
                    img2[0][1] = 1  # 设置为已拼接
                    colimgno.insert(0, img2[0][0])  # 往上拼接
                    img1 = img2  # 更新当前拼接图片
                else:  # 不能拼时退出循环
                    j = 0
            img1 = img
            j = 1
            while j:  # 仍能向下拼接时循环
                img2 = prematchtab[img1[2][0]]  # 取出当前拼接图片向下拼接的最佳匹配图片
                if img1[2][0] != -1 and img2[0][1] == 0 and img2[1][0] == img1[0][0] and img1[2][1] < simthr:
                    img2[0][1] = 1
                    colimgno.append(img2[0][0])  # 往下拼接
                    img1 = img2
                else:
                    j = 0
            colimgnolist.append(colimgno)
    return colimgnolist


# 最终列拼接
def simmatch_joint_col(imglist, imgnumlist, prematchtab, colimgnolist):
    colimglist, colimgnumlist = [], []
    for eachcol in colimgnolist:
        if len(eachcol) == 5:  # 图片正好5张
            if sum([imglist[i].shape[0] for i in eachcol]) == 2000:  # 宽为2000，就重新拼起来
                colimg = imglist[eachcol[0]]
                colimgnum = imgnumlist[eachcol[0]]
                for k in eachcol[1:]:
                    colimg = vstack((colimg, imglist[k]))
                    colimgnum = vstack((colimgnum, imgnumlist[k]))
                colimglist.append(colimg)
                colimgnumlist.append(colimgnum)
            else:
                for k in eachcol:  # 归零
                    prematchtab[k][0][1] = 0
        elif len(eachcol) < 5:  # 图片小于5张
            for k in eachcol:  # 归零
                prematchtab[k][0][1] = 0
        elif len(eachcol) > 5:  # 图片大于5张
            t = 1
            for j in range(len(eachcol) - 5):
                if sum([imglist[i].shape[0] for i in eachcol[j:j + 5]]) == 2000:  # 找到宽为2000的5张图片，就拼起来
                    colimg = imglist[eachcol[j]]
                    colimgnum = imgnumlist[eachcol[j]]
                    for k in eachcol[j + 1:j + 5]:
                        colimg = vstack((colimg, imglist[k]))
                        colimgnum = vstack((colimgnum, imgnumlist[k]))
                    colimglist.append(colimg)
                    colimgnumlist.append(colimgnum)
                    del eachcol[j:j + 5]
                    for k in eachcol:
                        prematchtab[k][0][1] = 0
                    t = 0
                    break
            if t:
                for k in eachcol:  # 将其他的归零
                    prematchtab[k][0][1] = 0
    return colimglist, colimgnumlist


def q0():
    imglist, imgnumlist = t1a1.readoriginimg('4')
    prematchtab = presimmatch(imglist)
    rowlist, collist, unklist = presort(prematchtab, 0.04)
    return imglist, imgnumlist, prematchtab, rowlist, collist


def q1_row(imglist, imgnumlist, prematchtab, rowlist):
    rowimgnolist = bestsimmatch_prejoint_row(prematchtab, rowlist, 0.1)
    rowimglist, rowimgnumlist = simmatch_joint_row(imglist, imgnumlist, prematchtab, rowimgnolist)
    t1a1.saveimg(rowimglist, 't1a4', 'step1row')
    return rowimgnumlist


def q2_row(rowimgnumlist):
    rowimglist = t1a1.readimg('t1a4', 'step1row')
    imglist_row, imgnumlist_row = t1a2.bestsimmatch_joint_col(rowimglist, rowimgnumlist, 0.1)
    t1a1.saveimg(imglist_row, 't1a4', 'step2row')
    return imgnumlist_row


def q1_col(imglist, imgnumlist, prematchtab, collist):
    colimgnolist = bestsimmatch_prejoint_col(prematchtab, collist, 0.1)
    colimglist, colimgnumlist = simmatch_joint_col(imglist, imgnumlist, prematchtab, colimgnolist)
    t1a1.saveimg(colimglist, 't1a4', 'step1col')
    return colimgnumlist


def q2_col(colimgnumlist):
    colimglist = t1a1.readimg('t1a4', 'step1col')
    imglist_col, imgnumlist_col = t1a2.bestsimmatch_joint_row(colimglist, colimgnumlist, 0.1)
    t1a1.saveimg(imglist_col, 't1a4', 'step2col')
    return imgnumlist_col


def q3(imgnumlist_row, imgnumlist_col):
    imglist_row = t1a1.readimg('t1a4', 'step2row')
    imglist_col = t1a1.readimg('t1a4', 'step2col')
    imglist = imglist_row + imglist_col
    imgnumlist = imgnumlist_row + imgnumlist_col
    imglist1, imgnumlist1 = t1a2.bestsimmatch_joint_row(imglist, imgnumlist, 0.1)
    imglist2, imgnumlist2 = t1a2.bestsimmatch_joint_col(imglist1, imgnumlist1, 0.1)
    t1a1.saveimg(imglist2, 't1a4', 'step3')
    return imgnumlist2


if __name__ == '__main__':
    imglist, imgnumlist, prematchtab, rowlist, collist = q0()
    rowimgnumlist = q1_row(imglist, imgnumlist, prematchtab, rowlist)
    imgnumlist_row = q2_row(rowimgnumlist)
    colimgnumlist = q1_col(imglist, imgnumlist, prematchtab, collist)
    imgnumlist_col = q2_col(colimgnumlist)
    imgnumlist = q3(imgnumlist_row, imgnumlist_col)

    # imglist = t1a1.readimg('t1a4', 'step3')
    # for img in imglist:
    #     print(img.shape)
    # img1 = vstack((imglist[2], imglist[1]))
    # imshow(img1), axis('off')
    # show()

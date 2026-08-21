# 第二题

from pylab import *
import cv2
import t1a1


# 相似度1：两行每个像素两两对应相减的绝对值的平均值
def similarity(list0, list1):
    return mean(cv2.absdiff(list0, list1))


# 预相似匹配 行
def presimmatch_row(imgl):
    prematchtab_row = []
    for i in range(len(imgl)):
        bestmatch_left, bestmatch_right = [-1, inf], [-1, inf]  # 最佳匹配的图片编号及相似度
        for j in range(len(imgl)):
            if j != i and imgl[i].shape[0] == imgl[j].shape[0]:  # 不是原图，行数相等
                sim_left = similarity(imgl[i][:, 0, :], imgl[j][:, -1, :])  # 计算左边界相似度
                if sim_left < bestmatch_left[1]:  # 和当前最佳匹配的相似度比较，如果相似度更小
                    bestmatch_left = [j, sim_left]  # 则将它替换
                sim_right = similarity(imgl[i][:, -1, :], imgl[j][:, 0, :])  # 计算右边界相似度
                if sim_right < bestmatch_right[1]:
                    bestmatch_right = [j, sim_right]
        prematchtab_row.append([[i, 0], bestmatch_left, bestmatch_right])
        # [[图片编号, 拼接状态（0未拼，1已拼）], 左方最佳匹配图片编号及相似度, 右方]
    return prematchtab_row


# 最佳相似度匹配，横向拼接图片及序号
def bestsimmatch_joint_row(imgl, imgnuml, simthr):
    prematchtab_row = presimmatch_row(imgl)
    imgl_row, imgnuml_row = [], []
    for img in prematchtab_row:  # 循环列表的每一张图片
        if img[0][1] == 0:  # 如果当前图片尚未拼接
            img[0][1] = 1  # 设置为已拼接
            img0, img0num = imgl[img[0][0]], imgnuml[img[0][0]]  # 设置行拼接的中心图片
            img1 = img  # 设置当前拼接图片
            j = 1
            while j:  # 仍能向左拼接时循环
                img2 = prematchtab_row[img1[1][0]]  # 取出当前拼接图片向左拼接的最佳匹配图片
                if img1[1][0] != -1 and img2[0][1] == 0 and img2[2][0] == img1[0][0] and img1[1][1] < simthr:  # 互为最佳匹配
                    img2[0][1] = 1  # 设置为已拼接
                    img0 = hstack((imgl[img2[0][0]], img0))  # 将图片往左拼接
                    img0num = hstack((imgnuml[img2[0][0]], img0num))  # 生成拼接序号表
                    img1 = img2  # 更新当前拼接图片
                else:  # 不能拼时退出循环
                    j = 0
            img1 = img
            j = 1
            while j:  # 仍能向右拼接时循环
                img2 = prematchtab_row[img1[2][0]]  # 取出当前拼接图片向右拼接的最佳匹配图片序号
                if img1[2][0] != -1 and img2[0][1] == 0 and img2[1][0] == img1[0][0] and img1[2][1] < simthr:
                    img2[0][1] = 1
                    img0 = hstack((img0, imgl[img2[0][0]]))  # 将图片往右拼接
                    img0num = hstack((img0num, imgnuml[img2[0][0]]))
                    img1 = img2
                else:
                    j = 0
            imgl_row.append(img0)
            imgnuml_row.append(img0num)
    return imgl_row, imgnuml_row


# 预相似匹配 列
def presimmatch_col(imgl):
    prematchtab_col = []
    for i in range(len(imgl)):
        bestmatch_up, bestmatch_down = [-1, inf], [-1, inf]  # 最佳匹配的图片编号及相似度
        for j in range(len(imgl)):
            if j != i and imgl[i].shape[1] == imgl[j].shape[1]:  # 不是原图，列数相等
                sim_up = similarity(imgl[i][0, :, :], imgl[j][-1, :, :])  # 计算上边界相似度
                if sim_up < bestmatch_up[1]:  # 和当前最佳匹配的相似度比较，如果相似度更小
                    bestmatch_up = [j, sim_up]  # 则将它替换
                sim_down = similarity(imgl[i][-1, :, :], imgl[j][0, :, :])  # 计算下边界相似度
                if sim_down < bestmatch_down[1]:
                    bestmatch_down = [j, sim_down]
        prematchtab_col.append([[i, 0], bestmatch_up, bestmatch_down])
        # [[图片编号, 拼接状态（0未拼，1已拼）], 上方最佳匹配图片编号及相似度, 下方]
    return prematchtab_col


# 最佳相似度匹配，纵向拼接图片及序号
def bestsimmatch_joint_col(imgl, imgnuml, simthr):
    prematchtab_col = presimmatch_col(imgl)
    imgl_col, imgnuml_col = [], []
    for img in prematchtab_col:  # 循环列表的每一张图片
        if img[0][1] == 0:  # 如果当前图片尚未拼接
            img[0][1] = 1  # 设置为已拼接
            img0, img0num = imgl[img[0][0]], imgnuml[img[0][0]]  # 设置拼接列的中心图片
            img1 = img  # 设置当前拼接图片
            j = 1
            while j:  # 仍能向上拼接时循环
                img2 = prematchtab_col[img1[1][0]]  # 取出当前拼接图片向上拼接的最佳匹配图片序号
                if img1[1][0] != -1 and img2[0][1] == 0 and img2[2][0] == img1[0][0] and img1[1][1] < simthr:  # 互为最佳匹配
                    img2[0][1] = 1  # 设置为已拼接
                    img0 = vstack((imgl[img2[0][0]], img0))  # 将图片往上拼接
                    img0num = vstack((imgnuml[img2[0][0]], img0num))  # 生成拼接序号表
                    img1 = img2  # 更新当前拼接图片
                else:  # 不能拼时退出循环
                    j = 0
            img1 = img
            j = 1
            while j:  # 仍能向下拼接时循环
                img2 = prematchtab_col[img1[2][0]]  # 取出当前拼接图片向下拼接的最佳匹配图片序号
                if img1[2][0] != -1 and img2[0][1] == 0 and img2[1][0] == img1[0][0] and img1[2][1] < simthr:
                    img2[0][1] = 1
                    img0 = vstack((img0, imgl[img2[0][0]]))  # 将图片往下拼接
                    img0num = vstack((img0num, imgnuml[img2[0][0]]))
                    img1 = img2
                else:
                    j = 0
            imgl_col.append(img0)
            imgnuml_col.append(img0num)
    return imgl_col, imgnuml_col


def q1(imgl, imgnuml, simthr):
    imgl_col, imgnuml_col = bestsimmatch_joint_col(imgl, imgnuml, simthr)
    t1a1.saveimg(imgl_col, 't1a2', 'column')
    return imgnuml_col


def q2(imgnuml, simthr):
    imgl = t1a1.readimg('t1a2', 'column')
    imgl_res, imgnuml_res = bestsimmatch_joint_row(imgl, imgnuml, simthr)
    t1a1.saveimg(imgl_res, 't1a2', 'result')
    t1a1.savematchtab(imgnuml_res[0], 't1a2')
    return imgl_res, imgnuml_res


if __name__ == '__main__':
    imgl, imgnuml = t1a1.readoriginimg('2')
    simthr = 0.05  # 相似度阈值

    imgnuml_col = q1(imgl, imgnuml, simthr)
    imgl_res, imgnuml_res = q2(imgnuml_col, simthr)

    # imgl_col, imgnuml_col = bestsimmatch_joint_col(imgl, imgnuml, simthr)
    # imgl_res, imgnuml_res = bestsimmatch_joint_row(imgl_col, imgnuml_col, simthr)
    # imshow(imgl_res[0]), axis('off')
    # show()

    # print(imgnuml_res, len(imgl_res))

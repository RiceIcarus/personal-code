# 第二题  拼接图片，生成图片序号拼接表

import t1a1


# 读取图片
def readimg(filename, name):
    imglist = []  # 声明一个存储待拼接图片的列表
    for each in os.listdir(f'{filename}\\{name}'):
        imglist.append(imread(f'{filename}\\{name}\\{each}'))
    return imglist


# 相似度1：两行每个像素两两对应相减的绝对值的平均值
def similarity1(list0, list1):
    return mean(abs(list0 - list1))


# 相似度匹配，纵向拼接图片及序号
def simmatchjoint_col(imglist, imgnumlist, simthreshold):
    imglist_col, imgnumlist_col = [], []
    while imglist:  # 还有未拼接的图片时循环
        img0, img0num = imglist.pop(), imgnumlist.pop()
        j = 1
        while j:
            bestmatch = [inf, -1]  # 最佳匹配的相似度及图片序号
            for i in range(len(imglist)):
                if img0[0, :, :].shape[0] == imglist[i][-1, :, :].shape[0]:  # 当底片第一行的列数和循环图片最后一行的列数相等
                    sim = similarity1(img0[0, :, :], imglist[i][-1, :, :])  # 计算底片第一行和循环图片最后一行的相似度
                    if sim < bestmatch[0]:  # 找出相似度更高的图片
                        bestmatch = [sim, i]  # 最佳匹配替换为它
            if bestmatch[0] < simthreshold:  # 设定相似度阈值，小于阈值判断为相似
                img0 = vstack((imglist[bestmatch[1]], img0))  # 将图片进行纵向拼接
                img0num = vstack((imgnumlist[bestmatch[1]], img0num))  # 按同样的顺序生成拼接顺序
                del imglist[bestmatch[1]], imgnumlist[bestmatch[1]]
            else:
                j = 0
        j = 1
        while j:
            bestmatch = [inf, -1]  # 最佳匹配的相似度及图片序号
            for i in range(len(imglist)):
                if img0[-1, :, :].shape[0] == imglist[i][0, :, :].shape[0]:  # 当底片最后一行的列数和循环图片第一行的列数相等
                    sim = similarity1(img0[-1, :, :], imglist[i][0, :, :])  # 计算底片最后一行和循环图片第一行的相似度
                    if sim < bestmatch[0]:
                        bestmatch = [sim, i]
            if bestmatch[0] < simthreshold:
                img0 = vstack((img0, imglist[bestmatch[1]]))
                img0num = vstack((img0num, imgnumlist[bestmatch[1]]))
                del imglist[bestmatch[1]], imgnumlist[bestmatch[1]]
            else:
                j = 0
        imglist_col.append(img0)
        imgnumlist_col.append(img0num)
    return imglist_col, imgnumlist_col


# 相似度匹配，横向拼接图片及序号
def simmatchjoint_row(imglist, imgnumlist, simthreshold):
    imglist_row, imgnumlist_row = [], []
    while imglist:  # 还有未拼接的图片时循环
        img0, img0num = imglist.pop(), imgnumlist.pop()
        j = 1
        while j:
            bestmatch = [inf, -1]  # 最佳匹配的相似度及图片序号
            for i in range(len(imglist)):
                if img0[:, 0, :].shape[0] == imglist[i][:, -1, :].shape[0]:
                    sim = similarity1(img0[:, 0, :], imglist[i][:, -1, :])  # 计算底片第一列和循环图片最后一列的相似度
                    if sim < bestmatch[0]:  # 找出相似度更高的图片进行替换
                        bestmatch = [sim, i]
            if bestmatch[0] < simthreshold:  # 设定相似度阈值，小于阈值判断为相似
                img0 = hstack((imglist[bestmatch[1]], img0))  # 将图片进行横向拼接
                img0num = hstack((imgnumlist[bestmatch[1]], img0num))  # 将同样的顺序生成拼接顺序
                del imglist[bestmatch[1]], imgnumlist[bestmatch[1]]
            else:
                j = 0
        j = 1
        while j:
            bestmatch = [inf, -1]  # 最佳匹配的相似度及图片序号
            for i in range(len(imglist)):
                if img0[:, -1, :].shape[0] == imglist[i][:, 0, :].shape[0]:
                    sim = similarity1(img0[:, -1, :], imglist[i][:, 0, :])  # 计算底片最后一列和循环图片第一列的相似度
                    if sim < bestmatch[0]:  # 找出相似度更高的图片进行替换
                        bestmatch = [sim, i]
            if bestmatch[0] < simthreshold:  # 设定相似度阈值，小于阈值判断为相似
                img0 = hstack((img0, imglist[bestmatch[1]]))  # 将图片进行横向拼接
                img0num = hstack((img0num, imgnumlist[bestmatch[1]]))  # 将同样的顺序生成拼接顺序
                del imglist[bestmatch[1]], imgnumlist[bestmatch[1]]
            else:
                j = 0
        imglist_row.append(img0)
        imgnumlist_row.append(img0num)
    return imglist_row, imgnumlist_row


def q1(imgnumlist):
    imglist = readoriginimg('2')
    imglist_col, imgnumlist_col = simmatchjoint_col(imglist, imgnumlist, 0.05)  # 设置相似度为0.033
    saveimg(imglist_col, 't1a2_1', 'column')
    return imgnumlist_col


# 人工干预
def q2(imgnumlist_col):
    imglist = readimg('t1a2_1', 'column')
    imglist[5] = vstack((imglist[12], imglist[5]))
    imgnumlist_col[5] = vstack((imgnumlist_col[12], imgnumlist_col[5]))
    del imglist[12], imgnumlist_col[12]
    saveimg(imglist, 't1a2_1', 'column_intervened')
    return imgnumlist_col


def q3(imgnumlist_col):
    imglist = readimg('t1a2_1', 'column_intervened')
    imglist_res, imgnumlist_res = simmatchjoint_row(imglist, imgnumlist_col, 0.05)  # 设置相似度为0.033
    saveimg(imglist_res, 't1a2_1', 'result')
    savematchtab(imgnumlist_res[0], 't1a2_1')
    return imgnumlist_res


if __name__ == '__main__':
    imgnumlist = readimgnum('2')
    imgnumlist_col = q1(imgnumlist)
    imgnumlist_col = q2(imgnumlist_col)
    imgnumlist_res = q3(imgnumlist_col)

    # imglist = readoriginimg('2')
    # imgnumlist = readimgnum('2')
    # imglist_col, imgnumlist_col = simmatchjoint_col(imglist, imgnumlist, 0.033)
    # imglist_col[5] = vstack((imglist_col[12], imglist_col[5]))
    # imgnumlist_col[5] = vstack((imgnumlist_col[12], imgnumlist_col[5]))
    # del imglist_col[12], imgnumlist_col[12]
    # imglist_res, imgnumlist_res = simmatchjoint_row(imglist_col, imgnumlist_col, 0.033)
    # print(imgnumlist_res, len(imglist_res))
    # imshow(imglist_res[0]), axis('off')
    # show()

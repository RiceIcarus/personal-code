# 第四题

from pylab import *
import t1a1
import t1a2


# 最佳相似度匹配，预行拼接
def bestsimmatch_prejoint_row(imgl, simthr):
    prematchtab_row = t1a2.presimmatch_row(imgl)
    rowimgnol = []
    for img in prematchtab_row:  # 循环拼接列表的每一张图片
        if img[0][1] == 0:  # 如果当前图片尚未拼接
            img[0][1] = 1  # 设置为已拼接
            rowimgno = [img[0][0]]  # 设置拼接编号表
            img1 = img  # 设置当前拼接图片
            j = 1
            while j:  # 仍能向左拼接时循环
                img2 = prematchtab_row[img1[1][0]]  # 取出当前拼接图片向左拼接的最佳匹配图片
                if img1[1][0] != -1 and img2[0][1] == 0 and img2[2][0] == img1[0][0] and img1[1][1] < simthr:  # 互为最佳匹配
                    img2[0][1] = 1  # 设置为已拼接
                    rowimgno.insert(0, img2[0][0])  # 往左拼接
                    img1 = img2  # 更新当前拼接图片
                else:  # 不能拼时退出循环
                    j = 0
            img1 = img
            j = 1
            while j:  # 仍能向右拼接时循环
                img2 = prematchtab_row[img1[2][0]]  # 取出当前拼接图片向右拼接的最佳匹配图片
                if img1[2][0] != -1 and img2[0][1] == 0 and img2[1][0] == img1[0][0] and img1[2][1] < simthr:
                    img2[0][1] = 1
                    rowimgno.append(img2[0][0])  # 往右拼接
                    img1 = img2
                else:
                    j = 0
            rowimgnol.append(rowimgno)
    return rowimgnol


# 最终行拼接
def simmatch_joint_row(imgl, imgnuml, simthr):
    rowimgnol = bestsimmatch_prejoint_row(imgl, simthr)
    rowimgl, rowimgnuml = [], []
    for eachrow in rowimgnol:
        if len(eachrow) == 5:  # 图片正好5张
            if sum([imgl[i].shape[1] for i in eachrow]) == 2000:  # 宽为2000，就拼起来
                rowimg = imgl[eachrow[0]]
                rowimgnum = imgnuml[eachrow[0]]
                for k in eachrow[1:]:  # 从左往右拼
                    rowimg = hstack((rowimg, imgl[k]))
                    rowimgnum = hstack((rowimgnum, imgnuml[k]))
                rowimgl.append(rowimg)
                rowimgnuml.append(rowimgnum)
        elif len(eachrow) > 5:  # 图片大于5张
            t = []
            for j in range(len(eachrow) - 5):
                if sum([imgl[i].shape[1] for i in eachrow[j:j + 5]]) == 2000:  # 找到所有宽为2000的5张图片，拼起来
                    t.append(j)
            for j in t:
                rowimg = imgl[eachrow[j]]
                rowimgnum = imgnuml[eachrow[j]]
                for k in eachrow[j + 1:j + 5]:
                    rowimg = hstack((rowimg, imgl[k]))
                    rowimgnum = hstack((rowimgnum, imgnuml[k]))
                rowimgl.append(rowimg)
                rowimgnuml.append(rowimgnum)
    return rowimgl, rowimgnuml


# 最佳相似度匹配，预列拼接
def bestsimmatch_prejoint_col(imgl, simthr):
    prematchtab_col = t1a2.presimmatch_col(imgl)
    colimgnol = []
    for img in prematchtab_col:  # 循环拼接列表的每一张图片
        if img[0][1] == 0:  # 如果当前图片尚未拼接
            img[0][1] = 1  # 设置为已拼接
            colimgno = [img[0][0]]  # 设置拼接编号表
            img1 = img  # 设置当前拼接图片
            j = 1
            while j:  # 仍能向上拼接时循环
                img2 = prematchtab_col[img1[1][0]]  # 取出当前拼接图片向上拼接的最佳匹配图片
                if img1[1][0] != -1 and img2[0][1] == 0 and img2[2][0] == img1[0][0] and img1[1][1] < simthr:  # 互为最佳匹配
                    img2[0][1] = 1  # 设置为已拼接
                    colimgno.insert(0, img2[0][0])  # 往上拼接
                    img1 = img2  # 更新当前拼接图片
                else:  # 不能拼时退出循环
                    j = 0
            img1 = img
            j = 1
            while j:  # 仍能向下拼接时循环
                img2 = prematchtab_col[img1[2][0]]  # 取出当前拼接图片向下拼接的最佳匹配图片
                if img1[2][0] != -1 and img2[0][1] == 0 and img2[1][0] == img1[0][0] and img1[2][1] < simthr:
                    img2[0][1] = 1
                    colimgno.append(img2[0][0])  # 往下拼接
                    img1 = img2
                else:
                    j = 0
            colimgnol.append(colimgno)
    return colimgnol


# 最终列拼接
def simmatch_joint_col(imgl, imgnuml, simthr):
    colimgnol = bestsimmatch_prejoint_col(imgl, simthr)
    colimgl, colimgnuml = [], []
    for eachcol in colimgnol:
        if len(eachcol) == 5:  # 图片正好5张
            if sum([imgl[i].shape[0] for i in eachcol]) == 2000:  # 宽为2000，就重新拼起来
                colimg = imgl[eachcol[0]]
                colimgnum = imgnuml[eachcol[0]]
                for k in eachcol[1:]:  # 从上往下拼
                    colimg = vstack((colimg, imgl[k]))
                    colimgnum = vstack((colimgnum, imgnuml[k]))
                colimgl.append(colimg)
                colimgnuml.append(colimgnum)
        elif len(eachcol) > 5:  # 图片大于5张
            t = []
            for j in range(len(eachcol) - 5):
                if sum([imgl[i].shape[0] for i in eachcol[j:j + 5]]) == 2000:  # 找到宽为2000的5张图片，就拼起来
                    t.append(j)
            for j in t:
                colimg = imgl[eachcol[j]]
                colimgnum = imgnuml[eachcol[j]]
                for k in eachcol[j + 1:j + 5]:
                    colimg = vstack((colimg, imgl[k]))
                    colimgnum = vstack((colimgnum, imgnuml[k]))
                colimgl.append(colimg)
                colimgnuml.append(colimgnum)
    return colimgl, colimgnuml


# 输出各种尺寸图片数量
def imgshapesort(imgl):
    imgshapel = [[[], [], []], [[], [], []], [[], [], []]]
    t = {320: 0, 400: 1, 480: 2}
    tr = {0: 320, 1: 400, 2: 480}
    for i in range(len(imgl)):
        s = imgl[i].shape[:2]
        imgshapel[t[s[0]]][t[s[1]]].append(i)
    for j in range(len(imgshapel)):
        for k in range(len(imgshapel[0])):
            print(f'({tr[j]}, {tr[k]}):', len(imgshapel[j][k]), end='    ')
        print('\n')
    return imgshapel


# 先横拼再竖拼，找出所有2000*2000的图片
def step1_row(imgl, imgnuml, simthr):
    rowimgl, rowimgnuml = simmatch_joint_row(imgl, imgnuml, simthr)
    imgl_1row, imgnuml_1row = simmatch_joint_col(rowimgl, rowimgnuml, simthr)
    t1a1.saveimg(imgl_1row, 't1a4', 'step1row')
    return imgnuml_1row


# 先竖拼再横拼，找出所有2000*2000的图片
def step1_col(imgl, imgnuml, simthr):
    colimgl, colimgnuml = simmatch_joint_col(imgl, imgnuml, simthr)
    imgl_1col, imgnuml_1col = simmatch_joint_row(colimgl, colimgnuml, simthr)
    t1a1.saveimg(imgl_1col, 't1a4', 'step1col')
    return imgnuml_1col


# 去掉上述已经拼好的图片，将剩下的图片在进行拼接
def step1_r(imgnuml_1row, imgnuml_1col):
    imgl, imgnuml = t1a1.readoriginimg('4')
    imgl_1row = t1a1.readimg('t1a4', 'step1row')
    imgl_1col = t1a1.readimg('t1a4', 'step1col')
    imgl_1res, imgnuml_1res = imgl_1row + imgl_1col, imgnuml_1row + imgnuml_1col
    l1 = list(ones(len(imgl)))
    for i in set(flatten(imgnuml_1res)):
        l1[i - 1] = 0
    imgl_2, imgnuml_2 = [], []
    for i in range(len(l1)):
        if l1[i]:
            imgl_2.append(imgl[i])
            imgnuml_2.append(imgnuml[i])
    t1a1.saveimg(imgl_1res, 't1a4', 'step1res')
    t1a1.saveimg(imgl_2, 't1a4', 'step2')
    return imgnuml_1res, imgl_2, imgnuml_2


# 先横拼再竖拼
def step2_row(imgl_2, imgnuml_2, simthr):
    rowimgl, rowimgnuml = simmatch_joint_row(imgl_2, imgnuml_2, simthr)
    imgl_2row, imgnuml_2row = t1a2.bestsimmatch_joint_col(rowimgl, rowimgnuml, simthr)
    t1a1.saveimg(imgl_2row, 't1a4', 'step2row')
    return imgnuml_2row


# 再去除已经拼好的
def step2_interv(imgl, imgnuml, imgnuml_1res, imgnuml_2row):
    l2 = list(ones(len(imgl)))
    for i in set(flatten(imgnuml_1res + imgnuml_2row)):
        l2[i - 1] = 0
    imgl_2i, imgnuml_2i = [], []
    for i in range(len(l2)):
        if l2[i]:
            imgl_2i.append(imgl[i])
            imgnuml_2i.append(imgnuml[i])
    return imgl_2i, imgnuml_2i


# 先竖拼再横拼
def step2_col(imgl_2i, imgnuml_2i, simthr):
    colimgl, colimgnuml = t1a2.bestsimmatch_joint_col(imgl_2i, imgnuml_2i, simthr)
    imgl_2col, imgnuml_2col = t1a2.bestsimmatch_joint_row(colimgl, colimgnuml, simthr)
    t1a1.saveimg(imgl_2col, 't1a4', 'step2col')
    return imgnuml_2col


# 最后将所有图片拼接起来
def step3(imgnuml_1res, imgnuml_2row, imgnuml_2col, simthr):
    imgl_1res = t1a1.readimg('t1a4', 'step1res')
    imgl_2row = t1a1.readimg('t1a4', 'step2row')
    imgl_2col = t1a1.readimg('t1a4', 'step2col')
    imgl_3 = imgl_1res + imgl_2row + imgl_2col
    imgnuml_3 = imgnuml_1res + imgnuml_2row + imgnuml_2col
    imgl_3col, imgnuml_3col = t1a2.bestsimmatch_joint_col(imgl_3, imgnuml_3, simthr)
    imgl_res, imgnuml_res = t1a2.bestsimmatch_joint_row(imgl_3col, imgnuml_3col, simthr)
    t1a1.saveimg(imgl_res, 't1a4', 'step3')
    t1a1.savematchtab(imgnuml_res[0], 't1a4')
    return imgl_res, imgnuml_res


# 完整过程
def step0(imgl, imgnuml, simthr):
    rowimgl, rowimgnuml = simmatch_joint_row(imgl, imgnuml, simthr)
    imgl_1row, imgnuml_1row = simmatch_joint_col(rowimgl, rowimgnuml, simthr)
    colimgl, colimgnuml = simmatch_joint_col(imgl, imgnuml, simthr)
    imgl_1col, imgnuml_1col = simmatch_joint_row(colimgl, colimgnuml, simthr)
    imgl_1res, imgnuml_1res = imgl_1row + imgl_1col, imgnuml_1row + imgnuml_1col
    l1 = list(ones(len(imgl)))
    for i in set(flatten(imgnuml_1res)):
        l1[i - 1] = 0
    imgl_2, imgnuml_2 = [], []
    for i in range(len(l1)):
        if l1[i]:
            imgl_2.append(imgl[i])
            imgnuml_2.append(imgnuml[i])
    rowimgl, rowimgnuml = simmatch_joint_row(imgl_2, imgnuml_2, simthr)
    imgl_2row, imgnuml_2row = t1a2.bestsimmatch_joint_col(rowimgl, rowimgnuml, simthr)
    l2 = list(ones(len(imgl)))
    for i in set(flatten(imgnuml_1res + imgnuml_2row)):
        l2[i - 1] = 0
    imgl_2i, imgnuml_2i = [], []
    for i in range(len(l2)):
        if l2[i]:
            imgl_2i.append(imgl[i])
            imgnuml_2i.append(imgnuml[i])
    colimgl, colimgnuml = t1a2.bestsimmatch_joint_col(imgl_2i, imgnuml_2i, simthr)
    imgl_2col, imgnuml_2col = t1a2.bestsimmatch_joint_row(colimgl, colimgnuml, simthr)
    imgl_3 = imgl_1res + imgl_2row + imgl_2col
    imgnuml_3 = imgnuml_1res + imgnuml_2row + imgnuml_2col
    imgl_3col, imgnuml_3col = t1a2.bestsimmatch_joint_col(imgl_3, imgnuml_3, simthr)
    imgl_res, imgnuml_res = t1a2.bestsimmatch_joint_row(imgl_3col, imgnuml_3col, simthr)
    return imgl_res, imgnuml_res


if __name__ == '__main__':
    imgl, imgnuml = t1a1.readoriginimg('4')
    simthr = 0.1  # 相似度阈值

    imgnuml_1row = step1_row(imgl, imgnuml, simthr)
    imgnuml_1col = step1_col(imgl, imgnuml, simthr)
    imgnuml_1res, imgl_2, imgnuml_2 = step1_r(imgnuml_1row, imgnuml_1col)
    imgnuml_2row = step2_row(imgl_2, imgnuml_2, simthr)
    imgl_2i, imgnuml_2i = step2_interv(imgl, imgnuml, imgnuml_1res, imgnuml_2row)
    imgnuml_2col = step2_col(imgl_2i, imgnuml_2i, simthr)
    imgnuml_res = step3(imgnuml_1res, imgnuml_2row, imgnuml_2col, simthr)

    # imgl_res, imgnuml_res = step0(imgl, imgnuml, simthr)

    # print(imgnuml_res, len(imgnuml_res))
    # imshow(imgl_res[0]), axis('off')
    # show()

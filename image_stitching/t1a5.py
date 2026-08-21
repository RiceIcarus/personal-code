# 第五题

from pylab import *
import cv2
import t1a1
import t1a2
import t1a4


# 读取原始图片及序号
def readoriginimg_jpg(a):
    imgl, imgnuml = [], []  # 声明一个存储待拼接图片的列表，一个存储待拼接图片序号的列表
    for i in range(1, 151):
        imgl.append(imread(f'2022数学建模培训题1A附件\\1A附件{a}\\{i}.jpg'))
        imgnuml.append(array([[i]]))
    return imgl, imgnuml


# 保存图片
def saveimg_jpg(imgl, filename, name):
    for i in range(len(imgl)):
        cv2.imwrite(f'{filename}\\{name}\\{str(i).zfill(3)}.jpg', imgl[i][:, :, (2, 1, 0)])


# 先横拼再竖拼，找出所有2000*2000的图片
def step1_row(imgl, imgnuml, simthr):
    rowimgl, rowimgnuml = t1a4.simmatch_joint_row(imgl, imgnuml, simthr)
    imgl_1row, imgnuml_1row = t1a4.simmatch_joint_col(rowimgl, rowimgnuml, simthr)
    saveimg_jpg(imgl_1row, 't1a5', 'step1row')
    return imgnuml_1row


# 先竖拼再横拼，找出所有2000*2000的图片
def step1_col(imgl, imgnuml, simthr):
    colimgl, colimgnuml = t1a4.simmatch_joint_col(imgl, imgnuml, simthr)
    imgl_1col, imgnuml_1col = t1a4.simmatch_joint_row(colimgl, colimgnuml, simthr)
    saveimg_jpg(imgl_1col, 't1a5', 'step1col')
    return imgnuml_1col


# 去掉上述已经拼好的图片，将剩下的图片在进行拼接
def step1_r(imgnuml_1row, imgnuml_1col):
    imgl, imgnuml = readoriginimg_jpg('5')
    imgl_1row = t1a1.readimg('t1a5', 'step1row')
    imgl_1col = t1a1.readimg('t1a5', 'step1col')
    imgl_1res, imgnuml_1res = imgl_1row + imgl_1col, imgnuml_1row + imgnuml_1col
    l1 = list(ones(len(imgl)))
    for i in set(flatten(imgnuml_1res)):
        l1[i - 1] = 0
    imgl_2, imgnuml_2 = [], []
    for i in range(len(l1)):
        if l1[i]:
            imgl_2.append(imgl[i])
            imgnuml_2.append(imgnuml[i])
    saveimg_jpg(imgl_1res, 't1a5', 'step1res')
    saveimg_jpg(imgl_2, 't1a5', 'step2')
    return imgnuml_1res, imgl_2, imgnuml_2


# 先竖拼再横拼
def step2_col(imgl_2, imgnuml_2, simthr):
    colimgl, colimgnuml = t1a4.simmatch_joint_col(imgl_2, imgnuml_2, simthr)
    imgl_2col, imgnuml_2col = t1a2.bestsimmatch_joint_row(colimgl, colimgnuml, simthr)
    saveimg_jpg(imgl_2col, 't1a5', 'step2col')
    return imgnuml_2col


# 再去除已经拼好的
def step2_interv(imgl, imgnuml, imgnuml_1res, imgnuml_2col):
    l2 = list(ones(len(imgl)))
    for i in set(flatten(imgnuml_1res + imgnuml_2col)):
        l2[i - 1] = 0
    imgl_2i, imgnuml_2i = [], []
    for i in range(len(l2)):
        if l2[i]:
            imgl_2i.append(imgl[i])
            imgnuml_2i.append(imgnuml[i])
    return imgl_2i, imgnuml_2i


# 先横拼再竖拼
def step2_row(imgl_2i, imgnuml_2i, simthr):
    rowimgl, rowimgnuml = t1a2.bestsimmatch_joint_row(imgl_2i, imgnuml_2i, simthr)
    imgl_2row, imgnuml_2row = t1a2.bestsimmatch_joint_col(rowimgl, rowimgnuml, simthr)
    saveimg_jpg(imgl_2row, 't1a5', 'step2row')
    return imgnuml_2row


# 最后将所有图片拼接起来
def step3(imgnuml_1res, imgnuml_2row, imgnuml_2col, simthr):
    imgl_1res = t1a1.readimg('t1a5', 'step1res')
    imgl_2row = t1a1.readimg('t1a5', 'step2row')
    imgl_2col = t1a1.readimg('t1a5', 'step2col')
    imgl_3 = imgl_1res + imgl_2row + imgl_2col
    imgnuml_3 = imgnuml_1res + imgnuml_2row + imgnuml_2col
    imgl_3col, imgnuml_3col = t1a2.bestsimmatch_joint_col(imgl_3, imgnuml_3, simthr)
    imgl_res, imgnuml_res = t1a2.bestsimmatch_joint_row(imgl_3col, imgnuml_3col, simthr)
    saveimg_jpg(imgl_res, 't1a5', 'step3')
    t1a1.savematchtab(imgnuml_res[0], 't1a5')
    return imgl_res, imgnuml_res


# 完整过程
def step0(imgl, imgnuml, simthr):
    rowimgl, rowimgnuml = t1a4.simmatch_joint_row(imgl, imgnuml, simthr)
    imgl_1row, imgnuml_1row = t1a4.simmatch_joint_col(rowimgl, rowimgnuml, simthr)
    colimgl, colimgnuml = t1a4.simmatch_joint_col(imgl, imgnuml, simthr)
    imgl_1col, imgnuml_1col = t1a4.simmatch_joint_row(colimgl, colimgnuml, simthr)
    imgl_1res, imgnuml_1res = imgl_1row + imgl_1col, imgnuml_1row + imgnuml_1col
    l1 = list(ones(len(imgl)))
    for i in set(flatten(imgnuml_1res)):
        l1[i - 1] = 0
    imgl_2, imgnuml_2 = [], []
    for i in range(len(l1)):
        if l1[i]:
            imgl_2.append(imgl[i])
            imgnuml_2.append(imgnuml[i])
    colimgl, colimgnuml = t1a4.simmatch_joint_col(imgl_2, imgnuml_2, simthr)
    imgl_2col, imgnuml_2col = t1a2.bestsimmatch_joint_row(colimgl, colimgnuml, simthr)
    l2 = list(ones(len(imgl)))
    for i in set(flatten(imgnuml_1res + imgnuml_2col)):
        l2[i - 1] = 0
    imgl_2i, imgnuml_2i = [], []
    for i in range(len(l2)):
        if l2[i]:
            imgl_2i.append(imgl[i])
            imgnuml_2i.append(imgnuml[i])
    rowimgl, rowimgnuml = t1a2.bestsimmatch_joint_row(imgl_2i, imgnuml_2i, simthr)
    imgl_2row, imgnuml_2row = t1a2.bestsimmatch_joint_col(rowimgl, rowimgnuml, simthr)
    imgl_3 = imgl_1res + imgl_2row + imgl_2col
    imgnuml_3 = imgnuml_1res + imgnuml_2row + imgnuml_2col
    imgl_3col, imgnuml_3col = t1a2.bestsimmatch_joint_col(imgl_3, imgnuml_3, simthr)
    imgl_res, imgnuml_res = t1a2.bestsimmatch_joint_row(imgl_3col, imgnuml_3col, simthr)
    return imgl_res, imgnuml_res


if __name__ == '__main__':
    imgl, imgnuml = readoriginimg_jpg('5')
    simthr = 20  # 相似度阈值

    imgnuml_1row = step1_row(imgl, imgnuml, simthr)
    imgnuml_1col = step1_col(imgl, imgnuml, simthr)
    imgnuml_1res, imgl_2, imgnuml_2 = step1_r(imgnuml_1row, imgnuml_1col)
    imgnuml_2col = step2_col(imgl_2, imgnuml_2, simthr)
    imgl_2i, imgnuml_2i = step2_interv(imgl, imgnuml, imgnuml_1res, imgnuml_2col)
    imgnuml_2row = step2_row(imgl_2i, imgnuml_2i, simthr)
    imgnuml_res = step3(imgnuml_1res, imgnuml_2row, imgnuml_2col, simthr)

    # imgl_res, imgnuml_res = step0(imgl, imgnuml, simthr)

    # print(imgnuml_res, len(imgnuml_res))
    # imshow(imgl_res[0]), axis('off')
    # show()

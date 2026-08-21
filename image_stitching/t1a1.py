# 第一题

from pylab import *
import cv2
import os


# 读取原始图片及序号
def readoriginimg(a):
    imgl, imgnuml = [], []  # 声明一个存储待拼接图片的列表，一个存储待拼接图片序号的列表
    for i in range(1, 151):
        imgl.append(imread(f'2022数学建模培训题1A附件\\1A附件{a}\\{i}.png'))
        imgnuml.append(array([i]))
    return imgl, imgnuml


# 读取图片
def readimg(filename, name):
    imgl = []  # 声明一个存储待拼接图片的列表
    for each in os.listdir(f'{filename}\\{name}'):
        imgl.append(imread(f'{filename}\\{name}\\{each}'))
    return imgl


# 保存图片
def saveimg(imgl, filename, name):
    for i in range(len(imgl)):
        cv2.imwrite(f'{filename}\\{name}\\{str(i).zfill(3)}.png', imgl[i][:, :, (2, 1, 0)] * 255)


# 保存图片序号拼接表
def savematchtab(imgnuml_res, filename):
    with open(f'{filename}\\matchtab.txt', 'w') as f:
        for eachline in imgnuml_res:
            f.write(str(eachline).replace('.', '').replace('\n', ''))
            f.write('\n')


# def savematchtab(imgnuml_res, filename):
#     df1 = pd.DataFrame(imgnuml_res)
#     df1.to_csv(f'{filename}\\matchtab.csv', index=False, encoding='gbk')


# 横向匹配并拼接图片及序号
def matchjoint_row(imgl, imgnuml):
    imgl_row, imgnuml_row = [], []  # 声明一个存储拼好的横向图片的列表
    while imgl:  # 还有未拼接的图片时循环
        img0, img0num = imgl.pop(), imgnuml.pop()  # 取出最后一张图片，作为待拼接的底片
        j = 1
        while j:
            j = 0
            for k in [2, 3]:
                for i in range(len(imgl)):  # 循环整个待拼接图片列表
                    if img0.shape[0] == imgl[i].shape[0] and all(img0[:, :k] == imgl[i][:, -k:]):  # 底片的前k列与循环图片的后k列相等
                        img0 = hstack((imgl[i][:, :-k], img0))  # 将循环图片与底片横向拼接
                        img0num = hstack((imgnuml[i], img0num))  # 将循环图片序号与底片序号横向拼接
                        del imgl[i], imgnuml[i]  # 删掉已经拼上的图片
                        j = 1
                        break  # 跳出最近一个for循环
        j = 1
        while j:
            j = 0
            for k in [2, 3]:
                for i in range(len(imgl)):
                    if img0.shape[0] == imgl[i].shape[0] and all(img0[:, -k:] == imgl[i][:, :k]):  # 底片的后k列与循环图片的前k列相等
                        img0 = hstack((img0[:, :-k], imgl[i]))
                        img0num = hstack((img0num, imgnuml[i]))
                        del imgl[i], imgnuml[i]
                        j = 1
                        break
        imgl_row.append(img0)
        imgnuml_row.append(img0num)
    return imgl_row, imgnuml_row


# 纵向匹配并拼接图片及序号
def matchjoint_col(imgl, imgnuml):
    imgl_col, imgnuml_col = [], []  # 声明一个存储拼好的纵向图片的列表
    while imgl:  # 还有未拼接的图片时循环
        img0, img0num = imgl.pop(), imgnuml.pop()  # 取出最后一张图片，作为待拼接的底片
        j = 1
        while j:
            j = 0
            for k in [2, 3]:
                for i in range(len(imgl)):  # 循环整个待拼接图片列表
                    if img0.shape[1] == imgl[i].shape[1] and all(img0[:k, :] == imgl[i][-k:, :]):  # 底片的前k行与循环图片的后k行相等
                        img0 = vstack((imgl[i][:-k, :], img0))  # 将循环图片与底片纵向拼接
                        img0num = vstack((imgnuml[i], img0num))  # 将循环图片序号与底片序号纵向拼接
                        del imgl[i], imgnuml[i]  # 删掉已经拼上的图片
                        j = 1
                        break  # 跳出最近一个for循环
        j = 1
        while j:
            j = 0
            for k in [2, 3]:
                for i in range(len(imgl)):
                    if img0.shape[1] == imgl[i].shape[1] and all(img0[-k:, :] == imgl[i][:k, :]):  # 底片的后k行与循环图片的前k行相等
                        img0 = vstack((img0[:-k, :], imgl[i]))
                        img0num = vstack((img0num, imgnuml[i]))
                        del imgl[i], imgnuml[i]
                        j = 1
                        break
        imgl_col.append(img0)
        imgnuml_col.append(img0num)
    return imgl_col, imgnuml_col


if __name__ == '__main__':
    imglist, imgnumlist = readoriginimg('1')
    imglist_row, imgnumlist_row = matchjoint_row(imglist, imgnumlist)
    imglist_res, imgnumlist_res = matchjoint_col(imglist_row, imgnumlist_row)

    print(imgnumlist_res)
    print(len(imglist_res))
    imshow(imglist_res[0]), axis('off')
    show()

    # saveimg(imglist_res, 't1a1', 'result')
    # savematchtab(imgnumlist_res[0], 't1a1')

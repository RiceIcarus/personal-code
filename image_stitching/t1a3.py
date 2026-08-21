# 第三题  拼接图片，生成图片序号拼接表

from pylab import *
import t1a1


# 向上匹配拼接
def matchjoint_up(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l):
    for i in range(len(imgl)):  # 循环待拼接的图片列表
        for r in range(16):  # 在16*16的范围内查找
            for c in range(16):
                if all(img_res[y, x + 8:x + 8 + l] == imgl[i][r - 16, c:c + l]):  # 查找到对应l长度的像素完全相同
                    y, x, yn = y - r - 392, x + 8 - c, yn - 1  # 移动光标的位置到[y - r - 392, x + 8 - c]
                    img_res[y:y + 408, x:x + 408] = imgl[i]  # 覆盖对应位置的图像
                    imgnum_res[yn, xn] = imgnuml[i]
                    del imgl[i], imgnuml[i]  # 删掉已拼接上的图片
                    return img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn  # 返回拼了这张图后的结果，待拼接的图片列表，光标所在位置
    return 0, 0, 0, 0, 0, 0, 0, 0  # 若未匹配到对应的图片，则全返回0


# 向下匹配拼接
def matchjoint_down(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l):
    for i in range(len(imgl)):  # 循环待拼接的图片列表
        for r in range(16):  # 在16*16的范围内查找
            for c in range(16):
                if all(img_res[y + 407, x + 8:x + 8 + l] == imgl[i][r, c:c + l]):  # 查找到对应l长度的像素完全相同
                    y, x, yn = y + 407 - r, x + 8 - c, yn + 1  # 移动光标的位置到[y + 407 - r, x + 8 - c]
                    img_res[y:y + 408, x:x + 408] = imgl[i]  # 覆盖对应位置的图像
                    imgnum_res[yn, xn] = imgnuml[i]
                    del imgl[i], imgnuml[i]  # 删掉已拼接上的图片
                    return img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn  # 返回拼了这张图后的结果，待拼接的图片列表，光标所在位置
    return 0, 0, 0, 0, 0, 0, 0, 0  # 若未匹配到对应的图片，则全返回0


# 向左匹配拼接
def matchjoint_left(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l):
    for i in range(len(imgl)):  # 循环待拼接的图片列表
        for r in range(16):  # 在16*16的范围内查找
            for c in range(16):
                if all(img_res[y + 8:y + 8 + l, x] == imgl[i][r:r + l, c + 392]):  # 查找到对应l长度的像素完全相同
                    y, x, xn = y + 8 - r, x - c - 392, xn - 1  # 移动光标的位置到[y + 8 - r, x - c - 392]
                    img_res[y:y + 408, x:x + 408] = imgl[i]  # 覆盖对应位置的图像
                    imgnum_res[yn, xn] = imgnuml[i]
                    del imgl[i], imgnuml[i]  # 删掉已拼接上的图片
                    return img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn  # 返回拼了这张图后的结果，待拼接的图片列表，光标所在位置
    return 0, 0, 0, 0, 0, 0, 0, 0  # 若未匹配到对应的图片，则全返回0


# 向右匹配拼接
def matchjoint_right(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l):
    for i in range(len(imgl)):  # 循环待拼接的图片列表
        for r in range(16):  # 在16*16的范围内查找
            for c in range(16):
                if all(img_res[y + 8:y + 8 + l, x + 407] == imgl[i][r:r + l, c]):  # 查找到对应l长度的像素完全相同
                    y, x, xn = y + 8 - r, x + 407 - c, xn + 1  # 移动光标的位置到[y + 8 - r, x + 407 - c]
                    img_res[y:y + 408, x:x + 408] = imgl[i]  # 覆盖对应位置的图像
                    imgnum_res[yn, xn] = imgnuml[i]
                    del imgl[i], imgnuml[i]  # 删掉已拼接上的图片
                    return img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn  # 返回拼了这张图后的结果，待拼接的图片列表，光标所在位置
    return 0, 0, 0, 0, 0, 0, 0, 0  # 若未匹配到对应的图片，则全返回0


# 纵向匹配拼接到边界
def matchjoint_updown(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l):
    yc, xc, ync, xnc, imgl_t = y, x, yn, xn, 1  # 保存光标初始位置，生成控制变量
    while imgl_t:  # 尚未拼到上边界时循环
        img_res_t, imgnum_res_t, imgl_t, imgnuml_t, y_t, x_t, yn_t, xn_t = matchjoint_up(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l)  # 往上拼一张图片
        if imgl_t:  # 如果成功拼上去了
            img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn = img_res_t, imgnum_res_t, imgl_t, imgnuml_t, y_t, x_t, yn_t, xn_t  # 光标移到拼上去这张图的左上角
    y, x, yn, xn, imgl_t = yc, xc, ync, xnc, 1  # 重设光标初始位置，生成控制变量
    while imgl_t:  # 尚未拼到下边界时循环
        img_res_t, imgnum_res_t, imgl_t, imgnuml_t, y_t, x_t, yn_t, xn_t = matchjoint_down(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l)  # 往下拼一张图片
        if imgl_t:  # 如果成功拼上去了
            img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn = img_res_t, imgnum_res_t, imgl_t, imgnuml_t, y_t, x_t, yn_t, xn_t  # 光标移到拼上去这张图的左上角
    return img_res, imgnum_res, imgl, imgnuml  # 返回拼了这张图后的结果，待拼接的图片列表


# 匹配拼接完整的图片
def matchjoint(imgl, imgnuml, l):
    img_res, imgnum_res = zeros((8000, 12000, 3)), zeros((20, 30))  # 生成一个较大的空白画布，每个值都设为0
    y, x, yn, xn, imgl_t = 4000, 6000, 10, 15, 1  # 设置光标初始位置，生成控制变量
    img_res[y:y + 408, x:x + 408], imgnum_res[yn, xn] = imgl.pop(), imgnuml.pop()  # 给画布画上第一张图片, 生成图片序号拼接表
    img_res, imgnum_res, imgl, imgnuml = matchjoint_updown(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l)  # 将第一张图片上下拼到边界
    while imgl_t:  # 尚未拼到左边界时循环
        img_res_t, imgnum_res_t, imgl_t, imgnumlist_t, y_t, x_t, yn_t, xn_t = matchjoint_left(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l)  # 往左拼一张图片
        if imgl_t:  # 如果成功拼上去了
            img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn = img_res_t, imgnum_res_t, imgl_t, imgnumlist_t, y_t, x_t, yn_t, xn_t  # 光标移到拼上去这张图的左上角
            img_res, imgnum_res, imgl, imgnuml = matchjoint_updown(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l)  # 将左边这张图片上下拼到边界
    y, x, yn, xn, imgl_t = 4000, 6000, 10, 15, 1  # 将光标还原到初始位置，生成控制变量
    while imgl_t:  # 尚未拼到右边界时循环
        img_res_t, imgnum_res_t, imgl_t, imgnumlist_t, y_t, x_t, yn_t, xn_t = matchjoint_right(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l)  # 往右拼一张图片
        if imgl_t:  # 如果成功拼上去了
            img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn = img_res_t, imgnum_res_t, imgl_t, imgnumlist_t, y_t, x_t, yn_t, xn_t  # 光标移到拼上去这张图的左上角
            img_res, imgnum_res, imgl, imgnuml = matchjoint_updown(img_res, imgnum_res, imgl, imgnuml, y, x, yn, xn, l)  # 将右边这张图片上下拼到边界

    img_res = img_res[[not all(img_res[i, :] == 0) for i in range(img_res.shape[0])], :]  # 去除掉全为0的行
    img_res = img_res[:, [not all(img_res[:, i] == 0) for i in range(img_res.shape[1])]]  # 去除掉全为0的列
    imgnum_res = imgnum_res[[not all(imgnum_res[i, :] == 0) for i in range(imgnum_res.shape[0])], :]  # 去除掉全为0的行
    imgnum_res = imgnum_res[:, [not all(imgnum_res[:, i] == 0) for i in range(imgnum_res.shape[1])]]  # 去除掉全为0的列
    return img_res, imgnum_res  # 返回拼了这张图后的结果，待拼接的图片列表


if __name__ == '__main__':
    imgl, imgnuml = t1a1.readoriginimg('3')
    img_res, imgnum_res = matchjoint(imgl, imgnuml, 50)

    # print(imgnum_res)
    # imshow(img_res), axis('off')
    # show()

    t1a1.saveimg([img_res], 't1a3', 'result')
    t1a1.savematchtab(imgnum_res, 't1a3')

"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
添加了自编的后处理
输入原图和mask图，输出正常语序图用于识别
"""

# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2 as cv
import math
import os
import time
import pdb

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv.applyColorMap(img, cv.COLORMAP_JET)
    return img
 
class InverseColorMap(object):
    def __init__(self, colormap=cv.COLORMAP_JET):
        self.colormap = colormap
        grays = np.arange(256, dtype=np.uint8)
        self.map = cv.applyColorMap(grays, colormap).squeeze()

    def __call__(self, false_color):
        return np.apply_along_axis(self.nearest_gray, 2, false_color).astype(np.uint8)

    def nearest_gray(self, color):
        diff = np.abs(color - self.map).sum(1)
        return diff.argmin()

    def draw_segment(self, segment):
        mask = (segment > 0).astype(np.uint8)
        scale = np.linspace(0, 255, segment.max() + 1).astype(np.uint8)
        scaled = np.apply_along_axis(lambda x: scale[x], 1, segment)
        false_color = cv.applyColorMap(scaled, self.colormap)
        false_color *= mask[:, :, None]
        return false_color


def write_fail_log(image_file_path, error_code):
    fail_path = './fail_log/'
    fail_file = './fail_log/fail_log.txt'
    if not os.path.isdir(fail_path):
        os.mkdir(fail_path)
    with open(fail_file, 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +
                ' 发生错误，文件' + image_file_path + '不达标，需人工除处理。（错误类型：' + error_code + '。）' + '\n')

# cv.distanceTransform() # to get center
# 实现输入原始印章图片目录，寻找中心点并使用圆标出,返回带标注的图片和中心坐标值
def find_center_point(image_file_path):
    src_Img = cv.imread(image_file_path, 1)
    image_Gray = cv.cvtColor(src_Img, cv.COLOR_RGB2GRAY)
    image_Gray = ~image_Gray  # 对灰度图取反，找色块的中心（灰度图）时不需要取反
    image_Blur = cv.GaussianBlur(image_Gray, (5, 5), 2)  # 滤波
    #image_Hold = cv.threshold(image_Blur, 20, 200, cv.THRESH_BINARY)[1]  # 阈值
    circles = cv.HoughCircles(image_Blur, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 80, 120)
    ret_Pt = []
    if circles.shape[0] == 0:
        write_fail_log(image_file_path, '找不到圆心')
    else:
        index = np.argmax(circles[0, :, 2], axis=0)
        cv.circle(src_Img, (circles[0][index][0], circles[0][index][1]), 2, (0, 255, 0), 3,
                  cv.LINE_AA)  # draw center of circle
        Pt = (circles[0][index][0], circles[0][index][1])
        ret_Pt = Pt
    return src_Img, ret_Pt

'''
外径为中心点到正上方字符上边界，找Y坐标最低（Y值最大）的白色区域坐标
内径为白色像素点与圆心距离最短的值，不够鲁棒，需要找新方法。
内外径均使用扩展值进行了微调
'''
#寻找内外径
def find_diameter(raw_image_file_path, res_image_file_path):
    src_Img, Pt = find_center_point(raw_image_file_path)  # 中心坐标
    ret_src_Img = []
    ret_Pt = []
    ret_internal_diameter = 0.0
    ret_outer_diameter = 0.0
    if Pt != []:
        inv_cm = InverseColorMap()
        im = cv.imread(raw_image_file_path)
        mask = cv.imread(res_image_file_path)
        H, W = im.shape[:2]
        w = mask.shape[1] // 2
        mask1 = cv.resize(mask[:, :w, :], (W, H))
        gray1 = inv_cm(mask1)
        gray = cv.medianBlur(gray1, 7)
        binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)[1]
        # binary = ~binary
        outer_d = 0  # 暂存
        outer_d_P = (0, 0)
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1] // 2, binary.shape[1]):
                if binary[i, j] == 255:
                    if j > outer_d:
                        outer_d = j
                        outer_d_P = (i, j)  # 坐标
        # 热力图不准，要往外扩展
        outer_diameter_expand = 20
        outer_diameter = math.sqrt(pow((Pt[0] - outer_d_P[0]), 2) + pow((Pt[1] - outer_d_P[1]), 2)) + outer_diameter_expand
        # 确定内径的方法：白色像素点与圆心距离最短的值为内径
        inter_d = outer_diameter  # 内径初始值为外径，逐步减小
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i, j] == 255:
                    inter_d_tmp = math.sqrt(pow((Pt[0] - i), 2) + pow((Pt[1] - j), 2))
                    if inter_d_tmp < inter_d:
                        inter_d = inter_d_tmp
        internal_diameter_expand = 0#之前设定为5，但是106.png由于存在横排文字，内径计算值小于5
        internal_diameter = inter_d + internal_diameter_expand
        cv.circle(src_Img, Pt, int(internal_diameter), (0, 255, 0), 3)
        cv.circle(src_Img, Pt, int(outer_diameter), (0, 255, 0), 3)
        ret_src_Img = src_Img
        ret_Pt = Pt
        ret_internal_diameter =  internal_diameter
        ret_outer_diameter = outer_diameter
    return ret_src_Img, ret_Pt, ret_internal_diameter, ret_outer_diameter

#圆环区域遮罩提取
def crop(filename, center, inner_Radius, outer_Radius):
    ret_crop_img = []
    if center != []:
        img = cv.imread(filename, 0)
        h, w = img.shape[:2]
        mask1 = np.zeros((h, w), np.uint8)  # 内圆遮罩
        mask2 = np.zeros((h, w), np.uint8)  # 外圆遮罩
        cv.circle(mask1, center, inner_Radius, (255, 0, 0), -1)
        cv.circle(mask2, center, outer_Radius, (255, 0, 0), -1)
        mask = cv.bitwise_xor(mask1, mask2)
        crop_img = cv.bitwise_and(img, img, mask=mask)
        ret_crop_img = crop_img
    return ret_crop_img


#逆时针旋转
def Nrotate(angle,valuex,valuey,pointx,pointy):
      angle = (angle/180)*math.pi
      valuex = np.array(valuex)
      valuey = np.array(valuey)
      nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
      nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
      return (nRotatex, nRotatey)


#顺时针旋转
def Srotate(angle,valuex,valuey,pointx,pointy):
      angle = (angle/180)*math.pi
      valuex = np.array(valuex)
      valuey = np.array(valuey)
      sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
      sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
      return (sRotatex,sRotatey)


#将四个点做映射
def rotatecordiate(angle,rectboxs,pointx,pointy):
      output = []
      for rectbox in rectboxs:
        if angle>0:
          output.append(Srotate(angle,rectbox[0],rectbox[1],pointx,pointy))
        else:
          output.append(Nrotate(-angle,rectbox[0],rectbox[1],pointx,pointy))
      return output


def imagecrop(image, box, i, is_mult_word):
    xs = [x[1] for x in box]  
    ys = [x[0] for x in box]
    #尝试a，单张图片测试，后面可以采用滴水法（drop.py，目前仅支持2个字符的分割，输入图片，自己找切割路径）
    if is_mult_word == True:
        cropimage = image[min(xs):max(xs), min(ys):max(ys)]
        cropimage1 = cropimage[0:cropimage.shape[0],0:int(cropimage.shape[1]/2),]
        cropimage2 = cropimage[0:cropimage.shape[0],int(cropimage.shape[1]/2):cropimage.shape[1]]
        cv.imwrite('./22-test-result/cropimage_' + str(i) + '_1_.png', cropimage1)
        cv.imwrite('./22-test-result/cropimage_' + str(i) + '_2_.png', cropimage2)
    cropimage = image[min(xs):max(xs), min(ys):max(ys)]
    cv.imwrite('./22-test-result/cropimage_'+str(i)+'_.png', cropimage)
    return cropimage


#切单字的方案：
#内外半径已知 角度加半径可确定一个坐标点 以圆心为原点，垂直向下的射线为起始射线 顺时针做个循环 一次加一度,重要的参数一直都只有一个，角度
#cv.imread()纵坐标在左边
def tps_test(origin_img,corp_img, Pt, internal_diameter, outer_diameter):
    img =corp_img
    cv.imshow('binary',img)
    w,h = origin_img.shape
    new_img = np.reshape(origin_img, (w,h,1))
    new_img = np.concatenate([new_img,new_img,new_img],axis=2)
    #cv.imshow('0', img)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    fusion_open = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    cv.imshow('open_process', fusion_open)
    img = fusion_open
    #对于这个图片，以圆心和竖直向下的那一点为端点
    P0 = Pt
    #角度记录
    word_angel = []
    begin_outer = []
    begin_inter = []
    end_outer = []
    end_inter = []
    is_mult_word = False
    #P0P为扫描线，线段中有一点碰到白色代表遇到了字，外径减内径剩下的圆环区域宽，整个线段都在黑色区域，代表扫过了文字。
    #获取碰到时候的线段端点和全出去时的线段端点，4个点构成裁剪矩形，适当扩展矩形范围，剪出矩形区域
    #目前整体逻辑如下：从0度开始累加，发现线段中有一点处于白色像素，记录此时的角度和外径端点坐标作为矩形外点，发现整个线段都是黑色像素时，
    # 记录此时的角度和内径端点坐标作为矩形内点，当内外点都存在时，在图片上绘制这个矩形（后面改成剪切这个四点构成的区域）并显示，
    # 然后内外点置空，继续循环，找下一个
    is_begin = False
    for i in range(50, 360):    ##旋转
        #此为不断移动的外径端点P
        P = (P0[0] - outer_diameter * math.sin(i * 3.14 / 180), P0[1] + outer_diameter * math.cos(i * 3.14 / 180))
        #表示线段
        if is_begin is False:
            for length in range(int(internal_diameter), int(outer_diameter)):
                x = P0[0] - length * math.sin(i * 3.14/180)
                y = P0[1] + length * math.cos(i * 3.14 / 180)
                line = y
                #整个线段有一点碰到了白色，则开始进入文字区域，记录此时的内外径点
                if img[int(y), int(x)] == 255:
                    #内径点，即为靠近圆心的端点
                    P_start_inter = (P0[0] - internal_diameter * math.sin(i * 3.14 / 180), P0[1] + internal_diameter * math.cos(i * 3.14 / 180))
                    #外径点，即为远离圆心的端点
                    P_start_outer = P
                    word_angel.append(i)
                    begin_inter.append(P_start_inter)
                    begin_outer.append(P_start_outer)
                    is_begin = True
                    break
        else:
            #找字体的终点
            #如何判断整个线段都接触到黑色？整个线段的像素值都为0。
            flag = True
            for length in range(int(internal_diameter), int(outer_diameter)):
                x = P0[0] - length * math.sin(i * 3.14/180)
                y = P0[1] + length * math.cos(i * 3.14 / 180)
                p = (x, y)
                if img[int(y), int(x)] != 0:
                    flag = False
                    break
            if flag:
                P_end_inter = (P0[0] - internal_diameter * math.sin(i * 3.14 / 180), P0[1] + internal_diameter * math.cos(i * 3.14 / 180))
                P_end_outer = P
                word_angel.append(i)
                end_inter.append(P_end_inter)
                end_outer.append(P_end_outer)
                is_begin = False
    #画始、末线
    rotate_angel = []
    if len(word_angel)%2==0:
        angel = word_angel
        for i in range(0,len(angel),2):
            tmpangel = (angel[i] + angel[i+1])//2
            rotate_angel.append(tmpangel)
    else:
        angel = word_angel[:-1]
        for i in range(0,len(angel),2):
            tmpangel = (angel[i] + angel[i+1])//2
            rotate_angel.append(tmpangel)
    for i in range(len(end_outer)):
        cnt = np.array([[int(begin_outer[i][0]), int(begin_outer[i][1])],[int(begin_inter[i][0]), int(begin_inter[i][1])],
                        [int(end_outer[i][0]), int(end_outer[i][1])],[int(end_inter[i][0]), int(end_inter[i][1])]])
        rect = cv.minAreaRect(cnt)
        # print(rect)
        box_origin = cv.boxPoints(rect)
        box = np.int0(box_origin)
        angel = rotate_angel[i]
        if angel<180:
            angel = 0 - (180-angel)
            M = cv.getRotationMatrix2D(rect[0],angel,1)  
            dst = cv.warpAffine(origin_img,M,(origin_img.shape[0],origin_img.shape[1]))
        else:
            angel = angel - 180
            M = cv.getRotationMatrix2D(rect[0],angel,1)  
            dst = cv.warpAffine(origin_img,M,(origin_img.shape[0],origin_img.shape[1]))
        #作差时，一个是偶数索引（终止角），一个是奇数索引（起始角），针对1.png，有两个字的为49度，单字一般为20度
        #print('第'+str(i)+'组文字角为：',word_angel[2 * i+1],'和',word_angel[2 * i])
        print('第'+str(i)+'组角差为：',word_angel[2 * i+1]-word_angel[2 * i])
        angle_difference = word_angel[2 * i+1]-word_angel[2 * i]
        if angle_difference > 30:
            print('第'+str(i)+'组内接矩形包含了多个文字，需要进行裁剪。')
            is_mult_word = True
        else:
            is_mult_word = False
        new_img = cv.drawContours(new_img,[box],0,(255,0,0),2)
        # cv.imshow('1',new_img)
        # cv.waitKey(0)
        box = rotatecordiate(rect[2], box_origin, rect[0][0], rect[0][1])
        imagecrop(dst, np.int0(box), i, is_mult_word)
    cv.imshow('test',new_img)
    cv.waitKey(0)
    #设置断点操作
    #pdb.set_trace()



def to_polar(img_name, crop_img, Pt, radius):
    #圆心和半径已知
    ret_cut_ltr_img = []
    #一开始的确使用的是is not None，但是crop_img为[]时，crop_img is not None返回为True
    #if crop_img == []:
        #print('crop_img为空')
        #print((crop_img is None))
    if crop_img != []:
        img = crop_img
        matLogPolar = cv.logPolar(img, Pt, radius / 3, cv.WARP_FILL_OUTLIERS)
        matLogPolar = cv.threshold(matLogPolar, 0, 255, cv.THRESH_TOZERO)[1]
        srcCopy = cv.transpose(matLogPolar)
        srcCopy = cv.flip(srcCopy, 0)
        #去掉上下黑边
        x = srcCopy.shape[0]
        y = srcCopy.shape[1]
        edges_x = []
        edges_y = []
        for i in range(x):
            for j in range(y):
                if srcCopy[i][j] == 255:
                    edges_x.append(i)
                    edges_y.append(j)
        if edges_x == []:
            write_fail_log(img_name, '找不到裁剪边界')
        else:
            left = min(edges_x)  # 左边界
            right = max(edges_x)  # 右边界
            width = right - left  # 宽度
            bottom = min(edges_y)  # 底部
            top = max(edges_y)  # 顶部
            height = top - bottom  # 高度
            ret_ltr_img = srcCopy[left:left + width, bottom:bottom + height]
            # 裁剪值能否根据起始点的极坐标变换，得到其x坐标？
            cut_x = 50
            left_img = ret_ltr_img[0:ret_ltr_img.shape[1], 0:cut_x]
            #print(ret_ltr_img.shape[0])
            right_img = ret_ltr_img[0:ret_ltr_img.shape[1], cut_x:ret_ltr_img.shape[1]]
            ret_cut_ltr_img = np.hstack((right_img, left_img))
    return ret_cut_ltr_img


'''
垂直投影分割法，有bug且暂不使用
def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # 绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255
    #cv.imshow('hProjection2', hProjection)

    return h_


def getVProjection(image):
    vProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0] * w
    # 循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    # 绘制垂直平投影图像
    for x in range(w):
        for y in range(h - w_[x], h):
            vProjection[y, x] = 255
    cv.imshow('vProjection',vProjection)
    return w_
'''
def post_process():
    img_file_path = './data/'
    #img_file_path = './new-data-test/'
    mask_img_file_path = './result/'
    file_names = os.listdir(img_file_path)
    for file_name in file_names:
        print(file_name+' is under post-processing')
        img_name = os.path.join(img_file_path, file_name)
        mask_file_name = 'res_'+file_name.split('.')[0]+'_mask.jpg'
        mask_img_name = os.path.join(mask_img_file_path, mask_file_name)
        test_Img, Pt, internal_diameter, outer_diameter = find_diameter(img_name, mask_img_name)
        corp_img = crop(img_name, Pt, int(internal_diameter), int(outer_diameter))
        #最初期的预处理，用于pdf
        #seal_img = remove_overlapped_seal_word(origin_img)
        #word_img = remove_seal(origin_img)
        #cv.imshow('seal_img', seal_img)
        #cv.imshow('word_img', word_img)
        #cv.waitKey(0)
        if corp_img is not None:
            ltr_Img = to_polar(img_name, corp_img, Pt, outer_diameter)
            #tps_test(corp_img,binary, test_Pt, internal_diameter, outer_diameter)
            if ltr_Img != []:
                ltr_Img_name = file_name.split('.')[0]
                cv.imwrite('./polar-img/polar_' + ltr_Img_name + '.jpg', ltr_Img)
    print('done')

import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb
from scipy.ndimage.filters import convolve, correlate
import matplotlib.pyplot as plt
import time # 時間測定

%matplotlib inline

def mse(y1, y2):
    return ((y1 - y2)**2).mean()

""" 1-1 平均化フィルタ """
def myaverage_naive(im, filter_size):
    ''' im : 入力画像, filter_size : 平均化フィルタのサイズ（奇数） '''
    iheight, iwidth = im.shape[:2]
    imout = np.zeros((iheight, iwidth))
    
    # ここにコードを追加
    padding = np.pad(im, (filter_size//2, filter_size//2), mode="symmetric") # パディング
    
    onearray = np.ones((filter_size, filter_size)) / (filter_size ** 2) # すべて1の行列を作る   
    h, w = onearray.shape
    
    for j in range(iheight):
        for i in range(iwidth):
            for m in range(h):
                for n in range(w):
                    imout[j, i] += padding[j + m, i + n] * onearray[m, n]
    return imout

""" 1-3 積分画像 """
def myaverage_integral(im, filter_size):
    ''' im : 入力画像, filter_size : 平均化フィルタのサイズ（奇数） '''
    def integral_image(im):
        ''' 積分画像の作成 '''
        s = np.zeros_like(im)
        
        # ここにコードを追加
        h, w = im.shape
        
        # 横の1行目と縦の1列目を先に計算する
        s[0, 0] = im[0, 0]
        for i in range(1, h, 1):
            s[0, i] = s[0, i - 1]+im[0, i]
        for i in range(1,w,1):
            s[i, 0] = s[i - 1, 0]+im[i, 0]

        for j in range(1, h, 1):
            for i in range(1, w, 1):
                s[j, i] = s[j, i - 1] + s[j - 1, i] - s[j - 1,i - 1] + im[j, i]
        
        return s
    
    iheight, iwidth = im.shape[:2]
    imout = np.zeros((iheight, iwidth))

    # ここにコードを追加
    fs = filter_size//2 # 例外処理で足したマスを計算する
    padding = np.pad(im, (fs + 1, fs + 1), mode="symmetric") # パディング（例外処理のために+1にする）
    #print("padding = \n",padding)
        
    s_padding = integral_image(padding) # 積分画像をとる
    
    # 畳み込み
    """  
    s_paddingをパディングしてたが，imoutはパディングしてないので，
    s_padding と imout は対応しているマスは filter_size ずれがある，
    それのずれを修正するために，
    横と縦をそれぞれ filter_size//2+1 を修正しなければならない．
    
    """
    mis = fs + 1 # ずれ修正
    for j in range(iheight):
        for i in range(iwidth):
            imout[j, i] = (s_padding[j + fs + mis, i + fs + mis] - s_padding[j - fs - 1 + mis,i + fs + mis] 
                           - s_padding[j + fs + mis,i - fs - 1 + mis] 
                           + s_padding[j - fs - 1 + mis,i - fs - 1 + mis]) / (filter_size ** 2)
    return imout

""" 1-6 分離可能フィルタ """
def myaverage_separable(im, filter_size):
    ''' im : 入力画像, filter_size : 平均化フィルタのサイズ（奇数） '''
    iheight, iwidth = im.shape[:2]
    imout = np.zeros((iheight, iwidth))

    # ここにコードを追加
    padding = np.pad(im, (filter_size//2, filter_size//2), mode="symmetric") # パディング
    
    x = np.ones((1, filter_size)) / (filter_size) # すべて1のi方向(1×filter_size)行列を作る
    y = np.ones((filter_size, 1)) / (filter_size) # すべて1のj方向(filter_size×i)行列を作る   
    
    for j in range(iheight):
        for i in range(iwidth):
            provision = 0
            for m in range(filter_size):
                for n in range(filter_size):
                    provision += padding[j + m, i + n] * x[0, n] * y[m, 0]
            imout[j, i] = provision
   
    return imout

im = 255 * rgb2gray(imread("exercise/Lenna2.png"))
#print("im = \n", im)

filter_size = 5

kernel = np.ones((filter_size, filter_size)) / (filter_size ** 2)

starta = time.time() # time start
im2a = myaverage_naive(im, filter_size)
elapsed_time_a = time.time() - starta # time finish

startb = time.time() # time start
im2b = myaverage_integral(im, filter_size)
elapsed_time_b = time.time() - startb # time finish

startc = time.time() # time start
im2c = myaverage_separable(im, filter_size) # オプション
elapsed_time_c = time.time() - startc # time finish

im2_gt = correlate(im, kernel, mode='reflect')

#print("im2b = \n", im2b)
#print("im2_gt =  \n", im2_gt)

print("mse_naive =", mse(im2_gt, im2a)) # 1未満であればOK
print ("time_naive = {0}".format(elapsed_time_a) + "[sec]\n")

print("mse_integral =", mse(im2_gt, im2b)) # 1未満であればOK
print ("time_integral = {0}".format(elapsed_time_b) + "[sec]\n")

print("mse_separable =", mse(im2_gt, im2c)) # オプション
print ("time_separable = {0}".format(elapsed_time_c) + "[sec]\n")

plt.subplot(1, 2, 1); 
plt.imshow(im2_gt)
plt.subplot(1, 2, 2);
plt.imshow(im2c)

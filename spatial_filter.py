import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize, rotate
from scipy.ndimage.filters import correlate
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgb
from sklearn import linear_model # 線形回帰を行うモジュール
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd

%matplotlib inline

plt.gray()

im = 255 * rgb2gray(imread("exercise/Lenna2.png"))
imout = 255 * rgb2gray(imread("exercise/Lenna2_edited.png"))

# ★ここにコードを追加
def mse(y1, y2):
    return ((y1 - y2)**2).mean()

h, w = im.shape

"""
Y = aX + b
Y と X は既知数で，a と b は未知数である．
X に関する Y の一次関数を求めることと同様にすればよい．
F(m×1) = A(m×9)×H(9×1)+B(m×1)
の行列を作る．
行列 H と B を求めたい．
つまり，線形回帰による教師なし学習を用いること．
回帰直線 Y = aX + b についてその回帰係数(a)と切片(b)を求める．
"""
H = [] # hの係数について空のリストを作る
F = [] # 等式の右辺について空のリストを作る
for j in range(1, h-1):
    for i in range(1, w-1):
        x = []
        for m in range(-1, 2):
            for n in range(-1, 2):
                x.append(im[j + m, i + n])
        H.append(x)
        F.append(imout[j, i])

# 行列式に変更

A = np.array(H, dtype = np.float)
data,data2 = A.shape
B = np.array(F, dtype = np.float)

e = linear_model.LinearRegression() #回帰を行うためのインスタンスを取得する
e.fit(A, B) # あてはめを行う
rc = e.coef_ # 回帰係数
s = e.intercept_ # 切片
rate = e.score(A,B) # 寄与率
se = np.sqrt((1-rate)/(data-2)) # 標準誤差
t = np.sqrt(rate)/se # t値

print("regression coefficient =\n",np.reshape(rc,(3, 3)),"\nsegment =",s)
print("r^2 =",e.score(A,B))
print("扱えないデータの割合：", 1-rate)
print("SE = ",se)
print("T = ",t)
print("t^2 = ", t**2)
print("95%信頼空間：",np.sqrt(rate)-1.96*se, np.sqrt(rate)+1.96*se)

Ans = np.reshape(rc, (3, 3)) # 3×3の2次元配列に変形
prediction = correlate(im, Ans) + s

plt.subplot(1, 2, 1); 
plt.imshow(imout)
plt.subplot(1, 2, 2);
plt.imshow(prediction)

print("mse =", mse(imout, prediction))

import cv2
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

def hybrid_image(im1, im2, cutoff1, cutoff2, visu = False):
    """ ハイブリッド画像の作成 
         cutoff1 -- ガウス型ローパスフィルタのカットオフ周波数(cycles/image)
         cutoff2 -- ガウス型ハイパスフィルタのカットオフ周波数(cycles/image)
    """
    # 低周波数領域，高周波領域のカットオフ周波数cutoff1, cutoff2から周波数フィルタ関数を作成
    def lowpass_gauss(u, v, sigma):
        """ ガウス分布型ローパスフィルタ """
        return np.exp(-1.0*(u**2 + v**2)/(2*sigma**2))

    def lowpass(u, v): 
        return lowpass_gauss(u, v, 1.2 * cutoff1 / im1.shape[0])

    def highpass(u, v): 
        return 1.0 - lowpass_gauss(u, v, 0.63 * cutoff2 / im1.shape[0])
    
    # (1) フーリエ変換
    fshift1 = do_fft(im1)
    fshift2 = do_fft(im2)
    
    # (2) 周波数フィルタリング
    fshift1 = do_ffilter(fshift1, lowpass)
    fshift2 = do_ffilter(fshift2, highpass)
    
    # (3) ハイブリッドイメージの合成
    hybrid_image = do_ifft(fshift1 + fshift2) 
    hybrid_image = array_normalize(hybrid_image)  # 画素値を0～1に正規化
    
    return hybrid_image

W = 512   # 画像サイズ（縦横の画素数)
im1 = cv2.imread("data/library1.jpg") # low
im2 = cv2.imread("data/tus.jpg") # hight

# RGB順に並べ替える
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

# 画像のサイズを同じに揃える
im1 = resize(im1, (W, W))
im2 = resize(im2, (W, W))

cutoff1 = 15
cutoff2 = 15
plt.figure(figsize=(12, 12))
him = hybrid_image(im1, im2, cutoff1=15, cutoff2=15, visu=False)
show_hybrid_image(him)

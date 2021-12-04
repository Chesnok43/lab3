import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
from matplotlib.colors import LogNorm as log


def DFFTnp(img):
    ft = np.fft.fft2(img)
    ftsh = np.fft.fftshift(ft) 
    return ftsh

def revDFFTnp(dfftn):
    ft_ish = np.fft.ifftshift(dfftn) 
    revImg = np.fft.ifft2(ft_ish) 
    return revImg


def Gauss(img, ftsh):
    ksize = 21
    kernel = np.zeros(img.shape) # ядро
    blur = cv.getGaussianKernel(ksize, -1)
    blur = np.matmul(blur, np.transpose(blur))
    kernel[0:ksize, 0:ksize] = blur
    fkshift = DFFTnp(kernel)
    mult = np.multiply(ftsh, fkshift)
    return revDFFTnp(mult)
    

images = glob.glob('00_65.png')
for image in images:
    img = np.float32(cv.imread(image, 0))
    fshift = DFFTnp(img)

    plt.subplot(221), plt.title('spectr'), plt.xticks([]), plt.yticks([])
    plt.imshow(np.abs(fshift), cmap = "gray", norm = log(vmin = 5))

    w, h = fshift.shape 
    mp = fshift[w//2][h//2]
    for i in range(w):
        for j in range(h):
            if i != w//2 and j != h//2:
                if abs(np.abs(fshift[i][j])-np.abs(mp)) < np.abs(mp) - 250000:
                    fshift[i][j] = 0

    plt.subplot(222), plt.title('Notch'), plt.xticks([]), plt.yticks([])
    plt.imshow(np.abs(fshift), cmap = "gray", norm = log(vmin = 5))

    revImg = Gauss(revDFFTnp(fshift), fshift)
    
    plt.subplot(223), plt.title('img'), plt.xticks([]), plt.yticks([])
    plt.imshow(abs(img), cmap = 'gray')
    plt.subplot(224), plt.title('GaussResult'), plt.xticks([]), plt.yticks([])
    plt.imshow(abs(revImg), cmap = 'gray')

    plt.show()
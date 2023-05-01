import tensorflow as tf
# import imageio
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
# from skimage.filters import gaussian
from scipy.linalg import sqrtm
import random
from skimage.transform import swirl

# im = imageio.imread('/Users/candicecai/Desktop/Sophomore_Spring_PIC_16B/PIC16B---GAN-Project-/cats/1.jpg')
# im_np = np.array(im)
# plt.imshow(im_np), plt.axis('off'), plt.show()

#####Gaussian_noise#####

def Gaussian_noise (image, alpha = 0.1): 
    """
    Add Gaussian noise to images

    Input Arguments: 
    image: one image (RGB channel range (0,255)) 
    alpha: a parameter for adding Gaussian noise 
        choices of alpha: 0, 0.1, 0.25, 0.3, 0.4 
    """
    image = (image - 127.5)/127.5
    mean = 0
    sigma = 1
    shape = image.shape
    gaussian = np.random.normal(mean, sigma, shape)  #generate gaussian noise with the same shape as the input images 

    interpolated = (1-alpha)*image + alpha*gaussian

    interpolated = (interpolated*127.5 + 127.5).astype(int)
    return interpolated



#####Gaussian_Blur#####

def Gaussian_Blur (images, ksize = 5, alpha = 0): 
  """
  Convolve images with a Gaussian kernel 

  Input Arguments: 
  image: all training data 
  ksize: kernel size 
  alpha: standard deviation of the Gaussian kernel 
    choices of alpha: 0, 1, 2, 4

  """
  
  if ksize == 0:
      return images
  # kernel = np.random.normal(mean, sigma, (ksize,ksize)) #Gaussian Kernel
  examp_num = images.shape[0]
  new_images = []
  for i in range(examp_num): 
    image = images[i].copy()
    new = cv.GaussianBlur(image, ksize=(int(ksize), int(ksize)), sigmaX=int(alpha))
    new_images.append(new)
  new_images = np.array(new_images)
  return new_images





#####Add Random Rectangles#####

def rect(res, share, hi=64, wi=64, chan=3):
    '''
    Apply n_rect numbers of black rectangles to images
    
    Input Arguments:
    image_num: number of images in input
    res: training data(RGD channel range(0,225),4d)
    share: control the size of implanted rectangles(0-1)
    hi,wi,chan: shape of images
    '''
    if share == 0:
        return res
    image_num = res.shape[0]
    result = np.zeros_like(res)
    for i in range(image_num):

        rhi = np.int(hi*share)
        rwi = np.int(wi*share)
        xpos = random.randint(0, hi-rhi)            
        ypos = random.randint(0, wi-rwi)
        xdim = xpos + rhi
        ydim = ypos + rwi
        
        img_i = res[i,:].copy()

        img_i[xpos:xdim,ypos:ydim,:] = np.ones((rhi, rwi, chan))*0.0
        result[i,:,:,:]=img_i
    return result

#####Swirl#####



# In[106]:
def apply_swirl(res, n_swirls, radius=30, strength=3, hi=64, wi=64, chan=3):
    '''
    Apply Swirl to images
    
    Input Arguments:
    image_num: number of images in the input
    res: training data(number of images, RGD channel range(0,225),4 dim)
    n_swirls: number of swirls applied
    hi,wi,chan: shape of images
    '''
    if n_swirls == 0:
        return res
    image_num = res.shape[0]
    result = np.zeros_like(res).astype(float)
    for i in range(image_num):
        img = res[i,:].copy()

        for j in range(n_swirls):

            xpos = hi // 2
            ypos = wi // 2
            center = (xpos,ypos)
            img = swirl(img, rotation=0, strength=strength, radius=radius, center=center)
        result[i,:,:,:]=img
    return result
#####Test#######
# plt.imshow(Gaussian_noise(im_np)), plt.axis('off'), plt.show()
# plt.imshow(Gaussian_Blur(im_np)), plt.axis('off'), plt.show()
# plt.imshow(rect(im_np,n_rect=2, share=0.10)), plt.axis('off'), plt.show()
# plt.imshow(apply_swirl(im_np,n_swirls=1,radius=70,strength=4.0)), plt.axis('off'), plt.show()






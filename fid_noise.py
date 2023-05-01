from gan import GAN
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm

def calculate_fid(model, images1, images2):
    """
    Calculating FID score between training images and training images with noise. 
    """
    print('calculating FID...')
    # calculate activations
    # model = gan.inception_classifier
    images1 = np.kron(images1, np.ones(shape=(1, 2, 2, 1)))
    images2 = np.kron(images2, np.ones(shape=(1, 2, 2, 1)))
    print('extracting features...')
    act1 = model.predict(images1)
    act2 = model.predict(images2)
	# calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
	    covmean = covmean.real
	# calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    print('finished!')
    return fid


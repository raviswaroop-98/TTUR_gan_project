import os
import numpy as np
from keras import models
import matplotlib.pyplot as plt
import seaborn as sns
from gan import GAN


SAMPLE_SIZE = 7000
fid_scores = []

# check if FID scores have already been recorded
if os.path.exists('fid_scores.npy'):
    fid_scores = np.load('fid_scores.npy')
# get FID scores from saved model stages
else:
    gan_model = GAN()
    for i in range(10, 501, 10):
        gan_model.generator = models.load_model('models/gen_model{}'.format(i))
        fid_scores.append(gan_model.FID(SAMPLE_SIZE))
    # save numpy array file
    fid_arr = np.array(fid_scores)
    np.save('fid_scores.npy', fid_arr)

# plot  FID scores
x = np.arange(10, 501, 10)
print(x)
fig = sns.lineplot(x=x, y=fid_scores)
fig.set(xlabel='Epochs',
        ylabel='Frechet Inception Distance',
        title='FID Score at Intervals of 10 Epochs')
plt.savefig('./images/fid_scores_per_epoch')

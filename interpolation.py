import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tensorflow as tf
from gan import GAN

# code that helps prevent my kernal from dying while training on gpu
# comment out if not needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# create and load model
gan_model = GAN()
gan_model.load_model()

plt.figure(figsize=(8, 6))

# generated a range of images to visualize interpolation
for i in range(5):
    input_arr = np.random.randn(1, 100)
    for j in range(10):
        plt.subplot(5, 10, i * 10 + j + 1)
        plt.axis('off')
        interp_image = gan_model.generator.predict(input_arr + j * 0.3 - 1)
        interp_image = (interp_image + 1.0) / 2.0
        plt.imshow(interp_image[0])

plt.savefig('interpolation.png')

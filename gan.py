import os
import numpy as np
from scipy import linalg
from scipy.ndimage import map_coordinates
from keras import optimizers, models, layers, Sequential, applications
from tensorflow.keras.utils import load_img
import matplotlib.pyplot as plt
import tensorflow


class GAN:
    """
    Generative Adversarial Network class used to load a cat dataset and
    train a model.
    """
    def __init__(self):
        """
        GAN class variables can be set through train() or load_model()
        """
        self.dis_lr = None
        self.gen_lr = None
        self.alpha = None
        self.epochs = None
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.dataset = None
        self.plot_generator_input = self.generator_input(16)
        self.inception_classifier = applications.inception_v3.InceptionV3(
            include_top=False,
            pooling='avg',
            input_shape=(128, 128, 3))

    @staticmethod
    def load_data():
        """
        Load cat dataset from folder with values in range [-1, 1]
        """

        if os.path.exists('data.npy'):
            images = np.load('data.npy')

        else:
            images = np.array([], dtype=np.uint8).reshape((0, 64, 64, 3))
            for file in os.listdir('./cats'):
##                import pdb;pdb.set_trace()
#                if file.path.endswith('.jpg'):
                img_file = os.path.join('./cats',file)
#                import pdb;pdb.set_trace()
                images = np.append(images, load_img(img_file))
            images = np.reshape(images, newshape=(15747, 64, 64, 3))
            images = (images.astype('float32') - 127.5) / 127.5

            np.save('data.npy', images)

        return images

    def generator_model(self):
        """
        Create a generator model
        """
        # tried to impliment batch normalization but didn't trail well
        model = Sequential([
            # layer 1 - 4x4 array
            layers.Dense(256 * 4 * 4, input_shape=(100, )),
            layers.Reshape((4, 4, 256)),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 2 - 8x8 array
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                   padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 3 - 16x16 array
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                   padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 4 - 32x32 array
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                   padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 5 - 64x64 array
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # output layer - 64x64x3 array
            layers.Conv2D(3, (3, 3), activation='tanh', padding='same')
        ])

        return model

    def discriminator_model(self):
        """
        Create a discriminator model
        """
        model = Sequential([
            # layer 1
            layers.Conv2D(64, (3, 3), padding='same', input_shape=(64, 64, 3)),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 2
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 3
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 4
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 5
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # output layer
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

        return model

    def gan_model(self, generator, discriminator):
        """
        Combine generator and discriminator to create a GAN model
        """
        disc_adam = optimizers.Adam(lr=self.dis_lr, beta_1=0.5)
        discriminator.compile(loss='binary_crossentropy', optimizer=disc_adam)
        discriminator.trainable = False
        model = Sequential([generator, discriminator])
        gan_adam = optimizers.Adam(lr=self.gen_lr, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=gan_adam)
        return model

    def real_samples(self, size):
        """
        Load real samples from the dataset
        """
        indexes = np.random.randint(self.dataset.shape[0], size=size)
        real_images = self.dataset[indexes]
        return real_images

    @staticmethod
    def generator_input(sample_size):
        """
        Create input to be used by the generator
        """
        return np.random.randn(sample_size, 100)

    def train(self, batch_size, dis_lr, gen_lr, alpha, epochs):
        """
        Train the GAN model
        """
        self.dis_lr = dis_lr
        self.gen_lr = gen_lr
        self.alpha = alpha
        self.epochs = epochs
        self.dataset = self.load_data()

        # create a generator, discriminator and GAN
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.gan = self.gan_model(self.generator, self.discriminator)

        # create half batch to set  number of training examples
        # from real and training data
        half_batch = int(batch_size / 2)

        # number of batches to train per epoch
        batches = int(self.dataset.shape[0] / batch_size)

        # training
        for i in range(self.epochs):
            print('Training Epoch: {}'.format(i))
            for _ in range(batches):
                # train on real samples
                real_images = self.real_samples(half_batch)
                real_labels = np.ones(shape=(half_batch, 1))
                self.discriminator.train_on_batch(real_images, real_labels)
                # trian on fake samples
                fake_input = self.generator_input(half_batch)
                fake_images = self.generator.predict(fake_input)
                fake_labels = np.zeros(shape=(half_batch, 1))
                self.discriminator.train_on_batch(fake_images, fake_labels)
                # train GAN
                gan_input = self.generator_input(batch_size)
                gan_label = np.ones(shape=(batch_size, 1))
                self.gan.train_on_batch(gan_input, gan_label)
            # save a png in epochs folder to show progress
            if (i + 1) % 5 == 0:
                self.plot_generated_images('epochs/Epoch_{}'.format(i + 1))
            # save models to use for later analysis
            if (i + 1) % 10 == 0:
                self.generator.save('models/gen_model{}'.format(i + 1))
                self.discriminator.save('models/dis_model{}'.format(i + 1))
                self.gan.save('models/gan_model{}'.format(i + 1))

    def save_model(self):
        """
        Save a GAN model for later use
        """
        self.generator.save('generator_model')
        self.discriminator.save('discriminator_model')
        self.gan.save('gan_model')

    def load_model(self):
        """
        Load a previously saved GAN model
        """
        self.generator = models.load_model('generator_model')
        print('generator loaded')
        self.discriminator = models.load_model('discriminator_model')
        print('discriminator loaded')
        self.gan = models.load_model('gan_model')
        print('gan loaded')

    def plot_generated_images(self, filename):
        """
        Save a png of generated images
        """
        # generator fake images
        images = self.generator.predict(self.plot_generator_input)
        images = (images + 1) / 2.0

        # plot generator images on subplots
        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.axis('off')
            plt.imshow(images[i])

        # save figure as png
        plt.savefig(filename)

    def summary(self):
        """
        Print model summary
        """
        print("Generator Model Summary:")
        self.generator.summary()
        print('Discriminator Model Summary:')
        self.discriminator.summary()
        print('GAN Model Summary:')
        self.gan.summary()

    def FID(self, sample_size):
        """
        Evaluate the current generator model with frechet inception
        distance score.
        """
        # load dataset into the class
        self.dataset = self.load_data()
        print('Calculating FID score...')
        # create real and generated images to compare
        fake_gen_input = self.generator_input(sample_size)
        fake_imgs = self.generator.predict(fake_gen_input)
        # stretch images to 128x128
        fake_imgs = np.kron(fake_imgs, np.ones(shape=(1, 2, 2, 1)))
        # get real samples
        real_imgs = self.real_samples(sample_size)
        # stretch images to 128x128
        real_imgs = np.kron(real_imgs, np.ones(shape=(1, 2, 2, 1)))
        # calculate activations
        fake_act = self.inception_classifier.predict(fake_imgs)
        real_act = self.inception_classifier.predict(real_imgs)
        # calculate mean and covariance statistics
        mu1, sigma1 = fake_act.mean(axis=0), np.cov(fake_act, rowvar=False)
        mu2, sigma2 = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        print('FID score: ', fid)
        return fid

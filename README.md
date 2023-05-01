### Load a Model
The `main.py` file demostrates how to load a previously trained GAN model
and output new images generated by the model.

```
from gan import GAN

# Load last trained model to skip training time
gan_model = GAN()
gan_model.load_model()
# save a 4x4 grid of images generated by the model
gan_model.plot_generated_images('generated_images')
# Prints the generator, discriminator, and GAN model summary
gan_model.summary()
```


### Train the Model
The `training.py` file demonstrates how to create a model with specified
hyperparameters. Every 5 epochs the model will output images will to `epochs`
to visualize trianing progress. The following code can take several hours to 
run.

```
from gan import GAN

# hyper parameters
BATCH_SIZE = 128
DISCRIMINATOR_LR = 5e-5
GENERATOR_LR = 2e-4
EPOCHS = 500
ALPHA = 5e-1


# training the model from scratch
gan_model = GAN()
# train the model wtih the given hyperparamters
gan_model.train(BATCH_SIZE, DISCRIMINATOR_LR, GENERATOR_LR, ALPHA, EPOCHS)
# save a 4x4 grid of images generated by the model
gan_model.plot_generated_images('generated_images')
# print a summary of the generator, discriminator, and GAN model
gan_model.summary()
# save model
gan_model.save_model()
```


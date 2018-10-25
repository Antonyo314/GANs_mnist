from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
batch_size = 32
epochs = 30000
sample_interval = 200


# os.chdir(os.getcwd())
# print(os.getcwd())


def get_generator():
    input_noise = Input(shape=(latent_dim,))

    x = Dense(256)(input_noise)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Dense(np.prod(img_shape), activation='tanh')(x)
    img = Reshape(img_shape)(x)

    return Model(input_noise, img)


def get_discriminator():
    img = Input(shape=img_shape)
    x = Flatten()(img)

    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(1, activation='sigmoid')(x)
    return Model(img, x)


def get_combined():
    z = Input(shape=(latent_dim,))
    img = generator(z)
    validity = discriminator(img)
    return Model(z, validity)


def sample_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig('generated_images/%d.png' % epoch)
    plt.close()


optimizer = Adam(0.0002, 0.5)

generator = get_generator()
discriminator = get_discriminator()

combined = get_combined()

discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

discriminator.trainable = False

combined.compile(loss='binary_crossentropy', optimizer=optimizer)

(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    gen_imgs = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Train the generator (to have the discriminator label samples as valid)
    g_loss = combined.train_on_batch(noise, valid)

    # Plot the progress
    print("%d [D loss: %f, ac   c.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    if epoch % sample_interval == 0:
        sample_images(epoch)

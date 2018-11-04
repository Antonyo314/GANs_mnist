from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import scipy
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Conv2D, UpSampling2D, Concatenate, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_contrib.layers.normalization import InstanceNormalization

img_rows = 32  # power of 2
img_cols = 32
channels = 1
img_shape = (img_rows, img_cols, channels)
batch_size = 32
epochs = 100_000
sample_interval = 500

mask_w = 10
mask_h = 10

gf = 32
k = 4
s = 2

num_classes = 10


def build_generator():
    """U-net"""

    # Generator input
    img_g = Input(shape=img_shape)

    # Downsampling
    d1 = Conv2D(gf, kernel_size=k, strides=s, padding='same')(img_g)
    d1 = LeakyReLU(alpha=0.2)(d1)

    d2 = Conv2D(gf * 2, kernel_size=k, strides=s, padding='same')(d1)
    d2 = LeakyReLU(alpha=0.2)(d2)
    d2 = BatchNormalization(momentum=0.8)(d2)

    d3 = Conv2D(gf * 4, kernel_size=k, strides=s, padding='same')(d2)
    d3 = LeakyReLU(alpha=0.2)(d3)
    d3 = BatchNormalization(momentum=0.8)(d3)

    d4 = Conv2D(gf * 8, kernel_size=k, strides=s, padding='same')(d3)
    d4 = LeakyReLU(alpha=0.2)(d4)
    d4 = BatchNormalization(momentum=0.8)(d4)

    # Upsampling
    u1 = UpSampling2D(size=2)(d4)
    u1 = Conv2D(gf * 4, kernel_size=k, strides=1, padding='same', activation='relu')(u1)
    u1 = BatchNormalization(momentum=0.8)(u1)

    u2 = Concatenate()([u1, d3])
    u2 = UpSampling2D(size=2)(u2)
    u2 = Conv2D(gf * 2, kernel_size=k, strides=1, padding='same', activation='relu')(u2)
    u2 = BatchNormalization(momentum=0.8)(u2)

    u3 = Concatenate()([u2, d2])
    u3 = UpSampling2D(size=2)(u3)
    u3 = Conv2D(gf, kernel_size=k, strides=1, padding='same', activation='relu')(u3)
    u3 = BatchNormalization(momentum=0.8)(u3)

    u4 = Concatenate()([u3, d1])
    u4 = UpSampling2D(size=2)(u4)
    u4 = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(img_g, u4)


def build_discriminator():
    img_d = Input(shape=img_shape)

    x = Conv2D(64, kernel_size=k, strides=2, padding='same', input_shape=img_shape)(img_d)
    x = LeakyReLU(alpha=0.8)(x)
    x = Conv2D(128, kernel_size=k, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = InstanceNormalization()(x)
    x = Conv2D(256, kernel_size=k, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = InstanceNormalization()(x)

    validity = Conv2D(1, kernel_size=k, strides=1, padding='same')(x)
    label = Flatten()(x)
    label = Dense(num_classes + 1, activation="softmax")(label)

    return Model(img_d, [validity, label])


def build_combined():
    img_masked = Input(shape=img_shape)

    validity, _ = discriminator(generator(img_masked))

    return Model(img_masked, validity)


def sample_images(epoch, imgs):
    r, c = 3, 6

    masked_imgs = random_mask(imgs)
    gen_imgs = generator.predict(masked_imgs)

    imgs = (imgs + 1.0) * 0.5
    masked_imgs = (masked_imgs + 1.0) * 0.5
    gen_imgs = (gen_imgs + 1.0) * 0.5

    gen_imgs = np.where(gen_imgs < 0, 0, gen_imgs)

    fig, axs = plt.subplots(r, c)
    for i in range(c):
        axs[0, i].imshow(imgs[i, :, :, 0], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(masked_imgs[i, :, :, 0], cmap='gray')
        axs[1, i].axis('off')
        axs[2, i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
        axs[2, i].axis('off')
    fig.savefig('generated_images/%d.png' % epoch)
    plt.close()


def random_mask(imgs_):
    imgs = imgs_.copy()

    for i in range(imgs.shape[0]):
        x = randrange(0, img_cols - mask_w)
        y = randrange(0, img_rows - mask_h)

        imgs[i][x:x + mask_w, y:y + mask_h, 0] = 0

    return imgs


optimizer = Adam(0.0002, 0.5)

generator = build_generator()
discriminator = build_discriminator()

combined = build_combined()

discriminator.compile(loss=['mse', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy'])

discriminator.trainable = False

combined.compile(loss='mse', optimizer=optimizer)

(X, y), (_, _) = mnist.load_data()

X = np.array([scipy.misc.imresize(x, [img_rows, img_cols]) for x in X])

# Rescale -1 to 1
X = X / 127.5 - 1.
X = np.expand_dims(X, axis=3)

X_masked = random_mask(X)

y_cat = to_categorical(y, num_classes=num_classes + 1).astype(
    np.float32)  # 11 classes (0-9 are digit labels and 10 is fake label

# Adversarial ground truths
valid = np.ones((batch_size, 4, 4, 1))
fake = np.zeros((batch_size, 4, 4, 1))

for epoch in range(epochs):
    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, X.shape[0], batch_size)
    imgs, imgs_masked = X[idx], X_masked[idx]
    labels = y_cat[idx]

    gen_imgs = generator.predict(imgs_masked)

    fake_labels = to_categorical([num_classes] * batch_size, num_classes=num_classes + 1)

    d_loss_real = discriminator.train_on_batch(imgs, [valid, labels])
    d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    # Train the generator (to have the discriminator label samples as valid)
    g_loss = combined.train_on_batch(imgs_masked, valid)

    # Plot the progress
    print("%d [D loss: %f, ac   c.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[4], g_loss))

    if epoch % sample_interval == 0:
        sample_images(epoch, X)

generator.save('saved_models/generator.h5')

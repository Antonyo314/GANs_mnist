import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
batch_size = 32
epochs = 100_000
sample_interval = 500
num_classes = 10


def build_generator():
    input_noise = Input(shape=(latent_dim,))
    input_lbl = Input(shape=(num_classes,), dtype='float32')
    x = concatenate([input_noise, input_lbl])

    x = Dense(256)(x)
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

    return Model([input_noise, input_lbl], img)


def build_discriminator():
    img = Input(shape=img_shape)
    input_lbl = Input(shape=(num_classes,), dtype='float32')

    x = Flatten()(img)
    x = concatenate([x, input_lbl])

    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(1, activation='sigmoid')(x)
    return Model([img, input_lbl], x)


def build_combined():
    z = Input(shape=(latent_dim,))
    input_lbl_generator = Input(shape=(num_classes,), dtype='float32')
    img = generator([z, input_lbl_generator])

    input_lbl_discriminator = Input(shape=(num_classes,), dtype='float32')
    validity = discriminator([img, input_lbl_discriminator])
    return Model([z, input_lbl_generator, input_lbl_discriminator], validity)


def sample_images(epoch):
    r, c = 10, 10
    fig, axs = plt.subplots(r, c)

    for i in range(r):
        noise = np.random.normal(0, 1, (1, latent_dim))
        for j in range(c):
            y = np.zeros((1, 10))
            y[0, j] = 1
            gen_imgs = generator.predict([noise, y])

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            axs[i, j].imshow(gen_imgs[0, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
    fig.savefig('generated_images/%d.png' % epoch)
    plt.close()


optimizer = Adam(0.0002, 0.5)

generator = build_generator()
discriminator = build_discriminator()

combined = build_combined()

discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

discriminator.trainable = False

combined.compile(loss='binary_crossentropy', optimizer=optimizer)

(X_train, y_train), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

y_train_cat = to_categorical(y_train).astype(np.float32)
# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs, labels = X_train[idx], y_train_cat[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    gen_imgs = generator.predict([noise, labels])
    d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
    d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Train the generator (to have the discriminator label samples as valid)
    g_loss = combined.train_on_batch([noise, labels, labels], valid)

    # Plot the progress
    print("%d [D loss: %f, ac   c.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    if epoch % sample_interval == 0:
        sample_images(epoch)

generator.save('saved_models/generator.h5')

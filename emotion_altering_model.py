import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def load_images_from_folder(base_folder):
    images = []
    labels = []
    label_map = {}
    for idx, folder in enumerate(os.listdir(base_folder)):
        label_map[folder] = idx
        folder_path = os.path.join(base_folder, folder)
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(idx)
    return np.array(images), np.array(labels), label_map

base_folder = 'data_for_GAN/'
images, labels, label_map = load_images_from_folder(base_folder)
images = images / 255.0


def build_generator(noise_dim, num_labels):
    noise_input = layers.Input(shape=(noise_dim,))
    label_input = layers.Input(shape=(num_labels,))


    x = layers.concatenate([noise_input, label_input], axis=1)  # axis=1 to concatenate along the feature axis

    # Start with a smaller Dense layer, so the reshaped dimensions are smaller
    x = layers.Dense(128 * 32 * 32, use_bias=False)(x)  # Adjusted to 32 * 32
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((32, 32, 128))(x)  # Adjusted to 32x32
    # Conv2DTranspose to upscale to 64x64
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Conv2DTranspose to upscale to 128x128
    x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)  # Now 128x128x3

    model = models.Model(inputs=[noise_input, label_input], outputs=x)
    return model

def build_discriminator(image_shape, num_labels):
    image_input = layers.Input(shape=image_shape)
    label_input = layers.Input(shape=(num_labels,))
    label_dense = layers.Dense(np.prod(image_shape))(label_input)
    label_reshape = layers.Reshape(image_shape)(label_dense)

    x = layers.concatenate([image_input, label_reshape])
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    model = models.Model(inputs=[image_input, label_input], outputs=x)
    return model

generator = build_generator(100, len(label_map))
discriminator = build_discriminator((128, 128, 3), len(label_map))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
@tf.function
def train_step(images, labels, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy, noise_dim):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, noise_dim])

    one_hot_labels = tf.one_hot(labels, depth=len(label_map))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, one_hot_labels], training=True)

        real_output = discriminator([images, one_hot_labels], training=True)
        fake_output = discriminator([generated_images, one_hot_labels], training=True)

        gen_loss = generator_loss(fake_output, cross_entropy)
        disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    real_accuracy = tf.reduce_mean(tf.cast(tf.math.greater_equal(real_output, 0.5), tf.float32))
    fake_accuracy = tf.reduce_mean(tf.cast(tf.math.less(fake_output, 0.5), tf.float32))
    disc_accuracy = (real_accuracy + fake_accuracy) / 2

    return gen_loss, disc_loss, disc_accuracy


def train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy, noise_dim):
    gen_losses = []
    disc_losses = []
    disc_accuracies = []

    for epoch in range(epochs):
        total_gen_loss = 0
        total_disc_accuracy = 0
        total_disc_loss = 0
        num_batches = 0

        for image_batch, label_batch in dataset:
            gen_loss, disc_loss, disc_accuracy = train_step(image_batch, label_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy, noise_dim)
            total_gen_loss += gen_loss
            total_disc_loss += disc_loss
            total_disc_accuracy += disc_accuracy
            num_batches += 1

        # Average losses and accuracy over all batches
        avg_gen_loss = total_gen_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches
        avg_disc_accuracy = total_disc_accuracy / num_batches

        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        disc_accuracies.append(avg_disc_accuracy)

        print(
            f"Epoch {epoch + 1}, Gen Loss: {avg_gen_loss}, Disc Loss: {avg_disc_loss}, Disc Accuracy: {avg_disc_accuracy}")

    return gen_losses, disc_losses, disc_accuracies

noise_dim = 100
train_dataset = tf.data.Dataset.from_tensor_slices((images, tf.cast(labels, tf.int32))).batch(32)
gen_losses, disc_losses, disc_accuracies = train(train_dataset, 20, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy, noise_dim)

# Plotting
epochs_range = range(1, 21)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, gen_losses, label='Generator Loss')
plt.plot(epochs_range, disc_losses, label='Discriminator Loss')
plt.title('Generator and Discriminator Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, disc_accuracies, label='Discriminator Accuracy', color='orange')
plt.title('Discriminator Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
def alter_emotion(image, target_emotion, generator, noise_dim, label_map):
    noise = tf.random.normal([1, noise_dim])
    label_index = label_map[target_emotion]
    one_hot_label = tf.one_hot([label_index], depth=len(label_map))

    altered_image = generator([noise, one_hot_label], training=False)
    altered_image = (altered_image[0, :, :, :] * 127.5 + 127.5).numpy()
    altered_image = np.clip(altered_image, 0, 255).astype(np.uint8)
    return altered_image



def plot_altered_image(image):
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and labels
    plt.show()

# Example usage of the alter_emotion and plot_altered_image functions
input_image_path = 'image_2.jpg'  # Path to the image you want to alter
target_emotion = 'happy'  # The desired emotion
noise_dim = 100  # Make sure this matches the dimension used in the generator

# Load and preprocess the user image
user_image = cv2.imread(input_image_path)
user_image = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)
user_image = cv2.resize(user_image, (128, 128))  # Resize to match the generator's expected input
user_image = user_image / 255.0  # Normalize pixel values

# Alter the emotion of the image
altered_image = alter_emotion(user_image, target_emotion, generator, noise_dim, label_map)

# Plot the altered image
plot_altered_image(altered_image)


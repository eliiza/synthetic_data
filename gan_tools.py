import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def add_dense_layer(model,
                    nodes,
                    activation,
                    use_bias=False):
    """
    Function to add a dense layer within a defined model.
    NB: LeakyReLU is atm a special activation function in Keras and has to be added as a separate layer, hence
    all such activations must be passed as a layer/activation function and cannot be specified through their
    string identifier.
    Args:
        model: [object] Keras sequential model
        nodes: [list] list of number of nodes for hidden layers
        activation: [object] activation function for hidden layers (default: LeakyReLU)

    Returns:
        model
    """
    model.add(layers.Dense(nodes, use_bias=use_bias))
    model.add(layers.BatchNormalization())
    model.add(activation())
    return model


def make_dense_model(dims,
                     activation,
                     inputs: int,
                     outputs: int = 1,
                     output_activation: str = 'tanh'):
    """
    Make generator based on number of nodes given for hidden layers and activation function specified.
    Number of noise inputs is given as well as expected number of outputs.
    TODO: compare number of outputs to number of columns in data - do inside class before passing on values
    TODO: add choice between dense and 1D-convolution?
    TODO: add choice of activation for final layer
    Args:
        dims: [list] numbers of nodes for dense layers
        activation: [object] activation function for dense layers
        inputs: [int] number of inputs
        outputs: [int] number of outputs

    Returns:
        model
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(inputs))

    for nodes in dims:
        add_dense_layer(model, nodes, activation)

    model.add(layers.Dense(outputs, use_bias=False, activation=output_activation))

    return model


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_step(data: object,
               generator: object,
               discriminator: object,
               batch_size: int,
               noise_dim: int,
               generator_optimizer: object=tf.keras.optimizers.Adam(1e-5),
               discriminator_optimizer: object=tf.keras.optimizers.Adam(5e-6)) -> object:
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)

        real_output = discriminator(data, training=True)
        fake_output = discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train_gan(dataset: object,
              generator: object,
              discriminator: object,
              epochs: int,
              batch_size: int,
              noise_dim: int,
              generator_optimizer: object=tf.keras.optimizers.Adam(1e-5),
              discriminator_optimizer: object=tf.keras.optimizers.Adam(5e-6),
              save_model: object = True,
              checkpoint_dir: object = './training_checkpoints') -> object:
    for epoch in range(epochs):
        start = time.time()
        g_losses = []
        d_losses = []

        for data_batch in dataset:
            g_loss, d_loss = train_step(data_batch,
                                       generator=generator,
                                       discriminator=discriminator,
                                       batch_size=batch_size,
                                       noise_dim=noise_dim,
                                       generator_optimizer=generator_optimizer,
                                       discriminator_optimizer=discriminator_optimizer)
            g_losses.append(g_loss)
            d_losses.append(d_loss)

        # Save the model every 15 epochs
        if save_model:
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                             discriminator_optimizer=discriminator_optimizer,
                                             generator=generator,
                                             discriminator=discriminator)
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print("Generator loss: {}".format(np.mean(g_losses)))
        print("Discriminator loss: {}".format(np.mean(d_losses)))

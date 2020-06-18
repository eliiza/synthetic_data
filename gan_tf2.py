########################
# Tensorflow 2.2 implementation of the 1D-GAN found at
# https://github.com/dialnd/imbalanced-algorithms
########################

import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils import check_random_state


class GAN(object):
    """
    Generative Adversarial Network (GAN) implemented using TensorFlow.

     The GAN framework uses two iteratively trained adversarial networks to
     estimate a generative process. A generative model, G, captures the data
     distribution, while a discriminative model, D, estimates the probability
     that a sample came from the training data rather than from G, the
     generative model [1].

     Parameters
     ----------
     num_epochs : int
         Passes over the training dataset.
     batch_size : int
         Size of minibatches for stochastic optimizers.
     d_hidden_dim : list
         Discriminator number of units per hidden layer.
     g_hidden_dim : list
         Generator number of units per hidden layer.
     n_input : int
         Number of inputs to initial layer.
     d_transfer_fct : object
         Discriminator transfer function for hidden layers.
     g_transfer_fct : object
         Generator transfer function for hidden layers.
     d_learning_rate : float
         Discriminator learning rate schedule for weight updates.
     g_learning_rate : float
         Generator learning rate schedule for weight updates.
     random_state : int or None, optional (default=None)
         If int, random_state is the seed used by the random number generator.
         If None, the random number generator is the RandomState instance used
         by np.random.
     log_every : int
         Print loss after this many steps.

     References
     ----------
     .. [1] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D.
            Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. "Generative
            Adversarial Nets". Advances in Neural Information Processing
            Systems 27 (NIPS), 2014.

     Notes
     -----
     Based on related code:
         - https://www.tensorflow.org/tutorials/generative/dcgan
     """

    def __init__(self, num_epochs: int,
                 batch_size: int,
                 d_hidden_dims: list,
                 g_hidden_dims: list,
                 n_inputs: int,
                 g_outputs: int,
                 d_activation: object = layers.LeakyReLU,
                 g_activation: object = layers.LeakyReLU,
                 d_learning_rate: float = 0.01,
                 g_learning_rate: float = 0.0005,
                 d_optimiser: object = tf.keras.optimizers.Adam,
                 g_optimiser: object = tf.keras.optimizers.Adam,
                 random_state: int = 42,
                 log_every: int = None):

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.network_dims = {
            'd_hidden_dims': d_hidden_dims,
            'g_hidden_dims': g_hidden_dims,
            'n_inputs': n_inputs,
            'g_outputs': g_outputs
        }

        self.d_activation = d_activation
        self.g_activation = g_activation

        self.d_optimiser = d_optimiser(learning_rate=d_learning_rate)
        self.g_optimiser = g_optimiser(learning_rate=g_learning_rate)

        self._initialise_gan()

        # This method returns a helper function to compute cross entropy loss
        # TODO: replace with choice of custom loss function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.random_state = check_random_state(random_state)
        tf.random.set_seed(random_state)

        self.log_every = log_every

    def _add_dense_layer(self,
                         model,
                         nodes,
                         activation,
                         use_bias=False):
        """
        Function to add a dense layer within a defined model.
        NB: LeakyReLU is atm an advanced activation function in Keras and has to be added as a separate layer, hence
        all activations for the hidden layers must be passed as a layer/activation function and cannot be specified
        through their string identifier.
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

    def _make_dense_model(self,
                          dims,
                          activation,
                          inputs: int,
                          outputs: int = 1,
                          output_activation: str = 'tanh'):
        """
        Make generator based on number of nodes given for hidden layers and activation function specified.
        Number of noise inputs is given as well as expected number of outputs.
        TODO: compare number of outputs to number of columns in data - do inside class before passing on values
        TODO: add choice between dense and 1D-convolution?
        Args:
            dims: [list] numbers of nodes for dense layers
            activation: [object] activation function for dense layers
            inputs: [int] number of inputs
            outputs: [int] number of outputs
            output_activation: [str] default='tanh', string identifier for final activation function

        Returns:
            model
        """

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(inputs))

        for nodes in dims:
            self._add_dense_layer(model, nodes, activation)

        model.add(layers.Dense(outputs, use_bias=False, activation=output_activation))

        return model

    def _initialise_gan(self):
        self.generator = self._make_dense_model(dims=self.network_dims['g_hidden_dims'],
                                                activation=self.g_activation,
                                                inputs=self.network_dims['n_inputs'],
                                                outputs=self.network_dims['g_outputs'])
        self.discriminator = self._make_dense_model(dims=self.network_dims['d_hidden_dims'],
                                                    activation=self.d_activation,
                                                    inputs=self.network_dims['g_outputs'],
                                                    output_activation='linear')

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def _train_step(self,
                    data: object):
        noise = tf.random.normal([self.batch_size, self.network_dims['n_inputs']])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)

            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimiser.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimiser.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def _train_gan(self,
                   dataset: object,
                   save_model: bool = True,
                   checkpoint_dir: str = './training_checkpoints') -> object:

        for epoch in range(self.num_epochs):
            start = time.time()
            g_losses = []
            d_losses = []

            for data_batch in dataset:
                g_loss, d_loss = self._train_step(data_batch)

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

    def fit(self,
            data,
            buffer_size: int = None,
            save_model: bool = False,
            checkpoint_dir: str = './training_checkpoints') -> object:
        """
        Fit GAN to data.
        #TODO: check size of dataset vs buffersize/batch size
        #TODO: check number of columns vs number of outputs in GAN
        Args:
            data: numpy array with real data to be imitated
            buffer_size: amount of data to be trained on
            save_model:
            checkpoint_dir:

        Returns:

        """
        if not buffer_size:
            buffer_size = data.shape[0]

        if data.shape[1] != self.network_dims['g_outputs']:
            raise Exception("Number of variables in data and generator output do not match.")

        # Shuffle and batch data
        training_data = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size).batch(self.batch_size)

        self._train_gan(dataset=training_data,
                        save_model=save_model,
                        checkpoint_dir=checkpoint_dir)

        return self

    def sample(self,
               n_samples: int = 1):
        """
        lalala sampling
        Args:
            n_samples: int, default=1, number of samples to produce

        Returns:
            generated data
        """
        noise = tf.random.normal([n_samples, self.network_dims['n_inputs']])
        samples = self.generator.predict(noise)
        return samples

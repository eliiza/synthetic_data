########################
# Tensorflow 2.2 implementation of the 1D-GAN found at
# https://github.com/dialnd/imbalanced-algorithms
########################


import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mmd
from matplotlib.colors import Normalize


class MMD_GAN(object):
    """
    Generative Adversarial Network (GAN) implemented using TensorFlow.

     The GAN framework uses two iteratively trained adversarial networks to
     estimate a generative process. A generative model, G, captures the data
     distribution, while a discriminative model, D, estimates the probability
     that a sample came from the training data rather than from G, the
     generative model [1].

     Parameters
     ----------
     num_epochs: int
         Passes over the training dataset.
     batch_size: int
         Size of minibatches for stochastic optimizers.
     num_data_cols: int
         Number of columns in training data.
     d_hidden_layers: list
         List of dictionaries with type and dictionary of arguments per layer of discriminator.
         Example: [{'type': 'dense', {'units': 16, 'activation': 'relu'}},
                   {'type': 'dense', {'units': 10, 'activation': 'relu'}}]
     g_hidden_layers: list
        List of dictionaries with type and dictionary of arguments per layer of generator.
        Example: [{'type': 'dense', {'units': 16, 'activation': 'relu'}},
                  {'type': 'dense', {'units': 10, 'activation': 'relu'}}]
     noise_inputs: int, default = 100
         Number of inputs to generator, dimension of noise space Z.
     d_outputs: int, default = 1
         Number of outputs for discriminator.
     d_output_activation: str, default = 'linear'
         String name for discriminator output activation function as per Keras string codes.
     g_output_activation: str, default = 'sigmoid'
         String name for generator output activation function as per Keras string codes.
     d_learning_rate: float, default = 0.001
         Discriminator base learning rate for weight updates.
     g_learning_rate: float, default = 0.0005
         Generator base learning rate for weight updates.
     d_optimiser: object, default = tf.keras.optimizers.Adam
         Optimisation function for discriminator.
     g_optimiser: object, default = tf.keras.optimizers.Adam
         Optimisation function for generator.
     kernel: str, default = 'dot'
         Kernel function for loss. Example: 'dot' or 'mix_rq_dot'
     random_state: int or None, optional (default=42)
         If int, random_state is the seed used by the random number generator.
         If None, the random number generator is the RandomState instance used
         by np.random.
     log_every: int, default = 5
         Print loss after this many steps.
     image_generation: bool, default = False
         Whether to save an example image upon evaluation. Only useful for comparison to image training data.
     image_dir: str, default = 'img'
         Directory to same result images.
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
         - https://github.com/mbinkowski/MMD-GAN
     """

    def __init__(self, num_epochs: int,
                 batch_size: int,
                 num_data_cols: int,
                 d_hidden_layers: list,
                 g_hidden_layers: list,
                 noise_inputs: int = 100,
                 d_outputs: int = 1,
                 d_output_activation: str = 'linear',
                 g_output_activation: str = 'sigmoid',
                 d_learning_rate: float = 0.001,
                 g_learning_rate: float = 0.0005,
                 d_optimiser: object = tf.keras.optimizers.Adam,
                 g_optimiser: object = tf.keras.optimizers.Adam,
                 kernel: str = 'dot',
                 random_state: int = 42,
                 log_every: int = 5,
                 image_generation: bool = False,
                 image_dir: str = 'img'):

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.d_architecture = { 'num_inputs': num_data_cols,
                                'hidden_layers': d_hidden_layers,
                                'optimiser': d_optimiser,
                                'learning_rate': d_learning_rate,
                                'num_outputs': d_outputs,
                                'output_activation': d_output_activation
                                }

        self.g_architecture = { 'num_inputs': noise_inputs,
                                'hidden_layers': g_hidden_layers,
                                'optimiser': g_optimiser,
                                'learning_rate': g_learning_rate,
                                'num_outputs': num_data_cols,
                                'output_activation': g_output_activation
                                }

        self.d_optimiser = d_optimiser(learning_rate=d_learning_rate)
        self.g_optimiser = g_optimiser(learning_rate=g_learning_rate)

        self._initialise_gan()

        # Import appropriate kernel from mmd
        # TODO: replace with choice of custom loss function
        self.kernel = getattr(mmd, "_{}_kernel".format(kernel))

        self.random_state = check_random_state(random_state)
        tf.random.set_seed(random_state)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        self.log_every = log_every

        self.image_generation = image_generation
        self.image_dir = image_dir

    def _initialise_gan(self):
        self.generator = GAN.make_model(self.g_architecture)
        print("Initialised Generator: \n")
        print(self.generator.summary())
        self.discriminator = GAN.make_model(self.d_architecture)
        print("Initialised Discriminator: \n")
        print(self.discriminator.summary())


    @staticmethod
    def add_layer(model,
                  type: str,
                  layer: dict,
                  batch_normalisation: bool = False,
                  dropout: bool = False,
                  dropout_rate: float = 0.3,
                  advanced_activation: bool = False,
                  activation_layer: object = tf.keras.layers.LeakyReLU):
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
        if type == "dense":
            model.add(layers.Dense(**layer))
        elif type == 'conv':
            model.add(layers.Conv1D(**layer))
        else:
            raise ValueError('Layer type must be dense (dense) or convolutional (conv).')

        if batch_normalisation:
            model.add(layers.BatchNormalization())

        if dropout:
            model.add(layers.Dropout(rate=dropout_rate))

        if advanced_activation:
            model.add(activation_layer())

        return model

    @staticmethod
    def make_model(architecture: dict):
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
        if architecture['hidden_layers'][0]['type'] == 'dense':
            model.add(tf.keras.Input(architecture['num_inputs']))
        else:
            model.add(tf.keras.Input((architecture['num_inputs'], 1)))

        for layer in architecture['hidden_layers']:
            GAN.add_layer(model, **layer)

        model.add(layers.Dense(architecture['num_outputs'],
                               use_bias=False,
                               activation=architecture['output_activation']))

        return model

    def _loss_via_kernel(self, generator_features, data_features):
        kernel_function = self.kernel(generator_features, data_features)
        generator_loss = mmd.mmd2(kernel_function)
        discriminator_loss = -generator_loss
        return generator_loss, discriminator_loss

    def _train_step(self,
                    data: object):
        # TODO: Implement counter to differentiate discriminator and generator training

        noise = tf.random.normal([self.batch_size, self.g_architecture['num_inputs']])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)

            data_features = self.discriminator(data, training=True)
            generator_features = self.discriminator(generated_data, training=True)

            gen_loss, disc_loss = self._loss_via_kernel(generator_features, data_features)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimiser.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimiser.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def _plot_generated_images(self, epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
        generated_images = self.sample(examples)
        generated_images = generated_images.reshape(examples, 28, 28)
        print(generated_images[1] - generated_images[0])
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        image_name = os.path.join(self.image_dir, 'gan_generated_image_{epoch:04}.png'.format(epoch=epoch))
        plt.savefig(image_name)
        plt.close('all')

    def _train_gan(self,
                   dataset: object,
                   save_model: bool = True,
                   checkpoint_dir: str = './training_checkpoints') -> object:

        # Initialise running variables and lists
        start = time.time()
        total_time = 0
        g_losses = []
        d_losses = []

        for epoch in range(self.num_epochs):

            for data_batch in dataset:
                g_loss, d_loss = self._train_step(data_batch)

                g_losses.append(g_loss)
                d_losses.append(d_loss)

            # Save the model every 15 epochs
            if save_model:
                checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
                checkpoint = tf.train.Checkpoint(generator_optimizer=self.g_optimiser,
                                                 discriminator_optimizer=self.d_optimiser,
                                                 generator=self.generator,
                                                 discriminator=self.discriminator)
                if (epoch + 1) % 15 == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)

            if (epoch + 1) % self.log_every == 0:
                now = np.round(time.time() - start, 2)
                total_time = np.round(now + total_time, 2)
                print('Time for epoch {} is {} seconds.'.format(epoch + 1, now))
                print("Total time passed: {} seconds".format(total_time))
                print("Generator loss: {}".format(np.mean(g_losses)))
                print("Discriminator loss: {}".format(np.mean(d_losses)))
                if self.image_generation:
                    self._plot_generated_images(epoch=epoch)
                start = time.time()
                g_losses = []
                d_losses = []

    def fit(self,
            data,
            data_unscaled: bool = False,
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
        if data_unscaled:
            data = self.scaler.fit_transform(data)

        if not buffer_size:
            buffer_size = data.shape[0]

        if data.shape[1] != self.d_architecture['num_inputs']:
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
        noise = tf.random.normal([n_samples, self.g_architecture['num_inputs']])
        samples = self.generator.predict(noise)
        return samples



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
         - https://github.com/mbinkowski/MMD-GAN
     """

    def __init__(self, num_epochs: int,
                 batch_size: int,
                 num_data_cols: int,
                 d_hidden_layers: list,
                 g_hidden_layers: list,
                 noise_inputs: int = 100,
                 d_outputs: int = 1,
                 d_output_activation: str = 'linear',
                 g_output_activation: str = 'tanh',
                 d_learning_rate: float = 0.001,
                 g_learning_rate: float = 0.0005,
                 d_optimiser: object = tf.keras.optimizers.Adam,
                 g_optimiser: object = tf.keras.optimizers.Adam,
                 smoothing_noise: float = 0.1,
                 smoothing_noise_decay_steps: int = 100,
                 random_state: int = 42,
                 log_every: int = 5,
                 flip_label: bool = False,
                 d_noise: bool = False,
                 d_noise_stddev: float = 0.004,
                 d_noise_decay_steps: int = 100,
                 image_generation: bool = False,
                 image_dir: str = 'img'):

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.d_architecture = { 'num_inputs': num_data_cols,
                                'hidden_layers': d_hidden_layers,
                                'optimiser': d_optimiser,
                                'learning_rate': d_learning_rate,
                                'num_outputs': d_outputs,
                                'output_activation': d_output_activation
                                }

        self.g_architecture = { 'num_inputs': noise_inputs,
                                'hidden_layers': g_hidden_layers,
                                'optimiser': g_optimiser,
                                'learning_rate': g_learning_rate,
                                'num_outputs': num_data_cols,
                                'output_activation': g_output_activation
                                }

        self.d_optimiser = d_optimiser(learning_rate=d_learning_rate)
        self.g_optimiser = g_optimiser(learning_rate=g_learning_rate)

        self.smoothing_noise = smoothing_noise
        self.smooth_decay_steps = smoothing_noise_decay_steps

        self._initialise_gan()

        # This method returns a helper function to compute cross entropy loss
        # TODO: replace with choice of custom loss function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.random_state = check_random_state(random_state)
        tf.random.set_seed(random_state)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        self.flip_label = flip_label

        self.d_noise = d_noise
        self.d_noise_stddev = d_noise_stddev
        self.d_noise_decay_steps = d_noise_decay_steps

        self.log_every = log_every

        self.image_generation = image_generation
        self.image_dir = image_dir

    def _initialise_gan(self):
        self.generator = GAN.make_model(self.g_architecture)
        print("Initialised Generator: \n")
        print(self.generator.summary())
        self.discriminator = GAN.make_model(self.d_architecture)
        print("Initialised Discriminator: \n")
        print(self.discriminator.summary())


    @staticmethod
    def add_layer(model,
                  type: str,
                  layer: dict,
                  batch_normalisation: bool = False,
                  dropout: bool = False,
                  dropout_rate: float = 0.3,
                  advanced_activation: bool = False,
                  activation_layer: object = tf.keras.layers.LeakyReLU):
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
        if type == "dense":
            model.add(layers.Dense(**layer))
        elif type == 'conv':
            model.add(layers.Conv1D(**layer))
        else:
            raise ValueError('Layer type must be dense (dense) or convolutional (conv).')

        if batch_normalisation:
            model.add(layers.BatchNormalization())

        if dropout:
            model.add(layers.Dropout(rate=dropout_rate))

        if advanced_activation:
            model.add(activation_layer())

        return model

    @staticmethod
    def make_model(architecture: dict):
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
        if architecture['hidden_layers'][0]['type'] == 'dense':
            model.add(tf.keras.Input(architecture['num_inputs']))
        else:
            model.add(tf.keras.Input((architecture['num_inputs'], 1)))

        for layer in architecture['hidden_layers']:
            GAN.add_layer(model, **layer)

        model.add(layers.Dense(architecture['num_outputs'],
                               use_bias=False,
                               activation=architecture['output_activation']))

        return model

    def _discriminator_loss(self, real_output, fake_output):
        real_noise = 1 - self.smoothing_noise * self.smooth_decay_multiplier
        fake_noise = self.smoothing_noise * self.smooth_decay_multiplier
        real_loss = self.cross_entropy(tf.ones_like(real_output) * real_noise, real_output)
        fake_loss = self.cross_entropy(tf.ones_like(fake_output) * fake_noise, fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def _train_step(self,
                    data: object):
        noise = tf.random.normal([self.batch_size, self.g_architecture['num_inputs']])
        if self.d_noise:
            dtype_data = tf.keras.backend.dtype(data)
            disc_noise = tf.random.normal([self.batch_size, self.d_architecture['num_inputs']],
                                          stddev=self.d_noise_stddev,
                                          dtype=dtype_data)
            data = tf.math.add(data, disc_noise * self.d_noise_decay_multiplier)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)
            if self.d_noise:
                dtype_data = tf.keras.backend.dtype(generated_data)
                disc_noise = tf.random.normal([self.batch_size, self.d_architecture['num_inputs']],
                                              stddev=self.d_noise_stddev,
                                              dtype=dtype_data)
                generated_data = tf.math.add(generated_data, disc_noise * self.d_noise_decay_multiplier)

            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimiser.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimiser.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def _plot_generated_images(self, epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
        generated_images = self.sample(examples)
        generated_images = generated_images.reshape(examples, 28, 28)
        print(generated_images[1] - generated_images[0])
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        image_name = os.path.join(self.image_dir, 'gan_generated_image_{epoch:04}.png'.format(epoch=epoch))
        plt.savefig(image_name)
        plt.close('all')

    def _train_gan(self,
                   dataset: object,
                   save_model: bool = True,
                   checkpoint_dir: str = './training_checkpoints') -> object:

        # Initialise running variables and lists
        start = time.time()
        total_time = 0
        g_losses = []
        d_losses = []
        smooth_decay_list = list(np.linspace(1, 0, self.smooth_decay_steps))
        d_noise_decay_list = list(np.linspace(1, 0, self.d_noise_decay_steps))

        for epoch in range(self.num_epochs):

            if len(smooth_decay_list) > epoch:
                self.smooth_decay_multiplier = smooth_decay_list[epoch]
            else:
                self.smooth_decay_multiplier = 0

            if len(d_noise_decay_list) > epoch:
                self.d_noise_decay_multiplier = d_noise_decay_list[epoch]
            else:
                self.d_noise_decay_multiplier = 0

            for data_batch in dataset:
                g_loss, d_loss = self._train_step(data_batch)

                g_losses.append(g_loss)
                d_losses.append(d_loss)

            # Save the model every 15 epochs
            if save_model:
                checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
                checkpoint = tf.train.Checkpoint(generator_optimizer=self.g_optimiser,
                                                 discriminator_optimizer=self.d_optimiser,
                                                 generator=self.generator,
                                                 discriminator=self.discriminator)
                if (epoch + 1) % 15 == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)

            if (epoch + 1) % self.log_every == 0:
                now = np.round(time.time() - start, 2)
                total_time = np.round(now + total_time, 2)
                print('Time for epoch {} is {} seconds.'.format(epoch + 1, now))
                print("Total time passed: {} seconds".format(total_time))
                print("Generator loss: {}".format(np.mean(g_losses)))
                print("Discriminator loss: {}".format(np.mean(d_losses)))
                if self.image_generation:
                    self._plot_generated_images(epoch=epoch)
                start = time.time()
                g_losses = []
                d_losses = []

    def fit(self,
            data,
            data_unscaled: bool = False,
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
        if data_unscaled:
            data = self.scaler.fit_transform(data)

        if not buffer_size:
            buffer_size = data.shape[0]

        if data.shape[1] != self.d_architecture['num_inputs']:
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
        noise = tf.random.normal([n_samples, self.g_architecture['num_inputs']])
        samples = self.generator.predict(noise)
        return samples


class WeightClip(tf.keras.constraints.Constraint):
    def __init__(self, c):
        self.c = c

    def __call__(self, p):
        return tf.keras.backend.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__, 'c': self.c}


class WassersteinGAN(object):
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
                 noise_inputs: int,
                 g_outputs: int,
                 d_activation: object = layers.LeakyReLU,
                 g_activation: object = layers.LeakyReLU,
                 d_learning_rate: float = 0.001,
                 g_learning_rate: float = 0.0005,
                 d_optimiser: object = tf.keras.optimizers.RMSprop,
                 g_optimiser: object = tf.keras.optimizers.RMSprop,
                 clip_factor: float = 0.01,
                 init_stddev: float = 0.025,
                 init_function: object = tf.keras.initializers.TruncatedNormal,
                 smoothing_noise: float = 0.1,
                 smoothing_noise_decay_steps: int = 100,
                 random_state: int = 42,
                 log_every: int = 5,
                 flip_label: bool = False,
                 d_noise: bool = False,
                 d_noise_stddev: float = 0.004,
                 d_noise_decay_steps: int = 100,
                 image_generation: bool = False,
                 image_dir: str = 'img'):

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.network_dims = {
            'd_hidden_dims': d_hidden_dims,
            'g_hidden_dims': g_hidden_dims,
            'n_inputs': noise_inputs,
            'g_outputs': g_outputs
        }

        self.d_activation = d_activation
        self.g_activation = g_activation

        self.d_optimiser = d_optimiser(learning_rate=d_learning_rate)
        self.g_optimiser = g_optimiser(learning_rate=g_learning_rate)

        self.clip_factor = clip_factor

        self.init_fctn = init_function
        self.init_stddev = init_stddev

        self.smoothing_noise = smoothing_noise
        self.smooth_decay_steps = smoothing_noise_decay_steps

        self._initialise_gan()

        # This method returns a helper function to compute cross entropy loss
        # TODO: replace with choice of custom loss function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.random_state = check_random_state(random_state)
        tf.random.set_seed(random_state)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        self.flip_label = flip_label

        self.d_noise = d_noise
        self.d_noise_stddev = d_noise_stddev
        self.d_noise_decay_steps = d_noise_decay_steps

        self.log_every = log_every

        self.image_generation = image_generation
        self.image_dir = image_dir


    def _add_conv_layer(self,
                         model,
                         nodes,
                         activation,
                         initialiser: object = None,
                         kernel_con: object = None,
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
        model.add(layers.Conv1D(filters=nodes[0],
                                kernel_size=nodes[1],
                                use_bias=use_bias,
                                kernel_constraint=kernel_con,
                                kernel_initializer=initialiser))
        # model.add(layers.BatchNormalization())
        model.add(activation())
        return model

    def _make_conv_model(self,
                          dims,
                          activation,
                          inputs: int,
                          outputs: int = 1,
                          kernel_constraint: object = None,
                          initialiser: object = tf.keras.initializers.glorot_normal,
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
            self._add_conv_layer(model,
                                  nodes,
                                  activation,
                                  kernel_con=kernel_constraint,
                                  initialiser=initialiser)

        model.add(layers.Flatten())
        model.add(layers.Dense(outputs, use_bias=False, activation=output_activation))

        return model


    def _initialise_gan(self):
        self.generator = self._make_conv_model(dims=self.network_dims['g_hidden_dims'],
                                                activation=self.g_activation,
                                                inputs=(self.network_dims['n_inputs'], 1),
                                                outputs=self.network_dims['g_outputs'])
        self.discriminator = self._make_conv_model(dims=self.network_dims['d_hidden_dims'],
                                                    activation=self.d_activation,
                                                    inputs=(self.network_dims['g_outputs'], 1),
                                                    kernel_constraint=WeightClip(self.clip_factor),
                                                    initialiser=self.init_fctn(self.init_stddev),
                                                    output_activation='linear')

    def _wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    def _train_step(self,
                    data: object):
        noise = tf.random.normal([self.batch_size, self.network_dims['n_inputs'], 1])
        if self.d_noise:
            dtype_data = tf.keras.backend.dtype(noise)
            disc_noise = tf.random.normal([self.batch_size, self.network_dims['g_outputs'], 1],
                                          stddev=self.d_noise_stddev,
                                          dtype=dtype_data)
            data = tf.math.add(data, disc_noise * self.d_noise_decay_multiplier)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # gen_tape.watch(self.generator.trainable_variables)
            generated_data = self.generator(noise, training=True)
            if self.d_noise:
                dtype_data = tf.keras.backend.dtype(generated_data)
                disc_noise = tf.random.normal([self.batch_size, self.network_dims['g_outputs']],
                                              stddev=self.d_noise_stddev,
                                              dtype=dtype_data)
                generated_data = tf.math.add(generated_data, disc_noise * self.d_noise_decay_multiplier)

            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            real_loss = self._wasserstein_loss(real_output, -1 * tf.ones_like(real_output))
            fake_loss = self._wasserstein_loss(fake_output, tf.ones_like(fake_output))

            disc_loss = tf.math.subtract(fake_loss, real_loss)
            gen_loss = -1 * fake_loss

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimiser.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimiser.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def _plot_generated_images(self, epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
        generated_images = self.sample(examples)
        generated_images = generated_images.reshape(examples, 28, 28)
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        image_name = os.path.join(self.image_dir, 'gan_generated_image_{epoch:04}.png'.format(epoch=epoch))
        plt.savefig(image_name)
        plt.close('all')

    def _train_gan(self,
                   dataset: object,
                   save_model: bool = True,
                   checkpoint_dir: str = './training_checkpoints') -> object:

        # Initialise running variables and lists
        start = time.time()
        total_time = 0
        g_losses = []
        d_losses = []
        smooth_decay_list = list(np.linspace(1, 0, self.smooth_decay_steps))
        d_noise_decay_list = list(np.linspace(1, 0, self.d_noise_decay_steps))

        for epoch in range(self.num_epochs):

            if len(smooth_decay_list) > epoch:
                self.smooth_decay_multiplier = smooth_decay_list[epoch]
            else:
                self.smooth_decay_multiplier = 0

            if len(d_noise_decay_list) > epoch:
                self.d_noise_decay_multiplier = d_noise_decay_list[epoch]
            else:
                self.d_noise_decay_multiplier = 0

            for data_batch in dataset:
                g_loss, d_loss = self._train_step(data_batch)

                g_losses.append(g_loss)
                d_losses.append(d_loss)

            # Save the model every 15 epochs
            if save_model:
                checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
                checkpoint = tf.train.Checkpoint(generator_optimizer=self.g_optimiser,
                                                 discriminator_optimizer=self.d_optimiser,
                                                 generator=self.generator,
                                                 discriminator=self.discriminator)
                if (epoch + 1) % 15 == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)

            if (epoch + 1) % self.log_every == 0:
                now = np.round(time.time() - start, 2)
                total_time = np.round(now + total_time, 2)
                print('Time for epoch {} is {} seconds.'.format(epoch + 1, now))
                print("Total time passed: {} seconds".format(total_time))
                print("Generator loss: {}".format(np.mean(g_losses)))
                print("Discriminator loss: {}".format(np.mean(d_losses)))
                if self.image_generation:
                    self._plot_generated_images(epoch=epoch)
                start = time.time()
                g_losses = []
                d_losses = []

    def fit(self,
            data,
            data_unscaled: bool = False,
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
        if data_unscaled:
            data = self.scaler.fit_transform(data)

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
        noise = tf.random.normal([n_samples, self.network_dims['n_inputs'], 1])
        samples = self.generator.predict(noise)
        return samples

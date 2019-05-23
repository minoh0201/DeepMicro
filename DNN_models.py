
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, Flatten, Reshape, Cropping2D
from keras import backend as K
from keras.losses import mse, binary_crossentropy
import math
import numpy as np

# create MLP model
def mlp_model(input_dim, numHiddenLayers=3, numUnits=64, dropout_rate=0.5):

    model = Sequential()

    #Check number of hidden layers
    if numHiddenLayers >= 1:
        # First Hidden layer
        model.add(Dense(numUnits, input_dim=input_dim, activation='relu'))
        model.add(Dropout(dropout_rate))

        # Second to the last hidden layers
        for i in range(numHiddenLayers - 1):
            numUnits = numUnits // 2
            model.add(Dense(numUnits, activation='relu'))
            model.add(Dropout(dropout_rate))

        # output layer
        model.add(Dense(1, activation='sigmoid'))

    else:
        # output layer
        model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', )#metrics=['accuracy'])

    return model


# Autoencoder
def autoencoder(dims, act='relu', init='glorot_uniform', latent_act = False, output_act = False):
    """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """

    # whether put activation function in latent layer
    if latent_act:
        l_act = act
    else:
        l_act = None

    if output_act:
        o_act = 'sigmoid'
    else:
        o_act = None

    # The number of internal layers: layers between input and latent layer
    n_internal_layers = len(dims) - 2

    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_internal_layers):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # bottle neck layer, features are extracted from here
    h = Dense(dims[-1], activation=l_act, kernel_initializer=init, name='encoder_%d_bottle-neck' % (n_internal_layers))(h)

    y = h

    # internal layers in decoder
    for i in range(n_internal_layers, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], activation=o_act, kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

def conv_autoencoder(dims, act='relu', init='glorot_uniform', latent_act = False, output_act = False, rf_rate = 0.1, st_rate = 0.25):
    # whether put activation function in latent layer
    if latent_act:
        l_act = act
    else:
        l_act = None

    if output_act:
        o_act = 'sigmoid'
    else:
        o_act = None

    # receptive field and stride size
    rf_size = init_rf_size = int(dims[0][0] * rf_rate)
    stride_size = init_stride_size = int(rf_size * st_rate) if int(rf_size * st_rate) > 0 else 1
    print("receptive field (kernel) size: %d" % rf_size)
    print("stride size: %d" % stride_size)

    # The number of internal layers: layers between input and latent layer
    n_internal_layers = len(dims) - 1

    if n_internal_layers < 1:
        print("The number of internal layers for CAE should be greater than or equal to 1")
        exit()

    # input
    x = Input(shape=dims[0], name='input')
    h = x

    rf_size_list = []
    stride_size_list = []

    # internal layers in encoder
    for i in range(n_internal_layers):
        print("rf_size: %d, st_size: %d" % (rf_size, stride_size))
        h = Conv2D(dims[i + 1], (rf_size,rf_size), strides=(stride_size, stride_size), activation=act, padding='same', kernel_initializer=init, name='encoder_conv_%d' % i)(h)
        #h = MaxPool2D((2,2), padding='same')(h)
        rf_size = int(K.int_shape(h)[1] * rf_rate)
        stride_size = int(rf_size /2.) if int(rf_size /2.) > 0 else 1
        rf_size_list.append(rf_size)
        stride_size_list.append(stride_size)

    reshapeDim = K.int_shape(h)[1:]

    # bottle neck layer, features are extracted from h
    h = Flatten()(h)

    y = h

    y = Reshape(reshapeDim)(y)

    print(rf_size_list)
    print(stride_size_list)

    # internal layers in decoder
    for i in range(n_internal_layers - 1, 0, -1):
        y = Conv2DTranspose(dims[i], (rf_size_list[i-1],rf_size_list[i-1]), strides=(stride_size_list[i-1], stride_size_list[i-1]), activation=act, padding='same', kernel_initializer=init, name='decoder_conv_%d' % i)(y)
        #y = UpSampling2D((2,2))(y)

    y = Conv2DTranspose(1, (init_rf_size, init_rf_size), strides=(init_stride_size, init_stride_size), activation=o_act, kernel_initializer=init, padding='same', name='decoder_1')(y)

    # output cropping
    if K.int_shape(x)[1] != K.int_shape(y)[1]:
        cropping_size = K.int_shape(y)[1] - K.int_shape(x)[1]
        y = Cropping2D(cropping=((cropping_size, 0), (cropping_size, 0)), data_format=None)(y)

    #print("dims[0]: %s" % str(dims[0]))

    # output
    # y = Conv2D(1, (rf_size, rf_size), activation=o_act, kernel_initializer=init, padding='same', name='decoder_1')(y)
    #
    # outputDim = reshapeDim * (2 ** n_internal_layers)
    # if outputDim != dims[0][0]:
    #     cropping_size = outputDim - dims[0][0]
    #     #print(outputDim, dims[0][0], cropping_size)
    #     y = Cropping2D(cropping=((cropping_size, 0), (cropping_size, 0)), data_format=None)(y)


    return Model(inputs=x, outputs=y, name='CAE'), Model(inputs=x, outputs=h, name='encoder')


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_sigma = args
    #z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + z_sigma * epsilon
    #return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Variational Autoencoder
def variational_AE(dims, act='relu', init='glorot_uniform', output_act = False, recon_loss = 'mse', beta=1):

    if output_act:
        o_act = 'sigmoid'
    else:
        o_act = None

    # The number of internal layers: layers between input and latent layer
    n_internal_layers = len(dims) - 2

    ## build encoder model
    inputs = Input(shape=(dims[0],), name='input')

    h = inputs

    # internal layers in encoder
    for i in range(n_internal_layers):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # latent layer
    z_mean = Dense(dims[-1], name='z_mean')(h)
    z_sigma = Dense(dims[-1], name='z_sigma')(h)

    # z_log_var = Dense(dims[-1], name='z_log_var')(h)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(dims[-1],), name='z')([z_mean, z_sigma])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_sigma, z], name='encoder')

    ## build decoder model
    latent_inputs = Input(shape=(dims[-1],), name='z_sampling')

    y = latent_inputs

    # internal layers in decoder
    for i in range(n_internal_layers, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    outputs = Dense(dims[0], kernel_initializer=init, activation=o_act)(y)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')


    ## loss function
    if recon_loss == 'mse':
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= dims[0]

    kl_loss = 1 + K.log(1e-8 + K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma)
    #kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + (beta * kl_loss))
    vae.add_loss(vae_loss)

    vae.compile(optimizer='adam', )

    vae.metrics_tensors.append(K.mean(reconstruction_loss))
    vae.metrics_names.append("recon_loss")
    vae.metrics_tensors.append(K.mean(beta * kl_loss))
    vae.metrics_names.append("kl_loss")

    return vae, encoder, decoder


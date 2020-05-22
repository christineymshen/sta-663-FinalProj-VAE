# load VAE package
from VAE import VAE

# import mnist dataset
from keras.datasets import mnist

(trainX_raw, trainy), (testX_raw, testy) = mnist.load_data()
trainX = trainX_raw.reshape(trainX_raw.shape[0],-1)


# Due to use of numba jit, the first function call will trigger
# warnings during jit compilation

# model training for 100 mini-matches, each batch of size = 100
W, b, loss = VAE.train_AEVB(trainX, trainy, 1000)

# plot total loss for 1000 batch
plt.plot(loss)

# compare sample data vs reconstructed images based on model parameters
VAE.plot_samples(trainX, trainy, W, b)



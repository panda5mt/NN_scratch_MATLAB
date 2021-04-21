# MNIST_CNN_from_scratch
CNN to classify digits coded from scratch using cross-entropy loss and Adam optimizer.

This CNN has two convolutional layers, one max pooling layer, and two fully connected layers, employing cross-entropy as the loss function. To use this, load the mnist data into your Workspace, and run main_cnn. Parameters for training (number of epochs, batch size) can be adapted, as well as parameters pertaining to the Adam optimizer.

Trained on 1 epoch, the CNN achieves an accuracy of 95% on the test set. Accuracy may be improved by parameter tuning/ training on more epochs, but I coded this to construct the components of a typical CNN. Functions for the calculation of convolutions, max pooling, gradients (through backpopagation), etc. can be adapted for other architectures.

Neural-Network
==============

Basic Java implementation of a neural network on character recognition. Attempts to recognize characters that are black/white based on the feature definition = RGB value of each pixel.

How to use
==============
Download the Apache Commons Math API. http://commons.apache.org/proper/commons-math/download_math.cgi

1. Open the Preprocessor program in the console via <java Preprocessor <char>>, where <char> is the true character value of all the training set's picture. For example, you would specify a folder with all the pictures of L's and run the Preprocessor with <java Preprocessor L>. 
2. Run NeuralNetwork and specify the training set folder and the test set folder. The console will print its progress then the recognized letter. (Note NeuralNetworkOO requires that you instantiate NeuralNetwork and specify all the parameters e.g. number of units, outputs, features, regularization rate, learning rate, runs of gradient descent).

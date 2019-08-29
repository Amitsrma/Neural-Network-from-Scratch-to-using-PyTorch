This is an exercise to identify two digits in an image, sometimes overlapping, first only using multi-layer perceptron and then convolutional neural network.

Trick here was to create a parallel architecture where as we move through the network, we divide the flow of information such that last layer gives two output and those two outputs are compared with actual label to train the network.

Following is the diagram for architecture used.

![Alt text](Parallel_architecture.PNG?raw=true "Title")

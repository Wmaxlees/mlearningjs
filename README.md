#mlearning.js

A machine learning library written in JavaScript.

Note: This package requires cblas to be installed on your system.

For ubuntu, you can use `sudo apt-get install libblas-dev`

#Usage

First you have to initialize a network

    var Network = require('mlearning').Network;
    
    var myNetwork = new Network(50, 250);
    
The first argument is the size of the input vector. The second number is the size
of the batches during training.

Next you can create your network

    myNetwork.addLayer(5000, 0.01);
    myNetwork.addReLU();
    myNetwork.addLayer(2, 0.1);
    myNetwork.addReLU();
    myNetwork.addSoftmax();
    myNetwork.useCrossEntropyLoss();
    
The above code creates a Deep Neural Network with a hidden layer of size 5000. The output is of size 2. In between each layer is a rectifier linear unit non-linearity. The final layer is a softmax. The loss function the network will use is the cross entropy loss function.

Finally, we can train the network

    myNetwork.train(100000, (batchSize) => {
        return {
            X: someInputMatrix,
            Y: someOutputMatrix
        };
    }

The network will train for the number of iterations stated using the second parameter as the function to generate the input and expected values.

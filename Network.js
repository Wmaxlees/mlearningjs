'use strict';

{
    var NNLayer = require(__dirname + '/layers/NNLayer')
        , ConvLayer = require(__dirname + '/layers/ConvLayer')
        , ReLULayer = require(__dirname + '/layers/ReLULayer')
        , SoftmaxLayer = require(__dirname + '/layers/SoftmaxLayer')
        , Im2ColLayer = require(__dirname + '/layers/Im2ColLayer')
        , Row2ImLayer = require(__dirname + '/layers/Row2ImLayer')
        , Matrix2VectorLayer = require(__dirname + '/layers/Matrix2VectorLayer')
        , Tensor3D2VectorLayer = require(__dirname + '/layers/Tensor3D2VectorLayer')
        , Layer = require(__dirname + '/Layer')
        , CrossEntropyLoss = require(__dirname + '/loss/CrossEntropyLoss')
        , Tensor3D = require(__dirname + '/tensors/Tensor3D')
        , Tensor4D = require(__dirname + '/tensors/Tensor4D')
        , _ = require('lodash')
        , v = require('vectorious')
        , Matrix = v.Matrix;

    function Network(inputSize) {
        if (!_.isArray(inputSize)) {
            throw new Error('[Network]: Attempting to create a new network with bad parameters');
        }

        this.layers[0] = new Layer(false, inputSize, null);
    }

    module.exports = Network;

    Network.prototype = {
        addLayer: function (outputSize, gradientStepSize) {
            if (outputSize < 1) {
                throw new Error('[Network.addLayer]: Output size less than 1');
            }

            if (this.getLastShape().length !== 2 || this.getLastShape()[0] !== 1) {
                throw new Error('[Network.addLayer]: Cannot connect new layer to previous layer based on tensor dimensions');
            }

            var layer = new NNLayer(this.getLastShape()[1], outputSize, gradientStepSize);
            this.layers.push(new Layer(true, [1, outputSize], layer));

        }

        , addConvolutionalLayer: function (numberOfFilters, filterSize, depth, strideSize, gradientStepSize) {
            var inputSize = this.getLastShape();
            var im2ColWidth = ((inputSize[0]+1-filterSize)/strideSize)*((inputSize[1]+1-filterSize)/strideSize);
            var im2colLayer = new Im2ColLayer(filterSize, inputSize, strideSize);
            this.layers.push(new Layer(false, [im2ColWidth, filterSize*filterSize*depth], im2colLayer));

            var convLayer = new ConvLayer(numberOfFilters, filterSize, this.getLastShape(), strideSize, gradientStepSize);
            this.layers.push(new Layer(true, [numberOfFilters, im2ColWidth], convLayer));

            var shape = (inputSize[0]+1-filterSize)/strideSize;
            var row2imLayer = new Row2ImLayer(shape);
            this.layers.push(new Layer(false, [shape, shape, numberOfFilters], row2imLayer));
        }

        , addTensor3D2VectorLayer: function () {
            var lastShape = this.getLastShape();

            if (lastShape.length !== 3) {
                throw new Error('[Network.addTensor3D2Vector]: Cannot connect new layer to previous layer based on tensor dimensions');
            }

            var shape = [
                1,
                lastShape[0]*lastShape[1]*lastShape[2]
            ];

            var layer = new Tensor3D2VectorLayer();
            this.layers.push(new Layer(false, shape, layer));
        }

        , addMatrix2VectorLayer: function () {
            var lastShape = this.getLastShape();

            if (lastShape.length !== 2) {
                throw new Error('[Network.addMatrix2Vector]: Cannot connect new layer to previous layer based on tensor dimensions');
            }

            var shape = [
                1,
                lastShape[0]*lastShape[1]
            ];

            var layer = new Matrix2VectorLayer();
            this.layers.push(new Layer(false, shape, layer));
        }

        , addReLU: function () {
            var layer = new ReLULayer();
            this.layers.push(new Layer(false, this.getLastShape(), layer));
        }

        , addSoftmax: function () {
            var layer = new SoftmaxLayer();
            this.layers.push(new Layer(false, this.getLastShape(), layer));
        }

        , useCrossEntropyLoss: function () {
            var loss = new CrossEntropyLoss();

            this.lossFunction = loss;
        }

        , train: function (iterations, batchSize, inputFunction) {
            for (var i = 0; i < iterations; i += batchSize) {
                var X = [];
                var Y = [];
                for (var j = 0; j < batchSize; ++j) {
                    var temp = inputFunction();
                    X.push(temp.X);
                    Y.push(temp.Y);
                }

                // Get the input shape
                if (!X[0].shape) {
                    if (_.isArray(X[0])) {
                        X = new Matrix(X);
                        X = X.transpose();
                    } else {
                        for (var e = 0; e < X.length; ++e) {
                            X[e] = X[e].toArray();
                        }
                        X = new Matrix(X);
                        X = X.transpose();
                    }
                } else {
                    switch (X[0].shape.length) {
                        case 2: {
                            X = new Tensor3D(X);
                        } case 3: {
                            X = new Tensor4D(X);
                        } default: {
                            new Error('[Network.train]: Cannot create tensors of greater than 4th degree');
                        }
                    }
                }

                // Forward Pass
                X = this.forwardPass(X);

                // --------------------------------
                // Calculate Loss
                // --------------------------------
                var checker = X.transpose().toArray();
                var total = 0;
                for (var j = 0; j < batchSize; ++j) {
                    var prediction = (checker[j][0] > checker[j][1]) ? 0 : 1;
                    if (Y[j] == prediction) {
                        ++total;
                    }
                }

                X = this.lossFunction.forwardPass(X, Y);
                var loss = _.sum(X)/batchSize;

                console.log('Iterations: ' + (i+batchSize));
                console.log('% Accuracy: ' + (total/batchSize)*100);
                console.log('Loss: ' + loss);
                console.log('----------------------------------');

                var back = this.lossFunction.backwardPass(X, Y);

                this.backwardPass(back, batchSize);
            }
        }

        , forwardPass: function (x) {
            for (var i in this.layers) {
                x = this.layers[i].forwardPass(x);
            }

            return x;
        }

        , backwardPass: function (input, batchSize) {
            for (var i = 0; i < this.layers.length; ++i) {
                var index = this.layers.length - (i + 1);
                input = this.layers[index].backwardPass(input, batchSize);
            }

            return input;
        }

        , getLastShape: function () {
            return this.layers[this.layers.length-1].getOutputShape();
        }

        , lossFunction: {}
        , inputSize: 0
        , batchSize: 0
        , layers: []
    }

}
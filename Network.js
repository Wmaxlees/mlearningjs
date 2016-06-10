'use strict';

{
    var NNLayer = require(__dirname + '/layers/NNLayer')
        , ReLULayer = require(__dirname + '/layers/ReLULayer')
        , SoftmaxLayer = require(__dirname + '/layers/SoftmaxLayer')
        , CrossEntropyLoss = require(__dirname + '/loss/CrossEntropyLoss')
        , _ = require('lodash');

    function Network(inputSize, batchSize) {
        this.currentSize = inputSize
        this.batchSize = batchSize
    }

    module.exports = Network;

    Network.prototype = {
        addLayer: function (outputSize, stepSize) {
            if (outputSize < 1) {
                throw new Error('[Network.addLayer]: Output size less than 1');
            }

            var layer = new NNLayer(this.currentSize, outputSize, stepSize, this.batchSize);
            this.currentSize = outputSize;
            this.layers.push({
                calcRegLoss: true,
                layer: layer
            });
        }

        , addReLU: function () {
            var layer = new ReLULayer(this.batchSize);
            this.layers.push({
                calcRegLoss: false,
                layer: layer
            });
        }

        , addSoftmax: function () {
            var layer = new SoftmaxLayer(this.batchSize);
            this.layers.push({
                calcRegLoss: false,
                layer: layer
            });
        }

        , useCrossEntropyLoss: function () {
            var loss = new CrossEntropyLoss(this.batchSize);

            this.lossFunction = loss;
        }

        , train: function (iterations, inputFunction) {
            for (var i = 0; i < iterations; i += this.batchSize) {
                var input = inputFunction(this.batchSize);

                // Forward Pass
                input.X = this.forwardPass(input);

                // --------------------------------
                // Calculate Loss
                // --------------------------------
                var checker = input.X.transpose().toArray();
                var total = 0;
                for (var j = 0; j < this.batchSize; ++j) {
                    var prediction = (checker[j][0] > checker[j][1]) ? 0 : 1;
                    if (input.Y.toArray()[0][j] == prediction) {
                        ++total;
                    }
                }

                input.X = this.lossFunction.forwardPass(input.X, input.Y)
                var loss = _.sum(input.X)/this.batchSize;

                console.log('Iterations: ' + (i+this.batchSize));
                console.log('% Accuracy: ' + (total/this.batchSize)*100);
                console.log('Loss: ' + loss);
                console.log('----------------------------------');

                this.backwardPass(input);
            }
        }

        , forwardPass: function (input) {
            var x = input.X;
            for (var i in this.layers) {
                x = this.layers[i].layer.forwardPass(x);
            }

            return x;
        }

        , backwardPass: function (input) {
            var back = this.lossFunction.backwardPass(input.X, input.Y);
            for (var i = 0; i < this.layers.length; ++i) {
                var index = this.layers.length - (i + 1);
                back = this.layers[index].layer.backwardPass(back);
            }

            return back;
        }

        , lossFunction: {}
        , currentSize: 0
        , batchSize: 0
        , layers: []
    }

}
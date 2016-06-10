'use strict';

{
    var outerProduct = require(__dirname + '/../outerProduct.helper')
        , v = require('vectorious')
        , Matrix = v.Matrix;

    const lamda = 0.0001;

    function ConvLayer (numberOfFilters, inputSize, stepSize, batchSize) {
        if (!numberOfFilters || !inputSize || !stepSize || !batchSize) {
            throw new Error('ConvLayer constructor missing arguments');
        }

        this.W = Matrix.random(inputSize, numberOfFilters, 0.000001);
        this.stepSize = stepSize;
        this.batchSize = batchSize;
    }

    module.exports = ConvLayer;

    ConvLayer.prototype = {
        forwardPass: function (input) {
            if (this.batchSize !== input.length) {
                throw new Error('[ConvLayer.forwardPass]: Invalid input batch size: ' + input.length);
            }
            this.x = [];
            for (var i = 0; i < this.batchSize; ++i) {
                this.x[i] = input[i];
                this.out[i] = Matrix.multiply(this.W, input[i]);
            }

            return this.out;
        }

        , getRegularizationLoss: function () {
            var total = 0.0;
            this.W.each( (x, i, j) => {
                total += x;
            });

            return total*lamda;
        }

        , backwardPass: function (input) {
            if (this.batchSize !== back.length) {
                throw new Error('[ConvLayer.backwardPass]: Invalid back batch size: ' + back.length);
            }

            throw Error('ConvLayer.BackwardPass: Not Implemented');

            var result = [];
            var dW = [];
            for (var i = 0; i < this.batchSize; ++i) {
                dW[i] = Matrix.multiply(this.x[i], back[i].transpose()); // Dot product
                result[i] = Matrix.multiply(this.W, back[i]);
            }

            var avgdW = dW[0]; 
            for (var i = 1; i < this.batchSize; ++i) {
                avgdW.add(dW[i]);
            }
            avgdW.scale(1/this.batchSize);

            this.W.subtract(dW.scale(stepSize));

            return result;
        }

        , toConsole: function () {
            console.log(this.W.toString());
        }

        , x: []

        , out: []
    }
}
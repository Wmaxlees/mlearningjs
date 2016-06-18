'use strict';

{
    var outerProduct = require(__dirname + '/../outerProduct.helper')
        , Tensor3D = require(__dirname + '/../tensors/Tensor3D')
        , v = require('vectorious')
        , Matrix = v.Matrix;

    const lamda = 0.0001;

    var it = 0;

    function ConvLayer (numberOfFilters, filterSize, inputSize, strideSize, gradientStepSize) {
        if (!numberOfFilters || !filterSize || !inputSize || !strideSize || !gradientStepSize) {
            throw new Error('ConvLayer constructor missing arguments');
        }

        this.W = Matrix.random(numberOfFilters, inputSize[1], 0.001);
        this.filterSize = filterSize;
        this.strideSize = strideSize;
        this.numberOfFilters = numberOfFilters;
        this.gradientStepSize = gradientStepSize;
    }

    module.exports = ConvLayer;

    ConvLayer.prototype = {
        forwardPass: function (input) {
            delete this.x;
            delete this.out;
            this.x = [];
            for (var i = 0; i < input.shape[2]; ++i) {
                this.x[i] = input.getLayer(i);
                this.out[i] = Matrix.multiply(this.W, this.x[i]);
            }

            this.out = new Tensor3D(this.out).transpose();

            return this.out;
        }

        , getRegularizationLoss: function () {
            var total = 0.0;
            this.W.each( (x, i, j) => {
                total += x;
            });

            return total*lamda;
        }

        , backwardPass: function (input, batchSize) {
            var result = [];
            var dW = [];

            for (var i = 0; i < batchSize; ++i) {
                dW[i] = Matrix.multiply(this.x[i], input.getLayer(i).transpose()); // Dot product
                result[i] = Matrix.multiply(this.W.transpose(), input.getLayer(i));
            }

            var avgdW = dW[0]; 
            for (var i = 1; i < batchSize; ++i) {
                avgdW.add(dW[i]);
            }
            avgdW.scale(this.gradientStepSize/batchSize);

            this.W.add(avgdW);

            return result;
        }

        , toConsole: function () {
            console.log(this.W.toString());
        }

        , x: []

        , out: []
    }

    function im2col (tensor, filterWidth, filterHeight, depth) {
        var X = [];
        for (var i = 0; i < 8; ++i) {
            for (var j = 0; j < 8; ++j) {
                X.push(im2colFilter(tensor, i, j, filterWidth, filterHeight, filterDepth));
            }
        }
        
    }

    function im2colFilter (tensor, xOffset, yOffset, filterWidth, filterHeight, depth) {
        var builderArray = [];
        for (var i = 0; i < filterHeight; ++i) {
            for (var j = 0; j < filterWidth; ++j) {
                builderArray = builderArray.concat(tensor[i+yOffset][j+xOffset]);
            }
        }

        return builderArray;
    }
}
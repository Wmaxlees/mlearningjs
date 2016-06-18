'use strict';

{
    var Tensor3D = require(__dirname + '/../tensors/Tensor3D')
        , v = require('vectorious')
        , Matrix = v.Matrix;

    function Im2ColLayer (filterSize, inputSize, strideSize) {
        this.widthSteps = (inputSize[0]+1-filterSize)/strideSize;
        this.heightSteps = (inputSize[1]+1-filterSize)/strideSize;

        this.filterSize = filterSize;
        this.inputSize = inputSize;
        this.strideSize = strideSize;
    }

    module.exports = Im2ColLayer;

    Im2ColLayer.prototype = {
        forwardPass: function (input) {
            var usableInput = [];
            if (input.shape.length == 4) {
                // Training with batch
                for (var i = 0; i < input.shape[3]; ++i) {
                    usableInput.push(input.getLayer(i));
                }
            } else {
                usableInput = [input];
            }

            var result = [];
            for (var unit = 0; unit < usableInput.length; ++unit) {
                result[unit] = [];
                for (var i = 0; i < this.widthSteps; i += this.strideSize) {
                    for (var j = 0; j < this.heightSteps; j += this.strideSize) {
                        result[unit].push(apply(usableInput[unit], i, j, this.filterSize, this.inputSize[2]));
                    }
                }
                result[unit] = new Matrix(result[unit]).transpose();
            }

            return new Tensor3D(result);
        }

        , backwardPass: function (input, batchSize) {
            // throw new Error('[Im2ColLayer.backwardPass]: Not implemented yet');
        }

    }

    function apply (tensor, xOffset, yOffset, filterSize, depth) {
        var builderArray = [];
        tensor = tensor.toArray();

        for (var i = 0; i < filterSize; ++i) {
            for (var j = 0; j < filterSize; ++j) {
                builderArray = builderArray.concat(tensor[i+yOffset][j+xOffset]);
            }
        }

        return builderArray;
    }

}
'use strict';

{
    var Tensor3D = require(__dirname + '/../tensors/Tensor3D')
        , Tensor4D = require(__dirname + '/../tensors/Tensor4D')  
        , v = require('vectorious')
        , Matrix = v.Matrix;

    function Row2ImLayer (shape) {
        this.shape = shape;
    }

    module.exports = Row2ImLayer;

    Row2ImLayer.prototype = {
        forwardPass: function (input) {
            this.inputShape = input.shape;

            var result = [];
            for (var i = 0; i < input.shape[2]; ++i) {
                result[i] = [];
                for (var j = 0; j < input.shape[0]; ++j) {
                    result[i].push(apply(input.getLayer(i), j, this.shape));
                }

                result[i] = new Tensor3D(result[i]);
            }

            return new Tensor4D(result);
        }

        , backwardPass: function (input, batchSize) {
            var result = [];
            for (var i = 0; i < batchSize; ++i) {
                result.push(undo(input.getLayer(i)));
            }

            return new Tensor3D(result);
        }

        , inputShape: []
    };


    function apply (matrix, row, shape) {
        matrix = matrix.toArray();
        if (shape*shape !== matrix[row].length) {
            throw new Error('Row cannot be converted to correct shape');
        }

        var result = [];
        for (var i = 0; i < matrix[row].length/shape; ++i) {
            result[i] = [];
            for (var j = 0; j < shape; ++j) {
                result[i].push(matrix[row][i*j]);
            }
        }

        return new Matrix(result);
    }

    function undo (tensor) {
        var result = [];

        for (var depth = 0; depth < tensor.shape[2]; ++depth) {
            result[depth] = [];
            for (var i = 0; i < tensor.shape[1]; ++i) {
                for (var j = 0; j < tensor.shape[0]; ++j) {
                    result[depth].push(tensor.get(i, j, depth));
                }
            }
        }

        return new Matrix(result);
    }
}
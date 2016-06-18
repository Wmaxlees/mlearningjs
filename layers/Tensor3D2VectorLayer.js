'use static';

{
    var Tensor3D = require(__dirname + '/../tensors/Tensor3D')
        , Tensor4D = require(__dirname + '/../tensors/Tensor4D')
        , v = require('vectorious')
        , Matrix = v.Matrix;

    function Tensor3D2VectorLayer () {}

    module.exports = Tensor3D2VectorLayer;

    Tensor3D2VectorLayer.prototype = {
        forwardPass: function (input) {
            this.inputShape = input.shape;

            var result = [];
            for (var i = 0; i < input.shape[3]; ++i) {
                result.push(vectorize(input.getLayer(i)));
            }

            return new Matrix(result).transpose();
        }

        , backwardPass: function (input, batchSize) {
            input = input.transpose().toArray();

            var result = [];
            for (var i = 0; i < batchSize; ++i) {
                result.push(tensorize(input[i], this.inputShape));
            }

            return new Tensor4D(result);
        }

        , inputShape: []

    }

    function vectorize (tensor) {
        var result = [];
        for (var layer = 0; layer < tensor.shape[2]; ++layer) {
            for (var i = 0; i < tensor.shape[1]; ++i) {
                for (var j = 0; j < tensor.shape[0]; ++j) {
                    result.push(tensor.get(j, i, layer));
                }
            }
        }

        return result;
    }

    function tensorize (row, shape) {
        var result = [];

        for (var depth = 0; depth < shape[2]; ++depth) {
            var matrix = [];
            for (var i = 0; i < shape[1]; ++i) {
                matrix[i] = [];
                for (var j = 0; j < shape[0]; ++j) {
                    matrix[i][j] = row[i*j*depth];
                }
            }
            result.push(new Matrix(matrix));
        }

        return new Tensor3D(result);
    }
}
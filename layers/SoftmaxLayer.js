'use strict';

{
    var v = require('vectorious')
        , Matrix = v.Matrix;

    function SoftmaxLayer (batchSize) {
        if (!batchSize) {
            throw Error('SoftmaxLayer constructor missing arguments');
        }

        this.batchSize = batchSize;
    }

    module.exports = SoftmaxLayer;

    SoftmaxLayer.prototype = {
        forwardPass: function (input) {
            var shape = input.shape;
            if (this.batchSize !== shape[1]) {
                throw new Error('[SoftmaxLayer.forwardPass]: Invalid input batch size: ' + shape[1]);
            }

            var result = input.toArray();
            var denom = [];

            for (var row = 0; row < shape[0]; ++row) {
                for (var col = 0; col < shape[1]; ++col) {
                    var exponentiation = Math.exp(result[row][col]);

                    result[row][col] = exponentiation;

                    if (!denom[col]) {
                        denom[col] = 0.0;
                    } 
                    denom[col] += exponentiation;
                }
            }

            for (var row = 0; row < shape[0]; ++row) {
                for (var col = 0; col < shape[1]; ++col) {
                    result[row][col] /= denom[col];
                }
            }

            this.out = new Matrix(result);

            return this.out;
        }

        , backwardPass: function (input) {
            var shape = input.shape;
            if (this.batchSize !== shape[1]) {
                throw new Error('[SoftmaxLayer.backwardPass]: Invalid input batch size: ' + shape[1]);
            }

            var result = [];

            for (var layer = 0; layer < this.batchSize; ++layer) {
                var dLayer = Matrix.zeros(shape[0], shape[0]);
                for (var i = 0; i < shape[0]; ++i) {
                    for (var j = 0; j < shape[0]; ++j) {
                        if (i === j) {
                            dLayer.set(i, j, this.out.get(i, layer)*(1-this.out.get(i, layer)));
                        } else {
                            dLayer.set(i, j, this.out.get(i, layer)*this.out.get(j, layer));
                        }
                    }
                }

                var inputSlice = Matrix.zeros(2, 1);
                for (var row = 0; row < shape[0]; ++row) {
                    inputSlice.set(row, 0, input.get(row, layer));
                }
                result[layer] = Matrix.multiply(dLayer, inputSlice).transpose().toArray()[0];
            }

            return new Matrix(result).transpose();
        }

        , out: {}

    }

}
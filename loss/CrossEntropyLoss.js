'use strict';

{
    var v = require('vectorious')
        , Matrix = v.Matrix;

    function CrossEntropyLoss (batchSize) {
        if (!batchSize) {
            throw Error('ReLULayer constructor missing arguments');
        }

        this.batchSize = batchSize;
    }

    module.exports = CrossEntropyLoss;

    CrossEntropyLoss.prototype = {
        forwardPass: function (input, expected) {
            if (!expected || this.batchSize !== expected.shape[1]) {
                throw new Error('[CrossEntropyLoss.forwardPass]: Invalid expected values');
            }
            var expected = expected.toArray()[0];

            var shape = input.shape;
            if (this.batchSize !== shape[1]) {
                throw new Error('[CrossEntropyLoss.forwardPass]: Invalid input batch size: ' + shape[1]);
            }

            this.x = input;

            var result = [];
            for (var i = 0; i < this.batchSize; ++i) {
                result[i] = -Math.log(input.get(expected[i], i));
            }

            return result;
        }

        , backwardPass: function (input, expected) {
            if (this.batchSize !== expected.shape[1]) {
                throw new Error('[CrossEntropyLoss.backwardPass]: Invalid expected values');
            }

            var shape = this.x.shape;
            var expected = expected.toArray()[0];

            var result = Matrix.zeros(shape[0], shape[1]);

            for (var i = 0; i < shape[1]; ++i) {
                result.set(expected[i], i, -1/this.x.get(expected[i], i));
            }

            return result;
        }

        , x: {}

    }


}
'use strict';

{
    var v = require('vectorious')
        , Matrix = v.Matrix;

    function CrossEntropyLoss () {}

    module.exports = CrossEntropyLoss;

    CrossEntropyLoss.prototype = {
        forwardPass: function (input, expected) {
            if (!expected) {
                throw new Error('[CrossEntropyLoss.forwardPass]: Invalid expected values');
            }

            var shape = input.shape;

            this.x = input;
            input = input.toArray();

            var result = [];
            for (var i = 0; i < input[0].length; ++i) {
                result[i] = -Math.log(input[expected[i]][i]);
            }

            return result;
        }

        , backwardPass: function (input, expected) {
            var shape = this.x.shape;

            var result = Matrix.zeros(shape[0], shape[1]);

            for (var i = 0; i < shape[1]; ++i) {
                result.set(expected[i], i, -1/this.x.get(expected[i], i));
            }

            return result;
        }

        , x: {}

    }


}
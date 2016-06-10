'use strict';

{
    var v = require('vectorious')
        , Matrix = v.Matrix;

    function ReLULayer (batchSize) {
        if (!batchSize) {
            throw Error('ReLULayer constructor missing arguments');
        }

        this.batchSize = batchSize;
    }

    module.exports = ReLULayer;

    ReLULayer.prototype = {
        forwardPass: function (input) {
            if (this.batchSize !== input.shape[1]) {
                throw new Error('[ReLULayer.forwardPass]: Invalid input batch size: ' + input.shape[1]);
            }

            this.out = input;

            this.out.map( x => (x < 0) ? 0 : x );

            return this.out;
        }

        , backwardPass: function (input) {
            if (this.batchSize !== input.shape[1]) {
                throw new Error('[ReLULayer.backwardPass]: Invalid input batch size: ' + input.shape[1]);
            }

            var back = input;

            input.each( (x, i, j) => {
                if (this.out.get(i, j) === 0) {
                    back.set(i, j, 0);
                }
            });

            return back;
        }

        , out: {}
    }
}
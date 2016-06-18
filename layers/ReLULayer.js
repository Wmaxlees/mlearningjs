'use strict';

{
    var v = require('vectorious')
        , Matrix = v.Matrix;

    function ReLULayer () {}

    module.exports = ReLULayer;

    ReLULayer.prototype = {
        forwardPass: function (input) {
            this.out = input;

            this.out.map( x => (x < 0) ? 0 : x );

            return this.out;
        }

        , backwardPass: function (input) {
            input.each( (x, i, j) => {
                if (this.out.get(i, j) === 0) {
                    input.set(i, j, 0);
                }
            });

            return input;
        }

        , out: {}
    }
}
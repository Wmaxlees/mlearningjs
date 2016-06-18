'use strict';

{

    function Layer (calcRegLoss, outputShape, network) {
        this.calcRegLoss = calcRegLoss;
        this.outputShape = outputShape;
        this.network = network;
    }

    module.exports = Layer;

    Layer.prototype = {
        getOutputShape: function () {
            return this.outputShape;
        }

        , forwardPass: function (input) {
            if (!this.network) {
                return input;
            } else {
                return this.network.forwardPass(input);
            }
        }

        , backwardPass: function (input, batchSize) {
            if (!this.network) {
                return input;
            } else {
                return this.network.backwardPass(input, batchSize);
            }
        } 


    }
}
'use strict';

{
    var outerProduct = require(__dirname + '/../outerProduct.helper')
        , v = require('vectorious')
        , Vector = v.Vector
        , Matrix = v.Matrix;

    const lamda = 0.0001;

    function NNLayer (inputSize, outputSize, gradientStepSize) {
        if (!outputSize || !inputSize || !gradientStepSize) {
            throw new Error('NNLayer constructor missing arguments');
        }

        this.W = Matrix.random(outputSize, inputSize, 0.001);
        this.B = Matrix.random(outputSize, 1, 0.01);

        this.gradientStepSize = gradientStepSize;
    }
    
    module.exports = NNLayer;

    NNLayer.prototype = {
        forwardPass: function (input) {
            this.x = input;

            this.out = Matrix.multiply(this.W, input);
            this.out.each( (x, i, j) => {
                this.out.set(i, j, x + this.B.get(i, 0));
            })

            return this.out;
        }

        , getRegularizationLoss: function () {
            var total = 0.0;
            this.W.each( (x, i, j) => {
                total += x;
            });

            return total*lamda;
        }

        , backwardPass: function(input, batchSize) {
            var inputArray = input.transpose().toArray();
            var x = this.x.transpose().toArray();
            var dW = Matrix.zeros(this.W.shape[0], this.W.shape[1]);
            for (var i = 0; i < batchSize; ++i) {
                dW.add(outerProduct(x[i], inputArray[i]));
            }

            dW.scale(this.gradientStepSize/batchSize);
            this.W.add(dW);

            var averageLoss = new Matrix([averageRows(input.toArray())]);
            this.B.add(Matrix.scale(averageLoss, this.gradientStepSize).transpose());

            return Matrix.multiply(this.W.transpose(), input);

            this.gradientStepSize -= this.gradientStepSize * 0.001;
        }

        , toConsole: function () {
            console.log(this.W.toString());
        }

        , x: []

        , out: []
    }

    function averageRows (input) {
        for (var i = 0; i < input.length; ++i) {
            var length = input[i].length;
            input[i] = input[i].reduce( (total, num) => {
                return total + num;
            });

            input[i] /= length;
        }

        return input;
    }
}
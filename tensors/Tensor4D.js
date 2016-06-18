'use strict';

{
    function Tensor4D (matrixArray) {
        this.value = matrixArray;
        this.shape = matrixArray[0].shape;
        this.shape.push(matrixArray.length);
    }

    module.exports = Tensor4D;

    Tensor4D.prototype = {
        get: function (x, y, z, w) {
            return this.value[w].get(x, y, z);
        }

        , set: function (x, y, z, w, value) {
            this.value[w].set(x, y, z, value);
        }

        , shape: []

        , getLayer: function (layer) {
            if (layer >= this.value.length) {
                throw new Error('[Tensor4D.getLayer]: Layer does not exist')
            }

            return this.value[layer];
        }

        , toArray: function () {
            var result = [];
            for (var i = 0; i < this.value.length; ++i) {
                result.push(this.value[i].toArray());
            }
            return result;
        }

        , toString: function () {
            var result = '[';
            for (var i = 0; i < this.value.length; ++i) {
                result += this.value[i].toString();
            }
            result += ']';

            return result;
        }
    }

}
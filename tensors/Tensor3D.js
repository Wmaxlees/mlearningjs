'use strict';

{
    var _ = require('lodash');

    function Tensor3D (matrixArray) {
        this.value = matrixArray;
        this.shape = matrixArray[0].shape;
        this.shape.push(matrixArray.length);
    }

    module.exports = Tensor3D;

    Tensor3D.prototype = {
        get: function (x, y, z) {
            return this.value[z].get(x, y);
        }

        , set: function (x, y, z, value) {
            this.value[z].set(x, y, value);
        }

        , shape: []

        , transpose: function () {
            var result = _.cloneDeep(this);

            for (var i = 0; i < this.value.length; ++i) {
                result.value[i] = this.value[i].transpose();
            }

            return result;
        }

        , getLayer: function (layer) {
            if (layer >= this.value.length) {
                throw new Error('[Tensor3D.getLayer]: Layer does not exist')
            }

            var result = this.value[layer];
            result.shape = [
                this.shape[0],
                this.shape[1]
            ];

            return result;
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
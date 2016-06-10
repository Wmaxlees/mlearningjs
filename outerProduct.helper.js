'use strict';

{
    var v = require('vectorious')
        , Matrix = v.Matrix;

    module.exports = (a, b) => {
        var builderArray = [];

        for (var i = 0; i < a.length; ++i) {
            builderArray[i] = [];
            for (var j = 0; j < b.length; ++j) {
                builderArray[i].push(a[i]*b[j]);
            }
        }

        return new Matrix(builderArray);
    }

}
'use static';

{
    function Matrix2VectorLayer () {}

    module.exports = Matrix2VectorLayer;

    Matrix2VectorLayer.prototype = {
        forwardPass: function (input) {
            var result = [];
            for (var i = 0; i < input.shape[2]; ++i) {
                result.push(vectorize(input.getLayer[i]));
            }

            return new Matrix(result).transpose();
        }

        , backwardPass: function (input, batchSize) {
            throw new Error('[Matrix2VectorLayer.backwardPass]: Not implemented yet');
        }

    }

    function vectorize (matrix) {
        var result = [];

        for (var i = 0; i < matrix.shape[1]; ++i) {
            for (var j = 0; j < matrix.shape[0]; ++j) {
                result.push(matrix.get(j, i));
            }
        }

        return result;
    }
}
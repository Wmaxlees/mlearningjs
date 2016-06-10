'use strict';

{

    module.exports = {
        apply: (tensor, xOffset, yOffset, filterWidth, filterHeight, depth) => {
            var builderArray = [];
            for (var i = 0; i < filterHeight; ++i) {
                for (var j = 0; j < filterWidth; ++j) {
                    builderArray = builderArray.concat(tensor[i+yOffset][j+xOffset]);
                }
            }

            return builderArray;
        }
    }

}
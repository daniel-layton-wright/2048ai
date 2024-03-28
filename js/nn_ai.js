

NNAI = function(game) {
    this.game = game;
    this.session = new onnx.InferenceSession();
    this.sessionPromise = this.session.loadModel('network.onnx').then(() => {
        console.log('Model loaded');
    });
};

NNAI.prototype.nextMove = async function() {
    await this.sessionPromise;
    tensor = gridToTensor(this.game.grid);
    output = await this.session.run([tensor]);
    var outputTensor = output.values().next().value;
    var outputArray = Array.from(outputTensor.data);
    console.log('Output: ' + outputArray);
    var maxIndex = outputArray.indexOf(Math.max(...outputArray));
    var direction = maxIndex % 4;

    // direction currently is  0:left 1:right 2:up 3:down
    // convert to 0: up, 1:right, 2: down, 3: left
    switch (direction) {
        case 0:
            direction = 3;
            break;
        case 1:
            direction = 1;
            break;
        case 2:
            direction = 0;
            break;
        case 3:
            direction = 2;
            break;
    }

    return direction;
};

gridToTensor = function(grid) {
    // Make a 16 x 4 x 4 array of zeros
    gridArray = [];
    for (var i = 0; i < 16; i++) {
        for (var j = 0; j < 4; j++) {
            for (var k = 0; k < 4; k++) {
                gridArray.push(0);
            }
        }
    }

    // Fill in the values
    grid.eachCell(function(x, y, tile) {
        logTileValue = 0;
        if (tile) {
            logTileValue = Math.log2(tile.value);
        }
        ind = 16 * logTileValue + 4*y + x;
        gridArray[ind] = 1;
    });

    tensor = new onnx.Tensor(new Float32Array(gridArray), 'float32', [1, 16, 4, 4]);
    return tensor;
};
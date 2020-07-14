const nn = require('brain.js');
const smath = require('mathjs');

function sigmoid (x) {
    return 1/(1 + Math.E**(-x))
}

function z_gen (x, y) {
    return sigmoid(x) + Math.tanh(y)
}

const net = nn.NeuralNetwork()

function datagen (n) {
    range_x = smath.range(0, n, 0.1);
    range_y = smath.range(0, n, 0.1);
    inputs = [];

    for (let i = 0; i < r)


}
#include "neural_net.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
void init_network(NeuralNet *net) {
    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->weights_ih[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < OUTPUT_SIZE; j++)
            net->weights_ho[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->bias_h[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->bias_o[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

void forward(NeuralNet *net, float inputs[INPUT_SIZE]) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        net->hidden[j] = 0;
        for (int i = 0; i < INPUT_SIZE; i++)
            net->hidden[j] += inputs[i] * net->weights_ih[i][j];
        net->hidden[j] += net->bias_h[j];
        net->hidden[j] = sigmoid(net->hidden[j]);
    }
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        net->output[j] = 0;
        for (int i = 0; i < HIDDEN_SIZE; i++)
            net->output[j] += net->hidden[i] * net->weights_ho[i][j];
        net->output[j] += net->bias_o[j];
        net->output[j] = sigmoid(net->output[j]);
    }
}

void backward(NeuralNet *net, float inputs[], float target[], float learning_rate) {
    // Output layer errors
    float output_errors[OUTPUT_SIZE];
    for (int j = 0; j < OUTPUT_SIZE; j++)
        output_errors[j] = target[j] - net->output[j];

    // Update hidden->output weights
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < OUTPUT_SIZE; j++)
            net->weights_ho[i][j] += learning_rate * output_errors[j]
                                   * sigmoid_derivative(net->output[j])
                                   * net->hidden[i];

    // Update output biases
    for (int j = 0; j < OUTPUT_SIZE; j++)
        net->bias_o[j] += learning_rate * output_errors[j]
                        * sigmoid_derivative(net->output[j]);

    // Hidden layer errors - propagate backwards
    float hidden_errors[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_errors[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            hidden_errors[i] += output_errors[j] * net->weights_ho[i][j];
    }

    // Update input->hidden weights
    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->weights_ih[i][j] += learning_rate * hidden_errors[j]
                                   * sigmoid_derivative(net->hidden[j])
                                   * inputs[i];

    // Update hidden biases
    for (int j = 0; j < HIDDEN_SIZE; j++)
        net->bias_h[j] += learning_rate * hidden_errors[j]
                        * sigmoid_derivative(net->hidden[j]);
}

int get_action(NeuralNet *net) {
    int best = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (net->output[i] > net->output[best])
            best = i;
    return best;
}

void save_network(NeuralNet *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (f) {
        fwrite(net, sizeof(NeuralNet), 1, f);
        fclose(f);
    }
}

void load_network(NeuralNet *net, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (f) {
        fread(net, sizeof(NeuralNet), 1, f);
        fclose(f);
    }
}

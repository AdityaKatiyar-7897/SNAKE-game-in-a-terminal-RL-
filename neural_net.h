#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#define INPUT_SIZE 8
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 4

typedef struct{
	float weights_ih[8][16];
	float weights_ho[16][4];
	float hidden[16];
	float output[4];
	float bias_h[16];
	float bias_o[4];
} NeuralNet;


void init_network(NeuralNet *net);
void forward(NeuralNet *net, float inputs[INPUT_SIZE]);
int get_action(NeuralNet *net);
void backward(NeuralNet *net, float inputs[], float target[], float learning_rate);

void save_network(NeuralNet *net, const char *filename);
void load_network(NeuralNet *net, const char *filename);

#endif

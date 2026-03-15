#ifndef AGENT_H
#define AGENT_H

#include "game.h"
#include "neural_net.h"

typedef struct{
	float inputs[INPUT_SIZE];
	int action;
	float reward;
	float next_inputs[INPUT_SIZE];
	int done;
} Experience;

typedef struct{
	Experience arr[10000];
	int size;
	int index;
} ReplayMemory;

typedef struct {
    NeuralNet net;
    ReplayMemory memory;
    float learning_rate;
    float epsilon;
} Agent;

void init_agent(Agent *agent);
void remember(Agent *agent, float inputs[], int action, float reward, float next_inputs[], int done);
void train(Agent *agent);
int agent_action(Agent *agent, float inputs[]);


#endif

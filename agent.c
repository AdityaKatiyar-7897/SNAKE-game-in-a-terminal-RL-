#include "agent.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void init_agent(Agent *agent){
	agent->epsilon = 1.0f;
	agent->learning_rate = 0.001f;
	agent->memory.size =0;
	agent->memory.index = 0;

	init_network(&agent->net);
}

void remember(Agent *agent, float inputs[], int action, float reward, float next_inputs[], int done) {
    int idx = agent->memory.index;
    
    memcpy(agent->memory.arr[idx].inputs, inputs, sizeof(float) * INPUT_SIZE);
    memcpy(agent->memory.arr[idx].next_inputs, next_inputs, sizeof(float) * INPUT_SIZE);
    agent->memory.arr[idx].action = action;
    agent->memory.arr[idx].reward = reward;
    agent->memory.arr[idx].done = done;

    agent->memory.index = (agent->memory.index + 1) % 10000;
    if (agent->memory.size < 10000) agent->memory.size++;
}

int agent_action(Agent *agent, float inputs[]) {
    if ((float)rand() / RAND_MAX < agent->epsilon) {
        return rand() % OUTPUT_SIZE;
    }
    forward(&agent->net, inputs);
    return get_action(&agent->net);
}

void train(Agent *agent){
	if (agent->memory.size <32) return;

	for (int b = 0; b < 32 ; b++){
		int idx = rand() % agent->memory.size;
		Experience *exp = &agent->memory.arr[idx];

		forward(&agent->net, exp->next_inputs);
		float max_next = agent->net.output[get_action(&agent->net)];
		float target = exp->reward + (exp->done ? 0.0f : 0.99f * max_next);
		
		forward(&agent->net, exp->inputs);
		float error = target - agent->net.output[exp->action];
		agent->net.output[exp->action] += agent->learning_rate * error;
		}
    if (agent->epsilon > 0.1f) agent->epsilon -= 0.001f;
}

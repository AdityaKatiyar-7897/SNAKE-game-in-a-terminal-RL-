#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "game.h"
#include "neural_net.h"
#include "agent.h"
#include "terminal.h"

void draw(GameState *state, int generation, float epsilon, int best_score) {
    clear_screen();
    printf("+");
    for (int j = 0; j < GRID_WIDTH; j++) printf("-");
    printf("+\n");

    for (int i = 0; i < GRID_HEIGHT; i++) {
        printf("|");
        for (int j = 0; j < GRID_WIDTH; j++) {
            if (state->grid[i][j] == SNAKE)     printf("\033[32mO\033[0m");
            else if (state->grid[i][j] == FOOD) printf("\033[31mF\033[0m");
            else printf(" ");
        }
        printf("|\n");
    }

    printf("+");
    for (int j = 0; j < GRID_WIDTH; j++) printf("-");
    printf("+\n");
    printf("Generation: %d | Score: %d | Best: %d | Epsilon: %.2f\n",
           generation, state->score, best_score, epsilon);
}

int main() {
    srand(42);

    Agent agent;
    init_agent(&agent);
    load_network(&agent.net, "snake_brain.bin");

    int generation = 1;
    int best_score = 0;

    while (1) {
        GameState state;
        init_game(&state);

        float inputs[INPUT_SIZE];
        float next_inputs[INPUT_SIZE];
        int steps = 0;
        int last_score = 0;

        while (state.alive && steps < 2000) {
            get_inputs(&state, inputs);
            int action = agent_action(&agent, inputs);

            if (action == 0) state.snake.direction = UP;
            if (action == 1) state.snake.direction = DOWN;
            if (action == 2) state.snake.direction = LEFT;
            if (action == 3) state.snake.direction = RIGHT;

            update_game(&state);
            get_inputs(&state, next_inputs);

            float reward = 0;
            if (!state.alive)                  reward = -10.0f;
            else if (state.score > last_score) reward = 10.0f;
            else                               reward = -0.01f;

            remember(&agent, inputs, action, reward, next_inputs, !state.alive);
            train(&agent);

            last_score = state.score;
            steps++;

            if (generation % 500 == 0) {
                draw(&state, generation, agent.epsilon, best_score);
                usleep(50000);
            }
        }

        if (state.score > best_score) best_score = state.score;

        printf("Gen %d done | Score: %d | Best: %d | Epsilon: %.2f\n",
               generation, state.score, best_score, agent.epsilon);

        if (generation % 100 == 0)
            save_network(&agent.net, "snake_brain.bin");

        generation++;
    }

    return 0;
}

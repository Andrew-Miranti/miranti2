// compile with:  g++ -I/class/cs232/REU/tbb/include -L/class/cs232/REU/tbb/lib -o conwaylife conwaylife.c -ltbb 
// run with: ./conwaylife 

#include <string.h>	/* for strerror() */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>

#include "tbb/tick_count.h"
using tbb::tick_count;

void
set(char *board, unsigned board_size, unsigned x, unsigned y) {
	// printf("setting %d %d\n", x, y);
	unsigned index = board_size * y + x;
	board[index] = 1;
}

void
clear(char *board, unsigned board_size, unsigned x, unsigned y) {
	unsigned index = board_size * y + x;
	board[index] = 0;
}

char
is_set(char *board, unsigned board_size, unsigned x, unsigned y) {
	unsigned index = board_size * y + x;
	return board[index];
}

void 
clear_board(char *board, unsigned board_size) {
	bzero(board, board_size * board_size * sizeof(char));
}

void
draw_diehard(char *board, unsigned board_size) {
	int basex = board_size/2 - 4, basey = board_size/2 - 2;
	assert(basex > 0);
	set(board, board_size, basex+0, basey+1);
	set(board, board_size, basex+1, basey+1);
	set(board, board_size, basex+1, basey+2);
	set(board, board_size, basex+5, basey+2);
	set(board, board_size, basex+6, basey+0);
	set(board, board_size, basex+6, basey+2);
	set(board, board_size, basex+7, basey+2);
}

int x_offsets[8] = {-1, -1, -1,  0,  0,  1,  1,  1};
int y_offsets[8] = {-1,  0,  1, -1,  1, -1,  0,  1};

int main(int argc, char *argv[]) {
	int board_size = 1024;
	if (argc > 1) {
		board_size = atoi(argv[1]);
	}

	char *boards[3];
	for (int i = 0 ; i < 3 ; i ++) {
		boards[i] = (char *) malloc(board_size * board_size * sizeof(char));
		clear_board(boards[i], board_size);
	}

	draw_diehard(boards[0], board_size);
	unsigned prev_board = 0;  
	unsigned prev2_board = 2;  

	tick_count start = tick_count::now();

	for (int steps = 0 ; steps < 2000 ; steps ++) {
		printf("step %d\n", steps);
		unsigned current_board = (prev_board + 1) % 3;
		clear_board(boards[current_board], board_size);
		bool matches_prev = true;
		bool matches_prev2 = true;

		for (int j = 1 ; j < board_size - 1 ; j ++) {
			for (int i = 1 ; i < board_size - 1 ; i ++) {
				bool was_set = is_set(boards[prev_board], board_size, i, j);
				unsigned neighbors_set = 0;
				for (int k = 0 ; k < 8 ; k ++) {
					neighbors_set += is_set(boards[prev_board], board_size, i+x_offsets[k], j+y_offsets[k]);
				} 
				bool set_in_next = ((was_set && neighbors_set == 2) || neighbors_set == 3);
				if (set_in_next) {
					set(boards[current_board], board_size, i, j);
				}
				if (set_in_next != was_set) { matches_prev = false; }
				if (set_in_next != is_set(boards[prev2_board], board_size, i, j)) { matches_prev2 = false; }
			}
		}

		if (matches_prev) {
			printf("stable after %d iterations\n", steps + 1);
			break;
		}
		if (matches_prev2) {
			printf("quasi-stable after %d iterations\n", steps + 1);
			break;
		}

		prev2_board = prev_board;
		prev_board = current_board;
	}

	tick_count end = tick_count::now();
	printf("time = %f seconds\n", (end - start).seconds());  
}

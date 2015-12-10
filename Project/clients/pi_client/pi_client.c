#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include "distributed.h"
#define THREAD_COUNT 8
#define POINTS_PER_ITERATION 4096

unsigned long total = 0, hits = 0;
pthread_mutex_t resource = PTHREAD_MUTEX_INITIALIZER;
sig_atomic_t term = 0;

void stop() {
    term = 1;
}

double randDouble() {
    return (double)rand() / RAND_MAX;
}

void * runSimulation(void * _) {
    while (!term) {
        unsigned innerHits = 0;
        for (int i = 0; i < POINTS_PER_ITERATION; ++i)
            double x = randDouble(), y = randDouble();
            if (x*x + y*y < 1.0) {
                innerHits++;
            }
        }
        pthread_mutex_lock(&resource);
        total += POINTS_PER_ITERATION;
        hits += innerHits;
        pthread_mutex_unlock(&resource);
    }
    return NULL;
}

void sendData() {
    unsigned long[2] data;
    data[0] = total;
    data[1] = hits;
    sendToServer(data, sizeof(data));
}

int main(int argc, char ** argv) {
    struct sigaction stopper;
    stopper.sa_handler = stop;
    sigemptyset(&stopper.sa_mask);
    stopper.sa_flags = 0;
    sigaction(SIGINT, &stopper, NULL);
    sigaction(SIGTERM, &stopper, NULL);

    pthread_t[THREAD_COUNT] threads;
    for (int i = 0; i < THREAD_COUNT; ++i) {
        pthread_create(&threads[i], NULL, runSimulation, NULL);
    }
    for (int i = 0; i < THREAD_COUNT; ++i) {
        pthread_join(threads[i], NULL);
    }
    sendData();
    return 0;
}

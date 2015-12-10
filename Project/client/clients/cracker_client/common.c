#define _GNU_SOURCE
#include <time.h>
#include <assert.h>
#include <string.h>

#include "common.h"

/* Returns a "wall clock" timer value in seconds.
   This timer keeps ticking even when the thread or process is idle. */
double getTime() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* Returns a process-wide CPU time value in seconds.
   This will tick faster than getTime() if mutiple threads are busy. */
double getCPUTime() {
  struct timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* Return a thread-specific CPU time value in seconds.
   This timer only ticks when the current thread is not idle. */
double getThreadCPUTime() {
  struct timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* Given a string like "abc.....", return the number of letters in it. */
int getPrefixLength(const char *str) {
  char *first_dot = strchr(str, '.');
  if (first_dot)
    return first_dot - str;
  else
    return strlen(str);
}

/* Set 'result' to the string that you'd get after replacing every
   character with 'a' and calling incrementString() on it 'n' times. */
void setStringPosition(char *result, long n) {
  char *p = result + strlen(result) - 1;

  while (p >= result) {
    *p-- = 'a' + (n % 26);
    n /= 26;
  }
}

/* Increment the letters in 'str'.
   Returns 1 if the increment is successful.
   Returns 0 if all the letters in the string are 'z'. */
int incrementString(char *str) {
  assert(str);

  char *p = str + strlen(str) - 1;

  // find the last character after the prefix that is not a z
  while (p >= str && *p == 'z')
    p--;

  // if we found one, increment it
  if (p >= str) {

    // increment this character and move to the next one
    (*p++)++;

    // and set all the remaining characters to 'a'
    while (*p)
      *p++ = 'a';

    return 1;

  } else {

    // reached the end
    return 0;
  }
}

/**
 *  Given a queue_element, place it on the queue.  Can be called by multiple threads.
 *  Blocks if the number of items on the queue is equal to the queue's max size
 */
void queue_push(queue_t* queue, void* data) {
    pthread_mutex_lock(&queue->m);
    while (queue->maxSize && queue->count == queue->maxSize) {pthread_cond_wait(&queue->cv, &queue->m);}

    ++queue->count;
    queue_node_t * newData = calloc(1, sizeof(queue_node_t));
    newData->data = data;
    if (queue->tail) {
        queue->tail->next = newData;
        queue->tail = newData;
    } else {
        queue->head = queue->tail = newData;
    }
    pthread_cond_broadcast(&queue->cv);

    pthread_mutex_unlock(&queue->m);
}

/**
 *  Retrieve the queue_element at the front of the queue.  Can be called by multiple threads.
 *  Blocks if there are no tasks on the queue.
 */
void* queue_pull(queue_t* queue) {
    pthread_mutex_lock(&queue->m);
    while (!queue->count) {pthread_cond_wait(&queue->cv, &queue->m);}

    --queue->count;
    queue_node_t * oldHead = queue->head;
    if (queue->head == queue->tail) {
        queue->tail = NULL;
    }
    queue->head = queue->head->next;
    void * result = oldHead->data;
    free(oldHead);
    pthread_cond_broadcast(&queue->cv);

    pthread_mutex_unlock(&queue->m);
    return result;
}

/**
 *  Initializes the queue
 */
void queue_init(queue_t* queue, int maxSize){
    queue->maxSize = maxSize;
    queue->count = 0;
    queue->head = queue->tail = NULL;
    pthread_cond_init(&queue->cv, NULL);
    pthread_mutex_init(&queue->m, NULL);
}

/**
 *  Destorys the queue, freeing any remaining nodes in it.
 */
void queue_destroy(queue_t* queue){
    queue_node_t * itr = queue->head;

    while (itr) {
        queue_node_t * nextitr = itr->next;
        free(itr);
        itr = nextitr;
    }

    pthread_cond_destroy(&queue->cv);
    pthread_mutex_destroy(&queue->m);
}

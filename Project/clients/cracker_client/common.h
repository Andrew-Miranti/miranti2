#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#ifndef __CRACKER_COMMON_H__
#define __CRACKER_COMMON_H__

/* Returns a "wall clock" timer value in seconds.
   This timer keeps ticking even when the thread or process is idle. */
double getTime();

/* Returns a process-wide CPU time value in seconds.
   This will tick faster than getTime() if mutiple threads are busy. */
double getCPUTime();

/* Return a thread-local CPU time value in seconds.
   This timer only ticks when the current thread is not idle. */
double getThreadCPUTime();

/* Given a string like "abc.....", return the number of letters in it. */
int getPrefixLength(const char *str);

/* Set 'result' to the string that you'd get after replacing every
   character with 'a' and calling incrementString() on it 'n' times. */
void setStringPosition(char *result, long n);

/* Increment the letters in 'str'.
   Returns 1 if the increment is successful.
   Returns 0 if all the letters in the string are 'z'.

   For example:
     If str=="howdy", str will be changed to "howdz"
       and the function will return 1.
     If str=="jazz", str will be changed to "jbaa"
       and the function will return 1.
     If str=="zzzzzz", str will be unchanged and the
       function will return 0.

   'str' must contain only lowercase letters, and it will contain only
   lowercase letters when the function returns.
*/
int incrementString(char *str);

// Changes to this file will be ignored when your project is graded

#define USERNAME_MAX_LENGTH 8
#define HASHED_LENGTH 13
#define KNOWN_LENGTH 8

typedef struct { 
    char username[USERNAME_MAX_LENGTH + 1];
    char hashed[HASHED_LENGTH + 1];
    char known[KNOWN_LENGTH + 1]; 
} user_t;

typedef struct {
    long firstIndex;
    long count;
    user_t userInfo;
} parallel_user_t;

typedef struct queue_node_t {
  struct queue_node_t *next;
  void* data;
} queue_node_t;

typedef struct {
  queue_node_t *head, *tail;
  int count;
  int maxSize;
  pthread_cond_t cv;
  pthread_mutex_t m;
} queue_t;

/* Put a task onto the queue. */
void queue_push(queue_t* queue, void* data);
/* If the queue is empty, then this call will block until a put call completes. */
void* queue_pull(queue_t* queue);
void queue_init(queue_t* queue, int maxSize);
void queue_destroy(queue_t* queue);

#endif /* __CRACKER_COMMON_H__ */

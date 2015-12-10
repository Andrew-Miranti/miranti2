#include "cracker.h"

sig_atomic_t stop = 0;
sig_atomic_t term = 0;
sig_atomic_t totalHashes = 0;
int total = 0;
int success = 0;
int threadCount = 0;
const char * globalLine = NULL;
char * password = NULL;
pthread_mutex_t accumulatorLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t stopLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t termLock = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t barrier;

/* Compute the subrange that a thread should work on.
     unknown_letter_count - The number of letters in the password that
       are not known.
     thread_count - The number of worker threads.
     thread_id - My thread ID, where the IDs start at 1.
     start_index - (output parameter) This will be set to the index of
       the first possible password this thread should try.
     count - (output parameter) This will be set to the number of
       passwords this thread should try.
*/
void getSubrange(int unknown_letter_count, int thread_count, int thread_id,
                 long *start_index, long *count) {
  int i;
  long max = 1, end_index;
  for (i=0; i < unknown_letter_count; i++) max *= 26;
  *start_index = max * (thread_id - 1) / thread_count;
  end_index = max * thread_id / thread_count;
  if (end_index > max)
    end_index = max;
  *count = end_index - *start_index;
}

void doCrack(parallel_user_t * parallel_info, int id) {
    user_t * next = &parallel_info->userInfo;

    char test_pw[9];
    strcpy(test_pw, next->known);

    // unknown_chars will point to the part of the password that is unknown
    char *unknown_chars = test_pw + getPrefixLength(test_pw);
    setStringPosition(unknown_chars, parallel_info->firstIndex);
    printf("Thread %d: Start %s at %ld (%s)\n", id, next->username, parallel_info->firstIndex, test_pw);

    int found = 0;
    int hash_count = 0;
    struct crypt_data cdata;
    cdata.initialized = 0;

    do {
		pthread_mutex_lock(&stopLock);
		if (stop) {
			pthread_mutex_unlock(&stopLock);
			break;
		}
		pthread_mutex_unlock(&stopLock);
        const char *hashed = crypt_r(test_pw, "xx", &cdata);

        // uncomment this if you want to see the hash function doing its thing
        // printf("%s -> %s\n", test_pw, hashed);
        hash_count++;
        found = !strcmp(hashed, next->hashed);
    } while (!found && incrementString(unknown_chars) && hash_count < parallel_info->count);

	pthread_mutex_lock(&stopLock);
    if (found)
        stop = 1;
	pthread_mutex_unlock(&stopLock);
    
    char * exitString = (found ? "found" : (stop ? "cancelled" : "end"));
    printf("Thread %d: Stop after %d iterations (%s)\n", id, hash_count, exitString);

    if (found) {
        password = strdup(test_pw);
    }

    pthread_mutex_lock(&accumulatorLock);
    totalHashes += hash_count;
    pthread_mutex_unlock(&accumulatorLock);
}

void * runCracker(void * info) {
	int id = (int)(long)info;
	while (1) {
		pthread_barrier_wait(&barrier);
		pthread_mutex_lock(&termLock);
		if (term) {
			pthread_mutex_unlock(&termLock);
			break;
		}
		pthread_mutex_unlock(&termLock);
    	
		parallel_user_t toCrack;
    	sscanf(globalLine, "%8s %13s %8s", toCrack.userInfo.username, toCrack.userInfo.hashed, toCrack.userInfo.known);
    	int unknownLength = strlen(toCrack.userInfo.known + getPrefixLength(toCrack.userInfo.known));
    	getSubrange(unknownLength, threadCount, id, &toCrack.firstIndex, &toCrack.count);
    	doCrack(&toCrack, id);
		
		pthread_barrier_wait(&barrier);
	}
    return NULL;
}

int main() {
    const int thread_count = 8;
    int i;
    size_t buf_len = 0;
    char *line = NULL;
	pthread_barrier_init(&barrier, NULL, thread_count + 1);
	total = 0;
	success = 0;
	pthread_t * threads = calloc(thread_count, sizeof(pthread_t));
	threadCount = thread_count;

	for (i = 0; i < thread_count; ++i) {
		pthread_create(&threads[i], NULL, runCracker, (void*)(long)(i + 1));
	}

    while (getline(&line, &buf_len, stdin) != -1) {
        double start_cpu_time = getCPUTime();
        total++;
        stop = 0;
        totalHashes = 0;
        password = NULL;
        double start = getTime();
        char username [9]; 
        sscanf(line, "%8s", username);
        printf("Start %s\n", username);

		globalLine = line;

		pthread_barrier_wait(&barrier);
		pthread_barrier_wait(&barrier);

        double elapsed = getTime() - start;
    
		if (password) {
            success++;
            printf("Password for %s is %s (%d hashes in %.2f seconds)\n", username, password, totalHashes, elapsed);           
        } else {
            printf("Password for %s not found (%d hashes in %.2f seconds)\n", username, totalHashes, elapsed);
        }
        
        double total_cpu_time = getCPUTime() - start_cpu_time; 
        printf("Total CPU time: %.2f seconds.\n", total_cpu_time);
        printf("CPU usage: %.2fx\n\n", total_cpu_time / elapsed);
        free(password);
    }
	
	pthread_mutex_lock(&termLock);
	term = 1;
	pthread_barrier_wait(&barrier);
	pthread_mutex_unlock(&termLock);

	
	for (i = 0; i < thread_count; ++i) {
        pthread_join(threads[i], NULL);
    }
	free(threads);
    free(line);
	pthread_barrier_destroy(&barrier);
    return 0;
}

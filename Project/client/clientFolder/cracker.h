#define _GNU_SOURCE
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <crypt.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include "common.h"

int start(int thread_count);

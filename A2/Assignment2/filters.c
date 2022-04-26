#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>

typedef struct parallel_common_work{
    const filter *f;
    const int32_t *original_image;
    int32_t *output_image;
    int32_t width;
    int32_t height;
    int32_t num_threads;
    pthread_barrier_t *barrier;
    int32_t **min_max_arr;
    parallel_method method;
} parallel_common_work_t;

typedef struct parallel_work{
    parallel_common_work_t *common;
    int32_t id;
} parallel_work_t;


// #define _POSIX_BARRIERS 10

/************** FILTER CONSTANTS*****************/
/* laplacian */
int8_t lp3_m[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };
filter lp3_f = {3, lp3_m};

/* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */
/*************** COMMON WORK ***********************/
/* Process a single pixel and returns the value of processed pixel
 * you don't have to implement/use this function, but this is a hint
 * on how to reuse your code.
 * */
int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
    int32_t delta = (f->dimension - 1) / 2;
    int32_t cells_sum = 0;
    int32_t f_index = 0;

    /* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */

    int32_t i = (width * row) + column;
    target[i] = cells_sum;
    return cells_sum;
}

/********* SEQUENTIAL IMPLEMENTATIONS ***************/

// start and end are the indexes of the width
int32_t *apply_filter_row_by_row(const filter *f, const int32_t *original, int32_t *target,
                          int32_t width, int32_t height, int32_t start, int32_t end)
{

    int32_t max_value = INT32_MIN;
    int32_t min_value = INT32_MAX;

    for (int32_t x = start; x < end; x++)
    {
        for (int32_t y = 0; y < width; y++)
        {
            int32_t cur_value = apply2d(f, original, target, width, height, x, y);
            if (cur_value > max_value)
            {
                max_value = cur_value;
            }
            if (cur_value < min_value)
            {
                min_value = cur_value;
            }
        }
    }

    int32_t *result = malloc(sizeof(int32_t) * 2);
    result[0] = min_value;
    result[1] = max_value;
    return result;
}

/* TODO: your sequential implementation goes here.
 * IMPORTANT: you must test this thoroughly with lots of corner cases and 
 * check against your own manual calculations on paper, to make sure that your code
 * produces the correct image.
 * Correctness is CRUCIAL here, especially if you re-use this code for filtering 
 * pieces of the image in your parallel implementations! 
 */
void apply_filter2d(const filter *f, 
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height)
{
    int32_t *result = apply_filter_row_by_row(f, original, target, width, height, 0, height);

    // now normalize the target array based on min_value and max_value
    for (int32_t i = 0; i < width * height; i++){
        normalize_pixel(target, i, result[0], result[1]);
    }
    free(result);
}

/********* SHARDED_ROWS ***************/

int32_t *apply_sharded_rows(parallel_work_t *thread_work)
{

    const int32_t *original = thread_work->common->original_image;
    int32_t *target = thread_work->common->output_image;
    const filter *f = thread_work->common->f;
    int32_t width = thread_work->common->width;
    int32_t height = thread_work->common->height;
    int32_t num_threads = thread_work->common->num_threads;
    int num_sharded_rows = height / num_threads;
    // find corresponding row indexes for this thread
    int32_t start_row = num_sharded_rows * thread_work->id;
    int32_t end_row = num_sharded_rows + start_row;

    // last thread may get more
    if (end_row != height && thread_work->id == num_threads - 1){
        end_row = height;
    }

    int32_t *result = apply_filter_row_by_row(f, original, target, width, height, start_row, end_row);
    return result;
}

void normalize_sharded_rows(parallel_work_t *thread_work, int32_t min, int32_t max)
{
    int32_t width = thread_work->common->width;
    int32_t height = thread_work->common->height;
    int32_t num_sharded_rows = height / thread_work->common->num_threads;
     int32_t start_row = num_sharded_rows * thread_work->id;
    int32_t end_row = num_sharded_rows + start_row;

    // last thread may get more
    if (end_row != height && thread_work->id == thread_work->common->num_threads - 1){
        end_row = height;
    }

    for(int32_t x = start_row ; x < end_row; x++){
        for(int32_t y = 0 ; y < width; y++){
             int32_t i = (width * x) + y;
             normalize_pixel(thread_work->common->output_image, i, min, max);
        }
    }
}

/********* SHARDED_COLUMNS_COLUMN_MAJOR ***************/

int32_t *apply_sharded_columns_cols(parallel_work_t *thread_work){

    const int32_t *original = thread_work->common->original_image;
    int32_t *target = thread_work->common->output_image;
    const filter *f = thread_work->common->f;
    int32_t width = thread_work->common->width;
    int32_t height = thread_work->common->height;
    int32_t num_threads = thread_work->common->num_threads;
    int32_t num_sharded_columns = width / num_threads;
    // find corresponding column indexes for this thread
    int32_t start_col = num_sharded_columns * thread_work->id;
    int32_t end_col = num_sharded_columns + start_col;

    if (end_col != width && thread_work->id == num_threads - 1){
        end_col = width;
    }

    int32_t max_value = INT32_MIN;
    int32_t min_value = INT32_MAX;

    for(int32_t y = start_col; y < end_col; y++){
        for(int32_t x = 0 ; x < height; x++){
            int32_t cur_value = apply2d(f, original, target, width, height, x, y);
            if (cur_value > max_value)
            {
                max_value = cur_value;
            }
            if (cur_value < min_value)
            {
                min_value = cur_value;
            }
        }
    }

    int32_t *result = malloc(sizeof(int32_t) * 2);
    result[0] = min_value;
    result[1] = max_value;

    return result;
}

void normalize_sharded_columns_cols(parallel_work_t *thread_work, int32_t min, int32_t max)
{
     int32_t width = thread_work->common->width;
    int32_t height = thread_work->common->height;
    int32_t num_sharded_columns = width / thread_work->common->num_threads;
    int32_t start_col = num_sharded_columns * thread_work->id;
    int32_t end_col = num_sharded_columns + start_col;

    if (end_col != width && thread_work->id == thread_work->common->num_threads - 1){
        end_col = width;
    }

     for(int32_t y = start_col; y < end_col; y++){
        for(int32_t x = 0 ; x < height; x++){
             int32_t i = (width * x) + y;
             normalize_pixel(thread_work->common->output_image, i, min, max);
        }
    }
}

/********* SHARDED_COLUMNS_ROW_MAJOR ***************/
int32_t *apply_sharded_columns_rows(parallel_work_t *thread_work){
    int32_t num_columns = thread_work->common->width / thread_work->common->num_threads;
    int32_t start = num_columns * thread_work->id;
    int32_t end = start + num_columns;
    if (thread_work->id == thread_work->common->num_threads - 1) {
        end = thread_work->common->width;
    }

    int32_t max_value = INT32_MIN;
    int32_t min_value = INT32_MAX;

    for (int32_t i = 0; i < thread_work->common->height; i++) {
        for (int32_t j = start; j < end; j++) {
            int32_t value = apply2d(thread_work->common->f,
                                    thread_work->common->original_image,
                                    thread_work->common->output_image,
                                    thread_work->common->width,
                                    thread_work->common->height,
                                    i, j);
            if (value < min_value) {
                min_value = value;
            }
            if (value > max_value) {
                max_value = value;
            }
        }
    }

    int32_t *result = malloc(sizeof(int32_t) * 2);
    result[0] = min_value;
    result[1] = max_value;
    return result;
}

void normalize_sharded_columns_rows(parallel_work_t *thread_work, int32_t min, int32_t max){
    int32_t num_columns = thread_work->common->width / thread_work->common->num_threads;
    int32_t start = num_columns * thread_work->id;
    int32_t end = start + num_columns;
    if (thread_work->id == thread_work->common->num_threads - 1) {
        end = thread_work->common->width;
    }

    for (int32_t i = 0; i < thread_work->common->height; i++) {
        for (int32_t j = start; j < end; j++) {
            normalize_pixel(thread_work->common->output_image, i * thread_work->common->width + j, min, max);
        }
    }
}

/********* WORK_QUEUE ***************/
/* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */

inline int32_t get_num_chunk_1d(int32_t length, int32_t chunk_size) {
    int32_t result = length / chunk_size;
    if (length % chunk_size != 0)
        result++;
    return result;
}

inline int32_t get_num_chunk(int32_t width, int32_t height, int32_t chunk_size) {
    return get_num_chunk_1d(width, chunk_size) * get_num_chunk_1d(height, chunk_size);
}

int32_t get_task(work_queue_t *queue) {
    int result = pthread_mutex_lock(&queue->mutex);
    if (result != 0) {
        fprintf(stderr, "Work pool failed to lock mutex on fetching task.\n");
        exit(1);
    }

    int32_t index = queue->index;
    queue->index++;
    int32_t num_chunks = get_num_chunk(queue->width, queue->height, queue->chunk_size);
    if (index >= num_chunks) { // Wait for all filtering to complete
        if (queue->complete_count < num_chunks) {
            result = pthread_cond_wait(&(queue->cond), &(queue->mutex));
            if (result != 0) {
                fprintf(stderr, "Work pool failed to wait on conditional variable on fetching task.\n");
                exit(1);
            }
        }
    }

    result = pthread_mutex_unlock(&(queue->mutex));
    if (result != 0) {
        fprintf(stderr, "Work pool failed to unlock mutex on fetching task.\n");
        exit(1);
    }

    // If no task available, return -1 for worker threads to terminate itself.
    if (index >= 2 * num_chunks)
        return -1;

    return index;
}

void complete_task(work_queue_t *queue, int32_t min, int32_t max) {
    int result = pthread_mutex_lock(&(queue->mutex));
    if (result != 0) {
        fprintf(stderr, "Work pool failed to lock mutex on completing task.\n");
        exit(1);
    }
    if (min <= max) { // Setting min > max will not update the value
        if (queue->min > min)
            queue->min = min;
        if (queue->max < max)
            queue->max = max;
    }
    queue->complete_count++;

    // Signal cond var
    if (queue->complete_count == get_num_chunk(queue->width, queue->height, queue->chunk_size)) {
        result = pthread_cond_broadcast(&(queue->cond));
        if (result != 0) {
            fprintf(stderr, "Work pool failed to signal conditional variable.\n");
            exit(1);
        }
    }

    result = pthread_mutex_unlock(&(queue->mutex));
    if (result != 0) {
        fprintf(stderr, "Work pool failed to unlock mutex on completing task.\n");
        exit(1);
    }
}


/****************** ROW/COLUMN SHARDING ************/
/* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */
void* sharding_work(void *w)
{
    /* Your algorithm is essentially:
     *  1- Apply the filter on the image
     *  2- Wait for all threads to do the same
     *  3- Calculate global smallest/largest elements on the resulting image
     *  4- Scale back the pixels of the image. For the non work queue
     *      implementations, each thread should scale the same pixels
     *      that it worked on step 1.
     */
    parallel_work_t *thread_work = w;
    pthread_barrier_t *barrier = thread_work->common->barrier;
    int32_t **min_max_arr = thread_work->common->min_max_arr;
    int32_t num_threads = thread_work->common->num_threads;
    parallel_method method = thread_work->common->method;
    
    // default the function pointers to refers to the functions of SHARDED_ROWS
    int32_t* (*apply_function)(parallel_work_t *) = &apply_sharded_rows;
    void (*normalize_function)(parallel_work_t *, int32_t , int32_t) = &normalize_sharded_rows;

    switch (method){
        case SHARDED_ROWS:
            apply_function = &apply_sharded_rows;
            normalize_function = &normalize_sharded_rows;
            break;
        case SHARDED_COLUMNS_COLUMN_MAJOR:
            apply_function = &apply_sharded_columns_cols;
            normalize_function = &normalize_sharded_columns_cols;
            break;
        case SHARDED_COLUMNS_ROW_MAJOR:
            apply_function = &apply_sharded_columns_rows;
            normalize_function = &normalize_sharded_columns_rows;
            break;
        case WORK_QUEUE: // Should never reach here
            break;
    }

    // apply the corresponding method's apply_function 
    int32_t *result = (*apply_function)(thread_work);
    min_max_arr[thread_work->id] = result;

    // wait on barrier until all threads are done
    int res = pthread_barrier_wait(barrier);
    if(res != 0 && res != PTHREAD_BARRIER_SERIAL_THREAD){
        fprintf(stderr, "Failed to wait on pthread_barrier - %s.\n", strerror(errno));
        exit(1);
    }

    // now all threads are done, so find the global min and max and normalize
    int32_t min = INT32_MAX;
    int32_t max = INT32_MIN;
    for (int i = 0; i < num_threads; i++){
        for(int t = 0; t < 2; t++){
            if (min_max_arr[i][t] > max && min_max_arr[i][t] != INT32_MAX){
                max = min_max_arr[i][t];
            }
            if (min_max_arr[i][t] < min && min_max_arr[i][t] != INT32_MIN){
                min = min_max_arr[i][t];
            }
        }
    }

    // now corresponding method's normalize_function on target array based on global min and max
    (*normalize_function)(thread_work, min, max);

    return NULL;
}

/***************** WORK QUEUE *******************/
void* queue_work(void *work)
{
    work_queue_t *queue = (work_queue_t *)work;
    int32_t num_chunks = get_num_chunk(queue->width, queue->height, queue->chunk_size);

    while (1) {
        int32_t index = get_task(queue);
        if (index == -1)
            return NULL;
        
        if (index < num_chunks) {
            int32_t num_width = get_num_chunk_1d(queue->width, queue->chunk_size);
            int32_t i = index / num_width;
            int32_t j =  index % num_width;
            int32_t start_i = i * queue->chunk_size;
            int32_t end_i = (i + 1) * queue->chunk_size;
            if (end_i > queue->height)
                end_i = queue->height;
            int32_t start_j = j * queue->chunk_size;
            int32_t end_j = (j + 1) * queue->chunk_size;
            if (end_j > queue->width)
                end_j = queue->width;

            int32_t max_value = INT32_MIN;
            int32_t min_value = INT32_MAX;

            for (int32_t x = start_i; x < end_i; x++) {
                for (int32_t y = start_j; y < end_j; y++) {
                    int32_t value = apply2d(queue->f,
                                            queue->original_image,
                                            queue->output_image,
                                            queue->width,
                                            queue->height,
                                            x, y);
                    if (value < min_value) {
                        min_value = value;
                    }
                    if (value > max_value) {
                        max_value = value;
                    }
                }
            }

            complete_task(queue, min_value, max_value);
        } else {
            index -= num_chunks;
            int32_t num_width = get_num_chunk_1d(queue->width, queue->chunk_size);
            int32_t i = index / num_width;
            int32_t j =  index % num_width;
            int32_t start_i = i * queue->chunk_size;
            int32_t end_i = (i + 1) * queue->chunk_size;
            if (end_i > queue->height)
                end_i = queue->height;
            int32_t start_j = j * queue->chunk_size;
            int32_t end_j = (j + 1) * queue->chunk_size;
            if (end_j > queue->width)
                end_j = queue->width;

            for (int32_t x = start_i; x < end_i; x++) {
                for (int32_t y = start_j; y < end_j; y++) {
                    normalize_pixel(queue->output_image, x * queue->width + y, queue->min, queue->max);
                }
            }

            complete_task(queue, 1, 0);
        }
    }

}

void apply_filter2d_work_pool(const filter *f,
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int32_t num_threads, int32_t work_chunk) {
    // Set up the queue
    work_queue_t queue = {
        f, 
        original, 
        target, 
        width, 
        height,
        work_chunk,
        INT32_MAX,
        INT32_MIN,
        0,
        0,
        PTHREAD_MUTEX_INITIALIZER,
        PTHREAD_COND_INITIALIZER};

    pthread_t threads[num_threads];
    for (int32_t i = 0; i < num_threads; i++) {
        int result = pthread_create(threads + i, NULL, queue_work, (void *)&queue);
        if (result != 0) {
            fprintf(stderr, "Work pool failed to create worker threads.\n");
            exit(1);
        }
    }

    for (int32_t i = 0; i < num_threads; i++) {
        int result = pthread_join(threads[i], NULL);
        if (result != 0) {
            fprintf(stderr, "Work pool failed to join worker threads.\n");
            exit(1);
        }
    }

    int result = pthread_mutex_destroy(&(queue.mutex));
    if (result != 0) {
        fprintf(stderr, "Work pool failed to destroy the mutex.\n");
        exit(1);
    }
    result = pthread_cond_destroy(&(queue.cond));
    if (result != 0) {
        fprintf(stderr, "Work pool failed to destroy the conditional variable.\n");
        exit(1);
    }
}

/* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */
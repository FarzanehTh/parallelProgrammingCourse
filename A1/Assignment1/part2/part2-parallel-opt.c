#include <pthread.h>


/* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */

// Compute the historic average grade for a give course with stride. Updates value in the record
void *compute_average_stride(void *arguments) {
	/* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */
	int result __attribute__((unused));
	result = pthread_mutex_lock(args->mutex_ptr);
	assert(!result);
	args->course->average += sum;
	result = pthread_mutex_unlock(args->mutex_ptr);
	return NULL;
}

// Compute the historic average grade for a given course. Updates the average value in the record
void compute_average(course_record *course)
{
	assert(course != NULL);
	assert(course->grades != NULL);

	pthread_t threads[NUM_THREADS];
	pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
	struct argument arguments[NUM_THREADS];

	int result __attribute__((unused));

	course->average = 0.0;
	for (int i = 0; i < NUM_THREADS; i++) {
		arguments[i].course = course;
		arguments[i].mutex_ptr = &mutex;
		arguments[i].index = i;
		result = pthread_create(&threads[i], NULL, compute_average_stride, (void *)&arguments[i]);
		assert(!result);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		result = pthread_join(threads[i], NULL);
		assert(!result);
	}

	course->average /= course->grades_count;

	result = pthread_mutex_destroy(&mutex);
	assert(!result);
}

// Compute the historic average grades for all the courses
void compute_averages(course_record *courses, int courses_count)
{
    /* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */

    for (int i = 0; i < courses_count; i++) {
        compute_average(&(courses[i]));
	}
}


int main(int argc, char *argv[])
{
    /* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    compute_averages(courses, courses_count);
    clock_gettime(CLOCK_MONOTONIC, &end);

    /* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */
    return 0;
}

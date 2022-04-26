
#include <pthread.h>

// Compute the historic average grade for a given course. Updates the average value in the record
void *compute_average(void *course_ptr)
{
	course_record *course = course_ptr;
	assert(course != NULL);
	assert(course->grades != NULL);
    /* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */
}

// Compute the historic average grades for all the courses
void compute_averages(course_record *courses, int courses_count)
{
	assert(courses != NULL);

	pthread_t threads[courses_count] __attribute__((unused));
	int result __attribute__((unused));
	
	for (int i = 0; i < courses_count; i++) {
		result = pthread_create(&threads[i], NULL, compute_average, (void *)(&courses[i]));
		assert(!result);
	}

	for (int i = 0; i < courses_count; i++) {
		result = pthread_join(threads[i], NULL);
		assert(!result);
	}
}


int main(int argc, char *argv[])
{
    /* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */
    clock_gettime(CLOCK_MONOTONIC, &start);
    compute_averages(courses, courses_count);
    clock_gettime(CLOCK_MONOTONIC, &end);

   /* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */
}

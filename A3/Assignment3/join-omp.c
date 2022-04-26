#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "join.h"
#include "options.h"

/* Compute the ceiling of chunks that nthreads can divide len into. */
int get_chunk_size(int len, int nthreads){
	return (len + nthreads - 1) / nthreads;
}

/** symmertic partitioning based on two hash tables  **/
int symmertic_partitioning_hash(student_record *students, int students_count, ta_record *tas,
								int tas_count, join_func_hash_t *join_f_hash, int opt_nthreads)
{

	hash_table_t *student_hash_table = create_hash_table_students(students, students_count, students_count);
	hash_table_t *tas_hash_table = create_hash_table_tas(tas, tas_count, students_count); // same size as student hash table
	int chunk_size = get_chunk_size(student_hash_table->size, opt_nthreads);
	int count = 0;
	#pragma omp parallel for reduction(+ : count)
	for (int i = 0; i < opt_nthreads; i++){
		int start = i * chunk_size;
		int end = (i + 1) * chunk_size;
		if (end > student_hash_table->size)
		{
			end = student_hash_table->size;
		}
		count += join_f_hash(&(student_hash_table->buckets[start]),
							&(tas_hash_table->buckets[start]),
							end - start);
	}

	hash_destroy(student_hash_table);
	hash_destroy(tas_hash_table);
	return count;
}

/*********************** Fragment and replicate functions *********************/

int fragment_and_replicate_hash_coarse_grain(student_record *students, int students_count,
											ta_record *tas, int tas_count, int opt_nthreads)
{
	int count = 0;
	int chunk_size = (students_count > tas_count)
	 				? get_chunk_size(students_count, opt_nthreads) 
					: get_chunk_size(tas_count, opt_nthreads);

	// partition the larger array and hash the smaller one and let it be shared
	if (students_count > tas_count){
		hash_table_t *tas_hash_table = create_hash_table_tas(tas, tas_count, tas_count);

		#pragma omp parallel for firstprivate(tas) reduction(+ : count)
		for (int i = 0; i < opt_nthreads; i++){
			int start = i * chunk_size;
			int end = (i + 1) * chunk_size;
			if (end > students_count){
				end = students_count;
			}
			count += join_hashed_ta(tas_hash_table, &(students[start]), end - start);
		}
	}else{
		hash_table_t *student_hash_table = 
							create_hash_table_students(students, students_count, students_count);
		
		#pragma omp parallel for firstprivate(students) reduction(+: count)
		for (int i = 0; i < opt_nthreads; i++){
			int start = i * chunk_size;
			int end = (i + 1) * chunk_size;
			if (end > tas_count){
				end = tas_count;
			}
			count += join_hashed_student(student_hash_table, &(tas[start]), end - start);
		}
	}
	return count;
}


int fragment_and_replicate_coarse_grain(student_record *students, int students_count, ta_record *tas,
									 	int tas_count, join_func_t *join_f, int opt_nthreads)
{

	int count = 0;
	int chunk_size = (students_count > tas_count) 
					 ? get_chunk_size(students_count, opt_nthreads)
					 : get_chunk_size(tas_count, opt_nthreads);

	// partition the larger array
	if (students_count > tas_count){
		#pragma omp parallel for firstprivate(tas) reduction(+ : count)
		for (int i = 0; i < opt_nthreads; i++){
			int start = i * chunk_size;
			int end = (i + 1) * chunk_size;
			if (end > students_count){
				end = students_count;
			}
			count += join_f(&(students[start]), end - start, tas, tas_count);
		}
	}else{
		#pragma omp parallel for firstprivate(students) reduction(+: count)
		for (int i = 0; i < opt_nthreads; i++){
			int start = i * chunk_size;
			int end = (i + 1) * chunk_size;
			if (end > tas_count){
				end = tas_count;
			}
			count += join_f(students, students_count, &(tas[start]), end - start);
		}
	}
	return count;
}


/*  If the smaller array that is to be replicated does not fit into memory, this version of 
fragment_and_replicate will be used as it does finer grain partitions.*/
int fragment_and_replicate_fine_grain(student_record *students, int students_count, ta_record *tas,
									 int tas_count, join_func_t *join_f, int opt_nthreads)
{
	int count = 0;
	// partition two arrays such that n * m == num of threads available
	int m = opt_nthreads / 2;
	int n = opt_nthreads / m; // we want m > n

	if(students_count > tas_count){
		// m is number of partitions for student array and n is for the ta array
		// that is because we want to partition bigger array more
		int chunk_size_m = get_chunk_size(students_count , m);
		int chunk_size_n = get_chunk_size(tas_count , n);

		#pragma omp parallel for reduction(+ : count) collapse(2)
		for(int i = 0; i < m; i++){
			for(int t = 0; t < n; t++){
				int start_i = i * chunk_size_m;
				int end_i = (i + 1) * chunk_size_m;
				if(end_i > students_count){
					end_i = students_count;
				}
				int start_t = t * chunk_size_n;
				int end_t = (t + 1) * chunk_size_n;
				if(end_t > tas_count){
					end_t = tas_count;
				}
				count += join_f(&(students[start_i]), end_i - start_i, &(tas[start_t]), end_t - start_t);
			}
		}
	}else{
		// m is number of partitions for ta array and n is for the student array
		// that is because we want to partition bigger array more
		int chunk_size_m = get_chunk_size(tas_count, m);
		int chunk_size_n = get_chunk_size(students_count , n);

		#pragma omp parallel for reduction(+ : count) collapse(2)
		for(int i = 0; i < m; i++){
			for(int t = 0; t < n; t++){
				int start_i = i * chunk_size_m;
				int end_i = (i + 1) * chunk_size_m;
				if(end_i > tas_count){
					end_i = tas_count;
				}
				int start_t = t * chunk_size_n;
				int end_t = (t + 1) * chunk_size_n;
				if(end_t > students_count){
					end_t = students_count;
				}
				count += join_f(&(students[start_t]), end_t - start_t, &(tas[start_i]), end_i - start_i);
			}
		}
	}

	return count;
}


int fragment_and_replicate(student_record *students, int students_count, ta_record *tas, int tas_count,
							join_func_t *join_f, int opt_nthreads)
{
	// this is the default version of Fragment and Replicate is coarse grain, if you need fine grain,
	// uncomment the next one
	return fragment_and_replicate_coarse_grain(students, students_count, tas, tas_count,
												join_f, opt_nthreads);

	/* return fragment_and_replicate_fine_grain(students, students_count, tas, tas_count,
												 join_f, opt_nthreads); */
}


int fragment_and_replicate_hash(student_record *students, int students_count, ta_record *tas,
								int tas_count, int opt_nthreads)
{
	return fragment_and_replicate_hash_coarse_grain(students, students_count, tas, tas_count, opt_nthreads);
}

/*********************** Symmertic Partitioning functions *********************/

void build_partition(student_record *students, int students_count, ta_record *tas, int tas_count, 
					int nthreads, int *student_partition, int *ta_partition)
{
	int student_chunk_size = students_count / nthreads;
	for (int i = 0; i < nthreads; i++) {
		student_partition[i] = i * student_chunk_size;
	}
	student_partition[nthreads] = students_count;

	int ta_index = 0;
	ta_partition[0] = 0;
	for (int i = 1; i < nthreads; i++) {
		int sid = students[student_partition[i] - 1].sid;
		while (ta_index < tas_count && tas[ta_index].sid <= sid) {
			ta_index++;
		}
		ta_partition[i] = ta_index;
	}
	ta_partition[nthreads] = tas_count;
}

int symmertic_partitioning(student_record *students, ta_record *tas, int nthreads, join_func_t *join_f,
							int *student_partition, int *ta_partition) 
{
	int count = 0;
	#pragma omp parallel for reduction(+: count)
	for (int i = 0; i < nthreads; i++) {
		count += join_f(students + student_partition[i], student_partition[i + 1] - student_partition[i], tas + ta_partition[i], ta_partition[i + 1] - ta_partition[i]);
	}
	return count;
}

/*********************** Main *********************/

int main(int argc, char *argv[]) { 
    /* As this is a public repo, I am required to remove some parts. The complete implementation can be viewed on my private repo. */ 
}

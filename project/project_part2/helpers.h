#include <vector>
#include "common.h"

#define cutoff 0.01

using cell = std::vector<particle_t *>;
using cell_mpi = std::vector<particle_t>;
struct halo{
    int top;
    int bottom;
    int left;
    int right;
};

void apply_force_partical(particle_t *particle, cell **grids,
                          int src_row, int src_col,
                          int width, int height,
                          double *dmin, double *davg, int *navg);


void apply_force_grid(cell **grids, int row, int col, int width, int height,
                      double *dmin, double *davg, int *navg);


inline int get_grid_index(double x, double y, int width, int height)
{
    int row = (int)(x / cutoff);
    int col = (int)(y / cutoff);
    if (row < 0)
        row = 0;
    if (col < 0)
        col = 0;
    if (row >= height)
        row = height - 1;
    if (col >= width)
        col = width - 1;
    return row * width + col;
}


inline void get_global_dim(double x, double y, int global_width, int global_height, int &row, int &col)
{
    row = (int)(x / cutoff);
    col = (int)(y / cutoff);
    if (row < 0)
        row = 0;
    if (col < 0)
        col = 0;
    if (row >= global_height)
        row = global_height - 1;
    if (col >= global_width)
        col = global_width - 1;
}


// get process index based on global row and col
inline int get_proc_index(int row, int col, int num_of_chunks_rows,
                          int num_of_chunks_cols, int chunk_size_rows, int chunk_size_cols, int n_proc)
{
    int r = row / chunk_size_rows;
    int c = col / chunk_size_cols;
    int proc = r * num_of_chunks_cols + c;
    return proc;
}

inline void get_adjusted_local_dim(int global_row, int global_col, int chunk_size_rows,
                                   int chunk_size_cols, int num_of_chunks_rows, int &local_row,int &local_col,
                                   int num_of_chunks_cols, int n_proc, int rank,
                                   int local_width, int local_height, struct halo *halo)
{
    local_row = global_row % chunk_size_rows;
    local_col = global_col % chunk_size_cols;
    int main_proc_i = get_proc_index(global_row, global_col, num_of_chunks_rows, num_of_chunks_cols, chunk_size_rows,
                                     chunk_size_cols, n_proc);
    if (main_proc_i != rank)
    {
        // this is a halo element, findout which neighbour this belongs to
        if (main_proc_i == rank - 1)
        {
            local_row++;
            local_col = 0;
        }
        else if (main_proc_i == rank + 1)
        {
            local_row++;
            local_col = local_width - 1;
        }
        else if (main_proc_i == rank + num_of_chunks_cols)
        {
            local_row = local_height - 1;
            local_col++;
        }
        else if (main_proc_i == rank - num_of_chunks_cols)
        {
            local_row = 0;
            local_col++;
        }
    }
    else
    {
        // not a halo particle but adjust its size within the grid
        if (halo->top)
        {
            local_row++;
        }
        if (halo->left)
        {
            local_col++;
        }
    }
    if (local_row >= local_height)
    {
        local_row = local_height - 1;
    }
    if(local_col >= local_width){
        local_col = local_width - 1;
    }
}


// get local index based on global index
inline int get_local_grid_index(double x, double y, int global_width, int global_height,
                                int chunk_size_cols, int chunk_size_rows, int local_width,
                                int local_height, int num_of_chunks_rows, int num_of_chunks_cols,
                                int rank, int n_proc, struct halo* halo)
{
    int global_row;
    int global_col;
    get_global_dim(x, y, global_width, global_height, global_row, global_col);

    int local_row;
    int local_col;

    get_adjusted_local_dim(global_row, global_col, chunk_size_rows,
                           chunk_size_cols, num_of_chunks_rows, local_row, local_col,
                            num_of_chunks_cols, n_proc, rank, local_width, local_height, halo);

    return local_row * local_width + local_col;
}


cell **build_grid(particle_t *particles, int num_particles, double size,
                  int &width, int &height);

void move_particles_grid_serial(cell **grids, int width, int height);

void move_particles_grid_omp_chunk(cell **grids, int width, int height, int rows, int cols);

cell_mpi **build_partitions_vectors_MPI(particle_t *particles, int n, double global_grid_size,
                                        int global_width, int global_height, int n_proc, int rank, int num_of_chunks_rows,
                                        int num_of_chunks_cols, int chunk_size_rows, int chunk_size_cols);
void apply_force_grid_MPI(cell_mpi **grids, int row, int col, int width, int height,
                          double *dmin, double *davg, int *navg);
void apply_force_partical_MPI(particle_t *particle, cell_mpi **grids, int src_row, int src_col,
                              int width, int height, double *dmin, double *davg, int *navg);

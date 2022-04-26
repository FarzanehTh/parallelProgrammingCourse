#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "helpers.h"

#define density 0.0005
#define cutoff 0.01

void apply_force_partical(particle_t *particle, cell **grids,
                          int src_row, int src_col,
                          int width, int height,
                          double *dmin, double *davg, int *navg)
{
    cell *src = grids[src_row * width + src_col];
    for (auto s : *src)
    {
        apply_force(*particle, *s, dmin, davg, navg);
    }
}


void apply_force_grid(cell **grids, int row, int col, int width, int height,
                      double *dmin, double *davg, int *navg)
{
    for (auto p : *(grids[row * width + col]))
    {
        for (int src_row = row - 1; src_row < row + 2; ++src_row)
        {
            for (int src_col = col - 1; src_col < col + 2; ++src_col)
            {
                if (src_row >= 0 && src_row < height && src_col >= 0 && src_col < width)
                {
                    apply_force_partical(p,
                                         grids, src_row, src_col,
                                         width, height,
                                         dmin, davg, navg);
                }
            }
        }
    }
}


cell **build_grid(particle_t *particles, int num_particles, double size,
                  int &width, int &height)
{
    int grid_num = (int)(size / cutoff);
    width = grid_num;
    height = grid_num;
    cell **grids = (cell **)malloc(sizeof(cell *) * width * height);
    for (size_t i = 0; i < width * height; ++i)
    {
        grids[i] = new cell();
    }
    for (size_t i = 0; i < num_particles; ++i)
    {
        int index = get_grid_index(particles[i].x, particles[i].y, width, height);
        grids[index]->emplace_back(particles + i);
    }
    return grids;
}


void move_particles_grid_serial(cell **grids, int width, int height)
{
    for (int i = 0; i < width * height; ++i)
    {
        cell *grid = grids[i];
        for (auto it = grid->begin(); it != grid->end();)
        {   
            int index = get_grid_index((*it)->x, (*it)->y, width, height);
            if (index == i){
                ++it;
            }
            else{
                grids[index]->emplace_back(*it);
                grid->erase(it);
            }
        }
    }
}

cell_mpi **build_partitions_vectors_MPI(particle_t *particles, int n, double global_grid_size,
                                        int global_width, int global_height, int n_proc, int rank, int num_of_chunks_rows,
                                        int num_of_chunks_cols, int chunk_size_rows, int chunk_size_cols)
{

    cell_mpi **partitions_vectors = (cell_mpi **)malloc(sizeof(cell_mpi *) * n_proc);
    for (size_t i = 0; i < n_proc; i++)
    {
        partitions_vectors[i] = new cell_mpi();
    }

    // find prticles associated with every process's partition
    for (int p = 0; p < n; p++){
        int row;
        int col;
        get_global_dim(particles[p].x, particles[p].y, global_width, global_height, row, col);

        int main_proc_i = get_proc_index(row, col, num_of_chunks_rows, num_of_chunks_cols, chunk_size_rows, chunk_size_cols, n_proc);
        partitions_vectors[main_proc_i]->emplace_back(particles[p]);

        // also put this particle into all 4 processes that are neighbours, since this particle might be the halo of them
        int ranges[4][2] = {{-1, 0}, {+1, 0}, {0, -1}, {0, +1}};
        for(int i = 0; i < 4; i++){
            int d_r = ranges[i][0];
            int d_c = ranges[i][1];
            if (row + d_r >= 0 && col + d_c >= 0 && row + d_r < global_height && col + d_c < global_width)
            {
                int neighbour_proc_i = get_proc_index(row + d_r, col + d_c, num_of_chunks_rows, num_of_chunks_cols,
                                                      chunk_size_rows, chunk_size_cols, n_proc);
                if (neighbour_proc_i != main_proc_i) // only put it into a distinct neighbour process
                { 
                    partitions_vectors[neighbour_proc_i]->emplace_back(particles[p]);
                }
            }
        }
    }

    return partitions_vectors;
}


void apply_force_partical_MPI(particle_t &particle, cell_mpi **grids,
                              int src_row, int src_col,
                              int width, int height,
                              double *dmin, double *davg, int *navg)
{
    cell_mpi *src = grids[src_row * width + src_col];
    for (particle_t s : *src){
        apply_force(particle, s, dmin, davg, navg);
    }
}


void apply_force_grid_MPI(cell_mpi **grids, int row, int col, int width, int height,
                      double *dmin, double *davg, int *navg)
{
    cell_mpi* &cell = grids[row * width + col];
    for (auto it = cell->begin(); it != cell->end(); it++){
        for (int src_row = row - 1; src_row < row + 2; ++src_row)
        {
            for (int src_col = col - 1; src_col < col + 2; ++src_col)
            {
                if (src_row >= 0 && src_row < height && src_col >= 0 && src_col < width)
                {
                    apply_force_partical_MPI(*it,
                                             grids, src_row, src_col,
                                             width, height,
                                             dmin, davg, navg);
                }
            }
        }
    }
}
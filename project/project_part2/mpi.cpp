/* ------------
 * The code is adapted from the XSEDE online course Applications of Parallel Computing. 
 * The copyright belongs to all the XSEDE and the University of California Berkeley staff
 * that moderate this online course as well as the University of Toronto CSC367 staff.
 * This code is provided solely for the use of students taking the CSC367 course at 
 * the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * -------------
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <vector>
#include <math.h>
#include "common.h"

#define density 0.0005
#define cutoff 0.01

using cell = std::vector<particle_t>;

void get_global_pos(double x, double y, int num_grid_global, int &row, int &col)
{
    row = x / cutoff;
    col = y / cutoff;
    if (row >= num_grid_global)
        row = num_grid_global - 1;
    if (col >= num_grid_global)
        col = num_grid_global - 1; 
}

int get_owner_rank(double x, double y, int num_grid_global,
    int num_grid_local_width, int num_grid_local_height,
    int num_block_width, int num_block_height)
{
    int row = x / cutoff;
    int col = y / cutoff;
    if (row >= num_grid_global)
        row = num_grid_global - 1;
    if (col >= num_grid_global)
        col = num_grid_global - 1;
    
    int block_row = row / num_grid_local_height;
    int block_col = col / num_grid_local_width;
    if (block_row >= num_block_height)
        block_row = num_block_height - 1;
    if (block_col >= num_block_width)
        block_col = num_block_width - 1;
    
    return block_row * num_block_width + block_col;
}

bool is_halo(double x, double y, int num_grid_global,
    int num_grid_local_width, int num_grid_local_height,
    int num_block_width, int num_block_height, int rank) 
{
    int row = x / cutoff;
    int col = y / cutoff;
    if (row >= num_grid_global)
        row = num_grid_global - 1;
    if (col >= num_grid_global)
        col = num_grid_global - 1;
    
    int block_row = rank / num_block_width;
    int block_col = rank % num_block_width;
    int start_row = block_row * num_grid_local_height;
    int start_col = block_col * num_grid_local_width;
    int end_row = (block_row + 1) * num_grid_local_height;
    if (end_row > num_grid_global)
        end_row = num_grid_global;
    int end_col = (block_col + 1) * num_grid_local_width;
    if (end_col > num_grid_global)
        end_col = num_grid_global;

    bool row_hit = row == start_row - 1 || row == end_row;
    bool row_match = row >= start_row - 1 && row <= end_row;
    bool col_hit = col == start_col - 1 || col == end_col;
    bool col_match = col >= start_col - 1 && col <= end_col;
    
    return (row_hit && col_match) || (row_match && col_hit);
}

void apply_force_particle(particle_t &particle, cell **grid,
    int src_row, int src_col,
    int width, int height,
    double *dmin, double *davg, int *navg)
{
    cell *src = grid[src_row * width + src_col];
    for (auto &s: *src) {
        apply_force(particle, s, dmin, davg, navg);
    }
}

void apply_force_grid(cell **grid, int row, int col, int width, int height,
    double *dmin, double *davg, int *navg)
{
    for (auto &p: *(grid[row * width + col])) {
        p.ax = 0;
        p.ay = 0;
        for (int src_row = row - 1; src_row < row + 2; ++src_row) {
            for (int src_col = col - 1; src_col < col + 2; ++src_col) {
                if (src_row >= 0 && src_row < height && src_col >= 0 && src_col < width) {
                    apply_force_particle(p,
                                         grid, src_row, src_col,
                                         width, height,
                                         dmin, davg, navg);
                }
            }
        }
    }
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  set up the data partitioning across processors
    //
    int size = sqrt(density * n);
    int num_grid_global = size / cutoff;
    int num_block_height = sqrt(n_proc);
    int num_block_width = n_proc / num_block_height;
    int num_grid_local_width = (num_grid_global + num_block_width - 1) / num_block_width;
    int num_grid_local_height = (num_grid_global + num_block_height - 1) / num_block_height;

    int block_row = rank / num_block_width;
    int block_col = rank % num_block_width;
    int start_row = block_row * num_grid_local_height;
    int start_col = block_col * num_grid_local_width;
    int end_row = (block_row + 1) * num_grid_local_height;
    if (end_row > num_grid_global)
        end_row = num_grid_global;
    int end_col = (block_col + 1) * num_grid_local_width;
    if (end_col > num_grid_global)
        end_col = num_grid_global;
    
    bool halo_left = block_col > 0;
    bool halo_right = block_col < num_block_width - 1;
    bool halo_top = block_row > 0;
    bool halo_bottom = block_row < num_block_height - 1;

    int start_row_view = halo_top ? start_row - 1 : start_row;
    int end_row_view = halo_bottom ? end_row + 1 : end_row;
    int start_col_view = halo_left ? start_col - 1 : start_col;
    int end_col_view = halo_right ? end_col + 1 : end_col;

    int local_start_row = start_row - start_row_view;
    int local_end_row = end_row - start_row_view;
    int local_start_col = start_col - start_col_view;
    int local_end_col = end_col - start_col_view;

    //
    //  allocate storage for local partition
    //
    int view_height = end_row_view - start_row_view;
    int view_width = end_col_view - start_col_view;
    
    cell **local_grid = (cell **)malloc(sizeof(cell *) * view_height * view_width);
    for (int i = 0; i < view_height * view_width; ++i) {
        local_grid[i] = new cell();
    }

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    cell **local_send_array = (cell **)malloc(sizeof(cell *) * n_proc);
    for (int i = 0; i < n_proc; ++i) {
        local_send_array[i] = new cell();
    }
    if( rank == 0 ) {
        init_particles( n, particles );

        for (int i = 0; i < n; ++i) {
            int proc = get_owner_rank(particles[i].x, particles[i].y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height);
            local_send_array[proc]->emplace_back(particles[i]);
            // If is halo of neighbot processors, also send them
            int neighbor = proc - num_block_width;
            if (neighbor >= 0) {
                if (is_halo(particles[i].x, particles[i].y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                    local_send_array[neighbor]->emplace_back(particles[i]);
                }
            }
            neighbor = proc - num_block_width - 1;
            if (neighbor >= 0) {
                if (is_halo(particles[i].x, particles[i].y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                    local_send_array[neighbor]->emplace_back(particles[i]);
                }
            }
            neighbor = proc - num_block_width + 1;
            if (neighbor >= 0) {
                if (is_halo(particles[i].x, particles[i].y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                    local_send_array[neighbor]->emplace_back(particles[i]);
                }
            }
            neighbor = proc - 1;
            if (neighbor >= 0) {
                if (is_halo(particles[i].x, particles[i].y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                    local_send_array[neighbor]->emplace_back(particles[i]);
                }
            }
            neighbor = proc + 1;
            if (neighbor < n_proc) {
                if (is_halo(particles[i].x, particles[i].y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                    local_send_array[neighbor]->emplace_back(particles[i]);
                }
            }
            neighbor = proc + num_block_width - 1;
            if (neighbor < n_proc) {
                if (is_halo(particles[i].x, particles[i].y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                    local_send_array[neighbor]->emplace_back(particles[i]);
                }
            }
            neighbor = proc + num_block_width;
            if (neighbor < n_proc) {
                if (is_halo(particles[i].x, particles[i].y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                    local_send_array[neighbor]->emplace_back(particles[i]);
                }
            }
            neighbor = proc + num_block_width + 1;
            if (neighbor < n_proc) {
                if (is_halo(particles[i].x, particles[i].y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                    local_send_array[neighbor]->emplace_back(particles[i]);
                }
            }
        }
    }

    int send_particles_count[n_proc];
    int send_particles_disp[n_proc];
    int recv_particles_count[n_proc]; // For later use
    int recv_particles_disp[n_proc]; // For later use
    int total_send_count = 0;
    if (rank == 0) {
        for (int i = 0; i < n_proc; ++i) {
            send_particles_disp[i] = total_send_count;
            total_send_count += local_send_array[i]->size();
            send_particles_count[i] = local_send_array[i]->size();
        }
    }
    int recv_count;
    MPI_Scatter(send_particles_count, 1, MPI_INT, &recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    particle_t *local_array = (particle_t *)malloc(sizeof(particle_t) * recv_count);
    particle_t *global_send_array;
    if (rank == 0) {
        global_send_array = (particle_t *)malloc(sizeof(particle_t) * total_send_count);
        for (int i = 0; i < n_proc; ++i) {
            memcpy(global_send_array + send_particles_disp[i], local_send_array[i]->data(), sizeof(particle_t) * send_particles_count[i]);
        }
    }
    MPI_Scatterv(global_send_array, send_particles_count, send_particles_disp, PARTICLE, local_array, recv_count, PARTICLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
        free(global_send_array);

    //
    //  build local grids using the particles received
    //
    for (int i = 0; i < recv_count; ++i) {
        int row;
        int col;
        get_global_pos(local_array[i].x, local_array[i].y, num_grid_global, row, col);
        row -= start_row_view;
        col -= start_col_view;
        int local_index = row * view_width + col;
        local_grid[local_index]->emplace_back(local_array[i]);
    }
    free(local_array);

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
        //
        //  compute all forces
        //
        for (int row = local_start_row; row < local_end_row; ++row) {
            for (int col = local_start_col; col < local_end_col; ++col) {
                apply_force_grid(local_grid, row, col, view_width, view_height, &dmin, &davg, &navg);
            }
        }
     
        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        //
        //  move particles
        //
        for (int row = local_start_row; row < local_end_row; ++row) {
            for (int col = local_start_col; col < local_end_col; ++col) {
                for (auto &p: *(local_grid[row * view_width + col])) {
                    move(p);
                }
            }
        }

        // Clear halo zone
        if (halo_top) {
            for (int col = 0; col < view_width; ++col) {
                local_grid[col]->clear();
            }
        }
        if (halo_bottom) {
            for (int col = 0; col < view_width; ++col) {
                local_grid[(view_height - 1) * view_width + col]->clear();
            }
        }
        if (halo_left) {
            for (int row = 0; row < view_height; ++row) {
                local_grid[row * view_width]->clear();
            }
        }
        if (halo_right) {
            for (int row = 0; row < view_height; ++row) {
                local_grid[row * view_width + view_width - 1]->clear();
            }
        }

        // Move particles locally and to other processors
        for (int i = 0; i < n_proc; ++i) {
            local_send_array[i]->clear();
        }
        for (int row = local_start_row; row < local_end_row; ++row) {
            for (int col = local_start_col; col < local_end_col; ++col) {
                cell *vec = local_grid[row * view_width + col];
                for (auto p = vec->begin(); p != vec->end();) {
                    int new_rank = get_owner_rank(p->x, p->y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height);
                    // Send to neighbor if the particle is in halo
                    int neighbor = new_rank - num_block_width;
                    if (neighbor >= 0) {
                        if (is_halo(p->x, p->y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                            local_send_array[neighbor]->emplace_back(*p);
                        }
                    }
                    neighbor = new_rank - num_block_width - 1;
                    if (neighbor >= 0) {
                        if (is_halo(p->x, p->y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                            local_send_array[neighbor]->emplace_back(*p);
                        }
                    }
                    neighbor = new_rank - num_block_width + 1;
                    if (neighbor >= 0) {
                        if (is_halo(p->x, p->y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                            local_send_array[neighbor]->emplace_back(*p);
                        }
                    }
                    neighbor = new_rank - 1;
                    if (neighbor >= 0) {
                        if (is_halo(p->x, p->y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                            local_send_array[neighbor]->emplace_back(*p);
                        }
                    }
                    neighbor = new_rank + 1;
                    if (neighbor < n_proc) {
                        if (is_halo(p->x, p->y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                            local_send_array[neighbor]->emplace_back(*p);
                        }
                    }
                    neighbor = new_rank + num_block_width - 1;
                    if (neighbor < n_proc) {
                        if (is_halo(p->x, p->y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                            local_send_array[neighbor]->emplace_back(*p);
                        }
                    }
                    neighbor = new_rank + num_block_width;
                    if (neighbor < n_proc) {
                        if (is_halo(p->x, p->y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                            local_send_array[neighbor]->emplace_back(*p);
                        }
                    }
                    neighbor = new_rank + num_block_width + 1;
                    if (neighbor < n_proc) {
                        if (is_halo(p->x, p->y, num_grid_global, num_grid_local_width, num_grid_local_height, num_block_width, num_block_height, neighbor)) {
                            local_send_array[neighbor]->emplace_back(*p);
                        }
                    }
                    // Send to new owner, of move into new grid if remains owned
                    if (new_rank == rank) {
                        int new_row;
                        int new_col;
                        get_global_pos(p->x, p->y, num_grid_global,new_row, new_col);
                        new_row -= start_row_view;
                        new_col -= start_col_view;
                        if (new_row != row || new_col != col) {
                            local_grid[new_row * view_width + new_col]->emplace_back(*p);
                            p = vec->erase(p);
                        } else {
                            ++p;
                        }
                    } else {
                        local_send_array[new_rank]->emplace_back(*p);
                        p = vec->erase(p);
                    }
                }
            }
        }

        //
        // send particles
        //
        // first send number of particles
        total_send_count = 0;
        for (int i = 0; i < n_proc; ++i) {
            send_particles_disp[i] = total_send_count;
            total_send_count += local_send_array[i]->size();
            send_particles_count[i] = local_send_array[i]->size();
        }
        MPI_Alltoall(send_particles_count, 1, MPI_INT, recv_particles_count, 1, MPI_INT, MPI_COMM_WORLD);
        // compute displacement
        int total_recv_count = 0;
        for (int i = 0; i < n_proc; ++i) {
            recv_particles_disp[i] = total_recv_count;
            total_recv_count += recv_particles_count[i];
        }
        // build send and recv buffer
        particle_t *send_buff = (particle_t *)malloc(sizeof(particle_t) * total_send_count);
        particle_t *recv_buff = (particle_t *)malloc(sizeof(particle_t) * total_recv_count);
        for (int i = 0; i < n_proc; ++i) {
            memcpy(send_buff + send_particles_disp[i], local_send_array[i]->data(), sizeof(particle_t) * send_particles_count[i]);
        }
        MPI_Alltoallv(send_buff, send_particles_count, send_particles_disp, PARTICLE, recv_buff, recv_particles_count, recv_particles_disp, PARTICLE, MPI_COMM_WORLD);
        free(send_buff);

        // Move received particles into its grid
        for (int i = 0; i < total_recv_count; ++i) {
            particle_t p = recv_buff[i];
            int row;
            int col;
            get_global_pos(p.x, p.y, num_grid_global, row, col);
            row -= start_row_view;
            col -= start_col_view;
            int local_index = row * view_width + col;
            local_grid[local_index]->emplace_back(p);
        }
        free(recv_buff);
    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -the minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}

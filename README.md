## Parallel Programming

**Note**: These are the assignments that my teammate and I completed in this course. That means the skeleton of the project was given to us by the instructor and we worked on the parallelization tasks. The description of each assignment’s tasks which needed to be done are present as pdf in this repo. In all of the assignments for this course we used the Scinet supercomputer nodes to run and optimize our code.

As the starter codes of the assignments of this course are the instructor's property, I have removed most of the implementations from the files. That means this public repo is only an extremely *partial* view of the implementations. Please request to view the complete impelmentation on my private repo.

### **`Assignment 1`**

This assignment consists of 2 parts:

**part 1**

**Measuring memory bandwidth**: In this part we measured the bandwidth and latency of CPU caches and memory. We implemented our own custom measurement functions so that we can issue write operations on the memory/caches so that we can predicate and measure these quantities.

<p align="center">
    <kbd><img src="/A1/bandwidth.png" style="width:450px;height:300px;"></kbd>
</p>

**Measuring cache sizes and their latencies**: In this part we measured the cache sizes `(L1, L2, L3)` in the function. To measure the cache sizes, we define a large constant int number as:
`CONST_ACCESS = 64 × 1024 × 1024` and then we tried to access arrays of different sizes, exactly `CONST_ACCESS` number of times. The idea is that, for each array, if the whole array can be held by the cache at the level whose size we are measuring, the time of every single access to that array should be roughly constant, as shown in the results below.

<p align="center">
    <kbd><img src="/A1/cache_levels.png" style="width:450px;height:300px;"></kbd>
</p>

So we can conclude the sizes of the caches are as follows:
- L1​: 32 KiB
- L2​: 256 KiB
- L3​: 16 MiB

We also measured the latencies of cache levels as follows:

<p align="center">
    <kbd><img src="/A1/latencies.png" style="width:450px;height:300px;"></kbd>
</p>

**part 2**

We used these 3 tools to optimze our code:
- *gprof* to find the most time-intensive functions.
- *callgrind* tool of valgrind to find the call stack of the functions and the frequency of calls.
- *perf* tool to capture architectural performance counters(cache misses in particular).

So in our optimization, we focused on decreasing cache misses (and increasing spatial locality) by creating `Pthread` threads that work on one specific array at a time. This way we decreased the runtime of 5 executions of the program from `20.89` sec to `5.2` sec.

You can view more detailed descriptions in our report [*here*](/A1/Assignment1/report_A1.pdf).

---

### **`Assignment 2`**

In this assignment, we applied several sized Laplacian filters on pgm images which were stored as a 1-d array. We tried the Pthread parallelization performance of this application using these methods:

- SEQUENTIAL
- SHARDED_ROWS
- SHARDED_COLUMNS_COLUMN_MAJOR
- SHARDED_COLUMNS_ROW_MAJOR
- WORK_QUEUE

The performance results depended slightly also on the size of filter used but overall we concluded that using **SHARDED_ROWS** parallelization method is the most efficient one in terms of runtime.

<p align="center">
    <kbd><img src="/A2/sharded_rows.png" style="width:450px;height:300px;"></kbd>
</p>

You can view more detailed descriptions in our report [*here*](/A2/Assignment2/report_A2.pdf).

---

### **`Assignment 3`**

In this assignment, we used OpenMP, which is an application programming interface that can be used for parallel programming in shared-memory. Specifically, we tried to employ shared memory techniques to parallelize database join operations(equi-join operations to be specific). Given a equi-join query on two database tables, we first looked into possible serial implementations  of this operation which are:
- Nested-loop join
- Merge-sort join
- Hash join

And then we tried to parallelize it using two frameworks of data parallelization:

- Fragment-and-replicate
- Symmetric partitioning

We concluded that both of these data parallelization methods can be appropriate depending on the size of two tables and how similar sized they are. Also Since we had our data tables sorted, our results showed a faster performance using merge sort. That is because the hash join method has a great overhead of hash table creation and calls to malloc. However, if the arrays are not sorted, the time complexity of merge sort would predictably be as high as the hash table.

Here is one result from one of our specfic datasets:

<p align="center">
    <kbd><img src="/A3/join.png" style="width:450px;height:300px;"></kbd>
</p>

You can view more detailed descriptions in our report [*here*](/A3/Assignment3/report_A3.pdf).

---

### **`Assignment 4`**

In this assignment, we implemented  implementations to apply a discrete Laplacian filter kernel on the GPU, using CUDA. We tried to  process some images in parallel, and take advantage of the GPU's massive parallelism and memory bandwidth. We implemented 5 different kernels for this purpose and we compared their performance along with our best CPU implementation(sharded rows) in our report. Overall, our best kernel (kernel 5) has a speedup of around/more than 2 times faster than the best CPU implementation.

<p align="center">
    <kbd><img src="/A4/gpu_kernels.png" style="width:450px;height:300px;"></kbd>
</p>

<p align="center">
    <kbd><img src="/A4/gpu_runtime.png" style="width:450px;height:300px;"></kbd>
</p>

You can view more detailed descriptions in our report [*here*](/A4/Assignment4/report_A4.pdf).

---

### **`Project`**

In this project, we optimized the performance of a scientific particle simulation program using `MPI`. `MPI` (Message Passing Interface) is a communication protocol for parallel programming across a number of separate computers connected by a network (non-shared memory paradigm). Particle simulations are used in Mechanics, Astronomy, and biology. In our simulation, we have a certain number of particles and they repel one another, but only when closer than a cutoff distance. The goal is to have a fast simulation that can compute the new location of each particle at every sec of time. As the number of particles is large, the naive implementation will take `O(n^2)` every sec to compute the updated locations.

<p align="center">
    <kbd><img src="/project/simulation.png" style="width:450px;height:300px;"></kbd>
</p>

Our goal in this project was to implement:

**part 1** A faster serial implementation that runs in  O(n) using one single processor

**part 2** Given the serial implementation has the runtime of `O(n)`, the parallel implementation can run the whole simulation in `O(n / p)` (at every sec). (`p` is the number of cores of processors available). We implemented the parallel version using OpenMP and MPI in two separate attempts. 


We used the fact that particles within a cutoff distance will repel each other. Given the cutoff (and so cell side) value of 0.01, every 9 cells’ computation can be done by one core of processor, hence parallelization.

<p align="center">
    <kbd><img src="/project/grid_cells.png" style="width:450px;height:300px;"></kbd>
</p>

We can see the result of parallelization on runtime over multiple cores.

<p align="center">
    <kbd><img src="/project/strong_scaling_runtime.png" style="width:450px;height:300px;"></kbd>
</p>

You can view more detailed descriptions in our report [*here*](/project/project_part2/report_project.pdf).




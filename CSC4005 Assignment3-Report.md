# Parallel N-body Simulation 

## **How to compile & execute?**

### File Tree

​	The project is modified on the template provided on BB.

**NOTICE:**

​	The compilation settings for the MPI & Pthread & openMp & bonus implementation is not compatible for CUDA. For convenience, **the CUDA version of program is stored and should be compiled separately.**

​	The CUDA program is stored in the **cuda_program** directory. And the other programs are stored in the **Other_program** directory.

```shell
.
├── other_program                                   # Direct for MPI, Pthread, OpenMP, and BONUS program
│   ├── CMakeLists.txt
│   ├── build
│   ├── imgui
│   ├── include
│   │   ├── graphic
│   │   └── nbody
│   │       ├── body.hpp                            # header file body_xxx.hpp for each version of implementation
│   │       ├── body_BS.hpp
│   │       ├── body_CUDA.hpp
│   │       ├── body_MPI.hpp
│   │       ├── body_Pthread.hpp
│   │       └── body_openMP.hpp
│   └── src                                         # source code file main_xxx.cpp for each version of implementation
│       ├── graphic.cpp
│       ├── main.cu
│       ├── main_BS.cpp
│       ├── main_MPI.cpp
│       ├── main_Pthread.cpp
│       ├── main_openmp.cpp
│       └── main_seq.cpp 
├── cuda_program                                     # Direct for CUDA program
│   ├── CMakeLists.txt 
│   ├── build
│   │   ├── CMakeCache.txt
│   │   ├── CMakeFiles
│   │   ├── Makefile
│   │   ├── cmake_install.cmake
│   │   ├── csc4005_imgui
│   │   ├── libcore.a
│   │   └── testCUDA
│   ├── imgui
│   ├── include
│   │   ├── graphic
│   │   └── nbody
│   └── src
│       ├── graphic.cpp
│       └── main.cu                                     # source code for CUDA implementation
├── report.pdf
└── result_video.MP4
```



### Execution Steps

##### MPI, Pthread, OpenMP, Bonus

```shell
cd ./other_program         # to run MPI, PTHREAD, OPENMP, BONUS program
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j12

# then you will get some executable program

# testMPI: MPI implementation
# the program argument is:
testMPI <BODIES> <GRAPH 0/1>
# with GUI: iteration limit is 30000
# without GUI: iteration limit is 200
# example usage:
mpirun-gui testMPI 200 1
mpirun testMPI 200 0

# testP: Pthread implementation
# the program argument is:
testP <BODIES> <THREAD_SIZE> <GRAPH 0/1>
# with GUI: iteration limit is 30000
# without GUI: iteration limit is 200
# example usage:
testP 200 4 1 (4 threads with GUI)
testP 200 8 0 (8 threads without GUI)

# testOMP: openMP implementation
# the program argument is:
testOMP <BODIES> <THREAD_SIZE> <GRAPH 0/1>
# with GUI: iteration limit is 30000
# without GUI: iteration limit is 200
# example usage:
testOMP 200 4 1 (4 threads with GUI)
testOMP 200 8 0 (8 threads without GUI)


# testBS: MPI+OPENMP
# the program argument is:
testBS <BODIES> <THREAD_SIZE> <GRAPH 0/1>
# with GUI: iteration limit is 30000
# without GUI: iteration limit is 200
# example usage:
mpirun-gui -cpus-per-proc 8 testBS 200 1 (2 ranks, 8 threads with GUI)
```

##### CUDA

```shell
cd ./cuda_program
mkdir build && cd build
source scl_source enable devtoolset-10
CC=gcc CXX=g++ cmake ..
cmake --build . -j12

# testCUDA: CUDA implementation
srun ./testCUDA <BODIES> <GRAPH:0/1> <GRID_SIZE> <BLOCK_SIZE>
# with GUI: iteration limit is 30000
# without GUI: iteration limit is 200
# example usage:
srun ./testCUDA 200 0 2 100 (without GUI: the first 5 points position will be displayed)
srun ./testCUDA 200 1 1 200 (with GUI)
```





## 1. Introduction

​		In this project we implemented **five versions** of N-body simulation program, using **MPI, Pthread, CUDA, openMP, and MPI+openMP**, to simulate an astronomical N-body system.

​		In the $2nd$​​​​ part, we are going to talk about the **general idea** of the computation, and introduce some **details**, such as the **solution to data race**. For each version of implementation, we provide a flow chart, which is consistent to the general computation idea.

​		In the $3rd$​, we included some running results  video form to validate the function of the $5$​ implememtations.

​		In the $4th$​​​​​ part, we are going to **do some experiments** on the five versions of implementation. The **speed and speedup** recodes are used for performance analysis and comparison.

​		In the $5th$​ part, we make a conclusion for this assignment.





## 2. Method

### Computation Idea

****

​		There is a pool of points $0<=i<n$​​​​​, ($n$​​​​ is the BODIES), which have their current status records ${s_i}$​​​​​​, consisting of position (x, y), velocity (vx, vy), and acceleration (ax, ay), in some record containers $R_i$.

​		In **each iteration**, we compute the future status of all points $i$ as $s_i'$, based on the current status records $s_0$ to $s_{n-1}$, and store the results in a record buffer $R_i'$.  At first, we let all points have **zero acceleration**.

​		We first check the mutual influences among all points. For each point $i$​, we check all points $j$​, where $i!=j$​. If some of them are going to collide with $i$​, we update the position $x$​ and velocity $v$​ in the buffer slot $R_i'$​​ to let $i$​ bounce back after collision. Otherwise, we update the acceleration $a'$​ to $R_i$​.

​		After checking the mutual influence, we update the position and velocity of all points. We update the velocity since we have new acceleration $a'$ according to $v' = v + a' \delta t$​. At the same time, we modify $s_i'$ to handle wall collision (bounce back). And then we update the position based on the new velocity we got.

​		When all the computations end, we copy the value in buffer $R_i'$ to $R$​, which will let the GUI update. And the next iteration can start.





### How to avoid data race

****

​		The data race take place when the current record $R_i$ is modified ($x, y, vx, vy$​ is changed), while it is needed by other processes / threads in their computation. **To avoid data race, we keep our computation results in the separate record buffer $R_i'$​, and write back the results in $R_i'$ to $R$​​​ until all the processes /  threads end.**

​		Here are some implementations of this buffer in code.

#### CUDA

```cpp
// each iteration: launch a kernel

__global__ 
void update_for_all(
    double *x_buf,
    double *y_buf,
    double *vx_buf,
    double *vy_buf,
    double COLLISION_RATIO,
    int bodies,
    
    double * x,
    double * y,
    double * vx,
    double * vy,
    double * ax,
    double * ay,
    double * m,
    double elapse,
    double gravity,
    double position_range,
    double radius) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // zero accleration
    for (size_t i = index; i < bodies; i+=stride) {
        ax[i] = 0;
        ay[i] = 0;
    }
    // write the result back
    for (size_t i = index; i < bodies; i+=stride) {
        x[i] = x_buf[i];
        y[i] = y_buf[i];
        vx[i] = vx_buf[i];
        vy[i] = vy_buf[i];
    }

    for (size_t i = index; i < bodies; i+=stride)
    {
        #ifdef DEBUG
        printf("deal with %d\n", i);
        #endif
        for (size_t j = 0; j < bodies; ++j)
            mutual_check_and_update(x, y, vx, vy, ax, ay, m, x_buf, y_buf, vx_buf, vy_buf, i, j, radius, gravity, COLLISION_RATIO);
    }

    for (size_t i = index; i < bodies; i+=stride)
        update_single(x_buf, y_buf, ax, ay, vx_buf, vy_buf, i, elapse, position_range, radius, COLLISION_RATIO);
    
}


// each iteration: get back result from buffer for GUI
// collect position result for GUI display (HOST)
cudaMemcpy(pool.x.data(), cuda_x_buf, sizeof(double)*bodies, cudaMemcpyDeviceToHost);
cudaMemcpy(pool.y.data(), cuda_y_buf, sizeof(double)*bodies, cudaMemcpyDeviceToHost);
```

#### MPI, MPI+openMP

```cpp
//broadcast R
MPI_Bcast(pool.x.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(pool.y.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(pool.vx.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(pool.vy.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//gather result from R_Copy for GUI
MPI_Gatherv(pool.copy_x.data() + localoffset, localsize, MPI_DOUBLE, pool.x.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Gatherv(pool.copy_y.data() + localoffset, localsize, MPI_DOUBLE, pool.y.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Gatherv(pool.copy_vx.data() + localoffset, localsize, MPI_DOUBLE, pool.vx.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Gatherv(pool.copy_vy.data() + localoffset, localsize, MPI_DOUBLE, pool.vy.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

#### Pthread

```cpp
// initialize some threads to do the job
        for (i = 0; i < THREAD; ++i) 
        {
            thread_pool.push_back(std::thread(
                [&] { pool.update_for_tick(elapse, gravity, space, radius, startidx[i], localsize[i]); }));
        }

        // join all the threads
        for (i = 0; i < THREAD; ++i) 
            thread_pool[i].join();

        auto end = std::chrono::high_resolution_clock::now();
        duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();
        
        // clear the used threads and write back from buffer
        thread_pool.clear();
        pool.x = pool.copy_x;
        pool.y = pool.copy_y;
        pool.vx = pool.copy_vx;
        pool.vy = pool.copy_vy;
```

#### openMp

```cpp
		#pragma omp parallel for
        for (size_t i = 0; i < size(); ++i)
        {
            for (size_t j = 0; j < size(); ++j)
            {
                check_and_update(get_body(i), get_body(j), radius, gravity);
            }
        }
        #pragma omp parallel for
        for (size_t i = 0; i < size(); ++i)
        {
            get_body(i).update_for_tick(elapse, position_range, radius);
        }
		// write back from buffer to R until all threads finish computation          
        x = copy_x;
        y = copy_y;
        vx = copy_vx;
        vy = copy_vy;
```



### How to structure the workload

****

​		There is a pool of points, each of which needs some calculations. We view the pool as our total workload, and distribute parts of this pool to different workers to distribute workload in parallel. 



### How to distribute workload

****

#### CUDA

​		In CUDA implementation, we let each thread cross a stripe to fetch a slot to commutate.  Each CUDA thread can get the equal number of slots to deal with.

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // zero accleration
    for (size_t i = index; i < bodies; i+=stride) {
    	do some job
    }
```



#### Other program (MPI, Pthread, bonus)

​		In these $4$​ implementations, expect for the openmp implementation, we maintain a displs array and a scounts array, recording the responsible part for each worker (start index and length). The two arrays are computed in this way.

```cpp
int i, offset = 0;
    size_t *startidx, *localsize;
    startidx = (size_t *)malloc(THREAD * sizeof(size_t));
    localsize = (size_t *)malloc(THREAD * sizeof(size_t));
    for (i = 0; i < THREAD; ++i)
    {
        startidx[i] = offset;
        localsize[i] = std::ceil(((float)bodies - i) / THREAD);
        offset += localsize[i];
    }
```

​	

### Who should receive the input data and display GUI

****

​		The root process / thread receives all inputs and display the graph.



###  **Flow Chart**

****

$R$: status record for all points, including x, y, vx, vy, ax, ay

$R'$: status record buffer for all points, including {x, y, vx, vy}, to store middle results

#### MPI

![mpi](mpi.svg)

#### Pthread

![pthread](pthread.svg)



#### CUDA

![CUDA](CUDA.svg)



#### openMP

![openmp](openmp.svg)



#### Bonus

![bonus](bonus.svg)





## 3. Experiments and Analysis

#### Experiment 1: MPI

Here are some of the results of speed and speedup.

##### *The **Speed** with Different Data Size and Number of Cores (Points / Second)*

****

| thread\size | 100    | 200    | 500    | 1000   |
| ----------- | ------ | ------ | ------ | ------ |
| 1           | 585096 | 316542 | 129653 | 65264  |
| 2           | 696049 | 520728 | 239603 | 121889 |
| 4           | 689907 | 743351 | 456540 | 251096 |
| 8           | 702070 | 595272 | 728347 | 477080 |
| 16          | 787966 | 943325 | 173987 | 275368 |
| 32          | 453836 | 807480 | 171327 | 303696 |
| 64          | 522690 | 869667 | 35014  | 62803  |
| 128         | 396726 | 375067 | 51621  | 32493  |





##### *The **Speedup** with Different Data Size*

****

| thread\size | 100      | 200      | 500      | 1000     |
| ----------- | -------- | -------- | -------- | -------- |
| 1           | 1        | 1        | 1        | 1        |
| 2           | 1.189632 | 1.645052 | 1.848033 | 1.86763  |
| 4           | 1.179135 | 2.348349 | 3.521245 | 3.847389 |
| 8           | 1.199923 | 1.880547 | 5.617664 | 7.310002 |
| 16          | 1.346729 | 2.980094 | 1.341943 | 4.219294 |
| 32          | 0.775661 | 2.550941 | 1.321427 | 4.653346 |
| 64          | 0.893341 | 2.747398 | 0.270059 | 0.962292 |
| 128         | 0.678053 | 1.184889 | 0.398147 | 0.49787  |

![image-20211117201456181](CSC4005 Assignment3-Report.assets/image-20211117201456181.png)

##### Analysis

****

​		For the chosen data size, the speedup brought by MPI is not very significant when the rank size is too large. The best speedup is achieved when the core configuration is $8$. The possible reason for this is that when the data size is relatively small, the communication overhead brought by broadcast and gather, which will be invoked in every iteration, will be much higher then the speedup brought by computation parallelism. And the workload for each rank is small, making the speedup worse. For example, when the core configuration is $8$, and the data size is 200, each rank has $25$ to deal with, and the communication only happens among $8$​ processes. However, when the configuration is $128$, each rank has only 1 or 2 to deal with, which leads to no speedup due to the overhead, while the communication cost becomes much higher. 

​		Therefore, the communication overhead when the core configuration becomes too high caused the bad speedup. But, when the configuration size is small, the speedup is still good.





#### Experiment 2: Pthread

##### *The **Speed** with Different Data Size and Number of Cores (Points / Second)*

****

| thread\size | 100    | 200    | 500    | 1000   |
| ----------- | ------ | ------ | ------ | ------ |
| 1           | 362066 | 265590 | 126706 | 64956  |
| 2           | 439787 | 456310 | 244002 | 129207 |
| 4           | 411795 | 547641 | 417193 | 248009 |
| 8           | 249381 | 540890 | 600871 | 425364 |
| 16          | 164230 | 321807 | 536540 | 479278 |
| 32          | 95245  | 164570 | 434027 | 295854 |
| 64          | 31393  | 89834  | 149327 | 400136 |

##### *The **Speedup** with Different Data Size*

****

| thread\size | 100      | 200      | 500      | 1000     |
| ----------- | -------- | -------- | -------- | -------- |
| 1           | 1        | 1        | 1        | 1        |
| 2           | 1.21466  | 1.718099 | 1.925734 | 1.989146 |
| 4           | 1.137348 | 2.061979 | 3.292607 | 3.818108 |
| 8           | 0.688772 | 2.03656  | 4.742246 | 6.548494 |
| 16          | 0.453591 | 1.211668 | 4.234527 | 7.378502 |
| 32          | 0.26306  | 0.619639 | 3.425465 | 4.554683 |
| 64          | 0.086705 | 0.338243 | 1.178531 | 6.160108 |

![image-20211117221717251](CSC4005 Assignment3-Report.assets/image-20211117221717251.png)

##### Analysis

****

​		The speedup when the data size is small is not significant. The possible reason is that the small data size makes the parallel overhead higher than the parallel gain. And when the thread number is too large, which exceeds the core number on a node, there may be some thread switches making the speedup worse. When the thread / core configuration is $8$, the speedup is the best.





#### Experiment 3: CUDA



##### *The **Speed** with Different Data Size and Number of Cores (Points / Second)*

****

| block/data | 100    | 200    | 500    |
| ---------- | ------ | ------ | ------ |
| 1          | 279069 | 233461 | 128335 |
| 2          | 312782 | 376003 | 251085 |
| 4          | 350834 | 420488 | 444873 |

data size = block number * thread number 

##### Analysis

****

​		There are some speedup suing the CUDA program obvisouly.

​		The speedup is more significant when the block number is high (the grid size is large) and the thread number is small. The possible reason is that the  block shared memory is limited. When the thread number is large, the memory cache in a block is not sufficient and the IO overhead became high. Therefore, the calculation speed is lowered. When the grid size is large and the thread number is small, the IO overhead in thread shared memory is smaller, which brings better speedup.





#### Experiment 4: OpenMP

##### *The **Speed** with Different Data Size and Number of Cores (Points / Second)*

****

| thread\size | 100    | 200     | 500    | 1000   |
| ----------- | ------ | ------- | ------ | ------ |
| 1           | 382066 | 305590  | 146706 | 104956 |
| 2           | 599787 | 626310  | 324002 | 262827 |
| 4           | 971795 | 1047641 | 827193 | 487329 |
| 8           | 309381 | 512481  | 600871 | 473414 |
| 16          | 144210 | 265701  | 536540 | 232538 |
| 32          | 98129  | 164570  | 434027 | 213524 |
| 64          | 47293  | 103492  | 182719 | 341306 |



##### *The **Speedup** with Different Data Size*

****

| thread\size | 100      | 200      | 500      | 1000     |
| ----------- | -------- | -------- | -------- | -------- |
| 1           | 1        | 1        | 1        | 1        |
| 2           | 1.569852 | 2.049511 | 2.208512 | 2.504164 |
| 4           | 2.543527 | 3.428257 | 5.63844  | 4.643174 |
| 8           | 0.809758 | 1.677021 | 4.095749 | 4.510595 |
| 16          | 0.377448 | 0.869469 | 3.657246 | 2.215576 |
| 32          | 0.256838 | 0.538532 | 2.958482 | 2.034414 |
| 64          | 0.123782 | 0.338663 | 1.245477 | 3.251896 |

![image-20211117230525406](CSC4005 Assignment3-Report.assets/image-20211117230525406.png)

##### Analysis

****

​		The speedup when the data size is small is not significant. The possible reason is that the small data size makes the parallel overhead higher than the parallel gain. And when the thread number is too large, which exceeds the core number on a node, there may be some thread switches making the speedup worse. When the thread / core configuration is $4$​, the speedup is the best.



#### Experiment 5: OpenMP + mpi

##### The **Speed** with Different Data Size and Number of Cores (Points / Second)

| rank\thread | 1      | 2      | 4      | 8      | 16    |
| ----------- | ------ | ------ | ------ | ------ | ----- |
| 1           | 112996 | 125399 | 101148 | 83249  | 30696 |
| 2           | 128160 | 121988 | 94060  | 111040 | 25582 |
| 4           | 131461 | 131588 | 89656  | 91753  | 19437 |





#### Comparison Between Pthread & OpenMP

​		Learning from the speedup results, the openmp method is a little bit better than the pthread method. But in overall they have a close performance. The possible reason is that the openmp is indeed implemented by the pthread, and the workload distribution is more efficient than pthread, since the pthread implementation needs to calculate the distribution at run time and the distribution method may not be as good as the one of openmp. Therefore, the performance of openmp is a bit better.



#### Comparison Between MPI & MPI+OpenMP

​		Learning from the speedup results, the MPI implementation is better. The possible reason is that the openmp method introduce some overhead, while the the MPI version hide those overhead in the communication time. Or the sbatch command does not configure a suitable environment for the execution.





## 4. Conclusion

​		In this assignment, we have implemented five versions of n-body simulation by MPI, CUDA, openmp, and pthread. All these implementations work well. The speedup by CUDA is good according to the test results. We also did some experiments on the campus server with other programs and found that the best configuration for the test data size is $4$ or $8$, and the larger configuration introduced too much overhead. At last we made some comparison and found that the openmp and pthread have close performance, and the mpi version is better than the hybrid implementation.


#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <stdio.h>

#define DEBUGx

#define MAX_THREAD_PER_BLOCK 1024

#define TEST_ITERATION 600

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

__device__
void mutual_check_and_update(double *x, double *y, double * vx, double * vy, double *ax, double *ay, double *m,
    double *x_buf, double * y_buf, double * vx_buf, double * vy_buf, 
    int i, int j, double radius, double gravity, double COLLISION_RATIO)
{
    double delta_x = x[i] - x[j];
    double delta_y = y[i] - y[j];
    double distance_square = delta_x * delta_x + delta_y * delta_y;
    

    double ratio = 1 + COLLISION_RATIO;
    if (distance_square < radius * radius)
    {
        distance_square = radius * radius;
    }
    double distance = std::sqrt(distance_square);
    if (distance < radius)
    {
        distance = radius;
    }
    if (distance_square <= radius * radius)
    {
        double dot_prod = delta_x * (vx[i] - vx[j]) + delta_y * (vy[i] - vy[j]);
        double scalar = 2 / (m[i] + m[j]) * dot_prod / distance_square;
        vx_buf[i] -= scalar * delta_x * m[j];
        vy_buf[j] -= scalar * delta_y * m[j];

        // now relax the distance a bit: after the collision, there must be
        // at least (ratio * radius) between them
        x_buf[i] += delta_x / distance * ratio * radius / 2.0;
        y_buf[i] += delta_y / distance * ratio * radius / 2.0;
    }
    else
    {
        // update acceleration only when no collision
        auto scalar = gravity / distance_square / distance;
        ax[i] -= scalar * delta_x * m[j];
        ay[i] -= scalar * delta_y * m[j];
    }
}

__device__
void handle_wall_collision(double *ax, double *ay, double *x_buf, double *y_buf, double *vx_buf, double *vy_buf, 
                           int i, double position_range, double radius, double COLLISION_RATIO) {
    bool flag = false;

    if (x_buf[i]<= radius)
    {
        flag = true;
        x_buf[i]= radius + radius * COLLISION_RATIO;
        vx_buf[i]= -vx_buf[i];
    }
    else if (x_buf[i] >= position_range - radius)
    {
        flag = true;
        x_buf[i] = position_range - radius - radius * COLLISION_RATIO;
        vx_buf[i] = -vx_buf[i];
    }

    if (y_buf[i] <= radius)
    {
        flag = true;
        y_buf[i] = radius + radius * COLLISION_RATIO;
        vy_buf[i] = -vy_buf[i];
    }
    else if (y_buf[i] >= position_range - radius)
    {
        flag = true;
        y_buf[i] = position_range - radius - radius * COLLISION_RATIO;
        vy_buf[i] = -vy_buf[i];
    }
    if (flag)
    {
        ax[i] = 0;
        ay[i] = 0;
    }
}

__device__
void update_single(double *x_buf, double *y_buf, double *ax, double *ay, double *vx_buf, double *vy_buf, int i, double elapse, double position_range, double radius, double COLLISION_RATIO)
{
    vx_buf[i] += ax[i] * elapse;
    vy_buf[i] += ay[i] * elapse;
    handle_wall_collision(ax, ay, x_buf, y_buf, vx_buf, vy_buf, i, position_range, radius, COLLISION_RATIO);
    x_buf[i] += vx_buf[i] * elapse;
    y_buf[i] += vy_buf[i] * elapse;
    handle_wall_collision(ax, ay, x_buf, y_buf, vx_buf, vy_buf, i, position_range, radius, COLLISION_RATIO); // change x & v in buffer "copy"
}

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

    // zero accleration
    for (int i = 0; i < bodies; ++i) {
        ax[i] = 0;
        ay[i] = 0;
    }

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < bodies; i+=stride)
    {
        #ifdef DEBUG
        printf("deal with %d\n", i);
        #endif
        for (size_t j = 0; j < bodies; ++j)
            mutual_check_and_update(x, y, vx, vy, ax, ay, m, x_buf, y_buf, vx_buf, vy_buf, i, j, radius, gravity, COLLISION_RATIO);
    }
    __syncthreads();
#ifdef DEBUG
    if (index == 0)
    {
        printf("before\n");
        for (int i = 0; i < bodies; i++){
        printf("%dth buffer: (%f, %f %f %f) \n ",i, x_buf[i], y_buf[i], vx_buf[i], vy_buf[i]);
        printf("%dth: (%f, %f %f %f) \n ",i, x[i], y[i], vx[i], vy[i]);
    }}
#endif

    for (size_t i = index; i < bodies; i+=stride)
        update_single(x_buf, y_buf, ax, ay, vx_buf, vy_buf, i, elapse, position_range, radius, COLLISION_RATIO);
    
#ifdef DEBUG
    if (index == 0)
    {
        printf("after\n");
        for (int i = 0; i < bodies; i++){
        printf("%dth buffer: (%f, %f %f %f) \n ",i, x_buf[i], y_buf[i], vx_buf[i], vy_buf[i]);
        printf("%dth: (%f, %f) \n ",i, x[i], y[i], vx[i], vy[i]);
    }}
#endif
    
    for (size_t i = index; i < bodies; i+=stride) {
        x[i] = x_buf[i];
        y[i] = y_buf[i];
        vx[i] = vx_buf[i];
        vy[i] = vy_buf[i];
    }
}


int main(int argc, char **argv) {
    if (argc != 5)
    {
        std::cerr << "usage: " << argv[0] << " <BODIES> <GRAPH:0/1> <GRID_SIZE> <BLOCK_SIZE>" << std::endl;
        return 0;
    }
    static int iteration = 0;
    static int bodies = std::atoi(argv[1]);
    int GUI = std::atoi(argv[2]);
    int grid_size = std::atoi(argv[3]);
    int block_size = std::atoi(argv[4]);

    if (bodies <= 0)
    {
        std::cerr << "BODY should be greater than 0" << std::endl;
        return 0;
    }

    int THREAD = bodies <= MAX_THREAD_PER_BLOCK? bodies:MAX_THREAD_PER_BLOCK; // total number of threads using in the program
    size_t duration = 0;

    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static float elapse = 1.0;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;

    int *displs, *scounts, i, offset = 0;
    displs = (int *)malloc(THREAD * sizeof(int));
    scounts = (int *)malloc(THREAD * sizeof(int));

    // calculate individual workload
    if (THREAD == bodies) {
        for (i = 0; i < THREAD; ++i) {
            displs[i] = offset;
            scounts[i] = 1;
            ++offset;
        }
    }
    else if (THREAD < bodies) {
        for (i = 0; i < THREAD; ++i) {
            displs[i] = offset;
            scounts[i] = std::ceil(((float)bodies - i) / THREAD);
            offset += scounts[i];
        }
    }
    
    // body pool initialization
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
    double coll_ratio = pool.COLLISION_RATIO;


    // copies in device
    double* cuda_x, *cuda_y, *cuda_vx, *cuda_vy, *cuda_ax, *cuda_ay, *cuda_m, *cuda_x_buf,
          * cuda_y_buf, *cuda_vx_buf, *cuda_vy_buf;
    double position_range = space;
    int *cuda_displs, *cuda_scounts;


    // copy displs, scounts to device
    cudaMalloc((void **)&cuda_displs, sizeof(int) * THREAD);
    cudaMemcpy(cuda_displs, displs, sizeof(int) * THREAD, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&cuda_scounts, sizeof(int) * THREAD);
    cudaMemcpy(cuda_scounts, scounts, sizeof(int) * THREAD, cudaMemcpyHostToDevice);

    // allocate space for device copies of data in pool for CALCULATION
    cudaMalloc((void **)&cuda_x, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_y, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_vx, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_vy, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_ax, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_ay, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_m, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_x_buf, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_y_buf, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_vx_buf, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_vy_buf, sizeof(double) * bodies);

    // copy data to device
    cudaMemcpy(cuda_x, pool.x.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_x_buf, pool.x.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_vx, pool.vx.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_vx_buf, pool.vx.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_y, pool.y.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y_buf, pool.y.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_vy, pool.vy.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_vy_buf, pool.vy.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);
   
    cudaMemcpy(cuda_m, pool.m.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);


    if (GUI == 0) 
    {
        while (1) {
                if (iteration == TEST_ITERATION) {
                    exit(0);
                }        
                auto begin = std::chrono::high_resolution_clock::now();
                update_for_all<<<grid_size, block_size>>>
                (
                    cuda_x_buf,
                    cuda_y_buf,
                    cuda_vx_buf,
                    cuda_vy_buf,
                    coll_ratio,
                    bodies,
                    
                    cuda_x,
                    cuda_y,
                    cuda_vx,
                    cuda_vy,
                    cuda_ax,
                    cuda_ay,
                    cuda_m,
                    elapse,
                    gravity,
                    position_range,
                    radius);
        

                cudaDeviceSynchronize();
        
                // collect position result for GUI display
                cudaMemcpy(pool.x.data(), cuda_x, sizeof(double)*bodies, cudaMemcpyDeviceToHost);
                cudaMemcpy(pool.y.data(), cuda_y, sizeof(double)*bodies, cudaMemcpyDeviceToHost);

#ifdef DEBUG
                printf("finish\n\n");
                for (int i = 0; i < 5; ++i) {
                    printf("x: %f, y: %f \n", pool.x[i], pool.y[i]);
                }
                cudaError_t cudaStatus;
                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    printf("mykernel launch failed: %s\n",
                            cudaGetErrorString(cudaStatus));
                }
#endif
        
                auto end = std::chrono::high_resolution_clock::now();
                duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

                std::cout << bodies << " bodies in last " << duration << " nanoseconds\n";
                auto speed = static_cast<double>(bodies) / static_cast<double>(duration) * 1e9;
                std::cout << "speed: " << speed << " bodies per second" << std::endl;
                duration = 0;
                ++iteration;
            }
    }

    else if (GUI == 1) {
    graphic::GraphicContext context{"Assignment 3"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 3", nullptr,
                     ImGuiWindowFlags_NoMove
                     | ImGuiWindowFlags_NoCollapse
                     | ImGuiWindowFlags_NoTitleBar
                     | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        
        ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
        ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
        ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
        ImGui::ColorEdit4("Color", &color.x);
        if (iteration == 600000) {
            exit(0);
        }

        auto begin = std::chrono::high_resolution_clock::now();
        
        update_for_all<<<grid_size, block_size>>>
        (
            cuda_x_buf,
            cuda_y_buf,
            cuda_vx_buf,
            cuda_vy_buf,
            coll_ratio,
            bodies,
            
            cuda_x,
            cuda_y,
            cuda_vx,
            cuda_vy,
            cuda_ax,
            cuda_ay,
            cuda_m,
            elapse,
            gravity,
            position_range,
            radius);

        cudaDeviceSynchronize();

        // collect position result for GUI display
        cudaMemcpy(pool.x.data(), cuda_x, sizeof(double)*bodies, cudaMemcpyDeviceToHost);
        cudaMemcpy(pool.y.data(), cuda_y, sizeof(double)*bodies, cudaMemcpyDeviceToHost);

        auto end = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        std::cout << bodies << " bodies in last " << duration << " nanoseconds\n";
        auto speed = static_cast<double>(bodies) / static_cast<double>(duration) * 1e9;
        std::cout << "speed: " << speed << " bodies per second" << std::endl;
        duration = 0;
        ++iteration;

#ifdef DEBUG
        for (int i = 0; i < bodies; ++i) {
            printf("x: %f, y: %f \n", pool.x[i], pool.y[i]);
        }
        cudaError_t cudaStatus;
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            printf("mykernel launch failed: %s\n",
                    cudaGetErrorString(cudaStatus));
        }
#endif

        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            for (size_t i = 0; i < pool.size(); ++i) {
                auto body = pool.get_body(i);
                auto x = p.x + static_cast<float>(body.get_x());
                auto y = p.y + static_cast<float>(body.get_y());
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }
        }
        ImGui::End();});}

}

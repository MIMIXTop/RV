#include "Pointer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <sstream>
#include <format>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int K = 10;
constexpr int DIMS = 9;

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Константная память для центров
__constant__ double c_centers[K * DIMS];

// Ядро назначения кластеров
__global__ void assign_clusters(
    const double *points,
    int *cluster_ids,
    int *d_changed,
    int num_points) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    double min_dist = 1e20;
    int best_id = 0;

    // Ищем ближайший центр
    for (int c = 0; c < K; ++c) {
        double dist = 0.0;
        for (int i = 0; i < DIMS; ++i) {
            double diff = points[idx * DIMS + i] - c_centers[c * DIMS + i];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_id = c;
        }
    }

    // Если новый кластер отличается от старого
    if (cluster_ids[idx] != best_id) {
        cluster_ids[idx] = best_id;
        // Атомарно ставим флаг в 1.
        // Это гарантирует, что цикл продолжится, если хотя бы одна точка "двигается".
        // Цикл остановится, только если d_changed останется 0 (все точки на местах).
        atomicExch(d_changed, 1);
    }
}

// Ядро суммирования (Shared Memory)
__global__ void compute_sums_shared(
    const double *points,
    const int *cluster_ids,
    double *global_sums,
    int *global_counts,
    int num_points) {

    extern __shared__ double s_sums[];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;

    // Обнуление shared memory
    for (int i = tid; i < K * DIMS; i += blockSize) {
        s_sums[i] = 0.0;
    }
    __syncthreads();

    int objIndex = blockIdx.x * blockDim.y + threadIdx.y;
    int featureIndex = threadIdx.x;

    if (objIndex < num_points && featureIndex < DIMS) {
        int cluster = cluster_ids[objIndex];
        double val = points[objIndex * DIMS + featureIndex];

        atomicAdd(&s_sums[cluster * DIMS + featureIndex], val);

        if (featureIndex == 0) {
            atomicAdd(&global_counts[cluster], 1);
        }
    }
    __syncthreads();

    // Сброс в global memory
    for (int i = tid; i < K * DIMS; i += blockSize) {
        if (abs(s_sums[i]) > 1e-15) {
            atomicAdd(&global_sums[i], s_sums[i]);
        }
    }
}

// Ядро обновления центров
__global__ void update_centers(
    double *centers,
    const double *new_sums,
    const int *counts) {
    int c = threadIdx.x;
    if (c >= K) return;

    int count = counts[c];
    if (count > 0) {
        for (int i = 0; i < DIMS; ++i) {
            centers[c * DIMS + i] = new_sums[c * DIMS + i] / count;
        }
    }
}

std::vector<Point> load_pointers(const std::string &path_to_dir) {
    std::vector<Point> points;
    for (const auto &entry: std::filesystem::directory_iterator(path_to_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::ifstream file(entry.path());
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                Point p;
                for (int i = 0; i < DIMS; ++i) iss >> p.coords[i];
                points.push_back(p);
            }
        }
    }
    return points;
}

void save_all_results(const std::vector<Point> &points, const std::vector<Point> &centers) {
    std::filesystem::create_directories("../Data/gpu_opt");

    std::ofstream out_points("../Data/gpu_opt/kmeans_result.txt");
    for (const auto &p: points) {
        for (double coord: p.coords) out_points << coord << " ";
        out_points << p.cluster_id << "\n";
    }
    out_points.close();

    std::ofstream out_centers("../Data/gpu_opt/kmeans_centers.txt");
    for (int i = 0; i < centers.size(); ++i) {
        for (double coord: centers[i].coords) out_centers << coord << " ";
        out_centers << i << "\n";
    }
    out_centers.close();

    std::cout << "Результаты и центры сохранены в ../Data/gpu_strict" << std::endl;
}

std::vector<Point> kmeans(std::vector<Point> &points) {
    int num_points = points.size();
    if (num_points == 0) return {};

    std::array<Point, K> centers;
    std::uniform_int_distribution dis(0, num_points - 1);
    std::mt19937 gen(std::random_device{}());

    for (int i = 0; i < K; ++i) {
        centers[i] = points[dis(gen)];
        centers[i].cluster_id = i;
    }

    double *h_points, *h_centers, *h_centers_new_buf;
    int *h_cluster_ids, *h_changed_ptr;

    cudaCheck(cudaMallocHost((void**)&h_points, num_points * DIMS * sizeof(double)));
    cudaCheck(cudaMallocHost((void**)&h_centers, K * DIMS * sizeof(double)));
    cudaCheck(cudaMallocHost((void**)&h_centers_new_buf, K * DIMS * sizeof(double)));
    cudaCheck(cudaMallocHost((void**)&h_cluster_ids, num_points * sizeof(int)));
    cudaCheck(cudaMallocHost((void**)&h_changed_ptr, sizeof(int)));

    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < DIMS; ++j) h_points[i * DIMS + j] = points[i].coords[j];
        h_cluster_ids[i] = -1;
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < DIMS; ++j) h_centers[i * DIMS + j] = centers[i].coords[j];
    }

    double *d_points, *d_centers, *d_sums;
    int *d_cluster_ids, *d_counts, *d_changed;

    cudaCheck(cudaMalloc(&d_points, num_points * DIMS * sizeof(double)));
    cudaCheck(cudaMalloc(&d_centers, K * DIMS * sizeof(double)));
    cudaCheck(cudaMalloc(&d_cluster_ids, num_points * sizeof(int)));
    cudaCheck(cudaMalloc(&d_sums, K * DIMS * sizeof(double)));
    cudaCheck(cudaMalloc(&d_counts, K * sizeof(int)));
    cudaCheck(cudaMalloc(&d_changed, sizeof(int)));

    cudaCheck(cudaMemcpyAsync(d_points, h_points, num_points * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(d_centers, h_centers, K * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(d_cluster_ids, h_cluster_ids, num_points * sizeof(int), cudaMemcpyHostToDevice));

    // Синхронизация перед замером времени
    cudaDeviceSynchronize();

    std::cout << "Начало вычислительного цикла (Строгая сходимость)..." << std::endl;
    auto start_compute = std::chrono::high_resolution_clock::now();

    int blockSizeAssign = 256;
    int numBlocksAssign = (num_points + blockSizeAssign - 1) / blockSizeAssign;

    int threadsPerBlockY = 1024 / DIMS;
    if (threadsPerBlockY > 32) threadsPerBlockY = 32;

    dim3 blockDimSums(DIMS, threadsPerBlockY);
    dim3 gridDimSums((num_points + threadsPerBlockY - 1) / threadsPerBlockY, 1);
    size_t sharedMemSize = K * DIMS * sizeof(double);

    int iter = 0;
    int h_changed = 1;

    // Цикл работает пока h_changed == 1.
    // h_changed станет 0 только если НИ ОДНА точка не поменяет кластер.
    while (h_changed) {
        // 1. Сброс флага изменений
        h_changed = 0;
        *h_changed_ptr = 0;
        cudaCheck(cudaMemcpyAsync(d_changed, h_changed_ptr, sizeof(int), cudaMemcpyHostToDevice));

        // 2. Обновление константной памяти центров (с хоста, так как update_centers пишет в глобальную)
        cudaCheck(cudaMemcpyToSymbol(c_centers, h_centers, K * DIMS * sizeof(double)));

        // 3. Ядро назначения кластеров (ТУТ проверяется условие changed)
        assign_clusters<<<numBlocksAssign, blockSizeAssign>>>(d_points, d_cluster_ids, d_changed, num_points);
        cudaCheck(cudaGetLastError());

        // 4. Чтение флага изменений. Если он стал 1, значит цикл продолжится.
        cudaCheck(cudaMemcpy(h_changed_ptr, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        h_changed = *h_changed_ptr;

        // ВАЖНО: Мы НЕ делаем break по расстоянию сдвига центров.
        // Мы полагаемся только на h_changed.

        // 5. Пересчет сумм
        cudaCheck(cudaMemset(d_sums, 0, K * DIMS * sizeof(double)));
        cudaCheck(cudaMemset(d_counts, 0, K * sizeof(int)));
        compute_sums_shared<<<gridDimSums, blockDimSums, sharedMemSize>>>(d_points, d_cluster_ids, d_sums, d_counts, num_points);
        cudaCheck(cudaGetLastError());

        // 6. Обновление координат центров
        update_centers<<<1, K>>>(d_centers, d_sums, d_counts);
        cudaCheck(cudaGetLastError());

        // Копируем центры на хост, чтобы обновить c_centers на следующей итерации
        cudaCheck(cudaMemcpy(h_centers, d_centers, K * DIMS * sizeof(double), cudaMemcpyDeviceToHost));

        iter++;
        // Вывод каждые 10 итераций, чтобы не спамить в консоль
        if (iter % 10 == 0) std::cout << "Итерация: " << iter << " (точки все еще меняют кластеры)" << std::endl;
    }

    cudaDeviceSynchronize();
    auto end_compute = std::chrono::high_resolution_clock::now();

    std::cout << std::format(
        "⏱️ Чистое время вычислений: {} мс\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count());

    std::cout << std::format("✅ Алгоритм сошёлся за {} итераций (Все точки зафиксированы)\n", iter);

    cudaCheck(cudaMemcpy(h_cluster_ids, d_cluster_ids, num_points * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_points; ++i) {
        points[i].cluster_id = h_cluster_ids[i];
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < DIMS; ++j) {
            centers[i].coords[j] = h_centers[i * DIMS + j];
        }
    }

    for (const auto &p: centers) {
        std::cout << "Центроид " << p.cluster_id << " координаты: [";
        for (size_t i = 0; i < p.coords.size(); ++i) {
            std::cout << p.coords[i] << (i < p.coords.size() - 1 ? ", " : "");
        }
        std::cout << "]\n";
    }

    cudaFree(d_points); cudaFree(d_centers); cudaFree(d_cluster_ids);
    cudaFree(d_sums); cudaFree(d_counts); cudaFree(d_changed);

    cudaFreeHost(h_points); cudaFreeHost(h_centers); cudaFreeHost(h_centers_new_buf);
    cudaFreeHost(h_cluster_ids); cudaFreeHost(h_changed_ptr);

    return {centers.begin(), centers.end()};
}

int main() {
    try {
        std::vector<Point> points = load_pointers("../Data");
        std::cout << "Классификация начала (Strict convergence)\n";

        auto start_total = std::chrono::high_resolution_clock::now();
        std::vector<Point> centers = kmeans(points);
        auto end_total = std::chrono::high_resolution_clock::now();

        std::cout << std::format(
            "Полное время выполнения: {} мс\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count());

        save_all_results(points, centers);
    } catch (const std::exception &e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
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

// --- ПАРАМЕТРЫ ---
constexpr int K = 10;
constexpr int DIMS = 9;
constexpr int N_STREAMS = 10; // Количество потоков (streams)

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Центры в константной памяти
__constant__ double c_centers[K * DIMS];

// --- ЯДРА CUDA ---

// 1. Ядро назначения кластеров (добавлен аргумент offset)
__global__ void assign_clusters_stream(
    const double *points,
    int *cluster_ids,
    int *d_changed,
    int num_points,
    int offset) { // Смещение для текущего стрима

    // Глобальный индекс: смещение пачки + индекс внутри пачки
    int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points) return;

    double min_dist = 1e20;
    int best_id = 0;

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

    if (cluster_ids[idx] != best_id) {
        cluster_ids[idx] = best_id;
        atomicExch(d_changed, 1);
    }
}

// 2. Ядро суммирования (пишет в локальный буфер стрима)
__global__ void compute_sums_stream(
    const double *points,
    const int *cluster_ids,
    double *partial_sums,   // Указатель на часть памяти конкретного стрима
    int *partial_counts,    // Указатель на часть памяти конкретного стрима
    int num_points,
    int offset) {

    extern __shared__ double s_sums[];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;

    // Обнуление shared memory
    for (int i = tid; i < K * DIMS; i += blockSize) {
        s_sums[i] = 0.0;
    }
    __syncthreads();

    int objIndex = offset + blockIdx.x * blockDim.y + threadIdx.y;
    int featureIndex = threadIdx.x;

    if (objIndex < num_points && featureIndex < DIMS) {
        int cluster = cluster_ids[objIndex];
        double val = points[objIndex * DIMS + featureIndex];

        atomicAdd(&s_sums[cluster * DIMS + featureIndex], val);

        if (featureIndex == 0) {
            atomicAdd(&partial_counts[cluster], 1);
        }
    }
    __syncthreads();

    // Сброс результатов в глобальную память (в буфер стрима)
    for (int i = tid; i < K * DIMS; i += blockSize) {
        if (abs(s_sums[i]) > 1e-15) {
            atomicAdd(&partial_sums[i], s_sums[i]);
        }
    }
}

// --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

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
    std::filesystem::create_directories("../Data/gpu_stream");
    std::ofstream out_points("../Data/gpu_stream/kmeans_result.txt");
    for (const auto &p: points) {
        for (double coord: p.coords) out_points << coord << " ";
        out_points << p.cluster_id << "\n";
    }
    out_points.close();
    std::ofstream out_centers("../Data/gpu_stream/kmeans_centers.txt");
    for (int i = 0; i < centers.size(); ++i) {
        for (double coord: centers[i].coords) out_centers << coord << " ";
        out_centers << i << "\n";
    }
    out_centers.close();
    std::cout << "Результаты сохранены в ../Data/gpu_stream" << std::endl;
}

// --- ОСНОВНАЯ ФУНКЦИЯ K-MEANS ---

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

    // 1. Выделение Pinned Memory на Хосте (Критично для Stream!)
    double *h_points, *h_centers;
    int *h_cluster_ids, *h_changed_ptr;

    // Буферы для приема частичных сумм от каждого стрима
    // Размер = [Кол-во стримов] * [Размер данных одного стрима]
    double *h_stream_sums;
    int *h_stream_counts;

    // Используем cudaMallocHost (Pinned Memory)
    cudaCheck(cudaMallocHost((void**)&h_points, num_points * DIMS * sizeof(double)));
    cudaCheck(cudaMallocHost((void**)&h_centers, K * DIMS * sizeof(double)));
    cudaCheck(cudaMallocHost((void**)&h_cluster_ids, num_points * sizeof(int)));
    cudaCheck(cudaMallocHost((void**)&h_changed_ptr, sizeof(int)));

    cudaCheck(cudaMallocHost((void**)&h_stream_sums, N_STREAMS * K * DIMS * sizeof(double)));
    cudaCheck(cudaMallocHost((void**)&h_stream_counts, N_STREAMS * K * sizeof(int)));

    // Копирование данных
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < DIMS; ++j) h_points[i * DIMS + j] = points[i].coords[j];
        h_cluster_ids[i] = -1;
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < DIMS; ++j) h_centers[i * DIMS + j] = centers[i].coords[j];
    }

    // 2. Выделение памяти на GPU
    double *d_points, *d_centers;
    int *d_cluster_ids, *d_changed;

    // Буферы на GPU для каждого стрима (чтобы стримы не мешали друг другу atomicAdd)
    double *d_stream_sums;
    int *d_stream_counts;

    cudaCheck(cudaMalloc(&d_points, num_points * DIMS * sizeof(double)));
    cudaCheck(cudaMalloc(&d_centers, K * DIMS * sizeof(double)));
    cudaCheck(cudaMalloc(&d_cluster_ids, num_points * sizeof(int)));
    cudaCheck(cudaMalloc(&d_changed, sizeof(int)));

    cudaCheck(cudaMalloc(&d_stream_sums, N_STREAMS * K * DIMS * sizeof(double)));
    cudaCheck(cudaMalloc(&d_stream_counts, N_STREAMS * K * sizeof(int)));

    // Копируем точки один раз (они не меняются)
    cudaCheck(cudaMemcpy(d_points, h_points, num_points * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_cluster_ids, h_cluster_ids, num_points * sizeof(int), cudaMemcpyHostToDevice));

    // 3. Создание Streams (Как в вашем примере)
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; ++i) {
        cudaCheck(cudaStreamCreate(&streams[i]));
    }

    // Параметры запуска
    int batchSize = num_points / N_STREAMS; // Размер куска данных для одного стрима
    int threadsPerBlockY = 1024 / DIMS;
    if (threadsPerBlockY > 32) threadsPerBlockY = 32;
    dim3 blockDimSums(DIMS, threadsPerBlockY);
    size_t sharedMemSize = K * DIMS * sizeof(double);

    cudaDeviceSynchronize();

    std::cout << "Запуск с использованием " << N_STREAMS << " потоков (Streams)..." << std::endl;
    auto start_compute = std::chrono::high_resolution_clock::now();

    int iter = 0;
    int h_changed = 1;

    // --- ГЛАВНЫЙ ЦИКЛ ---
    while (h_changed) {
        h_changed = 0;
        *h_changed_ptr = 0;
        cudaCheck(cudaMemcpy(d_changed, h_changed_ptr, sizeof(int), cudaMemcpyHostToDevice));

        // Копируем центры в Constant Memory (синхронно для всех стримов)
        cudaCheck(cudaMemcpyToSymbol(c_centers, h_centers, K * DIMS * sizeof(double)));

        // Цикл по стримам: Запуск задач
        for (int i = 0; i < N_STREAMS; ++i) {
            int offset = i * batchSize;
            int currentBatchSize = (i == N_STREAMS - 1) ? (num_points - offset) : batchSize;

            // Указатели на область памяти ТЕКУЩЕГО стрима
            double* d_curr_sums = d_stream_sums + (i * K * DIMS);
            int* d_curr_counts = d_stream_counts + (i * K);

            // 1. Асинхронная очистка буферов стрима (с привязкой к стриму)
            cudaCheck(cudaMemsetAsync(d_curr_sums, 0, K * DIMS * sizeof(double), streams[i]));
            cudaCheck(cudaMemsetAsync(d_curr_counts, 0, K * sizeof(int), streams[i]));

            // 2. Запуск ядра Assign Clusters (с привязкой к стриму)
            int blockSize = 256;
            int numBlocks = (currentBatchSize + blockSize - 1) / blockSize;
            assign_clusters_stream<<<numBlocks, blockSize, 0, streams[i]>>>(
                d_points, d_cluster_ids, d_changed, num_points, offset
            );

            // 3. Запуск ядра Compute Sums (с привязкой к стриму)
            int gridY = (currentBatchSize + threadsPerBlockY - 1) / threadsPerBlockY;
            compute_sums_stream<<<gridY, blockDimSums, sharedMemSize, streams[i]>>>(
                d_points, d_cluster_ids, d_curr_sums, d_curr_counts, num_points, offset
            );

            // 4. Асинхронное копирование результатов (D2H) (с привязкой к стриму)
            // Копируем в соответствующее смещение на хосте
            cudaCheck(cudaMemcpyAsync(
                h_stream_sums + (i * K * DIMS),
                d_curr_sums,
                K * DIMS * sizeof(double),
                cudaMemcpyDeviceToHost,
                streams[i]
            ));

            cudaCheck(cudaMemcpyAsync(
                h_stream_counts + (i * K),
                d_curr_counts,
                K * sizeof(int),
                cudaMemcpyDeviceToHost,
                streams[i]
            ));
        }

        // Ждем завершения ВСЕХ стримов перед обработкой на CPU
        cudaDeviceSynchronize();

        // Проверяем флаг остановки
        cudaCheck(cudaMemcpy(h_changed_ptr, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        h_changed = *h_changed_ptr;

        if (!h_changed) break; // Все точки стабилизировались

        // АГРЕГАЦИЯ НА CPU (Сложение результатов всех стримов)
        std::vector<double> total_sums(K * DIMS, 0.0);
        std::vector<int> total_counts(K, 0);

        for (int i = 0; i < N_STREAMS; ++i) {
            for (int k = 0; k < K; ++k) {
                total_counts[k] += h_stream_counts[i * K + k];
                for (int d = 0; d < DIMS; ++d) {
                    total_sums[k * DIMS + d] += h_stream_sums[i * K * DIMS + k * DIMS + d];
                }
            }
        }

        // Вычисление новых центров
        for (int k = 0; k < K; ++k) {
            if (total_counts[k] > 0) {
                for (int d = 0; d < DIMS; ++d) {
                    h_centers[k * DIMS + d] = total_sums[k * DIMS + d] / total_counts[k];
                }
            }
        }

        iter++;
        if (iter % 10 == 0) std::cout << "Итерация: " << iter << "\r" << std::flush;
    }
    std::cout << std::endl;

    auto end_compute = std::chrono::high_resolution_clock::now();

    std::cout << std::format(
        "⏱️ Чистое время вычислений (Streams): {} мс\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count());
    std::cout << std::format("✅ Алгоритм сошёлся за {} итераций\n", iter);

    // Забираем финальные кластеры
    cudaCheck(cudaMemcpy(h_cluster_ids, d_cluster_ids, num_points * sizeof(int), cudaMemcpyDeviceToHost));

    // 5. Разрушение стримов (Как в вашем примере)
    for (int i = 0; i < N_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    // Сохранение в C++ структуры
    for (int i = 0; i < num_points; ++i) points[i].cluster_id = h_cluster_ids[i];
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < DIMS; ++j) centers[i].coords[j] = h_centers[i * DIMS + j];
    }

    // Очистка памяти
    cudaFree(d_points); cudaFree(d_centers); cudaFree(d_cluster_ids); cudaFree(d_changed);
    cudaFree(d_stream_sums); cudaFree(d_stream_counts);
    cudaFreeHost(h_points); cudaFreeHost(h_centers); cudaFreeHost(h_cluster_ids); cudaFreeHost(h_changed_ptr);
    cudaFreeHost(h_stream_sums); cudaFreeHost(h_stream_counts);

    return {centers.begin(), centers.end()};
}

int main() {
    try {
        std::vector<Point> points = load_pointers("../Data");
        std::cout << "Классификация начала (CUDA Streams Version)\n";

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
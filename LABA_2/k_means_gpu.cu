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
#include <format> // Для std::format

#include <cuda_runtime.h>

constexpr int K = 10;
constexpr int DIMS = 9;

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Базовое ядро назначения кластеров
__global__ void assign_clusters(
    const double *points,
    const double *centers,
    int *cluster_ids,
    int *d_changed,
    int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    double min_dist = 1e20;
    int best_id = 0;

    for (int c = 0; c < K; ++c) {
        double dist = 0.0;
        for (int i = 0; i < DIMS; ++i) {
            double diff = points[idx * DIMS + i] - centers[c * DIMS + i];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_id = c;
        }
    }

    // Если кластер изменился, ставим флаг
    if (cluster_ids[idx] != best_id) {
        cluster_ids[idx] = best_id;
        atomicExch(d_changed, 1);
    }
}

// Базовое ядро суммирования (Global Atomics)
__global__ void compute_sums(
    const double *points,
    const int *cluster_ids,
    double *new_sums,
    int *counts,
    int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    int cluster = cluster_ids[idx];

    atomicAdd(&counts[cluster], 1);

    for (int i = 0; i < DIMS; ++i) {
        atomicAdd(&new_sums[cluster * DIMS + i], points[idx * DIMS + i]);
    }
}

// Базовое ядро обновления центров
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
    std::filesystem::create_directories("../Data/gpu");

    std::ofstream out_points("../Data/gpu/kmeans_result.txt");
    for (const auto &p: points) {
        for (double coord: p.coords) out_points << coord << " ";
        out_points << p.cluster_id << "\n";
    }
    out_points.close();

    std::ofstream out_centers("../Data/gpu/kmeans_centers.txt");
    for (int i = 0; i < centers.size(); ++i) {
        for (double coord: centers[i].coords) out_centers << coord << " ";
        out_centers << i << "\n";
    }
    out_centers.close();

    std::cout << "Результаты и центры сохранены в ../Data/gpu" << std::endl;
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

    std::vector<double> h_points(num_points * DIMS);
    std::vector h_cluster_ids(num_points, -1);
    std::vector<double> h_centers(K * DIMS);

    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < DIMS; ++j) h_points[i * DIMS + j] = points[i].coords[j];
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < DIMS; ++j) h_centers[i * DIMS + j] = centers[i].coords[j];
    }

    double *d_points, *d_centers, *d_sums;
    int *d_cluster_ids, *d_counts, *d_changed;

    // Стандартное выделение памяти (не Pinned/Host)
    cudaCheck(cudaMalloc(&d_points, num_points * DIMS * sizeof(double)));
    cudaCheck(cudaMalloc(&d_centers, K * DIMS * sizeof(double)));
    cudaCheck(cudaMalloc(&d_cluster_ids, num_points * sizeof(int)));
    cudaCheck(cudaMalloc(&d_sums, K * DIMS * sizeof(double)));
    cudaCheck(cudaMalloc(&d_counts, K * sizeof(int)));
    cudaCheck(cudaMalloc(&d_changed, sizeof(int)));

    cudaCheck(cudaMemcpy(d_points, h_points.data(), num_points * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_centers, h_centers.data(), K * DIMS * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_cluster_ids, h_cluster_ids.data(), num_points * sizeof(int), cudaMemcpyHostToDevice));

    // Синхронизация перед началом чистого замера
    cudaDeviceSynchronize();

    // === НАЧАЛО ЗАМЕРА ВЫЧИСЛЕНИЙ ===
    std::cout << "Запуск чистого вычислительного цикла (Strict Convergence)..." << std::endl;
    auto start_compute = std::chrono::high_resolution_clock::now();

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;

    int iter = 0;
    int h_changed = 1;

    // Цикл продолжается, пока h_changed != 0
    while (h_changed) {
        h_changed = 0;
        cudaCheck(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        // 1. Назначение кластеров (здесь выставляется флаг d_changed)
        assign_clusters<<<numBlocks, blockSize>>>(d_points, d_centers, d_cluster_ids, d_changed, num_points);
        cudaCheck(cudaGetLastError());

        // 2. Чтение флага
        cudaCheck(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

        // 3. Обнуление и пересчет сумм
        cudaCheck(cudaMemset(d_sums, 0, K * DIMS * sizeof(double)));
        cudaCheck(cudaMemset(d_counts, 0, K * sizeof(int)));

        compute_sums<<<numBlocks, blockSize>>>(d_points, d_cluster_ids, d_sums, d_counts, num_points);
        cudaCheck(cudaGetLastError());

        // 4. Обновление центров
        update_centers<<<1, K>>>(d_centers, d_sums, d_counts);
        cudaCheck(cudaGetLastError());

        // Мы не копируем центры на хост для проверки max_shift,
        // так как критерий остановки теперь строго по h_changed.

        iter++;
        if (iter % 10 == 0) std::cout << "Итерация: " << iter << "\r" << std::flush;
    }
    std::cout << std::endl;

    // Синхронизация перед остановкой таймера
    cudaDeviceSynchronize();
    auto end_compute = std::chrono::high_resolution_clock::now();

    std::cout << std::format(
        "⏱️ Чистое время вычислений (Lab 2 Basic): {} мс\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count());
    std::cout << std::format("✅ Алгоритм сошёлся за {} итераций (Все точки зафиксированы)\n", iter);
    // === КОНЕЦ ЗАМЕРА ВЫЧИСЛЕНИЙ ===

    // Копируем финальные результаты
    cudaCheck(cudaMemcpy(h_cluster_ids.data(), d_cluster_ids, num_points * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_centers.data(), d_centers, K * DIMS * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_points);
    cudaFree(d_centers);
    cudaFree(d_cluster_ids);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_changed);

    for (int i = 0; i < num_points; ++i) {
        points[i].cluster_id = h_cluster_ids[i];
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < DIMS; ++j) {
            centers[i].coords[j] = h_centers[i * DIMS + j];
        }
    }

    // Вывод финальных центров
    for (const auto &p: centers) {
        std::cout << "Центроид " << p.cluster_id << " координаты: [";
        for (size_t i = 0; i < p.coords.size(); ++i) {
            std::cout << p.coords[i] << (i < p.coords.size() - 1 ? ", " : "");
        }
        std::cout << "]\n";
    }

    return {centers.begin(), centers.end()};
}

int main() {
    try {
        std::vector<Point> points = load_pointers("../Data");

        std::cout << "Классификация начала (Standard CUDA)\n";

        // Полный замер времени (включая память)
        auto start_total = std::chrono::high_resolution_clock::now();
        std::vector<Point> centers = kmeans(points);
        auto end_total = std::chrono::high_resolution_clock::now();

        std::cout << std::format(
            "Полное время выполнения (с учетом памяти): {} милисекунд\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count());

        save_all_results(points, centers);
    } catch (const std::exception &e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Неизвестная ошибка" << std::endl;
        return 1;
    }

    return 0;
}
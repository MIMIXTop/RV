#include "Pointer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;


void clearData() {
    if (fs::exists("../Data")) fs::remove_all("../Data");
    fs::create_directories("../Data");
}

void random_numbers(size_t count, double min_val, double max_val, std::string filename) {
    std::mt19937 gen(std::random_device{}());

    // 1. Для КАЖДОГО из 9 измерений определяем свой начальный центр и свой шаг блуждания
    struct DimState {
        double mean;
        double walk_sigma;
    };

    std::vector<DimState> dims(9);
    std::uniform_real_distribution<> start_dist(min_val, max_val);

    for (int i = 0; i < 9; ++i) {
        double d1 = start_dist(gen);
        double d2 = start_dist(gen);
        dims[i].mean = (d1 + d2) / 2.0;
        dims[i].walk_sigma = std::abs(d1 - d2) * 0.3; // Шаг блуждания зависит от размера области
    }

    std::vector<Point> points;
    points.reserve(count);

    int num_blobs = std::uniform_int_distribution<>(4, 15)(gen);
    size_t points_per_blob = count / num_blobs;

    for (int b = 0; b < num_blobs; ++b) {
        size_t current_batch = (b == num_blobs - 1) ? (count - points.size()) : points_per_blob;

        // Для каждого блоба в каждом измерении выбираем свою плотность (sigma)
        std::vector<std::normal_distribution<>> attr_dist;
        for (int i = 0; i < 9; ++i) {
            double sigma = dims[i].walk_sigma * 1.5; // Плотность кучности
            attr_dist.emplace_back(dims[i].mean, sigma);
        }

        for (size_t i = 0; i < current_batch; ++i) {
            Point p;
            for (int j = 0; j < 9; ++j) {
                p.coords[j] = attr_dist[j](gen);
            }
            points.push_back(p);
        }

        // Смещаем центры КАЖДОГО измерения независимо (Random Walk)
        for (int j = 0; j < 9; ++j) {
            std::normal_distribution<> walk(0, dims[j].walk_sigma);
            dims[j].mean += walk(gen);
            dims[j].mean = std::clamp(dims[j].mean, min_val, max_val);
        }
    }

    // Запись всех 9 координат
    std::ofstream out(filename);
    for (const auto& p : points) {
        for (int i = 0; i < 9; ++i) {
            out << p.coords[i] << (i == 8 ? "" : " ");
        }
        out << '\n';
    }
}

int main() {
    clearData();

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, 1000000);

    int num_threads;
    std::print("Количество потоков: ");
    std::cin >> num_threads;

    if (num_threads <= 0) return 1;

    size_t total_points = 80000;
    size_t per_thread = total_points / num_threads;

    std::vector<std::jthread> thread_pool;
    for (int i = 0; i < num_threads; ++i) {
        size_t count = (i == num_threads - 1) ? (total_points - per_thread * i) : per_thread;

        double x1 = dis(gen);
        double x2 = dis(gen);

        std::string filename = std::format("../Data/points_{}.txt", i);
        thread_pool.emplace_back(random_numbers, count, std::min(x1, x2), std::max(x1, x2), filename);
    }

    return 0;
}
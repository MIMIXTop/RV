#include "Pointer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <random>
#include <ranges>
#include <experimental/simd>
#include <vector>

constexpr int K = 10;

Point avg_point(const std::vector<Point>& points) {
    Point center;
    for (int i = 0; i < 9; ++i) {
        double sum = 0.0;
        for (const auto& point : points) {
            sum += point.coords[i];
        }
        center.coords[i] = sum / points.size();
    }
    return center;
}

double distance(const Point& point, const Point& center) {
    double dist = 0.0;

    for (auto&& [p_coord, c_coord] : std::views::zip(point.coords, center.coords)) {
        dist += (p_coord - c_coord) * (p_coord - c_coord);
    }

    return dist;
}

std::vector<Point> load_pointers(const std::string& path_to_dir) {
    std::vector<Point> points;

    for (const auto& entry : std::filesystem::directory_iterator(path_to_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::ifstream file(entry.path());

            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                Point p;
                for (int i = 0; i < 9; ++i) {
                    iss >> p.coords[i];
                }
                points.push_back(p);
            }
        }
    }
    return points;
}

void save_all_results(const std::vector<Point>& points, const std::vector<Point>& centers) {
    std::filesystem::create_directories("../Data/cpu");

    std::ofstream out_points("../Data/cpu/kmeans_result.txt");
    for (const auto& p : points) {
        for (double coord : p.coords) out_points << coord << " ";
        out_points << p.cluster_id << "\n";
    }
    out_points.close();

    std::ofstream out_centers("../Data/cpu/kmeans_centers.txt");
    for (int i = 0; i < centers.size(); ++i) {
        for (double coord : centers[i].coords) out_centers << coord << " ";
        out_centers << i << "\n";
    }
    out_centers.close();

    std::cout << "Результаты и центры сохранены в ../Data/cpu" << std::endl;
}
std::vector<Point> kmeans(std::vector<Point>& points) {
    std::array<Point, K> centers;

    std::uniform_int_distribution<> dis(0, points.size() - 1);
    std::mt19937 gen(std::random_device {}());

    for (int i = 0; i < K; ++i) {
        centers[i] = points[dis(gen)];
        centers[i].cluster_id = i;
    }

    int iter = 0;
    bool changed = false;
    while (!changed) {
        std::for_each(std::execution::par_unseq, points.begin(), points.end(), [&](Point& p) {
            double min_dist = std::numeric_limits<double>::max();
            int best_id = 0;
            for (int i = 0; i < K; ++i) {
                double d = distance(p, centers[i]);
                if (d < min_dist) {
                    min_dist = d;
                    best_id = i;
                }
            }
            p.cluster_id = best_id;
        });

        std::vector<std::array<double, 9>> new_sums(K, { 0.0 });
        std::vector<size_t> counts(K, 0);

        for (const auto& p : points) {
            counts[p.cluster_id]++;
            for (size_t i = 0; i < 9; ++i) {
                new_sums[p.cluster_id][i] += p.coords[i];
            }
        }

        for (int i = 0; i < K; ++i) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < 9; ++j) {
                    double new_val = new_sums[i][j] / counts[i];
                    if (std::abs(centers[i].coords[j] - new_val) < 1e-20)
                        changed = true;
                    centers[i].coords[j] = new_val;
                }
            }
        }
        ++iter;
    }

    std::println("Итерация: {}", iter);
    std::ranges::for_each(centers, [&](Point& p) {
        std::println("Центроид {} координаты: {}", p.cluster_id, p.coords);
    });
    return { centers.begin(), centers.end() };
}

int main() {
    try {
        std::vector<Point> points = load_pointers("../Data");
        std::println("Классификация начала");
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Point> centers = kmeans(points);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << std::format(
            "Классификация завершена за {} милисекунд\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        save_all_results(points, centers);
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Неизвестная ошибка" << std::endl;
        return 1;
    }

    return 0;
}
from matplotlib import pyplot as plt
import torch

def get_pca_projections(points_cpu: torch.Tensor, centers_cpu: torch.Tensor,
                        points_gpu: torch.Tensor, centers_gpu: torch.Tensor):
    # Вычисляем оси PCA ТОЛЬКО по точкам CPU (координаты точек одинаковы для обеих версий)
    # Находим среднее и центрируем данные вручную, чтобы использовать это же среднее для остальных
    mean = torch.mean(points_cpu, dim=0)
    centered_points = points_cpu - mean

    # Вычисляем матрицу трансформации V (оси главных компонент)
    # center=False, так как мы центрировали данные вручную
    _, _, V = torch.pca_lowrank(centered_points, q=2, center=False)

    # Проецируем все тензоры, вычитая то же самое среднее и умножая на матрицу V
    p_cpu = torch.matmul(points_cpu - mean, V)
    c_cpu = torch.matmul(centers_cpu - mean, V)

    p_gpu = torch.matmul(points_gpu - mean, V)
    c_gpu = torch.matmul(centers_gpu - mean, V)

    return p_cpu, c_cpu, p_gpu, c_gpu

def load_data(path):
    data = []
    labels = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            data.append([float(x) for x in parts[:9]])
            labels.append(int(parts[9]))
    return torch.tensor(data, dtype=torch.float32), labels

def make_combined_plot(p_cpu, labels_cpu, c_cpu, c_labels_cpu,
                       p_gpu, labels_gpu, c_gpu, c_labels_gpu,
                       name_output_file: str):

    # Создаем ОДНУ фигуру с ДВУМЯ графиками (1 строка, 2 колонки)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Рисуем график для CPU (Левый) ---
    axes[0].scatter(p_cpu[:, 0].numpy(), p_cpu[:, 1].numpy(),
                    c=labels_cpu, cmap='tab10', s=5, alpha=0.5)
    axes[0].scatter(c_cpu[:, 0].numpy(), c_cpu[:, 1].numpy(),
                    c=c_labels_cpu, cmap='tab10', marker='*', s=200,
                    edgecolors='black', linewidths=1.5, label="Centroids")
    axes[0].set_title("K-Means Results (CPU)")
    axes[0].legend()

    # --- Рисуем график для GPU (Правый) ---
    axes[1].scatter(p_gpu[:, 0].numpy(), p_gpu[:, 1].numpy(),
                    c=labels_gpu, cmap='tab10', s=5, alpha=0.5)
    axes[1].scatter(c_gpu[:, 0].numpy(), c_gpu[:, 1].numpy(),
                    c=c_labels_gpu, cmap='tab10', marker='*', s=200,
                    edgecolors='black', linewidths=1.5, label="Centroids")
    axes[1].set_title("K-Means Results (GPU)")
    axes[1].legend()

    # Общий заголовок
    plt.suptitle("Comparison of K-Means Results: CPU vs GPU (PCA 2D Projection)", fontsize=16)

    # Автоматически выравниваем отступы, чтобы ничего не обрезалось
    plt.tight_layout()
    plt.savefig(name_output_file)
    plt.close()

def main():
    # Загружаем точки и центры CPU
    points_cpu, labels_cpu = load_data("../Data/cpu/kmeans_result.txt")
    centers_cpu, c_labels_cpu = load_data("../Data/cpu/kmeans_centers.txt")

    # Загружаем точки и центры GPU
    points_gpu, labels_gpu = load_data("../Data/gpu/kmeans_result.txt")
    centers_gpu, c_labels_gpu = load_data("../Data/gpu/kmeans_centers.txt")

    # Получаем проекции в ЕДИНОМ базисе координат, чтобы графики были сопоставимы
    p_cpu, c_cpu, p_gpu, c_gpu = get_pca_projections(points_cpu, centers_cpu, points_gpu, centers_gpu)

    # Строим один объединенный график
    make_combined_plot(
        p_cpu, labels_cpu, c_cpu, c_labels_cpu,
        p_gpu, labels_gpu, c_gpu, c_labels_gpu,
        name_output_file="kmeans_comparison.png"
    )

    print("Объединенный график успешно сохранен как kmeans_comparison.png")

if __name__ == "__main__":
    main()
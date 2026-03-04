import os
from matplotlib import pyplot as plt
import torch

# Пути к файлам (проверьте, что они совпадают с тем, что в C++)
PATHS = {
    "CPU": "../Data/cpu",
    "GPU (Basic)": "../Data/gpu",
    "GPU (Optimized)": "../Data/gpu_opt",
    "GPU (CUDA Stream)": "../Data/gpu_stream"
}

def load_data(folder_path):
    """Загружает точки и центры из указанной папки."""
    res_path = os.path.join(folder_path, "kmeans_result.txt")
    cen_path = os.path.join(folder_path, "kmeans_centers.txt")

    if not os.path.exists(res_path) or not os.path.exists(cen_path):
        print(f"Warning: Files not found in {folder_path}. Skipping.")
        return None, None, None, None

    def read_file(path):
        data = []
        labels = []
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                # Предполагаем, что последние данные - это ID кластера
                data.append([float(x) for x in parts[:-1]])
                labels.append(int(parts[-1]))
        return torch.tensor(data, dtype=torch.float32), labels

    points, p_labels = read_file(res_path)
    centers, c_labels = read_file(cen_path)

    return points, p_labels, centers, c_labels

def get_pca_projection_basis(points):
    """Вычисляет матрицу проекции PCA и среднее значение по эталонному набору данных."""
    mean = torch.mean(points, dim=0)
    centered_points = points - mean
    # q=2 для проекции в 2D
    _, _, V = torch.pca_lowrank(centered_points, q=2, center=False)
    return mean, V

def project_data(points, centers, mean, V):
    """Проецирует данные, используя заранее вычисленный базис."""
    p_proj = torch.matmul(points - mean, V)
    c_proj = torch.matmul(centers - mean, V)
    return p_proj, c_proj

def make_grid_plot(datasets, output_file="kmeans_comparison_full.png"):
    """Рисует сетку графиков."""

    num_plots = len(datasets)
    cols = 2
    rows = (num_plots + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
    axes = axes.flatten() # Чтобы удобно обращаться по индексу

    for i, (name, data) in enumerate(datasets.items()):
        ax = axes[i]
        p_proj, p_labels, c_proj, c_labels = data

        # Рисуем точки
        ax.scatter(p_proj[:, 0].numpy(), p_proj[:, 1].numpy(),
                   c=p_labels, cmap='tab10', s=5, alpha=0.3)

        # Рисуем центры
        ax.scatter(c_proj[:, 0].numpy(), c_proj[:, 1].numpy(),
                   c=c_labels, cmap='tab10', marker='*', s=300,
                   edgecolors='black', linewidths=2, label="Centroids")

        ax.set_title(f"Result: {name}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Если графиков нечетное количество, скрываем лишние оси
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Comparison of K-Means Implementations (PCA 2D Projection)", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Оставляем место под заголовок
    plt.savefig(output_file)
    print(f"График сохранен как {output_file}")
    plt.close()

def main():
    loaded_datasets = {}

    # 1. Сначала пытаемся загрузить CPU данные, чтобы использовать их как эталон для PCA
    cpu_path = PATHS.get("CPU")
    ref_points, ref_pl, ref_centers, ref_cl = load_data(cpu_path)

    if ref_points is None:
        print("Error: CPU data not found! Cannot calculate reference PCA.")
        # Если CPU нет, можно попробовать взять GPU Basic как эталон,
        # но лучше, чтобы пользователь сначала запустил CPU версию.
        return

    # Вычисляем PCA базис на основе CPU данных
    mean, V = get_pca_projection_basis(ref_points)

    # Проецируем CPU данные
    p_proj, c_proj = project_data(ref_points, ref_centers, mean, V)
    loaded_datasets["CPU"] = (p_proj, ref_pl, c_proj, ref_cl)

    # 2. Загружаем и проецируем остальные датасеты
    for name, path in PATHS.items():
        if name == "CPU": continue # Уже загрузили

        points, p_labels, centers, c_labels = load_data(path)

        if points is not None:
            # Важно! Используем mean и V от CPU, чтобы графики не были повернуты относительно друг друга
            p_proj, c_proj = project_data(points, centers, mean, V)
            loaded_datasets[name] = (p_proj, p_labels, c_proj, c_labels)
        else:
            print(f"Skipping {name} (data not found)")

    # 3. Рисуем
    if len(loaded_datasets) > 0:
        make_grid_plot(loaded_datasets)
    else:
        print("No datasets loaded.")

if __name__ == "__main__":
    main()
from matplotlib import pyplot as plt
import torch

def apply_pca(full_data: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(full_data, dim=0)
    centered_data = full_data - mean
    
    U, S, V = torch.pca_lowrank(centered_data, q=2)
    
    projected_data = torch.matmul(centered_data, V[:, :2])
    
    return projected_data

def load_data(path):
    data = []
    labels = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            data.append([float(x) for x in parts[:9]])
            labels.append(int(parts[9]))
    return torch.tensor(data), labels

def main():
    # Загружаем точки и центры
    points_data, points_labels = load_data("../Data/kmeans_result.txt")
    centers_data, centers_labels = load_data("../Data/kmeans_centers.txt")

    # Объединяем, чтобы PCA применился ко всему пространству одинаково
    all_data = torch.cat([points_data, centers_data], dim=0)
    all_projected = apply_pca(all_data)

    # Разделяем обратно
    projected_points = all_projected[:-len(centers_data)]
    projected_centers = all_projected[-len(centers_data):]

    # Рисуем точки
    plt.scatter(projected_points[:, 0], projected_points[:, 1], 
                c=points_labels, cmap='tab10', s=5, alpha=0.5)

    # Рисуем центры крупными звездами
    plt.scatter(projected_centers[:, 0], projected_centers[:, 1], 
                c=centers_labels, cmap='tab10', marker='*', s=200, 
                edgecolors='black', linewidths=1.5, label="Centroids")

    plt.title("K-Means Results with Centroids (PCA Projection)")
    plt.savefig("kmeans_final.png")

if __name__ == "__main__":
    main()
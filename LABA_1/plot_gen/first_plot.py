import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Теперь нам не нужен класс Pointer для вычислений, 
# так как мы работаем с матрицами PyTorch.

def get_data_files() -> list[Path]:
    directory = Path("../Data")
    return list(directory.glob("*.txt"))

def load_data_from_file(path: Path) -> torch.Tensor:
    """Загружает данные из файла в тензор PyTorch (9 колонок)."""
    data = []
    with open(path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 9:
                # Превращаем строку из 9 чисел в список float
                data.append([float(p) for p in parts[:9]])
    
    return torch.tensor(data, dtype=torch.float32)

def apply_pca(full_data: torch.Tensor) -> torch.Tensor:
    """Применяет PCA для понижения размерности с 9D до 2D."""
    print(f"Применяю PCA к данным размерностью {full_data.shape}...")
    
    # 1. Центрирование данных (вычитаем среднее по каждой колонке)
    mean = torch.mean(full_data, dim=0)
    centered_data = full_data - mean
    
    # 2. Быстрый алгоритм PCA (lowrank SVD)
    # q=2 — до какого количества измерений сжимаем
    U, S, V = torch.pca_lowrank(centered_data, q=2)
    
    # 3. Проекция данных на 2 главные компоненты
    projected_data = torch.matmul(centered_data, V[:, :2])
    
    return projected_data

def gen_plot(projected_groups: list[torch.Tensor], file_names: list[str]):
    """Рисует график, где каждый тензор в списке — это отдельный файл (цвет)."""
    plt.figure(figsize=(12, 8))
    
    for i, data_2d in enumerate(projected_groups):
        # data_2d имеет размер [N, 2]
        # Извлекаем колонку 0 как X, колонку 1 как Y
        x = data_2d[:, 0].numpy()
        y = data_2d[:, 1].numpy()
        
        plt.scatter(x, y, label=file_names[i], s=5, alpha=0.6)

    plt.title("PCA Projection (9D -> 2D) of K-Means Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("pca_scatter_plot.png", dpi=300) # dpi=300 для высокого качества
    print("График сохранен в pca_scatter_plot.png")

def main():
    data_files = get_data_files()
    if not data_files:
        print("Данные не найдены в папке ../Data")
        return

    all_tensors = []
    names = []
    lengths = [] # Запоминаем, сколько точек в каждом файле

    # 1. Загружаем все файлы
    for full_path in data_files:
        tensor = load_data_from_file(full_path)
        if tensor.shape[0] > 0:
            all_tensors.append(tensor)
            names.append(full_path.name)
            lengths.append(tensor.shape[0])
            print(f"Loaded {full_path.name}: {tensor.shape[0]} points")

    if not all_tensors:
        return

    # 2. Объединяем в одну матрицу для PCA
    # Размер будет [~80000, 9]
    full_data = torch.cat(all_tensors, dim=0)

    # 3. Выполняем PCA
    full_projected = apply_pca(full_data)

    # 4. Разрезаем общую матрицу обратно на части, чтобы вернуть цвета файлам
    projected_groups = []
    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length
        projected_groups.append(full_projected[start_idx:end_idx])
        start_idx = end_idx

    # 5. Рисуем
    gen_plot(projected_groups, names)

if __name__ == "__main__":
    main()
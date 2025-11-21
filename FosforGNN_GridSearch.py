import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import random
import numpy as np
import sys

#Logger Function
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

#Set Seed & Device
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)

# Hyperparameter Grid
embedding_size = [32,64,128]
epochs = [250,300,350]
k = [5,7,9,11,13]             #knn
patience = [20,30,40,50]
num_augments = [4,5,6,7,8,9]
lr = [0.001, 0.005, 0.01]
noise_scale = [0.05,0.075, 0.1]

# Create Grid
from itertools import product
grid = list(product(embedding_size, epochs, k, patience, num_augments, lr, noise_scale))

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Read the data and labels from the dataset
data_df = pd.read_excel('data.xlsx', header=None)  # shape: (bands, samples)
labels_df = pd.read_excel('labels.xlsx', header=None)  # shape: (samples, 1)

# Mutual Info Regression
# Transpose data so that each row becomes a sample
from sklearn.feature_selection import mutual_info_regression
import numpy as np

mi = mutual_info_regression(data_df.values.T, np.squeeze(labels_df.values), random_state=seed)
selected_indices = np.argsort(mi)[-300:]  # En yüksek bilgi içeriğine sahip k bandı
X_selected = data_df.values.T[:, selected_indices]

# Create Torch Tensors
X = torch.tensor(X_selected, dtype=torch.float32)
y = torch.tensor(labels_df.values / 1000000, dtype=torch.float32)

# Connect to the logger
sys.stdout = DualLogger("gnn_training_log.txt")

# --- Data Augmentation Function ---
def augment_data(X, y, num_augments, noise_scale):
    augmented_X = [X]
    augmented_y = [y]

    # Calculate std for each band
    std_per_band = X.std(dim=0, keepdim=True)  # [1, num_features]

    for _ in range(num_augments):
        noise = torch.randn_like(X) * std_per_band * noise_scale
        X_noisy = torch.clamp(X + noise, min=0.0)
        augmented_X.append(X_noisy)
        augmented_y.append(y)

    return torch.cat(augmented_X, dim=0), torch.cat(augmented_y, dim=0)


# 3. Normalization Function
def row_normalize(X):
    min_vals, _ = X.min(dim=1, keepdim=True)
    max_vals, _ = X.max(dim=1, keepdim=True)
    return (X - min_vals) / (max_vals - min_vals + 1e-8)


# --- 4. KNN Graph Builder Function ---
def build_knn_graph(features, k):
    num_nodes = features.size(0)
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i in range(num_nodes):
        distances = torch.cdist(features[i].unsqueeze(0), features)[0]
        _, indices = torch.topk(distances, k=k + 1, largest=False)
        for j in indices[1:]:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
    edge_list = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) if
                 adjacency_matrix[i, j] == 1]
    edge_index = torch.tensor(list(zip(*edge_list)), dtype=torch.long)
    return edge_index.contiguous()

# --- 5. GNN Model ---
class Net(torch.nn.Module):
    def __init__(self, num_features, embedding_size):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.linear = torch.nn.Linear(embedding_size, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.linear(x)
        return x

# GridSearch
for idx, (embedding_size, epochs, k, patience, num_augments, lr, noise_scale) in enumerate(grid):
    print(f"\n--- Grid Search {idx+1}/{len(grid)} ---")
    print(f"embedding_size={embedding_size}, epochs={epochs}, k={k}, patience={patience}, num_augments={num_augments}, lr={lr}, noise_scale={noise_scale}")

    # --- seed is equal for each combination ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    fold_val_r2s = []
    fold_test_r2s = []

    # KFOLD
    from sklearn.model_selection import KFold

    all_X = X
    all_y = y
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)

    for fold, (train_index, test_index) in enumerate(kf.split(all_X)):
        print(f"\n--- Fold {fold + 1} ---")

        X_train_fold = all_X[train_index]
        y_train_fold = all_y[train_index]
        X_test_fold = all_X[test_index]
        y_test_fold = all_y[test_index]

        X_train_aug, y_train_aug = augment_data(X_train_fold, y_train_fold, num_augments,
                                                noise_scale)  # Orijinal + 2x augment

        #Shuffle the data
        torch.manual_seed(seed)
        idx = torch.randperm(X_train_aug.size(0))
        X_selected = X_train_aug[idx]
        y_selected = y_train_aug[idx]
        X_train_aug = X_selected
        y_train_aug = y_selected

        # Train and Val Split
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(X_train_aug, y_train_aug, test_size=0.3, random_state=seed)

        # 3. Normalize rows of train, test and validation sets
        X_train_norm = row_normalize(X_train)
        X_test_norm = row_normalize(X_test_fold)
        X_val_norm = row_normalize(X_val)

        # NumPy conversion
        X_train_norm = X_train_norm.cpu().numpy()
        X_val_norm = X_val_norm.cpu().numpy()
        X_test_norm = X_test_norm.cpu().numpy()

        # Tensor conversion
        X_train_norm = torch.tensor(X_train_norm, dtype=torch.float32)
        X_val_norm = torch.tensor(X_val_norm, dtype=torch.float32)
        X_test_norm = torch.tensor(X_test_norm, dtype=torch.float32)

        # upload the tensor to the device
        X_train_norm = X_train_norm.to(device)
        X_val_norm = X_val_norm.to(device)
        X_test_norm = X_test_norm.to(device)

        y_train = y_train.to(device)
        y_val = y_val.to(device)
        y_test = y_test_fold.to(device)

        edge_index_train = build_knn_graph(X_train_norm, k)
        edge_index_val = build_knn_graph(X_val_norm, k)
        edge_index_test = build_knn_graph(X_test_norm, k)

        train_data = Data(x=X_train_norm.to(device), edge_index=edge_index_train.to(device), y=y_train.to(device))
        val_data = Data(x=X_val_norm.to(device), edge_index=edge_index_val.to(device), y=y_val.to(device))
        test_data = Data(x=X_test_norm.to(device), edge_index=edge_index_test.to(device), y=y_test.to(device))

        torch.manual_seed(seed)
        model = Net(X.size(1), embedding_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()

        train_losses = []
        val_losses = []

        # --- Training ---
        best_val_loss = float('inf')

        counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(train_data)
            loss = criterion(out, train_data.y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(val_data)
                val_loss = criterion(val_pred, val_data.y)
                val_losses.append(val_loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

            # Early Stopping (patience)
            if val_loss.item() < best_val_loss - 1e-5:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()  # En iyi modeli kaydet
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Reload the best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("Best model state loaded.")

        # --- Evaluation (R2) ---
        model.eval()

        with torch.no_grad():
            val_pred = model(val_data)

        test_pred = model(test_data)

        from sklearn.metrics import r2_score

        y_true = val_data.y.cpu().numpy()
        y_pred = val_pred.cpu().numpy()

        test_true = test_data.y.cpu().numpy()
        test_pred = test_pred.cpu().detach().numpy()

        # Calculate R²
        r2 = r2_score(y_true, y_pred)
        r2_t = r2_score(test_true, test_pred)
        print(f"Fold {fold + 1}: Val R² = {r2:.4f}, Test R² = {r2_t:.4f}")
        fold_val_r2s.append(r2)
        fold_test_r2s.append(r2_t)

        # Write the results to a file
        with open("grid_results_2.txt", "a") as f:
            f.write(f"embedding size={embedding_size}, epochs={epochs}, patience={patience}, k={k}, fold={fold + 1}, augments={num_augments}, lr={lr}, noise={noise_scale}, Validation R2={r2:.4f}, Test R2={r2_t:.4f}\n")


sys.stdout.log.close()


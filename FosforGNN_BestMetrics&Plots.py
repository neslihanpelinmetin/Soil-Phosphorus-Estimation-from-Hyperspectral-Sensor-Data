"""
FosforGNN_Metrics&Plots.py

GCN-based regression model for soil phosphorus prediction.

- Reads spectra (data.xlsx) and labels (labels.xlsx)
- Selects top 300 bands via mutual information
- Trains a 2-layer GCN with K-Fold CV (3 folds by default)
- Applies data augmentation with band-wise noise
- Computes evaluation metrics (R², RMSE, RPD) in mg kg^-1
- Plots:
    * Training & validation loss
    * Observed vs. predicted (validation & test)
    * Residuals vs. predicted (validation & test)
- Prints K-Fold mean metrics with approximate 95% confidence intervals
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

# Directory where plots will be saved
PLOT_DIR = "GNN_Plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Random seed for full reproducibility
SEED = 42

# Scale factor for phosphorus:
#   - labels are divided by P_SCALE during training
#   - metrics and plots are computed in mg kg^-1 by multiplying back
P_TRAIN_SCALE = 1_000_000.0   # labels are ~1e5; this keeps y in ~0–0.2 for training
P_REPORT_SCALE = P_TRAIN_SCALE / 1000.0  # = 1000.0 → converts to mg kg^-1 for metrics

# GNN hyperparameters 
EMBEDDING_SIZE = 128
EPOCHS = 350
PATIENCE = 50
K_NEIGHBORS = 5
NUM_AUGMENTS = 6
LR = 0.001
NOISE_SCALE = 0.1
NUM_BANDS = 300           # after band selection (fixed)
DROPOUT = 0.3

# Device selection (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def augment_data(X: torch.Tensor,
                 y: torch.Tensor,
                 num_augments: int,
                 noise_scale: float):
    """
    Create augmented copies of the data by adding Gaussian noise to each band.

    Noise per band is scaled by that band's standard deviation.
    Returns concatenated original + augmented (X, y).
    """
    augmented_X = [X]
    augmented_y = [y]

    # Compute band-wise standard deviation: shape [1, num_features]
    std_per_band = X.std(dim=0, keepdim=True)

    for _ in range(num_augments):
        noise = torch.randn_like(X) * std_per_band * noise_scale
        X_noisy = torch.clamp(X + noise, min=0.0)
        augmented_X.append(X_noisy)
        augmented_y.append(y)

    return torch.cat(augmented_X, dim=0), torch.cat(augmented_y, dim=0)


def row_normalize(X: torch.Tensor) -> torch.Tensor:
    """
    Normalize each sample (row-wise) to [0, 1] based on its min and max.

    This keeps the relative shape of each spectrum while scaling intensities.
    """
    min_vals, _ = X.min(dim=1, keepdim=True)
    max_vals, _ = X.max(dim=1, keepdim=True)
    return (X - min_vals) / (max_vals - min_vals + 1e-8)


def mean_and_ci(values, conf: float = 0.95):
    """
    Compute mean and approximate confidence interval for a list of values.

    Uses normal approximation: mean ± z * standard error,
    where z ≈ 1.96 for 95% confidence.
    """
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    if len(values) <= 1:
        # Not enough folds to compute a CI
        return mean, np.nan, np.nan

    se = values.std(ddof=1) / np.sqrt(len(values))
    z = 1.96  # ~95% CI for normal distribution
    h = z * se
    return mean, mean - h, mean + h


def build_knn_graph(features: torch.Tensor, k: int) -> torch.Tensor:
    """
    Construct a k-NN graph based on Euclidean distance in feature space.

    Each node connects to its k nearest neighbors, and the graph is symmetrized.
    Returns:
        edge_index: LongTensor of shape [2, num_edges].
    """
    num_nodes = features.size(0)
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # Compute distances and build adjacency
    for i in range(num_nodes):
        # distances: [num_nodes]
        distances = torch.cdist(features[i].unsqueeze(0), features)[0]
        _, indices = torch.topk(distances, k=k + 1, largest=False)  # include self
        for j in indices[1:]:  # skip self at index 0
            adjacency_matrix[i, j] = 1.0
            adjacency_matrix[j, i] = 1.0  # make graph undirected

    # Convert adjacency matrix to edge_index list
    edge_list = [
        (i, j)
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
        if adjacency_matrix[i, j] == 1
    ]

    if len(edge_list) == 0:
        # Fallback if something goes wrong
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(list(zip(*edge_list)), dtype=torch.long)

    return edge_index.contiguous()


class Net(torch.nn.Module):
    """
    Simple 2-layer GCN with dropout and a final linear head for regression.
    """

    def __init__(self, num_features: int, embedding_size: int):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.linear = torch.nn.Linear(embedding_size, 1)
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------
# Main training + evaluation logic
# ---------------------------------------------------------------------

def main():
    # Ensure reproducibility
    set_seed(SEED)

    # -----------------------------
    # 1. Load data from Excel files
    # -----------------------------
    # data_df: shape (bands, samples)  -> we will transpose
    data_df = pd.read_excel("data.xlsx", header=None)
    labels_df = pd.read_excel("labels.xlsx", header=None)  # shape: (samples, 1)

    # Transpose spectra so each row corresponds to one sample
    #   original: [bands, samples]
    #   after T : [samples, bands]
    spectra = data_df.values.T  # numpy array, shape: (n_samples, n_bands)
    labels = labels_df.values.ravel()  # shape: (n_samples,)

    print("Original spectra shape:", spectra.shape)
    print("Original labels shape:", labels.shape)

    # ----------------------------------------------------
    # 2. Band selection via mutual information (top 300)
    # ----------------------------------------------------
    mi = mutual_info_regression(spectra, labels, random_state=SEED)
    selected_indices = np.argsort(mi)[-300:]  # indices of top 300 informative bands

    X_selected = spectra[:, selected_indices]  # shape: (n_samples, 300)

    # Convert to torch tensors
    X = torch.tensor(X_selected, dtype=torch.float32)  # features in original scale
    # Scale phosphorus labels for training, but keep original values for later
    y = torch.tensor(labels_df.values / P_TRAIN_SCALE, dtype=torch.float32)  # shape: (n_samples, 1)

    all_X = X
    all_y = y

    set_seed(SEED)

    # -----------------------------
    # 3. K-Fold cross-validation
    # -----------------------------
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)

    # Lists for metrics across folds
    fold_val_r2s = []
    fold_test_r2s = []

    fold_val_rmses = []
    fold_test_rmses = []

    fold_val_rpds = []
    fold_test_rpds = []

    for fold, (train_index, test_index) in enumerate(kf.split(all_X)):
        print(f"\n--- Fold {fold + 1} ---")

        # Split into train/test for this fold (still in scaled P space)
        X_train_fold = all_X[train_index]
        y_train_fold = all_y[train_index]
        X_test_fold = all_X[test_index]
        y_test_fold = all_y[test_index]

        # ---------------------------------------
        # 3.1 Data augmentation on the train set
        # ---------------------------------------
        X_train_aug, y_train_aug = augment_data(
            X_train_fold, y_train_fold,
            num_augments=NUM_AUGMENTS,
            noise_scale=NOISE_SCALE
        )

        # Shuffle augmented data
        set_seed(SEED)  # reset seed for reproducible permutation
        idx = torch.randperm(X_train_aug.size(0))
        X_train_aug = X_train_aug[idx]
        y_train_aug = y_train_aug[idx]

        # --------------------------------------
        # 3.2 Train/validation split on training
        # --------------------------------------
        set_seed(SEED)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_aug, y_train_aug,
            test_size=0.3,
            random_state=SEED
        )

        # --------------------------------------
        # 3.3 Row-normalization (per spectrum)
        # --------------------------------------
        X_train_norm = row_normalize(X_train)
        X_val_norm = row_normalize(X_val)
        X_test_norm = row_normalize(X_test_fold)

        # Convert to numpy and back to Tensor (optional, but keeps it clean)
        X_train_norm = torch.tensor(X_train_norm.cpu().numpy(), dtype=torch.float32)
        X_val_norm = torch.tensor(X_val_norm.cpu().numpy(), dtype=torch.float32)
        X_test_norm = torch.tensor(X_test_norm.cpu().numpy(), dtype=torch.float32)

        # Move to device
        X_train_norm = X_train_norm.to(DEVICE)
        X_val_norm = X_val_norm.to(DEVICE)
        X_test_norm = X_test_norm.to(DEVICE)

        y_train = y_train.to(DEVICE)
        y_val = y_val.to(DEVICE)
        y_test = y_test_fold.to(DEVICE)

        # --------------------------------------
        # 3.4 Build k-NN graphs for each split
        # --------------------------------------
        edge_index_train = build_knn_graph(X_train_norm, k=K_NEIGHBORS)
        edge_index_val = build_knn_graph(X_val_norm, k=K_NEIGHBORS)
        edge_index_test = build_knn_graph(X_test_norm, k=K_NEIGHBORS)

        train_data = Data(
            x=X_train_norm,
            edge_index=edge_index_train.to(DEVICE),
            y=y_train
        )
        val_data = Data(
            x=X_val_norm,
            edge_index=edge_index_val.to(DEVICE),
            y=y_val
        )
        test_data = Data(
            x=X_test_norm,
            edge_index=edge_index_test.to(DEVICE),
            y=y_test
        )

        # --------------------------------------
        # 3.5 Model, optimizer, loss
        # --------------------------------------
        set_seed(SEED)
        model = Net(num_features=X.size(1), embedding_size=EMBEDDING_SIZE).to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=1e-4
        )
        criterion = torch.nn.MSELoss()

        train_losses = []
        val_losses = []

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        # --------------------------------------
        # 3.6 Training loop with early stopping
        # --------------------------------------
        for epoch in range(EPOCHS):
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
                print(
                    f"Epoch {epoch + 1:3d} | "
                    f"Train Loss: {loss.item():.6f} | "
                    f"Val Loss: {val_loss.item():.6f}"
                )

            # Early stopping based on validation loss
            if val_loss.item() < best_val_loss - 1e-5:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model state before evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("Best model state loaded for evaluation.")

        # ---------------------------------------------------------
        # 4. Plots and metrics (per fold)
        # ---------------------------------------------------------

        # 4.1 Training & validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE, scaled)")
        plt.title(f"Training and Validation Loss (Fold {fold + 1})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"fold_{fold + 1}_loss_plot.png"))
        plt.close()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_pred = model(val_data)

        # Convert to mg kg^-1 for plots and metrics
        y_val_true = val_data.y.detach().cpu().numpy().ravel() * P_REPORT_SCALE
        y_val_pred = val_pred.detach().cpu().numpy().ravel() * P_REPORT_SCALE

        # 4.2 Validation scatter: observed vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(
            y_val_true,
            y_val_pred,
            alpha=0.6,
            edgecolor="none",
            label="Samples"
        )
        min_val = min(y_val_true.min(), y_val_pred.min())
        max_val = max(y_val_true.max(), y_val_pred.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            label="1:1 line"
        )
        plt.xlabel("Observed P (mg kg$^{-1}$)")
        plt.ylabel("Predicted P (mg kg$^{-1}$)")
        plt.title(f"Validation: Observed vs Predicted P (Fold {fold + 1})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"fold_{fold + 1}_validation_scatter.png"))
        plt.close()

        # 4.3 Validation residuals
        residuals_val = y_val_true - y_val_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(
            y_val_pred,
            residuals_val,
            alpha=0.6,
            edgecolor="none",
            label="Residuals"
        )
        plt.axhline(0.0, color="k", linestyle="--", label="Zero error")
        plt.xlabel("Predicted P (mg kg$^{-1}$)")
        plt.ylabel("Residual (Observed - Predicted) [mg kg$^{-1}$]")
        plt.title(f"Validation: Residuals vs Predicted P (Fold {fold + 1})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"fold_{fold + 1}_validation_residuals.png"))
        plt.close()

        # 4.4 Test set: predictions and residuals
        with torch.no_grad():
            test_pred = model(test_data)

        y_test_true = test_data.y.detach().cpu().numpy().ravel() * P_REPORT_SCALE
        y_test_pred = test_pred.detach().cpu().numpy().ravel() * P_REPORT_SCALE

        # Scatter: observed vs predicted (test)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            y_test_true,
            y_test_pred,
            alpha=0.6,
            edgecolor="none",
            label="Samples"
        )
        min_val_t = min(y_test_true.min(), y_test_pred.min())
        max_val_t = max(y_test_true.max(), y_test_pred.max())
        plt.plot(
            [min_val_t, max_val_t],
            [min_val_t, max_val_t],
            "k--",
            label="1:1 line"
        )
        plt.xlabel("Observed P (mg kg$^{-1}$)")
        plt.ylabel("Predicted P (mg kg$^{-1}$)")
        plt.title(f"Test: Observed vs Predicted P (Fold {fold + 1})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"fold_{fold + 1}_test_scatter.png"))
        plt.close()

        # Residuals: test
        residuals_test = y_test_true - y_test_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(
            y_test_pred,
            residuals_test,
            alpha=0.6,
            edgecolor="none",
            label="Residuals"
        )
        plt.axhline(0.0, color="k", linestyle="--", label="Zero error")
        plt.xlabel("Predicted P (mg kg$^{-1}$)")
        plt.ylabel("Residual (Observed - Predicted) [mg kg$^{-1}$]")
        plt.title(f"Test: Residuals vs Predicted P (Fold {fold + 1})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"fold_{fold + 1}_test_residuals.png"))
        plt.close()

        # 4.5 Evaluation metrics in mg kg^-1
        r2_val = r2_score(y_val_true, y_val_pred)
        rmse_val_mg = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
        sd_val = np.std(y_val_true, ddof=1)
        rpd_val = sd_val / rmse_val_mg

        r2_test = r2_score(y_test_true, y_test_pred)
        rmse_test_mg = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
        sd_test = np.std(y_test_true, ddof=1)
        rpd_test = sd_test / rmse_test_mg

        rmse_val_g = rmse_val_mg / 1000.0
        rmse_test_g = rmse_test_mg / 1000.0

        print(
            f"Fold {fold + 1}: "
            f"Val -> R²={r2_val:.4f}, RMSE={rmse_val_mg:.4f} mg kg^-1 ({rmse_val_g:.3f} g kg^-1), RPD={rpd_val:.4f} | "
            f"Test -> R²={r2_test:.4f}, RMSE={rmse_test_mg:.4f} mg kg^-1 ({rmse_test_g:.3f} g kg^-1), RPD={rpd_test:.4f}"
        )

        # Store metrics for this fold
        fold_val_r2s.append(r2_val)
        fold_test_r2s.append(r2_test)

        fold_val_rmses.append(rmse_val_mg)
        fold_test_rmses.append(rmse_test_mg)

        fold_val_rpds.append(rpd_val)
        fold_test_rpds.append(rpd_test)

    # -----------------------------------------------------------------
    # 5. Print K-Fold average metrics with 95% confidence intervals
    # -----------------------------------------------------------------
    print("\n=== K-Fold average metrics with 95% CI ===")

    for name, vals in [
        ("Val R²", fold_val_r2s),
        ("Val RMSE (mg kg^-1)", fold_val_rmses),
        ("Val RPD", fold_val_rpds),
        ("Test R²", fold_test_r2s),
        ("Test RMSE (mg kg^-1)", fold_test_rmses),
        ("Test RPD", fold_test_rpds),
    ]:
        mean, lo, hi = mean_and_ci(vals)
        if np.isnan(lo):
            print(f"{name}: mean = {mean:.4f} (CI: n/a, only one fold)")
        else:
            print(f"{name}: mean = {mean:.4f}, 95% CI = ({lo:.4f}, {hi:.4f})")


if __name__ == "__main__":
    main()

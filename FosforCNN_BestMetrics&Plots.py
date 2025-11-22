"""
FosforCNN_Metrics&Plots.py

CNN-based regression model for soil phosphorus prediction.

- Reads spectra (data.xlsx) and labels (labels.xlsx)
- Selects top 300 bands via mutual information (same as GNN script)
- Trains a 1D CNN with K-Fold CV (3 folds by default)
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

# Directory where plots will be saved
PLOT_DIR = "CNN_Plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Random seed for reproducibility (up to GPU nondeterminism)
SEED = 42

# Scale factor for phosphorus:
#   - labels are divided by P_TRAIN_SCALE during training
#   - metrics and plots are computed in mg kg^-1 by multiplying back
P_TRAIN_SCALE = 1_000_000.0      # keep labels in ~0–0.2 range during training
P_REPORT_SCALE = P_TRAIN_SCALE   # convert back to original mg kg^-1

# CNN hyperparameters
EMBEDDING_SIZE = 128      # best embedding size from grid search
EPOCHS = 250
PATIENCE = 50
KERNEL_SIZE = 13           # best kernel size "k" from grid search
NUM_AUGMENTS = 7
LR = 0.01
NOISE_SCALE = 0.075
NUM_BANDS = 300           # after band selection (fixed)
DROPOUT = 0.3

# Device selection (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def set_seed(seed: int):
    """
    Set random seeds for reproducibility (Python, NumPy, PyTorch).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
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


# ---------------------------------------------------------------------
# CNN model definition (same structure as your grid search)
# ---------------------------------------------------------------------

class CNNRegressor(nn.Module):
    """
    1D CNN regressor for spectral data.

    Input:  [batch_size, num_bands]
    Output: [batch_size, 1] (scaled phosphorus)

    The "embedding_size" controls both the number of channels and
    the size of the latent representation before the final linear layer.
    """

    def __init__(self, num_bands: int, embedding_size: int, kernel_size: int, dropout: float = DROPOUT):
        super(CNNRegressor, self).__init__()

        # Same padding: padding = kernel_size // 2
        padding = kernel_size // 2

        # First conv: 1 -> embedding_size channels
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=embedding_size,
            kernel_size=kernel_size,
            padding=padding,
        )
        # Second conv: embedding_size -> embedding_size channels
        self.conv2 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=embedding_size,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Dropout between conv layers and before final linear head
        self.dropout = nn.Dropout(dropout)

        # Global average pooling across the spectral dimension
        # Input:  [N, C, L] -> Output: [N, C, 1]
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final fully connected layer: embedding_size -> 1
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, num_bands]

        Returns:
            out: [batch_size, 1] (scaled phosphorus prediction)
        """
        # Add channel dimension: [N, 1, L]
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.dropout(x)

        # Global average pooling: [N, C, L] -> [N, C, 1]
        x = self.global_pool(x)
        x = x.squeeze(-1)  # [N, C]

        # Linear layer: [N, embedding_size] -> [N, 1]
        out = self.fc(x)
        return out


# ---------------------------------------------------------------------
# Main training + evaluation logic (single hyperparameter config)
# ---------------------------------------------------------------------

def main():
    # Ensure reproducibility (up to GPU nondeterminism)
    set_seed(SEED)

    print("Device:", DEVICE)

    # -----------------------------
    # 1. Load data from Excel files
    # -----------------------------
    data_df = pd.read_excel("data.xlsx", header=None)   # shape: (bands, samples)
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
    selected_indices = np.argsort(mi)[-NUM_BANDS:]  # indices of top informative bands

    X_selected = spectra[:, selected_indices]  # shape: (n_samples, NUM_BANDS)

    # Convert to torch tensors
    X = torch.tensor(X_selected, dtype=torch.float32)  # features in original scale
    # Scale phosphorus labels for training
    y = torch.tensor(labels_df.values / P_TRAIN_SCALE, dtype=torch.float32)  # shape: (n_samples, 1)

    print("Selected X shape:", X.shape)
    print("Scaled y shape:", y.shape)

    all_X = X
    all_y = y

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
        print(f"\n=== Fold {fold + 1} / 3 ===")

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
            noise_scale=NOISE_SCALE,
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
            random_state=SEED,
        )

        # --------------------------------------
        # 3.3 Row-normalization (per spectrum)
        # --------------------------------------
        X_train_norm = row_normalize(X_train)
        X_val_norm = row_normalize(X_val)
        X_test_norm = row_normalize(X_test_fold)

        # Convert to numpy and back to Tensor (optional, keeps it clean)
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
        # 3.4 Model, optimizer, loss
        # --------------------------------------
        set_seed(SEED)
        model = CNNRegressor(
            num_bands=NUM_BANDS,
            embedding_size=EMBEDDING_SIZE,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        # --------------------------------------
        # 3.5 Training loop with early stopping
        # --------------------------------------
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()

            out = model(X_train_norm)          # [N_train, 1] (scaled)
            loss = criterion(out, y_train)     # both in scaled space
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_norm)
                val_loss = criterion(val_pred, y_val)
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
            val_pred = model(X_val_norm)

        # Convert to mg kg^-1 for plots and metrics
        y_val_true = y_val.detach().cpu().numpy().ravel() * P_REPORT_SCALE
        y_val_pred = val_pred.detach().cpu().numpy().ravel() * P_REPORT_SCALE

        # 4.2 Validation scatter: observed vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(
            y_val_true,
            y_val_pred,
            alpha=0.6,
            edgecolor="none",
            label="Samples",
        )
        min_val = min(y_val_true.min(), y_val_pred.min())
        max_val = max(y_val_true.max(), y_val_pred.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            label="1:1 line",
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
            label="Residuals",
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
            test_pred = model(X_test_norm)

        y_test_true = y_test.detach().cpu().numpy().ravel() * P_REPORT_SCALE
        y_test_pred = test_pred.detach().cpu().numpy().ravel() * P_REPORT_SCALE

        # Scatter: observed vs predicted (test)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            y_test_true,
            y_test_pred,
            alpha=0.6,
            edgecolor="none",
            label="Samples",
        )
        min_val_t = min(y_test_true.min(), y_test_pred.min())
        max_val_t = max(y_test_true.max(), y_test_pred.max())
        plt.plot(
            [min_val_t, max_val_t],
            [min_val_t, max_val_t],
            "k--",
            label="1:1 line",
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
            label="Residuals",
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
            f"Val -> R²={r2_val:.4f}, RMSE={rmse_val_mg:.2f} mg kg^-1, RPD={rpd_val:.4f} | "
            f"Test -> R²={r2_test:.4f}, RMSE={rmse_test_mg:.2f} mg kg^-1, RPD={rpd_test:.4f}"
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
    print("\n=== CNN – K-Fold average metrics with 95% CI ===")

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
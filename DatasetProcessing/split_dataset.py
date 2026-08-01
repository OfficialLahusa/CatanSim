import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

def split_dataset(state_mat_path: Path, win_rate_mat_path: Path, group_vec_path: Path, output_dir: Path, test_proportion: float, val_proportion: float, seed: int):
    # Arrays aus Dateien laden
    print("Loading arrays...")
    state_matrix = np.load(state_mat_path)
    win_rate_matrix = np.load(win_rate_mat_path)
    groups_vector = np.load(group_vec_path)

    print(f"State matrix: shape {state_matrix.shape}, dtype {state_matrix.dtype}")
    print(f"Win rate matrix: shape {win_rate_matrix.shape}, dtype {win_rate_matrix.dtype}")
    print(f"Group label vector: shape {groups_vector.shape}, dtype {groups_vector.dtype}")

    # Group Split TrainVal/Test
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_proportion, random_state=seed)
    trainval_idx, test_idx = next(splitter.split(np.arange(groups_vector.shape[0]), groups=groups_vector))

    # Group Split Train/Val
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_proportion*(1/(1-test_proportion)), random_state=seed)
    train_idx, val_idx = next(splitter.split(trainval_idx, groups=groups_vector[trainval_idx]))
    train_idx, val_idx = trainval_idx[train_idx], trainval_idx[val_idx]

    print(f"Split sizes: train {train_idx.shape}, val {val_idx.shape}, test {test_idx.shape}")

    # Verify no overlaps
    print("Verifying integrity...")
    assert set(train_idx) & set(val_idx)  == set()
    assert set(train_idx) & set(test_idx) == set()
    assert set(val_idx)   & set(test_idx) == set()

    # Verify group cohesion
    assert set(groups_vector[train_idx]) & set(groups_vector[val_idx])  == set()
    assert set(groups_vector[train_idx]) & set(groups_vector[test_idx]) == set()
    assert set(groups_vector[val_idx])   & set(groups_vector[test_idx]) == set()
    print("Integrity verified!")

    # X_train, X_val, X_test, Y_train, Y_val, Y_test als Dateien speichern
    X_train = state_matrix[train_idx]
    X_val = state_matrix[val_idx]
    X_test = state_matrix[test_idx]

    y_train = win_rate_matrix[train_idx]
    y_val = win_rate_matrix[val_idx]
    y_test = win_rate_matrix[test_idx]

    #print(X_train.shape, X_val.shape, X_test.shape)
    #print(y_train.shape, y_val.shape, y_test.shape)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy",   X_train)
    np.save(output_dir / "X_val.npy",     X_val)
    np.save(output_dir / "X_test.npy",    X_test)
    np.save(output_dir / "y_train.npy",   y_train)
    np.save(output_dir / "y_val.npy",     y_val)
    np.save(output_dir / "y_test.npy",    y_test)
    np.save(output_dir / "idx_train.npy", train_idx)
    np.save(output_dir / "idx_val.npy",   val_idx)
    np.save(output_dir / "idx_test.npy",  test_idx)
    np.save(output_dir / "groups_train.npy", groups_vector[train_idx])
    np.save(output_dir / "groups_val.npy",   groups_vector[val_idx])
    np.save(output_dir / "groups_test.npy",  groups_vector[test_idx])
    print(f"Saved dataset splits to \"{output_dir}\".")
    return

if __name__ == "__main__":
    # Pfade zu numpy Array Dateien einlesen mit guten Defaults
    default_state_mat_path = Path("states.npy")
    state_mat_path = input(f"Enter the path of the game state matrix numpy array file. Default: \"{default_state_mat_path}\"\n").strip()
    state_mat_path = Path(state_mat_path) if state_mat_path else default_state_mat_path
    #print(state_mat_path)

    default_win_rate_mat_path = Path("win_rates.npy")
    win_rate_mat_path = input(f"Enter the path of the win rate matrix numpy array file. Default: \"{default_win_rate_mat_path}\"\n").strip()
    win_rate_mat_path = Path(win_rate_mat_path) if win_rate_mat_path else default_win_rate_mat_path
    #print(win_rate_mat_path)

    default_group_vec_path = Path("groups.npy")
    group_vec_path = input(f"Enter the path of the group label vector numpy array file. Default: \"{default_group_vec_path}\"\n").strip()
    group_vec_path = Path(group_vec_path) if group_vec_path else default_group_vec_path
    #print(group_vec_path)

    # Split proportions einlesen
    default_test_split_proportion = 0.15
    test_split_proportion = input(f"Enter test split proportion of total. Default: {default_test_split_proportion}\n").strip()
    test_split_proportion = float(test_split_proportion) if test_split_proportion else default_test_split_proportion

    default_val_split_proportion = 0.15
    val_split_proportion = input(f"Enter val split proportion of total. Default: {default_val_split_proportion}\n").strip()
    val_split_proportion = float(val_split_proportion) if val_split_proportion else default_val_split_proportion

    # Seed einlesen mit default
    default_seed = 42
    seed = input(f"Enter the seed for the split. Default: {default_seed}\n").strip()
    seed = int(seed) if seed else default_seed

    default_output_dir = Path(".")
    output_dir = input(f"Enter the output directory. Default: \"{default_output_dir}\"\n").strip()
    output_dir = Path(output_dir) if output_dir else default_output_dir
    
    split_dataset(state_mat_path, win_rate_mat_path, group_vec_path, output_dir, test_split_proportion, val_split_proportion, seed)
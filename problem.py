import os
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
import scanpy as sc

problem_title = "Single-cell RNA-seq cell types classification"
_target_attr_name = "standard_true_celltype_v5"
_prediction_label_names = [
    "Cancer_cells",
    "NK_cells",
    "T_cells_CD4+",
    "T_cells_CD8+",
]
# sanity check
_prediction_label_names.sort()

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names
)
# An object implementing the workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.Accuracy(name="acc"),
]


def get_cv(X, y):
    val_size = 0.3
    cv = StratifiedShuffleSplit(
        n_splits=5, test_size=val_size, random_state=57
    )
    return cv.split(X, y)


def _read_data(path, f_name):
    anndata = sc.read_h5ad(os.path.join(path, "data", f_name))
    y = anndata.obs[_target_attr_name].values
    X_sparse = anndata.X

    # only uses 1000 point for test mode, to accelerate computations
    if os.getenv("RAMP_TEST_MODE", 0):
        import numpy as np

        n_test_mode = 100
        quick_test_idx = np.random.permutation(range(X_sparse.shape[0]))[
            :n_test_mode
        ]
        X_sparse, y = X_sparse[quick_test_idx, :], y[quick_test_idx]
    return X_sparse, y


def get_train_data(path="."):
    f_name = "train/train.h5ad"
    return _read_data(path, f_name)


def get_test_data(path="."):
    f_name = "test/test.h5ad"
    return _read_data(path, f_name)


if __name__ == "__main__":
    import rampwf

    os.environ["RAMP_TEST_MODE"] = "1"
    rampwf.utils.testing.assert_submission()

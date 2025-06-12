import numpy as np

from roidims.bcv import BiCrossValidation
from roidims.utils import SubjectLoader

# ---------------------- Estimate optimal dimensionality --------------------- #
def run_bcv(subjects: list, rois: list, k_min: int, k_max: int, k_steps: int, n_perms: int):
    """Estimate optimal dimensionality using bi-cross-validation."""
    for subject in subjects:
        sub = SubjectLoader(subject)
        ks = np.arange(k_min, k_max+k_steps, k_steps)
        bcv = BiCrossValidation(ks, n_perms)

        for roi in rois:
            # Fit BCV
            V = sub.load_resp(roi, set="train")
            bcv.fit_parallel(V, joblib_kwargs={"n_jobs": 52, "verbose": 10})
            results = bcv.get_params()

            # Save results
            metrics = {
                name: (np.array(list(vals.values()))
                        if isinstance(vals, dict)
                        else np.asarray(vals))
                for name, vals in results.items()
            }
            metrics["ks"] = ks
            metrics["k_optim_subj"] = ks[np.argmin(metrics["test_errs"])]
            np.savez(sub.bcv_dir / f"bcv_metrics_{roi}.npz", **metrics)
            print(f"{subject} {roi} done")

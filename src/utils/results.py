import pandas as pd


def save_and_update_results(
    i,
    results,
    target_class,
    pred_class,
    t_type,
    magnitude,
    t_pred_class,
    ssim,
    spearman
):
    results.append([
        i, 
        target_class,
        pred_class,
        t_type,
        float(magnitude),
        t_pred_class,
        ssim,
        spearman
    ])
    df = pd.DataFrame(results, columns=[
        "test_idx",
        "target_class",
        "pred_class",
        "transform_type",
        "tranform_magnitude",
        "t_pred_class",
        "ssim",
        "spearman"
    ])
    return results, df
# SGU-Net
This is a project that explore the SGU-Net in thyroid ultrasound image segmentation.

1. First, execute `preprocess_data.py` under the segmentation folder to split the training and validation sets. Copy the generated datasets directly to `seg_tasks` for subsequent use.
2. `main.py` is used for training.
3. `show_results.py` is used to generate detection results. Both the generated ground truth (gt) and predictions (pred) are saved in `val_preds`.
4. `calculate_metric.py` is used to calculate metrics, which are saved in `evaluate_res.txt`.

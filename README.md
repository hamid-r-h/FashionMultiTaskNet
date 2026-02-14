# Fashion Multi-TaskNet — Multi-Task Fashion Attribute Classification (Notebook Project)

## 1) Project Overview and Problem Statement
This project tackles **multi-task image classification** for fashion product images. The goal is to train **one shared visual backbone** that predicts **multiple attributes at the same time** (multi-head classification). Depending on your dataset and `label_cols`, these tasks can include (examples):
- Product type / category
- Color
- Gender / target group
- Season, style, material, etc.

Multi-task learning helps the model learn **shared representations** across tasks, which often improves performance and reduces the need to maintain separate models.

---

## 2) Project Structure and How to Run
This is a **Notebook-based project**. You do **not** need a structured Python package layout to run it.

**Main file:**
- `fashion_multitasknet.ipynb` → full workflow: data loading, preprocessing, model definition, training, evaluation, and model comparisons.

### How to run
1. Install dependencies (see **Requirements** section).
2. Download / prepare the dataset (see **Input Data** section).
3. Update dataset paths inside the notebook.
4. Run the notebook from top to bottom (**Run All**).

---

## 3) Input Data Description
Your dataset typically contains:
- A tabular file (CSV/Parquet) with:
  - image path / image filename column
  - multiple label columns (tasks) — these are your `label_cols`
- A folder containing product images (JPG/PNG)

### Expected format (high level)
- **Images:** `.jpg` / `.png`
- **Labels:** categorical values per task (e.g., strings like “Dress”, “Red”) which are encoded to integers.

> Note: In your notebook, `label_cols` determines which columns are treated as tasks.

---

## 4) Preprocessing Steps
The notebook typically includes these preprocessing steps:

1. **Data cleaning**
   - remove invalid image paths / missing images
   - handle missing/invalid labels

2. **Label encoding**
   - convert categorical labels to numeric IDs for each task
   - build `num_classes_dict` mapping each task → number of classes

3. **Dataset split**
   - train / validation / test splitting

4. **Image preprocessing**
   - resize and normalization
   - optional augmentation transforms for training

---

## 5) Models and Architectures
This project uses a **shared backbone** + **multiple task-specific heads**.

Supported backbones (as used in your notebook):
- `convnext_tiny`
- `resnet50`
- `vit_b_16`
- `efficientnet_b5`

### Architecture idea
- The backbone extracts image features.
- Each task has its own classification head.
- The model outputs a dictionary of logits: one tensor per task.

---

## 6) Training and Evaluation Commands (Notebook Cells)
### Train & evaluate a single model
```python
result = train_and_evaluate_model(
    model_name="convnext_tiny",
    num_classes_dict=num_classes_dict,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    criterions=criterions,
    label_cols=label_cols,
    epochs=20,
    learning_rate=0.0001,
    extra_plots=True,          # enables Confusion Matrix + ROC + avg metric plots
    save_checkpoints=True      # saves checkpoints during training
)
```

### Compare multiple models (your comparison workflow cell)
```python
analysis_results = run_complete_model_analysis(
    num_classes_dict=num_classes_dict,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    criterions=criterions,
    label_cols=label_cols,
    epochs=7,
    learning_rate=0.001
)
```

### Evaluation outputs
- Per-task: Accuracy and Macro F1
- If `extra_plots=True`:
  - Confusion Matrix per task
  - ROC curve per task
  - Avg Accuracy / Avg F1-macro / Avg Precision / Avg Recall plots across epochs

---

## 7) Example Notebook Execution
Recommended steps:
1. Install requirements
2. Set dataset paths (CSV + image folder)
3. Run:
   - data loading
   - preprocessing + encoding + split
   - dataloaders
   - model and training/evaluation definitions
4. Choose one:
   - single-model training (deep evaluation + plots)
   - multi-model comparison workflow

---

## 8) Where Hyperparameter Tuning and Regularization Are Used
You implemented improvements in training strategy and regularization.

### Hyperparameter Tuning / Training Strategy (Where it’s used)
- **Differential learning rates** for backbone vs heads  
  (lower LR for pretrained backbone, higher LR for new heads)
- **Learning-rate scheduling** with `OneCycleLR`
- **Optimizer choice** (e.g., AdamW in tuned mode)
- Tuned values like `epochs` and `learning_rate`

### Regularization / Stabilization (Where it’s used)
- **Weight decay** (AdamW `weight_decay=1e-4`)
- **Gradient clipping** (`clip_grad_norm_` with `max_norm=1.0`)

---

## 9) Checkpoint Saving
If `save_checkpoints=True`, checkpoints are saved during training:

- `checkpoints/<model_name>_tuned_best.pt`  → best checkpoint (lowest validation loss)
- `checkpoints/<model_name>_tuned_last.pt`  → last epoch checkpoint

And for baseline training (no tuning/regularization):
- `checkpoints/<model_name>_baseline_best.pt`
- `checkpoints/<model_name>_baseline_last.pt`

---

## 10) Notebook-Only Requirements (requirements.txt)
Create a `requirements.txt` with:

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
Pillow
opencv-python
torch
torchvision
timm
```

Install:
```bash
pip install -r requirements.txt
```

---

## 11) Notes and Tips
- Use GPU if available: set `device="cuda"`.
- Make sure `label_cols` exactly matches your dataset label columns.
- Build `num_classes_dict` after label encoding.
- To keep your model-comparison cell output unchanged:
  - do not enable extra plotting inside the comparison workflow
  - use `extra_plots=True` only for single-model deep evaluation runs

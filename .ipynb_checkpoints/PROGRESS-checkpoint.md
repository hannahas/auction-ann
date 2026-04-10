# Project Progress

## Goal
Build a PyTorch ANN to predict shill (fraudulent) bidding behavior in eBay auctions,
using the UCI Shill Bidding Dataset. The project connects to game theory via mechanism
design — shill bidding is a mechanism violation that the ANN learns to detect from
behavioral signatures alone.

## Environment
- Python 3.11.9 (via pyenv)
- PyTorch 2.11.0
- Jupyter notebook kernel: "Python (auction-ann)"
- venv located at: `~/Desktop/auction-ann/venv`

## Activate environment
```bash
cd ~/Desktop/auction-ann
source venv/bin/activate
jupyter notebook
```
Then open `notebooks/auction_ann.ipynb`.

## Completed
- [x] Repo created and pushed to GitHub (https://github.com/hannahas/auction-ann)
- [x] Python 3.11.9 installed via pyenv
- [x] venv created with all dependencies
- [x] requirements.txt committed
- [x] Cell 1: imports and setup ✓

## Next session — pick up here
### Cell 2: Load the dataset
Fetch the UCI Shill Bidding Dataset directly from UCI. Add a synthetic
fallback in case the URL is unavailable. Print shape and show head().

URL to fetch:
https://archive.ics.uci.edu/ml/machine-learning-databases/00562/Shill%20Bidding%20Dataset.csv

Expected columns:
- Record_ID, Auction_ID, Bidder_ID (drop these — identifiers, not features)
- Bidder_Tendency, Bidding_Ratio, Successive_Outbidding, Last_Bidding
- Auction_Bids, Auction_Duration, Winning_Ratio
- Class (target: 0 = normal, 1 = shill)

### Cell 3: EDA
- Class balance check
- Histogram of each feature split by Class (normal vs shill)
- Correlation heatmap

### Cell 4: Preprocessing
- Drop identifier columns
- StandardScaler on features (fit on train only)
- 70/15/15 train/val/test split with stratify=y
- Convert to PyTorch TensorDatasets and DataLoaders (batch_size=64)

### Cell 5: Model architecture
- Class AuctionNet(nn.Module)
- Layers: 7 -> 64 -> 32 -> 16 -> 1
- ReLU activations, BatchNorm, Dropout(0.3), Sigmoid output
- Print model summary and parameter count

### Cell 6: Training loop
- Loss: BCELoss
- Optimizer: Adam(lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau(patience=5, factor=0.5)
- Early stopping: patience=10, restore best weights
- Track train/val loss and accuracy per epoch

### Cell 7: Training curves
- Side by side: loss curve and accuracy curve
- Train vs validation on each

### Cell 8: Evaluation
- Run on test set
- ROC-AUC score
- classification_report (precision, recall, F1)
- Confusion matrix heatmap + ROC curve side by side

### Cell 9: Feature importance
- Permutation importance: shuffle each feature, measure accuracy drop
- Horizontal bar chart ranked by importance
- Interpret top features through game theory lens

### Cell 10: Predicted probability distribution
- Histogram of model output probabilities split by true class
- Shows model calibration and decision boundary

### Cell 11: Markdown writeup cell
- Game theory interpretation of results
- Connect feature importances back to mechanism design and shill bidding strategy

### Cell 12: Save model
- torch.save checkpoint with model weights, scaler params, feature list

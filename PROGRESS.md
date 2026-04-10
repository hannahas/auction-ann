# Project Progress

## Status: COMPLETE

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

## Completed cells
- [x] Cell 1: Imports, working directory, reproducibility seed
- [x] Cell 2: Load UCI Shill Bidding Dataset (synthetic fallback if unavailable)
- [x] Cell 3a: Data dictionary and summary statistics
- [x] Cell 3b: Feature histograms (normal vs shill) and correlation heatmap
- [x] Cell 4: Preprocessing — StandardScaler, 70/15/15 split, DataLoaders
- [x] Cell 5: AuctionNet architecture (7->64->32->16->1), 3,329 parameters
- [x] Cell 6: Training loop — BCELoss, Adam, ReduceLROnPlateau, early stopping
- [x] Cell 7: Training curves — loss and accuracy, linear and log scale
- [x] Cell 8: Evaluation — ROC-AUC 0.9999, confusion matrix, classification report
- [x] Cell 9: Permutation feature importance
- [x] Cell 10: Predicted probability distributions (linear + log scale)
- [x] Cell 11: Game theory interpretation (markdown writeup)
- [x] Cell 12: Save model checkpoint to models/auction_ann.pt

## Key results
- Test accuracy: 99.79%
- ROC-AUC: 0.9999
- Top features: Bidder_Tendency (16.7% drop), Successive_Outbidding (13.7% drop)
- Early stopping at epoch 17

## Next steps
- Validate on real UCI dataset when server is available
- Add logistic regression baseline for comparison
- Explore SHAP values for per-prediction attribution
- Test adversarial robustness

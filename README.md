# Apple Stock Log-Return Forecasting with RNNs
A deep learning research project implementing and benchmarking different Recurrent Neural Network architectures for forecasting Apple (AAPL) stock log-returns. The project focuses on temporal feature engineering, training strategies, and financial-specific evaluation metrics.

## Features
- **8 Custom & Classic Architectures**: `Deep ReLU`, `BN Leaky`, `ELU Dropout`, `Wide`, `Residual RNN`, `Classic RNN`, `LSTM`, `GRU`
- **Advanced Feature Engineering**: Lag features, rolling statistics (MA7, MA21, Volatility5), log-returns, cyclical time encoding (sin/cos for day/month)
- **Robust Training Loop**: Huber Loss, AdamW optimizer, `ReduceLROnPlateau` scheduler, gradient clipping, early stopping, best-weights checkpointing
- **Financial Metrics**: MAE, Directional Accuracy, Cumulative R²
- **Comprehensive Visualization**: Correlation matrices, loss curves, actual vs predicted scatter, cumulative returns comparison

## Dataset
- **Source**: Historical Apple (AAPL) stock prices (Dec 1980 – Jun 2022)
- **Columns**: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- **Target Variable**: Next-day log-return: `log(Adj_Close_t+1 / Adj_Close_t)`
- **Preprocessing**: Chronological split (70/15/15), MinMax scaling, sequence windowing, batched loading (`batch_size=64`)

## Architecture Highlights
| Model | Key Characteristics | Parameters |
|-------|---------------------|------------|
| `Deep_ReLU` | 2-layer RNN, ReLU, Dropout(0.2) | ~16.1k |
| `BN_Leaky` | 1-layer RNN, BatchNorm, LeakyReLU | ~5.8k |
| `ELU_Dropout` | 1-layer RNN, ELU, Dropout(0.3) | ~5.7k |
| `Wide` | 1-layer RNN, hidden_dim=128 | ~19.6k |
| `Residual` | Skip connection from input to hidden state | ~7.2k |
| `Classic RNN` | Standard PyTorch RNN cell | ~5.7k |
| `LSTM` | Standard PyTorch LSTM cell with gating | ~22.6k |
| `GRU` | Standard PyTorch GRU cell with gating | ~17.0k |

## Results
| Model | Test MAE | Directional Accuracy | Cumulative R² |
|-------|----------|----------------------|---------------|
| Deep_ReLU | 0.0128 | 53.6% | 0.9213 |
| ELU_Dropout | 0.0144 | 46.4% | 0.3780 |
| BN_Leaky | 0.0146 | 46.2% | 0.3381 |
| GRU_Std | 0.0149 | 47.8% | -11.6206 |
| LSTM_Std | 0.0192 | 53.3% | -11.2200 |
| Wide | 0.0203 | 51.0% | -13.7584 |
| Classic RNN | 0.0237 | 48.0% | -140.3914 |
| Residual | 0.0324 | 51.5% | -207.5123 |

**Best Performer**: `Deep_ReLU` achieved `Cumulative R² = 0.9213`, `MAE = 0.0128`, and `Directional Accuracy = 53.6%`.

## Technologies

- **Framework:** PyTorch
- **Language:** Python 3.11
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt

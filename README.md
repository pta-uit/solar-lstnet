## Solar Generation Forecasting with LSTNet

### Mainly referenced paper
Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks.(https://arxiv.org/abs/1703.07015)

### Dataset
NeurIPS 2022: CityLearn Challenge ([starter-kit-team-Together](https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge/citylearn-2022-starter-kit-team-together/-/tree/master/data/citylearn_challenge_2022_phase_1?ref_type=heads)); Building_1.csv & weather.csv

### Requirements

```
torch==1.9.0
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
statsmodels==0.12.2
matplotlib==3.4.2
```

### Colab usage

Clone this repo:
```
!git clone https://github.com/quocanuit/lstnet-solar-gen.git
```
Data preprocessing stage:
```
!python preprocess_data.py --weather_data /your_path/weather.csv --building_data /your_path/Building_1.csv --output preprocessed_data.pkl
```
Training stage:
```
!python main.py --batch_size 32 --preprocessed_data preprocessed_data.pkl --save model.pt --loss_history loss_history.json
```
#### Optional arguments for training:
|--||
|-|-|
| `--gpu` | GPU to use (default: -1, i.e., CPU) |
| `--save` | Path to save the model (default: 'model.pt') |
| `--window` | Window size (default: 168) |
| `--horizon` | Forecasting horizon (default: 24) |
| `--hidRNN` | Number of RNN hidden units (default: 100) |
| `--hidCNN` | Number of CNN hidden units (default: 100) |
| `--hidSkip` | Number of skip RNN hidden units (default: 5) |
| `--CNN_kernel` | CNN kernel size (default: 6) |
| `--skip` | Skip length (default: 24) |
| `--highway_window` | Highway window size (default: 24) |
| `--dropout` | Dropout rate (default: 0.2) |
| `--output_fun` | Output function: sigmoid, tanh or None (default: sigmoid) |
| `--epochs` | Number of epochs (default: 100) |
| `--batch_size` | Batch size (default: 128) |
| `--lr` | Learning rate (default: 0.001) |
| `--loss_history` | Path to save the loss history (default: None, i.e., do not save) |

### Evaluating the model:
```
!python eval.py --model model.pt --preprocessed_data preprocessed_data.pkl --loss_history loss_history.json
```

#### Plot first sample:

![Forecast-plot](https://i.imgur.com/HHENEZd.png)
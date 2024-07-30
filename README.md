## Deep Learning for Solar Generation forecasting

### Mainly referenced paper
Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks.(https://arxiv.org/abs/1703.07015)

### Dataset
NeurIPS 2022: CityLearn Challenge ([starter-kit-team-Together](https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge/citylearn-2022-starter-kit-team-together/-/tree/master/data/citylearn_challenge_2022_phase_1?ref_type=heads)); Building_1.csv & weather.csv

### Colab usage
```
!git clone https://github.com/quocanuit/lstnet-solar-gen.git
```
```
!python /repo_path/main.py --gpu 0 --weather_data /your_path/weather.csv --building_data /your_path/Building_1.csv --save /your_path/model.pt --window 168 --horizon 24 --hidRNN 100 --hidCNN 100 --hidSkip 5 --CNN_kernel 6 --skip 24 --highway_window 24 --epochs 100 --batch_size 128 --lr 0.001 --gpu -1
```
#### args:
|--||
|-|-|
| --window | Input sequence length (168 hours = 7 days) |
| --horizon | Prediction horizon (24 hours = 1 day) |
| --hidRNN | Number of hidden units in the RNN |
| --hidCNN | Number of filters in the CNN |
| --hidSkip | Number of hidden units in the skip RNN |
| --CNN_kernel | Size of the CNN kernel |
| --skip | Skip length |
| --highway_window | Size of the highway window |
| --epochs | Number of training epochs |
| --batch_size | Batch size for training |
| --lr | Learning rate |
| --gpu | GPU to use (-1 means use CPU) |
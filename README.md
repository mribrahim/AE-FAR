# Time Series Anomaly Detection Leveraging MSE Feedback with AutoEncoder and RNN
[Paper link](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.TIME.2024.17)

The CompositeModel architecture (named as **AE-FAR** or **VAE-FAR**) includes a combination of an Autoencoder (AE) or Variational Autoencoder (VAE) and MSE Feedback with Attention and RNN(FAR). This architecture leverages error feedback to enhance model performance through a two-phase training process.

## Training Phases
The training is structured in two main phases:

- Pretraining Phase: The AE or VAE is trained independently to learn a compressed representation of the input data. This pretraining establishes a robust initialization for the CompositeModel.

- CompositeModel Training Phase (AE-FAR/VAE-FAR): The pretrained AE or VAE model is integrated into the CompositeModel with the FAR module. This phase refines the model to incorporate error feedback, enhancing its predictive capability.

# Main Results
Performance comparison of different anomaly detection methods in the literature.


![alt text](images/result.png)


# Description

- models folder contain pretrained models for proposed AE-FAR
- train-model.ipynb  for training code
- test-model.ipynb to evaluate the model with pretrained weights

# Citation
If you find this repo useful, please cite our paper.

```
@inproceedings{delibasoglu2024time,
  title={Time Series Anomaly Detection Leveraging MSE Feedback with AutoEncoder and RNN},
  author={Delibasoglu, Ibrahim and Heintz, Fredrik},
  booktitle={31st International Symposium on Temporal Representation and Reasoning (TIME 2024)},
  pages={17--1},
  year={2024},
  organization={Schloss Dagstuhl--Leibniz-Zentrum f{\"u}r Informatik}
}
```

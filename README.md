# **TaSNet: Time-domain Audio Separation Network**

Tensorflow 2.x (with Keras API) Implementation of the Conv-TasNet (Luo et al., 2019)

# **Training and Predicting**

1. Training example

```
python main.py --checkpoint=checkpoint --dataset_path=dataset_path
```

- checkpoint: path to save trained weights of model
- dataset_path: path of MUSDB18
- use `python main.py --help` to see all options

2. Predicting example

```
python predict.py --checkpoint=checkpoint --video_id=video_id
```

- checkpoint: path where trained weights of model is saved
- video_id: video id in youtube
- use `python predict.py --help` to see all options

# **License**

MIT License

# **References**

1. Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation ([IEEE][conv_tasnet_paper_link])
2. Tensorflow implementation by paxbun ([GitHub][paxbun_github_link])

[conv_tasnet_paper_link]: https://ieeexplore.ieee.org/document/8707065
[paxbun_github_link]: https://github.com/paxbun/Conv-TasNet

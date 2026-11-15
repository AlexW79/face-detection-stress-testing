# Data

The image datasets are **not included** in this repository because of their size and licensing. Download them from Kaggle and arrange them like this:

```
data/
├── train/                  # Skin Tone Classification Dataset (evaluation)
│   ├── Black/
│   ├── Brown/
│   └── White/
└── Faces/                  # Caltech Faces and Non Faces (HOG + SVM training only)
    ├── faces/
    └── nonfaces/
```

| Dataset | Used for | Link |
|---|---|---|
| Skin Tone Classification Dataset | Running the detectors (500 images per skin tone) | https://www.kaggle.com/datasets/usamarana/skin-tone-classification-dataset |
| Caltech Faces and Non Faces | Training the HOG + SVM classifier | https://www.kaggle.com/datasets/sagarkarar/caltech-faces-and-non-faces |

`src/facedetection.m` also searches `data/` recursively for a `train` folder if it isn't at `data/train`.

# OpenEDS Dataset

Dataset for pupil/iris segmentation from Facebook Research.

## Download

From Kaggle:
```bash
kaggle datasets download -d soumicksarker/openeds-dataset
unzip openeds-dataset.zip -d openeds
```

## Expected Structure

```
openeds/
├── train/
│   ├── images/
│   │   └── *.png
│   └── labels/
│       └── *.npy (or *.png)
├── validation/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Classes

- 0: background
- 1: sclera
- 2: iris
- 3: pupil

## Image Format

- Grayscale images, 400x640 pixels
- Labels as numpy arrays or PNG masks with class indices

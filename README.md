# Smiler

Facial emotion recognition model

## Getting started

1. Install requirements.txt
2. Run below code for best accuracy(90%)

```bash
python main.py model=convnext_tiny384 dataset=korean_face_emotion gpus=[0] train/optimizer=lion train.lr.lr=1e-6
```

## Dataset

You can download dataset
from [AIHUB](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=82)
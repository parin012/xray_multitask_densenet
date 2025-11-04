# X-ray MultiTask DenseNet (Classification + Segmentation)

이 프로젝트는 **소아 복부 X-ray 영상**을 이용해 병변 분류(Classification)와 병변 부위 세그멘테이션(Segmentation)을 **동시에 수행**하는 멀티태스크 DenseNet 모델입니다.


---

## 프로젝트 구조

```
 xray_multitask_densenet/
│
├── configs/ # 경로 및 하이퍼파라미터 설정
│ └── config.py
├── data/ # 데이터 전처리 및 Dataset 정의
│ ├── preprocess.py
│ └── dataset.py
├── models/ # 모델 구조 (DenseNet + SegDecoder)
│ └── densenet_multitask.py
├── train/ # 학습 및 유틸 함수
│ ├── train.py
│ └── utils.py
├── infer/ # 추론 스크립트
│ └── inference.py
├── results/ # 모델 및 출력물 저장 폴더
└── README.md 
```

## 데이터 구조

```
DATA_ROOT/
├── Training/
│   ├── 01.원천데이터/클래스A/*.png
│   └── 02.라벨링데이터/클래스A/*.json
├── Validation/
│   ├── 01.원천데이터/클래스A/*.png
│   └── 02.라벨링데이터/클래스A/*.json
└── Test/
    ├── 01.원천데이터/클래스A/*.png
    └── 02.라벨링데이터/클래스A/*.json
```

각 클래스 폴더 이름은 Train과 Val 모두 동일해야 하며,
JSON 라벨은 polygon(points) 기반 세그멘테이션 형식이어야 합니다.

---

## 학습 실행 

python3 train/train.py

출력물은 results/ 폴더에 저장됩니다
```
results/
 ├── best_model.pt
 ├── metrics.csv
 └── curves.png
```

## 추론 실행 

python3 infer/inference.py \
  --input " " \
  --weights " " \
  --save-dir "./results/infer_out" \
  --class-names "PyloricStenosis,FreeAir,AirFluid,Constipation,Normal"

출력물은 results/infer_out/ 폴더에 저장됩니다
```
results/infer_out/
 ├── inference_results.csv    # 각 이미지별 예측 결과
 ├── seg/                     # 세그멘테이션 마스크
 └── overlay/                 # 원본 + 히트맵 오버레이
```



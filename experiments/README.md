# 🧪 Experiments

이 디렉터리는 머신러닝 실험 코드를 보관하는 공간입니다.  
각 실험마다 별도 디렉터리를 갖고, 다음 내용을 포함할 수 있습니다:

- 실험 목적
- 사용 데이터셋
- 주요 코드 (전처리, 학습, 평가 등)
- 실험 결과 로그 및 그래프
- 하이퍼파라미터 설정

---

## 📋 디렉터리 구조 예시
```text
experiments/
├── feature-selection/
│   ├── README.md    # 실험 설명
│   ├── data/        # 실험용 데이터
│   ├── train.py     # 학습 코드
│   ├── result.log   # 실험 결과 로그
│   └── plots/       # 시각화 결과물
├── hyperparameter-tuning/
│   ├── README.md
│   ├── train.py
│   ├── tuning_results.csv

# AI-Assisted RF Capacitor Design Framework

This repository contains the source code and documentation for an AI-assisted framework designed to accelerate the inverse design of RF capacitors. The framework leverages Machine Learning (ML) surrogate models trained on data from Keysight's Advanced Design System (ADS) to rapidly predict capacitor performance and optimize its physical layout.

---

## 1. Project Overview
<!-- 한국어 주석: 프로젝트 개요 -->
<!-- AI 기반 RF 캐패시터 자동 설계 프레임워크로, ADS에서 얻은 데이터를 이용해 학습과 역설계를 수행합니다. -->

The primary objective of this project is to replace time-consuming, iterative electromagnetic (EM) simulations with a data-driven approach. By integrating simulation data from Keysight ADS (Momentum), we train a Multi-Layer Perceptron (MLP) surrogate model. This model accurately maps a capacitor's geometric parameters (e.g., finger length, width, spacing) to its electrical characteristics (e.g., Capacitance, Q-factor, Self-Resonant Frequency). Bayesian Optimization is then employed for efficient inverse design, finding the optimal geometry for a given set of target specifications.

---

## 2. Workflow (워크플로우)
<!-- 한국어 주석: 프레임워크 동작 흐름 -->

설계 자동화 프로세스는 다음 5단계로 구성됩니다:

1.  **ADS 데이터 생성 (Data Generation)**
    -   Keysight ADS에서 파라미터 스윕 EM 시뮬레이션을 실행하여 포괄적인 데이터셋을 생성합니다.
    -   주요 레이아웃 파라미터(예: 핑거 길이 `L`, 폭 `W`)를 지정된 범위 내에서 변경하며 시뮬레이션합니다.
    -   핵심 성능 지표인 커패시턴스(C), Q-factor, 자체 공진 주파수(SRF), S-파라미터를 추출합니다.

2.  **데이터 전처리 (Data Preprocessing)**
    -   ADS에서 출력된 원본 데이터를 CSV 또는 NumPy 배열과 같은 정형화된 포맷으로 변환합니다.
    -   안정적이고 효율적인 모델 학습을 위해 데이터를 정규화합니다.
    -   신뢰성 있는 모델 평가를 위해 데이터셋을 학습, 검증, 테스트 세트로 분할합니다.

3.  **대리 모델 학습 (Surrogate Modeling)**
    -   다층 퍼셉트론(MLP) 신경망을 학습시켜, 기하학적 파라미터(`L, W, S, N`)와 전기적 성능(`C, Q, SRF`) 간의 복잡한 관계를 모델링합니다.
    -   학습된 MLP 모델은 계산 비용이 높은 EM 시뮬레이터를 대체하는 빠르고 정확한 대리 모델(Surrogate Model) 역할을 합니다.

4.  **베이즈 최적화 (Bayesian Optimization)**
    -   학습된 대리 모델을 역설계(Inverse Design)에 활용합니다.
    -   목표 커패시턴스 또는 동작 주파수가 주어지면, 베이즈 최적화 기법을 사용하여 목표 사양을 가장 잘 만족하는 기하학적 파라미터를 효율적으로 탐색합니다.

5.  **ADS 검증 (Validation)**
    -   AI가 예측한 최적의 기하학적 구조를 Keysight ADS에서 최종 EM 시뮬레이션을 수행하여 검증합니다.
    -   시뮬레이션 결과를 대리 모델의 예측과 비교하여 프레임워크의 정확도를 검증하고 설계 루프를 완성합니다.

---

## 3. Directory Structure
<!-- 한국어 주석: 프로젝트 폴더 구조 -->

```
.
├── ads/                  # Keysight ADS project files and AEL scripts
├── data/
│   ├── raw/              # Raw data exported from ADS
│   └── processed/        # Processed and normalized data (CSV/NumPy)
├── results/
│   ├── models/           # Saved surrogate models (.pth)
│   └── plots/            # Validation plots and figures
├── src/
│   ├── preprocess.py     # Script for data preprocessing
│   ├── train.py          # Script for training the MLP surrogate model
│   └── optimize.py       # Script for Bayesian Optimization
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 4. Installation
<!-- 한국어 주석: 설치 및 환경 구성 -->

This project requires Python 3.9 or higher.

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/ai-rf-capacitor-design.git
    cd ai-rf-capacitor-design
    ```

2.  Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

The core dependencies are:
- `numpy`
- `pandas`
- `torch`
- `optuna`
- `matplotlib`
- `scikit-learn`

---

## 5. Usage
<!-- 한국어 주석: 사용 예시 -->
<!-- MLP 학습 → Bayesian Optimization 실행 → ADS 재검증 순서로 실행합니다. -->

The following examples demonstrate the primary workflow from the command line.

### 1. Preprocess the Data
First, process the raw data generated from ADS.
```bash
python src/preprocess.py --input-dir data/raw/ --output-dir data/processed/
```

### 2. Train the Surrogate Model
Train the MLP model on the preprocessed data. The trained model will be saved in the `results/models/` directory.
```bash
python src/train.py --data-path data/processed/dataset.csv --epochs 500 --batch-size 32
```

### 3. Run Bayesian Optimization for Inverse Design
Use the trained model to find the optimal geometry for a target capacitance (e.g., 1.5 pF).
```bash
python src/optimize.py --model-path results/models/mlp_surrogate.pth --target-capacitance 1.5e-12
```

### 4. Validate in ADS
The optimization script will output the predicted optimal parameters (e.g., `L=50um, W=10um, S=5um, N=8`). Use these values to run a final validation simulation in your Keysight ADS project.

---

## 6. Future Work
<!-- 한국어 주석: 향후 확장 방향 -->

This framework can be extended in several promising directions:

-   **Advanced Surrogate Models:** Implement Convolutional Neural Networks (CNNs) or U-Nets for pixel-based capacitor layouts, allowing for more flexible and complex geometries.
-   **Multi-Platform Integration:** Integrate with other EM simulation tools like Ansys HFSS or Sonnet for broader applicability.
-   **Multi-Objective Optimization:** Extend the optimization goal to handle trade-offs between multiple parameters, such as maximizing the Q-factor while achieving a target capacitance (C-Q trade-off).
-   **Transfer Learning:** Apply transfer learning techniques to adapt existing surrogate models to new manufacturing processes or technology nodes with minimal retraining effort.

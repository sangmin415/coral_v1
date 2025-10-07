# AI-Assisted RF Capacitor Design Framework

This repository contains the source code and documentation for an AI-assisted framework that accelerates the inverse design of RF capacitors. We train machine-learning surrogate models on Keysight ADS (Momentum) simulation data so that Bayesian optimization can quickly suggest capacitor layouts without repeated EM solves.

The team will stage training jobs on a supercomputing cluster, so this README emphasises the division of responsibilities between the local development environment (data preparation, experiment orchestration, inverse design) and the remote high-performance computing (HPC) system (model training at scale).

---

## 1. Project Overview
<!-- 한국어 주석: 프로젝트 개요 -->

### 1.1 Goals
1.  **Reduce simulation turnaround** by replacing iterative ADS sweeps with an accurate Multi-Layer Perceptron (MLP) surrogate.
2.  **Optimize capacitor geometry** (currently limited to length `L` and width `W` per the PDK constraint) using Bayesian optimization.
3.  **Establish a reusable ML pipeline** that can be extended to inductors and RF filters once the capacitor workflow is validated.

### 1.2 Artifacts
-   **Source code** for preprocessing, training, and inverse design scripts.
-   **Configuration files** describing dataset schemas and experiment metadata.
-   **Documentation** covering data-readiness checks, HPC usage patterns, and future extensions.

---

## 2. End-to-End Workflow
<!-- 한국어 주석: 전체 워크플로우 -->

The design automation workflow is split into five stages with clear local ↔ HPC boundaries.

| Stage | Owner | Description |
| --- | --- | --- |
| 1. **ADS 데이터 생성** | RF 설계 팀 | Keysight ADS에서 `L`, `W` 파라미터 스윕을 수행해 Excel/CSV 데이터셋을 생성합니다. 추가 파라미터는 PDK 제약에 따라 고정합니다. |
| 2. **데이터 수령 및 검증** | 로컬 개발 환경 | `docs/pre_data_readiness.md` 체크리스트에 따라 엑셀 스키마, 단위, 결측값을 확인하고 `configs/capacitor_baseline.yaml`을 업데이트합니다. |
| 3. **전처리 및 패키징** | 로컬 개발 환경 | `python src/preprocess.py ...` 명령으로 정규화·스플릿을 수행하고, 결과물을 `data/processed/`에 저장합니다. |
| 4. **HPC 모델 학습** | 슈퍼컴퓨터 | 전처리 산출물을 전송 후 SLURM/LSF 등 스케줄러에 학습 잡을 제출합니다. 학습 로그와 가중치를 `results/models/`에 저장합니다. |
| 5. **역설계 및 ADS 재검증** | 로컬 개발 환경 | 학습된 모델을 사용해 `python src/optimize.py ...` 로 최적의 `L`, `W`를 탐색하고 ADS에서 재시뮬레이션해 확인합니다. |

---

## 3. Directory Structure
<!-- 한국어 주석: 프로젝트 폴더 구조 -->

```
.
├── ads/                  # Keysight ADS project files and AEL scripts
├── configs/              # YAML configs describing dataset schema & experiments
├── data/
│   ├── raw/              # Raw data exported from ADS or lab measurements
│   └── processed/        # Normalized CSV/NPZ splits + scaler metadata
├── docs/                 # Additional documentation (e.g., pre-data checklist)
├── results/
│   ├── models/           # Saved surrogate models (.pth)
│   └── plots/            # Validation plots and figures
├── src/
│   ├── data_utils.py     # Torch dataset + scaler helpers
│   ├── preprocess.py     # Script for data preprocessing
│   ├── train.py          # Script for training the MLP surrogate model
│   └── optimize.py       # Script for Bayesian Optimization
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 4. Environment Setup
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

### 4.1 Local vs. HPC
-   **Local machine**: Run preprocessing, configuration edits, Bayesian optimization trials, and lightweight sanity checks.
-   **HPC cluster**: Execute large-scale training with the same Python environment. Use a virtual environment or module system to install `requirements.txt` on the cluster.

> 💡 **Tip**: Use `pip freeze > requirements-lock.txt` on the HPC side to capture the exact versions that produced a model artifact.

---

## 5. Standard Operating Procedure
<!-- 한국어 주석: 운영 절차 -->

The commands below outline the baseline experiment path. Replace placeholder paths with the actual Excel file name once received from the supervising professor.

### 1. Preprocess the Data (Local)
Run the preprocessing script locally to convert Excel/CSV exports into normalized PyTorch-ready tensors.
```bash
python src/preprocess.py \
  --input-path data/raw/ads_capacitor.xlsx \
  --config configs/capacitor_baseline.yaml \
  --output-dir data/processed/
```

### 2. Train the Surrogate Model (HPC)
Training occurs on the supercomputer. The example below assumes a SLURM scheduler; adapt it to your cluster environment.

1.  **Stage the dataset** to the HPC scratch directory (e.g., using `rsync`).
    ```bash
    rsync -avz data/processed/ user@hpc:/scratch/$USER/coral_v1/data/processed/
    ```

2.  **Submit a job script** (`scripts/train_capacitor.sbatch` example):
    ```bash
    #!/bin/bash
    #SBATCH --job-name=cap_train
    #SBATCH --time=04:00:00
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=32G

    module load python/3.10 cuda/12.1  # 클러스터 환경에 맞게 수정
    source ~/envs/coral/bin/activate

    cd $SCRATCH/coral_v1
    python src/train.py \
      --data-path data/processed/train.npz \
      --val-path data/processed/val.npz \
      --config configs/capacitor_baseline.yaml \
      --output-dir results/models/
    ```

3.  **Monitor and retrieve** results back to local storage when the job completes.
    ```bash
    rsync -avz user@hpc:/scratch/$USER/coral_v1/results/models/ results/models/
    ```

### 3. Run Bayesian Optimization (Local)
Back on the local machine, use the trained weights to search for target specifications (e.g., capacitance = 1.5 pF).
```bash
python src/optimize.py \
  --model-path results/models/mlp_surrogate.pth \
  --target-capacitance 1.5e-12 \
  --bounds "{'L': [10e-6, 120e-6], 'W': [4e-6, 40e-6]}"
```

### 4. Validate in ADS (RF Team)
Import the optimized `L`, `W` values into the ADS layout, run a final EM simulation, and compare the measured capacitance and Q-factor against the model prediction. Feed the validated result back into the dataset for continual improvement.

---

## 6. Future Work
<!-- 한국어 주석: 향후 확장 방향 -->

This framework can be extended in several promising directions:

-   **Advanced Surrogate Models:** Implement Convolutional Neural Networks (CNNs) or U-Nets for pixel-based capacitor layouts, allowing for more flexible and complex geometries once layout images become available.
-   **Multi-Platform Integration:** Integrate with other EM simulation tools like Ansys HFSS or Sonnet for broader applicability.
-   **Multi-Objective Optimization:** Extend the optimization goal to handle trade-offs between multiple parameters, such as maximizing the Q-factor while achieving a target capacitance (C-Q trade-off).
-   **Transfer Learning:** Apply transfer learning techniques to adapt existing surrogate models to new manufacturing processes or technology nodes with minimal retraining effort.

---

## 7. Reference Checklist
-   Review `docs/pre_data_readiness.md` before ingesting the professor's Excel dataset.
-   Confirm that the `configs/capacitor_baseline.yaml` file matches the actual column names and units.
-   Keep a changelog of HPC training runs (date, git commit, config hash, dataset version) for reproducibility.

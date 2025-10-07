# Pre-Data Readiness Checklist

이 문서는 실제 ADS/측정 데이터셋을 받기 전에 준비해야 할 항목을 정리합니다.

## 1. 데이터 스키마 확정
- 교수님이 제공할 엑셀 파일의 열 이름과 단위를 미리 확인합니다.
- 최소한 `L`/`W` 외에 고정 파라미터가 있다면 컬럼으로 포함되는지 확인합니다.
- 타깃 컬럼(예: `capacitance_pf`, `q_factor`, `srf_ghz`)이 명확한지 확인합니다.
- `configs/capacitor_baseline.yaml`의 `feature_columns`, `target_columns`를 실제 열 이름에 맞게 수정할 준비를 합니다.

## 2. 전처리 파이프라인 리허설
- `src/preprocess.py`는 엑셀/CSV 입력을 받아 정규화와 데이터 분할을 수행합니다.
- 더미 데이터를 이용해 아래 커맨드를 미리 실행해봅니다.
  ```bash
  python src/preprocess.py \
      --input-path data/raw/dummy.xlsx \
      --feature-cols L_um W_um \
      --target-cols capacitance_pf q_factor
  ```
- 실행 결과로 `data/processed/`에 `train/val/test.csv`와 `scalers.json`이 생성되는지 확인합니다.

## 3. 학습 스크립트 점검
- `src/train.py`는 전처리 결과를 이용해 MLP 대리모델을 학습합니다.
- 데이터가 없을 때는 난수로 생성한 CSV를 만들어 파이프라인을 검증할 수 있습니다.
- 체크포인트(`results/models/*.pth`)와 학습 곡선 로그(`*_history.json`)가 제대로 저장되는지 확인합니다.

## 4. 최적화 모듈
- `src/optimize.py`는 Optuna 기반의 베이즈 최적화 스텁입니다.
- 실제 사용 전, `configs/capacitor_baseline.yaml`의 `optimization.target_spec`을 원하는 값으로 조정합니다.
- 타깃 사양은 정규화되기 전 값이며, 스크립트가 이를 MLP 출력 단위에 맞춰 보고합니다.

## 5. 버전 관리 및 재현성
- 모든 실행 커맨드와 설정값은 `configs/`에 YAML로 기록합니다.
- 데이터 버전(예: `dataset_v1_202403.xlsx`)을 파일명에 반영하여 추후 추적이 쉽도록 합니다.
- 실험 결과는 `results/` 하위 폴더에 날짜/실험명으로 정리합니다.

## 6. 다음 액션 아이템
1. 교수님께 엑셀 열 이름, 단위, 샘플 수를 문의합니다.
2. 응답을 기반으로 `configs/capacitor_baseline.yaml` 업데이트.
3. 더미 데이터를 만들어 전체 파이프라인을 로컬에서 리허설합니다.
4. 실제 데이터 수신 즉시 `src/preprocess.py`부터 실행하여 학습 준비를 완료합니다.

이 체크리스트를 충족하면 실제 데이터가 도착하자마자 학습/역설계 실험을 시작할 수 있습니다.

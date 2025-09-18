# RF Capacitor Design Platform Roadmap

이 문서는 데이터 레이어, 서로게이트 모델, 역설계 파이프라인, HFSS 연동, 대규모 데이터 학습, 실용성 강화를 포함한 전반적인 고도화 계획을 정리한다. 현재 코드 베이스의 구성 요소와 연계하여 단계별 실행 방안을 기술한다.

## 1. 데이터 레이어 정리: 파라메트릭 시드 + 픽셀 하이브리드 구조

### 최신 진행 상황
- `generate_candidates.py`가 파라메트릭 시드 JSON과 메타데이터 CSV를 생성하여 하이브리드 데이터 구조를 구축했다.【F:ml_core/generate_candidates.py†L1-L74】
- `ml_core/datasets.HybridMaskDataset`가 마스크·파라미터·시뮬레이션 결과를 동시에 로딩하도록 추가되었다.【F:ml_core/datasets.py†L1-L205】
- `train_surrogate.py`는 `HybridUNetSurrogate`를 사용해 파라미터와 마스크를 융합하는 학습 루프를 지원한다.【F:ml_core/train_surrogate.py†L1-L94】【F:ml_core/unet.py†L1-L121】

### 현재 상태
- (레거시) `data/masks`에는 픽셀 기반 레이아웃만 저장되고 파라메트릭 정보는 `jobs_to_run.json` 생성 시에만 존재한다.【F:ml_core/generate_candidates.py†L13-L79】
- (레거시) 서로게이트 학습 데이터로는 단일 채널 마스크만 사용되며, 파라메트릭/메타 정보가 손실된다.【F:ml_core/train_surrogate.py†L16-L94】

### 개선 목표
1. **하이브리드 데이터 스키마**
   - `masks/`에는 기존과 동일하게 2D 바이너리 배열을 저장한다.
   - `params/` 디렉터리를 신설하여 각 설계에 대응하는 파라메트릭 시드(`.json` 또는 `.npy`)를 보관한다.
   - `metadata.csv` 또는 `metadata.parquet`에 시드, 생산 공정 정보, HFSS 시뮬레이션 설정을 정규화하여 기록한다.
2. **Dataset 계층화**
   - `ml_core/datasets.py`에 `HybridMaskDataset` 클래스를 추가하여 파라미터 벡터와 마스크 텐서를 동시에 로드한다.
   - 파라메트릭 시드를 encoder로 투영한 후 픽셀 인코더 출력과 결합하는 하이브리드 입력을 지원한다.
3. **데이터 버전 관리**
   - `data/schema.md`에 버전 정보 및 필드 정의를 문서화하고, `config.yaml`에 활성 데이터셋 버전을 명시한다.

### 실행 절차
1. `generate_candidates.py`를 확장하여 파라메트릭 시드 저장과 메타데이터 테이블 업데이트를 수행한다.
2. 신규 Dataset과 DataLoader를 도입하고, 학습 스크립트(`train_surrogate.py`)에서 이를 사용하도록 교체한다.
3. 데이터 검증 스크립트를 작성하여 누락/불일치 항목을 검사하고 CI에 포함한다.

## 2. 서러게이트 고도화: UNet + Physics-Aware Layer, Full S-Param 출력

### 현재 상태
- `UNetSmall` 구조는 단일 채널 마스크 입력을 받아 804 차원의 S-파라미터를 회귀한다.【F:ml_core/unet.py†L5-L48】
- 학습 루프는 기본적인 MSE + 패시비티 패널티만 포함한다.【F:ml_core/train_surrogate.py†L23-L45】

### 개선 목표
1. **하이브리드 입력 인코더**
   - 마스크 인코더(UNet 인코더)와 파라메트릭 인코더(MLP)를 병렬 구성 후, 브릿지 단계에서 피처를 융합한다.
2. **Physics-Aware Layer**
   - 출력단 직전에 전송선 임피던스/에너지 보존 제약을 반영하는 커스텀 레이어(`PhysicsAwareLayer`)를 삽입한다.
   - S-parameter의 Hermitian symmetry 및 패시비티를 보장하기 위한 soft constraint를 포함한다.
3. **Full S-parameter 구조화 출력**
   - 출력 텐서를 `(batch, n_ports, n_freq, 2)` 형태로 재구성하여 복소수 처리 일관성을 유지한다.
   - `train_surrogate.py`에서 새 손실 함수(예: 복소수 MSE + 물리 제약)를 지원하도록 업데이트한다.
4. **모델 버저닝 및 체크포인트**
   - PyTorch Lightning 혹은 자체 체크포인트 관리로 학습 상태를 기록하고, `models/` 디렉터리 구조를 정리한다.

### 실행 절차
1. `ml_core/unet.py`에 `HybridUNetSurrogate`와 `PhysicsAwareLayer`를 구현한다.
2. `train_surrogate.py`를 리팩터링하여 LightningModule 혹은 구조화된 학습 loop를 도입한다.
3. `surrogate.py`에 새 모델 팩토리를 추가하여 UI/컨트롤러가 모델 타입을 동적으로 선택할 수 있게 한다.
4. 통합 테스트: 임의 데이터로 forward pass, 손실 계산, backprop 검증 스크립트를 작성한다.

## 3. 역설계 강화: Bayesian Optimization + Generative 모델

### 현재 상태
- 파라메트릭 모드는 Optuna 기반 Bayesian optimization만 사용하며, freeform 모드는 Adam을 통해 픽셀을 직접 최적화한다.【F:main_controller.py†L129-L244】
- `inverse_design.py`는 gradient 기반 freeform 최적화만 포함한다.【F:ml_core/inverse_design.py†L1-L36】

### 개선 목표
1. **Bayesian Optimization 확장**
   - Optuna sampler를 Gaussian Process 기반으로 설정하고, multi-objective 지원(예: 삽입손실/반사손실 동시 최적화)을 도입한다.
   - Surrogate 예측 불확실도를 추정해 acquisition function(EI/UCB)을 조정한다.
2. **Generative 모델 도입**
    - Variational Autoencoder(VAE) 또는 Diffusion 기반 `MaskGenerator`를 훈련하여 freeform 초기 해를 생성한다.
    - 생성 모델 latent space와 Bayesian optimizer를 결합하여 탐색 효율을 향상한다.
3. **Mixed-initiative Loop**
    - Adam 기반 미세 조정은 생성된 후보에 대해 수행하고, 최적화 이력/메타데이터를 저장한다.

### 실행 절차
1. `ml_core/inverse_design.py`에 `BayesianFreeformDesigner`, `MaskGenerator` 클래스를 추가한다.
2. `main_controller.py`에 새로운 최적화 모드(예: "Freeform (Bayesian+Gen)")를 추가하고 UI에서 선택 가능하도록 한다.
3. 최적화 로그와 후보 저장을 위한 `results/history/` 디렉터리를 구성한다.

## 4. HFSS 연동 루프 구축: Surrogate 검증 + Active Learning

### 현재 상태
- HFSS 관련 스크립트는 `hfss_scripts/`와 `hfss_recorded_script.py`에 존재하지만 자동화 루프가 구성되어 있지 않다.

### 개선 목표
1. **Surrogate 검증 파이프라인**
   - `hfss_scripts/batch_simulator.py`를 작성하여 GDS/파라미터 입력에 대한 배치 HFSS 실행을 지원한다.
   - 시뮬레이션 결과를 `data/simulation_results/`에 버전별로 저장하고, surrogate 예측과 비교하는 검증 리포트를 자동 생성한다.
2. **Active Learning Loop**
   - Surrogate 불확실도(예: Monte Carlo dropout, ensemble)를 기반으로 재시뮬레이션할 후보를 선별한다.
   - HFSS 결과를 데이터셋에 합류시키고 재학습(또는 파인튜닝)을 트리거하는 `active_learning_loop.py`를 구현한다.
3. **스케줄링 및 모니터링**
   - Airflow/Prefect와 같은 워크플로 도구 또는 간단한 cron job으로 HFSS-학습 루프를 자동화한다.

### 실행 절차
1. HFSS 실행 wrapper와 결과 파싱 유틸리티를 구축한다.
2. Surrogate 예측 vs HFSS 결과 비교 대시보드를 작성한다.
3. Active learning 주기(예: weekly retrain)를 설정하고 로그/알림 시스템을 연동한다.

## 5. 대규모 데이터·학습 체계화: 10k → 50k+, Augmentation

### 현재 상태
- 현재 학습 스크립트는 수천 단위 데이터셋을 가정하고 기본적인 좌우 반전 augmentation만 제공한다.【F:ml_core/train_surrogate.py†L34-L45】

### 개선 목표
1. **데이터 생성 파이프라인 확장**
   - 병렬 파라메트릭 스윕 스케줄링(멀티프로세싱/클러스터)을 도입하여 50k 이상 샘플 생성.
   - HFSS 시뮬레이션 병렬화 및 결과 캐싱.
2. **Augmentation 다양화**
   - 좌우/상하 반전, 미세 변형(erosion/dilation), 노이즈 삽입, 파라메트릭 perturbation 추가.
   - 물리 제약을 위반하지 않는 범위 내에서 확률적 필터 적용.
3. **데이터 관리**
   - HDF5 혹은 Zarr 기반 대용량 저장 포맷을 도입하고, DataLoader에서 메모리 효율적 스트리밍을 지원한다.
   - Lightning DataModule 혹은 커스텀 batch sampler로 class balancing 및 curriculum 학습을 지원한다.

### 실행 절차
1. `generate_candidates.py`를 멀티프로세싱 구조로 리팩터링한다.
2. `train_surrogate.py`에 augmentation 파이프라인(Albumentations, Kornia 등)을 통합한다.
3. 학습 로깅/모니터링을 위해 `lightning_logs/` 디렉터리 구조를 정리하고 MLflow/W&B 연동을 검토한다.

## 6. 실용성 개선: GUI & 실행 속도 최적화

### 현재 상태
- Gradio UI는 기본적인 파라메트릭/픽셀 역설계 모드만 제공한다.【F:ui/app.py†L1-L117】
- 실행 속도는 단일 surrogate forward + 단순 optimizer에 의존한다.

### 개선 목표
1. **UI 개선**
   - 모드 확장: "Hybrid", "Active Learning", "Batch Validation" 탭을 추가하고, 데이터/모델 관리 기능을 제공한다.
   - 결과 대시보드: surrogate vs HFSS 비교 그래프, 학습 커브, 데이터셋 현황을 시각화.
   - 사용자 입력 검증과 로깅 다운로드 기능 강화.
2. **실행 속도 최적화**
   - PyTorch 모델의 TorchScript/ONNX 변환과 `torch.compile` 적용을 통해 추론 속도 향상.
   - CUDA 그래프 및 mixed precision inference 도입.
   - 파라메트릭 최적화의 병렬화(Optuna study 병렬 실행, 배치 평가).
3. **배포 친화성**
   - Docker Compose로 UI + 백엔드 + HFSS 인터페이스 컨테이너를 구성한다.
   - CI 파이프라인에서 기본 smoke test와 UI e2e 테스트를 자동화한다.

### 실행 절차
1. UI 레이아웃을 모듈화(`ui/components/`)하고 새로운 탭/컨트롤을 추가한다.
2. 모델 로딩 경로와 추론 캐시를 도입하여 반복 호출 시 지연을 최소화한다.
3. 성능 벤치마크 스크립트를 작성하고 최적화 효과를 모니터링한다.

## 단계별 마일스톤 요약

| 단계 | 기간 | 주요 산출물 |
| --- | --- | --- |
| Phase 1 | 주 1-2 | 데이터 스키마 개편, Hybrid Dataset 구현, 기본 Physics-aware UNet 프로토타입 |
| Phase 2 | 주 3-4 | Surrogate 학습 파이프라인 리팩터링, Bayesian+Generative 역설계 통합, UI 확장 초안 |
| Phase 3 | 주 5-6 | HFSS 액티브 러닝 루프, 대규모 데이터 생성 인프라, 모델/시뮬레이션 검증 자동화 |
| Phase 4 | 주 7+ | 속도 최적화, 배포 및 모니터링 체계, 지속적 개선 로드맵 수립 |

## 리스크 및 대응

- **데이터 일관성 문제**: 메타데이터 버전 관리 및 스키마 검증 스크립트로 예방.
- **HFSS 연동 안정성**: API 호출 실패 대비 재시도 로직과 로그 수집 강화.
- **대규모 학습 비용**: 클라우드 스팟 인스턴스 활용 및 mixed precision 학습 적용.
- **생성 모델 품질**: 초기에는 VAE 등 경량 모델로 시작하고, 필요 시 Diffusion으로 확장.

## 후속 조치

- 각 단계 완료 후 문서화(`docs/progress/`)와 회고를 진행한다.
- 핵심 모듈별 단위 테스트와 통합 테스트를 작성하여 회귀 방지 체계를 마련한다.
- 성능 지표(S-파라미터 MSE, HFSS vs surrogate delta, 최적화 시간 등)를 대시보드화한다.

RF Capacitor Design Platform – Roadmap

이 문서는 데이터 레이어, 서로게이트 모델, 역설계 파이프라인, HFSS 연동, 대규모 데이터 학습, 실용성 강화 등 플랫폼 고도화 계획을 정리한다. 각 단계별 실행 절차를 현재 코드 베이스와 연계하여 기술한다.

1. 데이터 레이어: 파라메트릭 시드 + 픽셀 하이브리드

현재 문제점

data/masks/에는 픽셀 기반 레이아웃만 저장, 파라메트릭 정보는 jobs_to_run.json 생성 시에만 존재

Surrogate 학습 데이터에 메타 정보 손실 발생

개선 목표

하이브리드 스키마

masks/: 기존 2D 바이너리 배열

params/: 각 설계의 파라메트릭 시드(.json/.npy)

metadata.csv/.parquet: 시드, 공정 정보, HFSS 설정 등 정규화 기록

Dataset 계층화

HybridMaskDataset: 파라미터 벡터 + 마스크 텐서 동시 로딩

파라메트릭 인코더 + 마스크 인코더 결합 지원

버전 관리

data/schema.md: 필드 정의/버전 기록

config.yaml: 활성 데이터셋 버전 명시

실행 절차

generate_candidates.py 확장 → 시드 저장 + 메타데이터 업데이트

ml_core/datasets.py에 HybridMaskDataset 추가 → train_surrogate.py에서 교체

데이터 검증 스크립트 작성 및 CI 연동

2. 서로게이트 고도화: Hybrid UNet + Physics-Aware Layer

현재 문제점

단일 마스크 입력 → 804차원 S-파라미터 회귀

학습 루프는 단순 MSE + 패시비티 패널티

개선 목표

하이브리드 입력 인코더: 마스크(UNet) + 파라미터(MLP) 피처 융합

Physics-Aware Layer: 출력단 직전 임피던스/에너지 보존 제약 반영

Full S-parameter 출력: (batch, n_ports, n_freq, 2) 형태 (복소수 일관성 유지)

모델 버저닝: Lightning 체크포인트 및 models/ 구조 정리

실행 절차

HybridUNetSurrogate, PhysicsAwareLayer 구현

train_surrogate.py → LightningModule 기반 리팩터링

surrogate.py에 모델 팩토리 추가

Forward/backprop 테스트 스크립트 작성

3. 역설계: Bayesian Optimization + Generative 모델

현재 문제점

Parametric: Optuna 기반 BO만 사용

Freeform: Adam 기반 gradient descent

개선 목표

Bayesian Optimization 확장

Gaussian Process, Multi-objective 지원

불확실도 기반 acquisition function(EI/UCB)

Generative 모델 도입

VAE/Diffusion 기반 Mask Generator

Latent space + BO 결합

Mixed-initiative Loop

생성 후보 → Adam 미세조정

최적화 이력/메타데이터 저장

실행 절차

BayesianFreeformDesigner, MaskGenerator 클래스 추가

main_controller.py에 신규 모드 연결

결과 저장: results/history/ 구성

4. HFSS 연동: Surrogate 검증 + Active Learning

현재 문제점

HFSS 스크립트는 있으나 자동화 루프 미구성

개선 목표

Surrogate 검증: 배치 HFSS 실행 → 결과 저장 및 리포트 자동 생성

Active Learning: 불확실도 기반 후보 선별 → 재시뮬레이션 + 재학습

자동화: cron/Prefect/Airflow 기반 주기적 실행

실행 절차

HFSS wrapper 및 결과 파서 작성

Surrogate vs HFSS 대시보드 구성

Active learning 주기 설정 및 로그 알림 연동

5. 대규모 데이터: 10k → 50k+, Augmentation

현재 문제점

데이터셋 수천 단위, 좌우 반전만 제공

개선 목표

데이터 생성 확장: 멀티프로세싱/클러스터 기반 병렬 생성

다양한 증강: 상하 반전, erosion/dilation, 노이즈 삽입, 파라메트릭 perturbation

대용량 관리: HDF5/Zarr 기반 저장, DataModule로 효율적 로딩

실행 절차

generate_candidates.py 병렬화

train_surrogate.py에 augmentation 파이프라인 통합

MLflow/W&B 연동으로 학습 모니터링

6. 실용성 개선: GUI & 실행 속도

현재 문제점

Gradio UI 단순, 추론 속도 최적화 부족

개선 목표

UI 개선: Hybrid/Active Learning/Validation 탭 추가, 데이터/모델 관리, 대시보드 제공

속도 최적화: TorchScript/ONNX, torch.compile, mixed precision inference

배포 친화성: Docker Compose + CI/CD smoke test

실행 절차

ui/components/ 구조화 및 신규 탭 추가

모델 캐싱 및 추론 경로 최적화

성능 벤치마크 작성 및 모니터링

단계별 마일스톤
단계	기간	산출물
Phase 1	주 1–2	데이터 스키마 개편, Hybrid Dataset, Physics-aware UNet 프로토타입
Phase 2	주 3–4	Surrogate 리팩터링, Bayesian+Generative 역설계, UI 확장 초안
Phase 3	주 5–6	HFSS Active Learning 루프, 대규모 데이터 인프라, 자동화 검증
Phase 4	주 7+	속도 최적화, 배포/모니터링 체계, 지속적 개선 로드맵
리스크 및 대응

데이터 일관성 → 메타데이터 버전 관리 + 검증 스크립트

HFSS 연동 실패 → 재시도 로직 + 로그 수집

학습 비용 → 클라우드 스팟 인스턴스, mixed precision 활용

생성 모델 품질 → VAE로 시작, 필요 시 Diffusion 확장

후속 조치

단계 완료 후 docs/progress/에 문서화 및 회고

핵심 모듈별 단위/통합 테스트 작성

성능 지표(S-파라미터 MSE, HFSS vs surrogate 오차, 최적화 시간 등) 대시보드화

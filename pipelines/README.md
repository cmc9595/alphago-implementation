# 실행 파이프라인 기록

`runs/`(git 제외)에서 실제로 돌린 실행 스크립트의 사본. 저장소 루트(`~/alphago-implementation`)에서 실행하는 전제다.

| 스크립트 | 내용 |
|---|---|
| `round3_20k_pipeline.sh` | **현재 진행 중**: 2바퀴 모델로 자가대국 20,000판(학습용 수 6.25%만 800회, 나머지 100회, 트리 재사용) → 가치 전부 사용 학습 → 2바퀴·1바퀴와 400회 100판 비교 → T5·가치 MSE 표 |
| `rounds_pipeline.sh` | 2바퀴 학습·비교 + KataGo 방식(최신 모델로 계속) 3바퀴 |
| `after_selfplay_pipeline.sh` | 1바퀴: 자가대국 → 학습 → 비교 대국 → 속도 측정 |
| `round3_800v_*.sh` | 800회 탐색 3바퀴 시도(중단) |
| `rl_cont_then_gen.sh` | RL 3,000→3,600 이어서 학습 → 비교 → 가치 데이터 150만 판 생성 |
| `value_capacity_20261020.sh` | 가치망 용량 시험(8×128, 6×96 이어서) |
| `value_then_shared.sh` | 별도 가치망 vs 공유모델(생성 데이터) 학습·비교 |
| `searchvalue_20261022.sh` | 가치 정답 = 0.5×승패 + 0.5×MCTS 승률 |
| `score_head_20261023.sh` | 집 차이 head 추가 |
| `eval_and_rating_20261022.sh`, `rating_final_20261023.sh` | KataGo 사람 모방 모델 상대 기력 측정 |

# alphago-implementation

AlphaGo(2016)의 SL 정책망 → RL 정책망 → 가치망 → MCTS 파이프라인을
Apple Silicon 맥북 한 대(M3 Pro)로 축소 재현해 19줄 아마추어 수준에 도달하는 것이 목표.
상세 계획과 벤치마크: [docs/PLAN.md](docs/PLAN.md) · **최종 결과와 기력 측정: [docs/RESULTS.md](docs/RESULTS.md)**

## 바둑판 실행

```bash
.venv/bin/python -u scripts/play_server.py --model checkpoints/sl_6x96/final.pt --value-model checkpoints/value_own/best.pt --port 8765
.venv/bin/python -u scripts/play_server.py --model checkpoints/search_finetune_20261019/final.pt --port 8768
.venv/bin/python -u scripts/play_server.py --model checkpoints/scorehead_20261023/final.pt --port 8767
```

## 구성

- `engine/src/` — C++ 바둑 엔진(규칙, 패, 활로, 축 읽기, 34평면 특징), pybind11 바인딩
- `alphago/` — Python 패키지 (`Board`, `features_batch`, `models.GoNet`)
- `scripts/bench.py` — 하드웨어 벤치마크 (케이스별 프로세스 격리, 메모리 상한)
- `tests/` — 규칙 테스트, 순수 Python 참조 구현과 랜덤 대국 교차검증

## 설치

```bash
uv venv --python 3.11 .venv
uv pip install --native-tls --python .venv/bin/python torch numpy pybind11 setuptools pytest mlx
uv pip install --native-tls --python .venv/bin/python --no-build-isolation -e .
```

C++ 수정 후 재빌드: `.venv/bin/python setup.py build_ext --inplace`

## 데이터 준비

```bash
# KGS (u-go.net): 한쪽 7단 이상 또는 양쪽 6단 대국, 55개 .tar.bz2, 48MB
# Fox (GitHub featurecat/go-dataset, GPL-3.0): 7d/8d/9d 폴더의 *.7z만 사용
#   주의: 같은 폴더의 7d2/8d2/9d2.7z.001은 실제로 7k/8k/9k(급) 기보라 제외
.venv/bin/python scripts/prepare_sgf.py data/raw/kgs --out data/kgs.npz
.venv/bin/python scripts/prepare_sgf.py data/raw/fox/7d.7z data/raw/fox/8d.7z data/raw/fox/9d.7z --out data/fox_dan.npz
```

수순만 저장하고(1수 2바이트), 특징은 학습 중에 C++ 엔진이 즉석에서 다시 계산한다(초당 약 7만 샘플).

## 지도학습 (SL)

```bash
.venv/bin/python scripts/train_sl.py --data data/kgs.npz data/fox_dan.npz   # checkpoints/sl_6x96/
```

`checkpoints/<run>/last.pt`에서 자동으로 이어서 학습하고, 지표는 `metrics.jsonl`에 기록된다.

## Sabaki로 대국

Sabaki → Engines → Manage Engines → Add
- Path: `<repo>/.venv/bin/python`
- Arguments: `<repo>/scripts/gtp_engine.py --model <repo>/checkpoints/sl_6x96/last.pt`

## 사용 예

```python
from alphago import Board, features_batch
b = Board(19, komi=7.5)
b.play(3 * 19 + 15)          # Q4 (action = y*19 + x, y=0이 1선)
f = b.features()             # (34, 19, 19) uint8
print(b)
```

## 메모리 주의

이 맥은 swap이 없어 OOM이 나면 바로 다운된다. GPU 작업에는 항상 아래 상한을 건다.

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
```

## 외부 비교 엔진 (2026-10-17 확인)

KataGo v1.16.3 OpenCL 실행 파일은 `data/katago/src/cpp/build/katago`, 인간 모방 모델은
`data/katago/b18c384nbt-humanv0.bin.gz`에 있다. `data/katago/human.cfg`와 함께
`humanSLProfile=rank_1d`를 지정하면 현대 초단 수 모방을 비교할 수 있다.
이 설정의 승률을 실제 서버 급수로 직접 환산하지 않는다.

Pachi 12.90은 `data/pachi/pachi`에 로컬 빌드했다. DCNN/JOSEKIFIX를 끈 순수 MCTS
버전이며, 시작할 때 `DATA_DIR`을 `data/pachi`의 절대 경로로 설정한다.
공식 릴리스 소스의 macOS `pcg32_set_state` 매크로 괄호 오타와 Git 없는 소스 배포용
빌드 메타데이터 규칙만 수정했다. 프로젝트 모델과 데이터에는 영향을 주지 않는다.

가치망을 같은 미사용 국면에서 비교하려면:

```bash
.venv/bin/python scripts/compare_values.py --positions 12000
```

결과는 `runs/value_comparison.json`에 저장한다. 회전·반전 민감도는 별도로 64국면에서
8개 대칭의 승률 최댓값과 최솟값 차이를 측정한 진단이다. 서로 다른 대국 단계나
라벨 분포에서 얻은 MSE는 직접 비교하지 않는다. 이 비교는 학습 때 사용한 판 분할을
재사용하므로 완전히 새로운 최종 테스트셋은 아니다.

## 대칭 평균 및 동일 시간 대국 실험

`scripts/run_engine_experiments.py`는 모델을 실험 폴더에 고정 복사하고 순차 실행한다.
대칭 평균 평가(8,192국면) 뒤 내부 비교 3종 각 20판, Pachi 및 KataGo 상대 각 50판을 수행한다.
내부 비교는 수당 0.35초의 연산 예산을 양쪽에 주며, 배치 종료 시점까지의 초과 시간도 기록한다.
최대 수에서 강제 종료된 판은 무효로 저장하고 성적에서 제외한다.
대칭 후보는 검증 MSE로 고르지만, 실제 채택은 동일 시간 대국의 Wilson 및 대국 쌍 부트스트랩
구간이 함께 개선을 지지할 때만 한다. 차이가 불분명하면 기존 설정을 유지한다.
검증 데이터는 최종 모델 학습 때의 검증 분할을 재사용하므로 새로운 최종 테스트라고 부르지 않는다.

진행 확인(GPU를 쓰지 않음):

```bash
.venv/bin/python scripts/progress_experiments.py --watch
```

중단 후에는 `scripts/run_engine_experiments.py`를 같은 인자로 다시 실행하면 된다.
중복 실행은 차단하고 완료한 판은 다시 두지 않는다. 조건이나 코드가 바뀌면 새 실험 이름을
사용해야 한다. 결과·명령·모델 해시는 `runs/experiments/ensemble_20261017/`에 저장된다.
대국 SGF와 판별 JSON은 `runs/matches/ensemble_20261017_*/`에 저장된다.
실험기는 현재 브라우저 바둑판의 모델이나 진행 중인 대국을 변경하지 않는다.

### 후보 엔진 비교 및 사활 재현 (2026-10-18)

종국(두 번 패스) 뒤의 탐색을 중단하도록 수정한 뒤, RL 정책+새 가치망과 SL 정책+SL 가치를
수당 2초, 흑백 교대 100판으로 비교합니다. 단일 방향이며 과거 코드의 결과와 분리합니다.

```sh
.venv/bin/python scripts/run_candidate_match.py --progress --watch
```

중단된 실험은 다음 명령으로 재개합니다. 실행 중에는 파일 잠금으로 중복 실행을 막습니다.

```sh
.venv/bin/python scripts/run_candidate_match.py
```

특정 국면은 SGF 또는 `/api/state`에서 저장한 JSON으로 재현합니다.
`--plies`는 해당 수가 둬진 직후의 국면입니다. 학습된 승률 추정이며 확정 사활 판정은 아닙니다.

```sh
.venv/bin/python scripts/analyze_position.py game.sgf --plies 120,121,122 --visits 400,1600,6400 --out runs/position_checks/example.json
```

### 흑백 직접 대국과 실시간 누적 분석

바둑판의 `대국 방식`에서 `흑·백 모두 직접 두기`를 선택하면 현재 기보를 유지하며
두 색을 직접 둘 수 있고 실시간 MCTS가 켜집니다. `새 대국 (양쪽 직접)`은 새 기보로 시작합니다.
수동 모드에서는 자동 AI 응수가 없으며 무르기는 한 수씩 동작합니다.

- 후보 색과 큰 숫자는 방문 순위, 아래 %는 현재 차례 기준 MCTS 승률 추정입니다.
- 후보 표에는 방문 횟수와 비중, 정책 사전확률, 주 변화도를 함께 표시합니다.
- 가치 head의 직접 예측과 MCTS를 거친 예상 승률은 구분해 바둑판 아래 표시합니다.
- 같은 국면에서는 트리와 루트 통계를 누적합니다. 착수/무르기/새 대국/모드 전환 시 취소·초기화합니다.
- 착수는 분석 중에도 가능하며, 이전 국면의 결과는 revision 확인 후 폐기합니다.
- 실시간 분석은 12800회 제한과 별개로 누적하지만, 저장 노드 20만 개 또는 추정 트리 메모리 2 GiB에 도달하면 가지와 통계를 보존한 채 정지합니다. 체크를 끄면 분석 트리를 해제합니다.
- 32768회마다 가지를 삭제하던 방식은 제거했습니다. 화면에 탐색 횟수와 별개인 실제 저장 노드 수, 추정 트리 메모리, 한도 도달 상태를 표시합니다. 트리 메모리 추정에는 노드/NumPy 배열/하위 노드 사전을 포함하며 GPU 추론과 프로세스 전체 메모리는 포함하지 않습니다.
- 분석 화면을 닫거나 다른 탭으로 옮겨 상태 조회가 10초 동안 없으면 계산을 쉽니다.
  같은 국면의 한도 내 트리는 유지하며 다시 조회하면 이어서 분석합니다. 착수/무르기/명시적 중지 시 해제합니다. 학습이나 빠른 롤아웃은 사용하지 않습니다.

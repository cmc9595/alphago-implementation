# M1 Max: 5바퀴 자가대국 실행 안내

이 문서만 보고 M1 Max에서 5바퀴 자가대국을 돌릴 수 있게 정리했다. 저장소 루트는 `~/alphago-implementation`.

## 무엇을 하나

| 항목 | 값 |
|---|---|
| 자가대국 모델 | 3바퀴 공유모델 (릴리스 `shared_round3_6x96.pt`) |
| 학습용 수 | 전체 수의 **12.5%**, 수당 **1,600회** 탐색 (새 트리, 루트 잡음, 처음 30수는 방문 비율대로 선택) |
| 그 밖의 수 | 수당 **200회** 탐색, 트리 재사용, 처음 30수는 방문 비율대로 선택 (판 다양화) |
| 기권 | 없음 |
| 판수 | **12,000판** (기본값, `GAMES=` 로 변경) |
| 시드 | 20507105부터 (M3 Pro 쪽과 겹치지 않음) |
| 결과 폴더 | `runs/experiments/selfplay_round5_1600v_p125_f200_seed20507105/` |
| 예상 시간 | 판당 약 13초 → 12,000판 약 **43시간** (±30%) |

4바퀴 대비 판당 탐색량이 약 2.4배다. 학습용 정답(1,600회)과 판 승패(200회)를 모두 강하게 해서 3바퀴처럼 큰 향상을 노린다.

## 1. 코드 받기

```bash
cd ~/alphago-implementation
git pull
```

이 버전에는 **GNU Go가 비정상 종료해도 판정 프로세스를 다시 띄우는 수정**이 들어 있다.
(4바퀴 때 M1 Max에서 GNU Go가 한 번 죽은 뒤 2,305판이 판정 없이 저장됐다.)

## 2. 모델 받기

```bash
mkdir -p checkpoints/round3_merged_20261102
gh release download models-v1 -p shared_round3_6x96.pt -R cmc9595/alphago-implementation \
  -O checkpoints/round3_merged_20261102/final.pt --clobber
```

`gh`가 없거나 로그인이 안 되어 있으면 https://github.com/cmc9595/alphago-implementation/releases/tag/models-v1 에서
`shared_round3_6x96.pt`를 받아 위 경로에 `final.pt`로 저장한다. 4바퀴 때 받은 파일이 이미 있으면 같은 파일이다.

## 3. 실행

```bash
nohup pipelines/round5_selfplay_m1max.sh > runs/round5_m1max.log 2>&1 &
```

판수를 바꾸려면: `GAMES=20000 nohup pipelines/round5_selfplay_m1max.sh > runs/round5_m1max.log 2>&1 &`

## 4. 확인

```bash
python3 scripts/progress_selfplay_parallel.py selfplay_round5_1600v_p125_f200_seed20507105 --watch
```

- 첫 32판이 동시에 끝나기 전(10~15분)에는 0판으로 보이는 게 정상이다.
- 판당 학습 국면이 약 **36개**로 나오면 설정이 맞다.
- 멈췄으면 3번 명령을 다시 실행하면 이어서 둔다.

## 5. 끝나면 옮기기

M3 Pro로 폴더 `runs/experiments/selfplay_round5_1600v_p125_f200_seed20507105/` 전체를 옮긴다(약 1.5~2GB).
압축하면 빠르다:

```bash
cd ~/alphago-implementation/runs/experiments
tar czf ~/Desktop/round5_m1max.tgz selfplay_round5_1600v_p125_f200_seed20507105
```

혹시 판정 없이 저장된 판이 있으면 M3 Pro에서 `scripts/rejudge.py <폴더 이름>`으로 복구한다.

## 주의

- 실행 중에는 맥이 잠들지 않게 `caffeinate`가 걸려 있다. 전원은 연결해 둔다.
- 이 맥은 swap이 거의 없을 수 있으니 다른 무거운 작업을 같이 돌리지 않는다.
- 학습은 M3 Pro에서 두 맥 데이터를 합쳐서 한다.

# TDOA 위치 추정 및 시각화

센서 4개의 초음파 파형에서 포락선 교차상관 시간차(TDOA)를 구해 음원의
2차원 위치를 추정합니다.

`tdoa.py`를 실행하면 다음 과정을 한 번에 처리합니다.

CSV 읽기 → 평균 제거 → 대역통과 필터 → Hilbert 포락선 →
교차상관 및 포물선 보간으로 시간차 계산 → 위치 추정 → 신호 및 위치 시각화.

- `tdoa.py`: 전체 실행과 신호처리·위치 추정 연결
- `optimization/linear_least_squares.py`: `estimate_position()` 선형 최소제곱 위치 추정
- `optimization/levenberg_marquardt.py`: 고정 lambda 감쇠를 적용한 LM 변형
- `optimization/gauss_newton.py`: 잔차 야코비안 기반 순수 Gauss–Newton 위치 추정
- `optimization/newton_method.py`: 정확한 헤시안을 사용하는 순수 Newton 위치 추정
- `optimization/gradient_descent.py`: 거리차 잔차의 비선형 최소제곱 gradient descent
- `signal_process/tdoa_signal_processing.py`: 필터링, 피크 검출, 시간 출력 및 신호 그래프
- `visualization/tdoa_visualization.py`: TDOA 위치 그래프
- `visualization/waveform_visualization.py`: CSV 읽기 및 원본 파형 그래프

합성 초음파 파형과 정답 도착 시간 CSV는 `signal_example/`에 있습니다.
생성 조건과 읽기 예제는 [합성 데이터 설명](signal_example/README.md)을 참고하세요.

`--solver`로 위치 추정 방법을 선택합니다. 기본값은 `linear_least_squares`입니다.

```bash
python3 tdoa.py --solver linear_least_squares
python3 tdoa.py --solver gradient_descent
python3 tdoa.py --solver newton_method
python3 tdoa.py --solver gauss_newton
python3 tdoa.py --solver levenberg_marquardt --lm-lambda 0.001
```

`gradient_descent`는 선형 최소제곱 해를 초기값으로 사용하고, 실제 거리차
잔차 제곱합을 Armijo backtracking으로 줄입니다. 그래디언트 노름이
`1e-8` 이하이면 종료하며 최대 반복 횟수는 10,000회입니다. 미수렴 시
오류를 보고합니다. 비볼록 문제이므로 전역 최적해는 보장하지 않습니다.

`newton_method`도 같은 초기값과 종료 조건을 사용합니다. 잔차의 2차 미분까지
포함한 헤시안으로 `H * delta_p = -gradient`를 풀고 `p += delta_p`로
갱신합니다. 감쇠와 backtracking 없이 전체 스텝을 적용하므로 목적함수
감소나 수렴을 보장하지 않습니다. 헤시안이 특이하거나 미수렴하면 오류를 보고합니다.

`gauss_newton`은 같은 초기값과 종료 조건을 사용하며, 잔차 야코비안 `J`로
`(J.T @ J) * delta_p = -J.T @ e`를 풀어 전체 스텝을 적용합니다.
감쇠나 backtracking은 없으며, 목적함수 감소나 수렴을 보장하지 않습니다.
야코비안의 열 랭크가 부족하거나 미수렴하면 오류를 보고합니다.

`levenberg_marquardt`는 `(J.T @ J + lambda * I) * delta_p = -J.T @ e`를
풀어 전체 스텝을 적용합니다. `--lm-lambda`는 유한한 0 이상의 값이며 기본값은
0.001입니다. 0이면 Gauss–Newton과 같습니다. 같은 초기값과 종료 조건을
사용하며, lambda 자동 조절이나 스텝 수락/거부를 포함하지 않는 고정 감쇠
변형입니다. 목적함수 감소나 수렴을 보장하지 않습니다.

신호처리만 단독 실행하려면 다음 명령을 사용합니다.
자세한 옵션은 [신호처리 설명](signal_process/README.md)에 있습니다.

```bash
python3 -m signal_process.tdoa_signal_processing
```

## 실행 결과 예시

잡음이 포함된 `signal_example/waveforms_noisy.csv`를 처리한 결과입니다.
신호 구간은 2.5–4.2 ms이며, 오차 평가용 실제 위치는 `(0.4, 0.6)` m로
지정했습니다.

### 신호처리 및 피크 검출

회색은 원본 파형, 파란색은 대역통과 필터 결과, 주황색은 포락선입니다.
빨간 점과 점선은 센서별 포락선 피크 및 그 시각을 표시합니다.

![센서 4개의 원본·필터링 파형, 포락선과 피크 시각](docs/images/processed_signals.png)

### TDOA 위치 추정

실선은 센서 쌍별 TDOA 곡선, 검은 삼각형은 센서, 초록 별은 실제 위치,
빨간 X는 추정 위치입니다. 오른쪽에는 S1 기준 교차상관 시간차를 표시합니다.
이 예제의 추정 위치는 약 `(0.4040, 0.5965)` m, 위치 오차는 약 `5.35 mm`입니다.

![TDOA 곡선과 추정 음원 위치 및 센서별 시간차](docs/images/tdoa_position.png)

위 이미지는 프로젝트 루트에서 다음 명령으로 다시 생성할 수 있습니다.
`docs/images/` 폴더에 저장된 README용 이미지를 갱신합니다.

```bash
python3 tdoa.py --csv signal_example/waveforms_noisy.csv --true-pos 0.4 0.6 --search-range-ms 2.5 4.2 --save docs/images/tdoa_position.png --save-signals docs/images/processed_signals.png --no-show
```

## 설치

Python 3.9 이상과 pip가 필요합니다. 프로젝트 폴더에서 가상환경을 만들고
NumPy, SciPy와 Matplotlib을 설치하세요.

Linux / macOS:

```bash
cd /path/to/8_TDOA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
cd C:\path\to\8_TDOA
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

위 경로는 실제 프로젝트 폴더 경로로 바꾸세요. Ubuntu / Debian에서
Python, pip 또는 `venv`가 설치되어 있지 않다면 먼저 설치합니다.

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

새 터미널에서는 가상환경 활성화 명령을 다시 실행하세요.
작업을 마친 뒤에는 `deactivate`로 가상환경을 종료할 수 있습니다.

## 실행

아래 명령은 프로젝트 루트에서 실행합니다. Linux / macOS:

```bash
python3 tdoa.py
```

Windows에서는 `python tdoa.py`로 실행하세요.

기본 입력은 `signal_example/waveforms_noisy.csv`입니다. 콘솔에 센서별
진단용 포락선 피크 시각(s/ms), S1 기준 교차상관 시간차, 추정 위치를 출력하고 두 그래프를
함께 엽니다. 신호 그래프에는 원본, 필터 결과, 포락선과 피크가 표시됩니다.

위치 그래프의 왼쪽에는 센서, 실제 음원 위치(알려진 경우), 추정 위치와 센서 1을 기준으로 한
TDOA 쌍곡선이 표시됩니다. 각 곡선은 해당 측정 시간차를 만족하는 위치의
집합입니다. 이상적인 측정에서는 음원 위치에서 만나며, 잡음이 있으면
곡선이 한 점에서 만나지 않거나 추정 위치와 차이가 생길 수 있습니다.

**TDOA 곡선은 시간차의 부호와 관계없이 모두 실선으로 표시합니다.**
색상과 범례로 S2−S1, S3−S1, S4−S1을 구분합니다. 오른쪽 막대그래프는
이 시간차를 밀리초 단위로 표시합니다.

```text
TDOA = 포락선 교차상관으로 추정한 해당 센서의 S1 대비 지연
양수: S1보다 늦은 신호 / 음수: S1보다 이른 신호
```

신호 그래프의 빨간 점선은 포락선 피크 시각을 나타냅니다.
정답 위치를 지정했을 때 위치 그래프에 표시되는 빨간 점선은 실제 위치와
추정 위치를 연결하는 오차 표시입니다.

이미지만 저장하려면 다음 명령을 사용하세요. GUI가 없는 환경에서도 실행됩니다.

```bash
python3 tdoa.py --save tdoa_visualization.png --save-signals processed.png --no-show
```

입력 파일과 교차상관 및 진단용 피크 탐색 구간을 지정할 수 있습니다.

```bash
python3 tdoa.py --csv signal_example/waveforms_clean.csv --search-range-ms 2.5 4.2
```

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `--csv PATH` | 입력 파형 CSV | `signal_example/waveforms_noisy.csv` |
| `--low-hz HZ` | 대역통과 필터 하한 | `30000` |
| `--high-hz HZ` | 대역통과 필터 상한 | `50000` |
| `--sound-speed VALUE` | 음속(m/s) | `343` |
| `--search-range-ms START END` | 교차상관·진단용 피크 탐색 및 신호 그래프 표시 구간(ms) | 전체 기록 |
| `--true-pos X Y` | 오차 평가용 실제 위치(m) | 미지정 |
| `--save PATH` | 위치 그래프 저장 | `results/tdoa_position.png` |
| `--save-signals PATH` | 처리 파형 그래프 저장 | `results/processed_signals.png` |
| `--no-show` | GUI 없이 실행 | GUI 표시 |

## 실제 데이터 연결

실제 데이터는 `--csv measurement.csv`로 입력하세요. CSV에는 `time_s`와
`s1_amplitude`부터 `s4_amplitude`까지 필요하며, 열 순서가 바뀌어도 센서
이름으로 맞춥니다. 실제 센서 좌표는 `tdoa.py`의 `SENSORS`에 S1부터 순서대로
미터 단위로 입력합니다. S1을 기준으로 위치를 계산합니다.

입력 헤더 예시:

```csv
time_s,s1_amplitude,s2_amplitude,s3_amplitude,s4_amplitude
```

각 행은 같은 시간 기준의 센서 샘플입니다. 시간은 초 단위이며 증가하는
균일한 간격이어야 합니다. 샘플링 주파수는 `time_s` 간격에서 계산합니다.
필터 상한은 샘플링 주파수의 절반 미만으로 설정하세요.
센서를 모두 일직선에 배치하면 현재 2차원 위치 추정에서 오류로 처리합니다.

실제 위치를 아는 경우 `--true-pos 0.4 0.6`을 추가하면 오차를 평가합니다.
현재 `EXAMPLE_TRUE_POS = None`이므로 기본 실행에서는 실제 위치와 오차가
표시되지 않습니다. 번들 합성 데이터의 실제 위치 `(0.4, 0.6)`을 기준으로
평가하려면 다음과 같이 실행하세요.

```bash
python3 tdoa.py --true-pos 0.4 0.6
```

정답 위치는 오차 평가와 그래프에만 사용하며 위치 추정 계산에는 사용하지
않습니다. `ground_truth.csv`와 `parameters.csv`는 통합 실행에서 자동으로
읽지 않습니다. 합성 조건을 바꾸면 `SENSORS`, 음속, 필터 대역과 평가용
정답 위치를 새 조건에 맞춰 설정하세요.

## 합성 데이터와 개별 실행

합성 파형을 재생성하려면 다음 명령을 사용합니다. 생성 스크립트 상단의
상수로 조건을 바꿀 수 있으며 기존 예제 CSV 4개를 덮어씁니다.

```bash
python3 signal_example/generate_synthetic.py
```

원본 파형과 정답 도착 시각만 확인하려면:

```bash
python3 -m visualization.waveform_visualization --truth signal_example/ground_truth.csv
```

자세한 데이터 형식은 [합성 데이터 설명](signal_example/README.md), 필터링과
피크 검출 방식은 [신호처리 설명](signal_process/README.md)을 참고하세요.

## 현재 처리 방식의 범위

포락선 교차상관 최대점에 3점 포물선 보간을 적용해 TDOA를 계산합니다.
출력과 신호 그래프의 포락선 피크는 진단용이며 위치 계산에는 사용하지 않습니다. 잡음·반사파·채널 응답 차이는
오차를 만들 수 있으며 신호 유무 판정은 아직 포함하지 않았습니다.
신호처리는 전체 기록을 사용하는 오프라인 방식입니다.

## 모든 솔버 비교

```bash
python3 run.py
python3 run.py --no-show
python3 run.py --csv signal_example/waveforms_clean.csv --output-dir results/clean --no-show
python3 run.py --csv measurement.csv --true-pos 0.4 0.6
```

`run.py`는 신호를 한 번 처리하고 동일한 TDOA로 모든 솔버를 실행합니다.
기본 clean/noisy 예제의 실제 위치는 `signal_example/ground_truth.csv`에서
평가용으로만 읽습니다. 다른 CSV에는 `--true-pos X Y`가 필요합니다.
센서 좌표는 `tdoa.py`의 `SENSORS`를 사용합니다.

기본 출력 폴더는 `results/solver_comparison/`이며, `comparison.png`와
`metrics.csv`를 저장합니다. 그래프는 센서/추정 위치, 솔버별 계산 시간,
위치 오차 막대그래프, 기준 거리 `r0` 절대오차 막대그래프를 포함합니다.
단일 기록의 오차 크기를 나타내며 통계적 신뢰구간은 아닙니다.
계산 시간은 각 솔버를 5회 워밍업한 후 매 라운드 실행 순서를 섞어 100회
측정합니다. 그래프 막대는 중앙값, 검은 오차 막대는 25~75 백분위 범위이며
신뢰구간은 아닙니다. 솔버 내부의 선형 초기화는 매 호출 포함하고 CSV 읽기·
공통 신호처리·오차 평가·시각화는 제외합니다. `--warmup`, `--repeats`,
`--timing-seed`로 횟수와 순서 생성 시드를 지정합니다. 실행 환경에 따라
시간은 달라질 수 있습니다. `metrics.csv`의 `solve_time_ms`는 중앙값이며,
`solve_time_p25_ms`, `solve_time_p75_ms`, 측정 횟수·설정·실패 상태도 저장합니다.

여기서 `r0`는 S1까지의 거리이며 기존 솔버의 `r0` 반환값입니다.
선형 솔버의 독립 미지수 `r0`을 그대로 평가하며, 그래프에는 솔버가 반환한
`r0`의 절대오차만 파란 막대로 표시합니다. `metrics.csv`에는 두 거리의
차이, 부호 있는 `r0` 오차, 절대오차와 목적함수도 기록합니다.
솔버 하나가 실패해도 다른 솔버를 실행하며 그래프에 FAILED를 표시하고
CSV에 원인을 기록합니다. 실패가 있으면 결과 저장 후 종료 코드 1을 반환합니다.

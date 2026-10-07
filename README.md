# TDOA 위치 추정 및 시각화

센서 4개의 도착 시간차(TDOA)로 음원의 2차원 위치를 추정합니다.

`tdoa.py`를 실행하면 다음 과정을 한 번에 처리합니다.

CSV 읽기 → 평균 제거 → 대역통과 필터 → 포락선 피크 검출 →
피크 시간차 계산 → 위치 추정 → 신호 및 위치 시각화.

- `tdoa.py`: 전체 실행과 `estimate_position()` 위치 추정
- `signal_process/process_ultrasound.py`: 필터링, 피크 검출, 시간 출력 및 신호 그래프
- `visualization/tdoa_visualization.py`: TDOA 위치 그래프
- `visualization/waveform_visualization.py`: CSV 읽기 및 원본 파형 그래프

합성 초음파 파형과 정답 도착 시간 CSV는 `signal_example/`에 있습니다.
생성 조건과 읽기 예제는 [합성 데이터 설명](signal_example/README.md)을 참고하세요.

필터링, 포락선 피크 시각 출력 및 그래프는 다음 명령으로 실행합니다.
자세한 옵션은 [신호처리 설명](signal_process/README.md)에 있습니다.

```bash
python3 -m signal_process.process_ultrasound
```

## 설치

Python 3와 pip가 필요합니다. 프로젝트 폴더에서 가상환경을 만들고
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

Linux / macOS:

```bash
python3 tdoa.py
```

Windows에서는 `python tdoa.py`로 실행하세요.

기본 입력은 `signal_example/waveforms_noisy.csv`입니다. 콘솔에 센서별
포락선 피크 시각(s/ms), S1 기준 시간차, 추정 위치를 출력하고 두 그래프를
함께 엽니다. 신호 그래프에는 원본, 필터 결과, 포락선과 피크가 표시됩니다.

위치 그래프의 왼쪽에는 센서, 실제 음원 위치(알려진 경우), 추정 위치와 센서 1을 기준으로 한
TDOA 쌍곡선이 표시됩니다. 각 곡선은 해당 측정 시간차를 만족하는 위치의
집합이며, 곡선들의 교점이 음원 위치에 해당합니다. 오른쪽 그래프는
센서별 시간차를 밀리초 단위로 표시합니다. 양수는 기준 센서보다 늦게,
음수는 더 일찍 도착했음을 뜻합니다.

이미지만 저장하려면 다음 명령을 사용하세요. GUI가 없는 환경에서도 실행됩니다.

```bash
python3 tdoa.py --save tdoa_visualization.png --save-signals processed.png --no-show
```

입력 파일과 피크 탐색 구간을 지정할 수 있습니다.

```bash
python3 tdoa.py --csv signal_example/waveforms_clean.csv --search-range-ms 2.5 4.2
```

`--low-hz`, `--high-hz`로 필터 대역(기본 30–50 kHz), `--sound-speed`로
음속(기본 343 m/s)을 변경합니다. `--search-range-ms`는 피크 탐색 및
신호 그래프 표시 구간에 모두 적용됩니다. `--no-show`는 화면 없이 실행합니다.

실제 데이터는 `--csv measurement.csv`로 입력하세요. CSV에는 `time_s`와
`s1_amplitude`부터 `s4_amplitude`까지 필요하며, 열 순서가 바뀌어도 센서
이름으로 맞춥니다. 실제 센서 좌표는 `tdoa.py`의 `SENSORS`에 S1부터 순서대로
미터 단위로 입력합니다. S1을 기준으로 위치를 계산합니다.

실제 위치를 아는 경우 `--true-pos 0.4 0.6`을 추가하면 오차를 평가합니다.
번들 clean/noisy CSV에는 기본 예제 정답 `EXAMPLE_TRUE_POS = [0.4, 0.6]`을
사용합니다. 합성 데이터의 음원 위치를 바꿨다면 이 값도 수정하거나
`--true-pos`로 지정하세요. 정답 위치는 추정 계산에 사용하지 않습니다.

현재 검출 시각은 포락선 최대점이며 첫 도착 시각과 다릅니다. 동일한 버스트
형태를 가정해 피크 차이를 TDOA로 사용합니다. 잡음·반사파·채널 응답 차이는
오차를 만들 수 있으며 신호 유무 판정은 아직 포함하지 않았습니다.
신호처리는 전체 기록을 사용하는 오프라인 방식입니다.

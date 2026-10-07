# TDOA 위치 추정 및 시각화

센서 4개의 도착 시간차(TDOA)로 음원의 2차원 위치를 추정합니다.

## 설치

Python 3와 pip가 필요합니다. 프로젝트 폴더에서 가상환경을 만들고
NumPy와 Matplotlib을 설치하세요.

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

왼쪽 그래프에는 센서, 실제 음원 위치, 추정 위치와 센서 1을 기준으로 한
TDOA 쌍곡선이 표시됩니다. 각 곡선은 해당 측정 시간차를 만족하는 위치의
집합이며, 곡선들의 교점이 음원 위치에 해당합니다. 오른쪽 그래프는
센서별 시간차를 밀리초 단위로 표시합니다. 양수는 기준 센서보다 늦게,
음수는 더 일찍 도착했음을 뜻합니다.

이미지만 저장하려면 다음 명령을 사용하세요. GUI가 없는 환경에서도 실행됩니다.

```bash
MPLBACKEND=Agg python3 tdoa.py --save tdoa_visualization.png --no-show
```

`tdoa.py`의 `sensors`, `true_pos`, `c` 값을 수정하여 센서 배치, 음원 위치,
음속을 바꿀 수 있습니다. 기본 예제는 잡음 없이 생성한 모의 측정값을 사용합니다.

# TDOA 위치 추정 및 시각화

센서 4개의 도착 시간차(TDOA)로 음원의 2차원 위치를 추정합니다.

```bash
python3 -m pip install -r requirements.txt
python3 tdoa.py
```

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

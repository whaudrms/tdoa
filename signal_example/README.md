# 합성 초음파 데이터

실제 측정 전 신호처리 개발용 모의 데이터입니다. 센서 배치와 음원 위치는
현재 `tdoa.py` 기본 예제와 같습니다. 생성기는 해당 설정을 독립적으로
보관하므로 `tdoa.py`를 수정해도 자동으로 바뀌지 않습니다.

| 파일 | 내용 |
| --- | --- |
| `waveforms_clean.csv` | 잡음 없는 동시 수신 파형, 4개 채널 |
| `waveforms_noisy.csv` | 같은 파형에 가우시안 잡음 추가, seed=42 |
| `ground_truth.csv` | 센서 좌표, 음원 좌표, 거리, 정답 도착 시각 및 TDOA |
| `parameters.csv` | 주파수, 샘플링, 음속, 잡음 등 생성 조건 |

가정: 40 kHz 반송파, Hann 포락선의 12주기 버스트(0.3 ms),
채널당 500 kHz 동시 샘플링, 8 ms 녹음(4,000개 샘플/채널).
송신은 녹음 시작 후 1 ms에 시작합니다. 진폭은 임의 단위이며 ADC 값이나
전압이 아닙니다. 반사파, 채널 시간차, ADC 양자화 및 센서 응답은 모델링하지
않았습니다. 샘플링 조건은 실제 아두이노의 지원 사양을 의미하지 않습니다.

파형 CSV의 `sample_index`는 0부터 시작하고 `time_s`는 공통 녹음 기준의
시간(초)입니다. `s1_amplitude`부터 `s4_amplitude`까지 센서 순서로 읽으세요.

정답 정의:

- `propagation_time_s = distance_m / sound_speed_m_per_s`
- `arrival_time_s = emission_time_s + propagation_time_s`: 직접파 버스트 시작
- `tdoa_to_s1_s = arrival_time_s - S1의 arrival_time_s`
- `arrival_sample_fractional`: 소수 샘플 단위의 이론적 도착 위치
- `first_sample_at_or_after_arrival`: 도착 시각 이상인 첫 샘플 인덱스

정답 시각은 반올림하지 않습니다. 버스트 시작에서 포락선은 0이므로
검출 임계값을 넘는 시각이나 상관 최대점이 정답 시작 시각과 반드시
일치하지는 않습니다. 검출 기준에 따른 지연을 따로 평가하세요.

재생성(프로젝트 루트에서):

```bash
python3 signal_example/generate_synthetic.py
```

같은 설정과 seed에서는 같은 데이터를 생성하며 기존 CSV 4개를 덮어씁니다.
조건을 바꾸려면 생성 스크립트 상단의 상수를 수정하세요.

읽기 예제:

```python
import numpy as np

data = np.genfromtxt('signal_example/waveforms_noisy.csv',
                     delimiter=',', names=True)
time_s = data['time_s']
raw = np.vstack([data[f's{i}_amplitude'] for i in range(1, 5)])
# raw.shape == (4, 4000)

truth = np.genfromtxt('signal_example/ground_truth.csv', delimiter=',',
                      names=True, dtype=None, encoding='utf-8')
tdoa_truth = truth['tdoa_to_s1_s'][1:]  # S2-S1, S3-S1, S4-S1; 단위 초
```

신호처리에는 파형을 입력하고, 정답 CSV는 검출 시간 및 위치 오차 평가에
사용하세요. 먼저 clean 파형으로 검증한 뒤 noisy 파형을 사용하세요.

## 파형 시각화

프로젝트 루트에서 실행하면 센서별 파형을 공통 시간축으로 표시합니다.
`--truth`를 지정하면 정답 도착 시각을 빨간 점선으로 함께 표시합니다.

```bash
python3 -m visualization.waveform_visualization --truth signal_example/ground_truth.csv
```

잡음 없는 파형을 보고 싶으면 `--csv signal_example/waveforms_clean.csv`를
추가하세요. 실제 데이터에는 정답이 없어도 `--csv`만 지정하여 사용할 수 있습니다.
진폭 표시는 현재 합성 데이터의 임의 단위입니다.

관심 구간 확대 및 이미지 저장(GUI 없는 환경):

```bash
MPLBACKEND=Agg python3 -m visualization.waveform_visualization --truth signal_example/ground_truth.csv --time-range-ms 2.5 4.2 --save waveforms.png --no-show
```

다른 코드에서 직접 호출할 수도 있습니다. 시간은 초, `raw`는
`(채널 수, 샘플 수)` 배열입니다.

```python
from visualization.waveform_visualization import visualize_waveforms

visualize_waveforms(time_s, raw)
```

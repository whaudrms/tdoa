# 초음파 신호처리와 피크 시간

위치 추정과 두 그래프까지 한 번에 실행하려면 프로젝트 루트에서
`python3 tdoa.py`를 사용하세요. 아래 명령은 신호처리만 단독 실행합니다.

프로젝트 루트에서 실행합니다.

```bash
python3 -m pip install -r requirements.txt
python3 -m signal_process.process_ultrasound
```

기본 입력은 `signal_example/waveforms_noisy.csv`입니다. 시간 열 `time_s`로
샘플링 주파수를 계산하고, 채널별 평균 제거 → 30–50 kHz 4차 Butterworth
대역통과 필터 → Hilbert 포락선 → 포락선 최대 샘플 탐색 순서로 처리합니다.
그래프에는 raw, 필터 결과, 포락선, 피크 시각을 표시합니다.
콘솔에는 각 센서의 피크 샘플 인덱스(0부터), 시각(s 및 ms), S1 기준
시간차(µs), 피크 기반 TDOA 배열(s)을 출력합니다.

구간을 제한하고 이미지로 저장하려면:

```bash
MPLBACKEND=Agg python3 -m signal_process.process_ultrasound --search-range-ms 2.5 4.2 --save processed.png --no-show
```

잡음 없는 데이터 비교:

```bash
python3 -m signal_process.process_ultrasound --csv signal_example/waveforms_clean.csv
```

`--low-hz`, `--high-hz`로 필터 대역을 변경할 수 있습니다. 입력 시간은
균일한 샘플 간격이어야 하고 상한 주파수는 샘플링 주파수의 절반 미만이어야 합니다.

피크 시각은 직접파 시작 시각이 아닙니다. 합성 버스트의 포락선 중심은
시작 후 약 0.15 ms이며, 같은 파형의 채널 간 피크 차이는 TDOA 비교에
사용할 수 있습니다. 이 코드는 선택 구간의 가장 큰 포락선 값을 고르므로
반사파나 잡음만 있는 입력에서도 피크가 출력됩니다. 신호 유무 검출과
첫 도착 판정은 아직 포함하지 않았습니다.

필터는 `sosfiltfilt`로 앞뒤 방향 처리하므로 전체 녹음 데이터를 사용하는
오프라인 처리입니다. 실제 스트리밍 처리와 검출 지연은 별도로 검증해야 합니다.

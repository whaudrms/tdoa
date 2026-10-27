# 초음파 신호처리와 교차상관 TDOA

기존 4채널 CSV(`time_s,s1_amplitude,s2_amplitude,s3_amplitude,s4_amplitude`)와
실행 명령, 옵션 기본값을 유지합니다. 위치 추정은 `python3 tdoa.py`,
솔버 비교는 `python3 run.py`, 신호처리만 실행하려면 아래 명령을 사용합니다.

```bash
python3 -m signal_process.tdoa_signal_processing
python3 -m signal_process.tdoa_signal_processing --search-range-ms 2.5 4.2 --save processed.png --no-show
```

처리 순서는 평균 제거 → 30–50 kHz 4차 Butterworth 대역통과 → Hilbert 포락선
→ 평균 제거·표준편차 정규화 → S1 기준 교차상관 → 3점 포물선 보간입니다.
핵심 필터링과 교차상관 함수는 `tdoa_signal_processing.py`를 공통 사용하며
채널 수에 맞게 동작합니다. 기존 진입점은 4채널 설정을 유지합니다.

`--low-hz`, `--high-hz`로 대역을 변경합니다. `--search-range-ms`는 전체 기록을
필터링한 뒤 교차상관과 진단용 피크 탐색에 사용할 공통 시간 구간을 선택하며,
신호 그래프 표시 범위에도 적용합니다. 구간에는 모든 채널의 버스트가 들어가야
합니다. 짧은 구간은 잘림과 평균 제거 영향으로 지연 추정에 편향을 만들 수 있습니다.

`process_signals()`의 기존 3개 반환값은 유지합니다. `process_measurement()`는
`filtered`, `envelope`, `peaks`, `peak_times`에 더해 `tdoa`, `lag_samples`,
`correlation_data`를 반환합니다. `tdoa.py`와 `run.py`는 이 함수를 함께 사용합니다.
피크 시각은 진단용이며 최초 도착 시각이나 교차상관 TDOA가 아닙니다.
콘솔과 신호 그래프에는 교차상관 TDOA를 별도로 표시하고, 위치 그래프와 솔버는
이 TDOA만 사용합니다. 양수는 대상 채널이 S1보다 늦다는 뜻입니다.

음속 기본값은 343 m/s이며 기존 `--sound-speed` 옵션은 유지합니다.
입력은 균일한 시간 간격이어야 합니다. `sosfiltfilt`는 전체 기록을 사용하는
오프라인 처리입니다. 반사파·서로 다른 버스트 형태는 오차를 만들 수 있으며,
첫 도착 판정과 신호 유무 판정은 별도입니다. 일정한 포락선은 오류 처리합니다.

모든 처리·검증·출력·그래프 함수는 `tdoa_signal_processing.py`에 통합되어 있습니다.
파일 직접 실행도 지원합니다:

```bash
python3 signal_process/tdoa_signal_processing.py --no-show
python3 -m signal_process.tdoa_signal_processing --demo --no-show
```

CSV는 기존 `s1_amplitude`~`s4_amplitude` 또는 `rx1`~`rx4` 열을 지원하며,
열 순서와 관계없이 센서 번호순으로 정렬합니다. 두 위치 추정 진입점도 같은
로더를 사용합니다. 원본의 `show_signal_plots()`와 `show_correlation_plots()`는
4채널을 지원하며 `show=False`로 Figure 목록을 받아 사용할 수 있습니다.

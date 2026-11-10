# Midterm 솔버별 사용 파라미터

이 문서는 `python3 run_midterm.py --no-show`로 생성한 현재 결과의 설정 기록입니다.
설정을 바꾸어 다시 실행하면 이 문서는 별도로 갱신해야 합니다.

## 공통 입력 및 신호처리

| 항목 | 적용값 |
|---|---|
| 음속 | 343 m/s |
| 샘플링 주파수 | 500,000 Hz (샘플 간격 2 µs) |
| 센서 순서 | 0_20, 20_0, 20_40, 0_30; 모두 sample_1 |
| 센서 좌표 [m] | (0, 0.20), (0.20, 0), (0.20, 0.40), (0, 0.30) |
| 기준 센서 | 0_20, 좌표 (0, 0.20) m |
| 입력 폴더 | real_signal/real_sample_261030/ |
| 필터 | Butterworth, butter의 N=4, 30–50 kHz, SOS 양방향 필터링 |
| 포락선 | Hilbert 해석 신호의 절댓값 |
| 시간차 | 평균 제거·표준편차 정규화 후 포락선 교차상관, 3점 포물선 보간 |
| 탐색 구간 | 전체 기록 (`search_range_ms=None`) |
| 전파 모드 | auto → 실측 개별 CSV 입력이므로 round-trip |
| 왕복 보정 | 측정 TDOA를 2로 나눈 후 위치 솔버에 전달 |
| 평가용 정답 위치 | (0.4, 0.2) m; 초기화나 최적화에는 사용하지 않음 |

## 솔버별 설정

| 파라미터 | Linear least squares | Gradient descent | Newton method | Gauss–Newton |
|---|---|---|---|---|
| OCP | linear_least_square | error_least_square | error_least_square | error_least_square |
| 최적화 변수 | x, y, r0 | x, y | x, y | x, y |
| 초기 위치 [m] | 불필요 | (0.1, 0.225) | (0.1, 0.225) | (0.1, 0.225) |
| 최대 업데이트 수 | 해당 없음 | 10,000 | 10,000 | 10,000 |
| 종료 기준 | 직접 계산 | gradient L2 norm ≤ 1e-8 | gradient L2 norm ≤ 1e-8 | gradient L2 norm ≤ 1e-8 |
| 업데이트 | NumPy pinv(A) @ b | -αg | solve(H, -g) | solve(JᵀJ, -g) |
| 스텝 크기 | 해당 없음 | 매 반복 α=1부터 backtracking | 1 고정 | 1 고정 |
| 스텝 축소 비율 | 해당 없음 | 0.5 | 없음 | 없음 |
| Armijo 계수 | 해당 없음 | 1e-4 | 없음 | 없음 |
| 선 탐색 최대 후보 평가 수 | 해당 없음 | 60 | 해당 없음 | 해당 없음 |
| Hessian | 해당 없음 | 사용 안 함 | 목적함수의 정확한 Hessian | JᵀJ 근사 |
| 감쇠 계수 | 해당 없음 | 해당 없음 | 없음 | 없음 |
| 거리 평활화 ε | 해당 없음 | 1e-9 m | 1e-9 m | 1e-9 m |
| 이번 실행의 실제 업데이트 수 | 0 (직접 계산) | 200 | 7 | 6 |

비선형 솔버의 초기값은 센서 좌표의 평균입니다. 초기값이 센서와 겹치면 공통
규칙으로 작은 오프셋을 적용하지만, 이번 배치에서는 오프셋이 필요하지 않습니다.

비선형 목적함수는 `F(p) = 0.5 * sum((d_i(p) - d_0(p) - c*Δt_i)^2)`이며,
`d_i(p) = sqrt(||p-s_i||² + ε²)`입니다. 모든 잔차에 동일한 가중치를 사용합니다.
선형 솔버는 별도의 `0.5 * ||Az-b||²` 문제를 풉니다. 선형 솔버의 r0는 독립 변수이며,
비선형 솔버의 r0는 추정 위치에서 기준 센서까지의 거리입니다.

반복 횟수는 완료한 위치 업데이트 수입니다. 수렴 확인과 Gradient descent의
내부 선 탐색 횟수는 포함하지 않습니다. 비선형 솔버는 초기점에서 이미 수렴하면
0회를 반환하며, 최대 반복 도달 또는 계산 실패는 성공 결과로 기록하지 않습니다.
`return_iterations=True`는 결과 수집 옵션으로, 최적화 경로를 바꾸지 않습니다.

## 시간 측정

- 솔버당 워밍업: 5회
- 솔버당 측정: 100회
- 솔버 실행 순서: 매 라운드 무작위, seed=42
- 표시값: 중앙값, 오차 막대: 25–75 백분위수
- 측정 범위: 초기화를 포함한 솔버 전체 호출; 신호처리와 그래프 생성은 제외
- `--lm-lambda=1e-3`는 Levenberg–Marquardt용이며 이 네 솔버에는 적용되지 않음

## 구현 근거

- `run_midterm.py`: 실행 옵션, 반복 횟수 수집, 벤치마크
- `sensor_config.py`: 센서 선택, 좌표, 샘플링 주파수
- `optimization/linear_least_squares.py`: 선형 계산 및 공통 초기화
- `optimization/gradient_descent.py`: Armijo 선 탐색 및 종료 기준
- `optimization/newton_method.py`: 순수 Newton 업데이트
- `optimization/gauss_newton.py`: 순수 Gauss–Newton 업데이트
- `ocp/error_least_square.py`: 비선형 목적함수와 거리 평활화

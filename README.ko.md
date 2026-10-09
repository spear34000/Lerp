<div align="center">

# Lerp

**언어 모델의 최적 배합을 후보당 몇 초 만에 찾고, 모든 주장을 검증 가능하게 남깁니다.**

[English](README.md) · [한국어](README.ko.md) · [레퍼런스](docs/REFERENCE.md) · [측정 결과](experiments/RESULTS.md) · [생태계 지도](ECOSYSTEM.md)

</div>

Lerp는 같은 베이스에서 나온 모델(**LoRA 어댑터 또는 전체 체크포인트, 밀집·멀티모달·MoE**)을 섞고, 모듈 그룹(attention, MLP, router 등)별 배합 비율을 탐색합니다.
`W = W0 + w · Δ`, 곧 선형 보간(linear interpolation)이 이름의 유래입니다. **새 모델을 처음부터 학습하지 않으며, 병합이 성공했다고 성능이 좋아졌다는 뜻도 아닙니다.**
좋은 배합을 싸게 찾고, 정말 좋아졌는지 증명하는 것이 목적입니다.

## 특징

- **후보당 몇 초**: `lerp search`는 모델 하나를 가속기에 올려 둔 채 후보를 제자리에서 덮어쓰고 바로 점수를 매깁니다. 0.5B 4초, Qwen3-4B 29초, OLMoE(MoE 69억, 전체 체크포인트) 44초입니다(후보마다 `lm-eval`을 새로 띄우면 각각 약 285초, 11~14분, 10분).
- **적은 평가로 탐색**: GP(가우시안 프로세스) 베이지안 탐색이 진화 탐색의 3분의 1 평가로 같은 품질을 찾았습니다.
- **모델 계열·벤치마크 독립**: 텐서 이름 규칙 하나로 MoE 라우터/전문가, 멀티모달 타워까지 처리하고, 낯선 구조는 설정(`tensor_rules`)으로, 객관식 벤치마크는 설정 파일의 `task:` 블록으로 추가합니다(코드 수정 없음).
- **검증 가능**: 입력과 출력을 SHA-256으로 고정하고, 탐색에 쓰지 않은 문항으로 다시 채점하며, 가짜 점수가 실제 결과에 섞이지 않게 막습니다.

## 빠른 시작

```bash
pip install "lerp[lora,gp,eval] @ git+https://github.com/spear34000/Lerp"
lerp --version

# 다운로드 없는 데모 (가짜 점수, 실제 결과로 보고되지 않음)
lerp init -c examples/multi_parent_demo.yaml -o runs/toy
lerp simulate -r runs/toy && lerp advance -r runs/toy --allow-simulated && lerp report -r runs/toy
```

같은 베이스에서 학습한 LoRA 두 개를 GP로 탐색하고 검증:

```bash
cp examples/gp_demo.yaml my.yaml            # base_model, parents 경로 수정
lerp check -c my.yaml
lerp init -c my.yaml -o runs/my && lerp freeze -r runs/my --strict
lerp cycle -r runs/my --rounds 3 --engine lora
lerp validate -r runs/my -c examples/holdout_evaluation.yaml --baseline all

# 또는 `lerp search`: 모델 하나를 올려 둔 채 후보를 제자리에서 섞고 몇 초 만에 채점 (LoRA·전체 체크포인트 모두)
lerp search -r runs/my --rounds 6 --baselines --device xpu
lerp build  -r runs/my -g 5 -i 0 --engine lora        # 우승 후보만 실제로 만들고 검증
```

Windows에서는 `PYTHONUTF8=1`을 설정하세요.

## 측정된 것 (작은 모델, 과제당 100~300문항이라 0.05 미만 차이는 노이즈)

| 질문 | 결과 |
|---|---|
| `lerp search` 속도 | 0.5B 285초 → 4초, Qwen3-4B 11~14분 → 29초, OLMoE 약 10분 → 44초. 점수는 `lm-eval`과 100문항당 1~3문항 이내로 일치 |
| 병합이 도움이 되나 | **능력이 서로 보완적일 때만.** ARC LoRA + BoolQ LoRA(Qwen2.5-0.5B)는 새 문항에서 0.78 대 부모 0.71/0.69(+0.07, 표준오차의 약 3배). 약하거나 중복되는 쌍은 더 나은 부모 대비 이득이 없었습니다 |
| GP vs 진화 | GP 8회 평가가 진화 24회 평가와 같은 수준(새 문항 적합도 0.739 vs 0.741). 진화는 무작위 탐색보다 낫지 않았습니다 |
| 큰/특이한 체크포인트 | Gemma 4 E4B(멀티모달 16GB) 5분, OLMoE-1B-7B(전문가 64개/층) 55초에 병합, 그룹별 가중치가 정확히 적용됨 |
| 다른 계열(Qwen3 + Gemma 4) | 가중치 병합 불가. `check`가 구조·어휘·토크나이저 불일치로 거부 |

## 지원 범위

| 지원 | 비고 |
|---|---|
| Linear, Task Arithmetic 병합 | 전체 체크포인트(`lite` 엔진 또는 MergeKit)와 LoRA(랭크 이어붙이기로 정확히 병합) |
| 층별, 모듈 그룹별 가중치 | attention / mlp / router / norm / embedding / other + 깊이 프로파일. MoE 라우터는 별도 그룹 |
| 탐색 | 진화, GP + 기대 개선, 무작위. `lerp search`는 상주 모델로 후보당 몇 초 |
| 평가 | YAML 블록으로 쓰는 객관식 로그확률 과제와 greedy 생성형 exact match 과제, `lerp cycle`로 lm-eval의 모든 과제 |

| 미지원 (현재) | 대신 일어나는 일 |
|---|---|
| SLERP | 미구현 |
| 자체 TIES / DARE | MergeKit 레시피로만 생성(`ties`, `dare_ties`, `dare_linear`). **MergeKit 실행은 이 환경에서 검증하지 않았고** 상주 평가기와 `lite` 엔진은 거부합니다 |
| 학습 CLI | `lerp train` 없음. LoRA 학습은 `experiments/train_lora.py` 스크립트 |
| DoRA, AdaLoRA, `rank_pattern`, bias / `modules_to_save`, 임베딩 LoRA | 어댑터를 오류로 거부(`check`, `build`) |
| 양자화 입력 (GPTQ, AWQ, GGUF, bitsandbytes 베이스 / QLoRA 베이스) | 먼저 dequantize 필요. 병합에는 완전 정밀도 베이스가 필요 |
| 구조 개조 (층 제거, 폭 변경, Dense <-> MoE 변환) | 미구현. 부모는 구조가 완전히 같아야 함 |
| 지식 편집 (ROME, MEMIT) | 범위 밖 |
| 다른 계열 (Qwen3 + Gemma 4) | 가중치 병합 불가, `check`가 거부 |

## 한계

- 파일만으로는 두 체크포인트가 같은 베이스에서 나왔는지 증명할 수 없습니다.
- 약 14GB를 넘는 병합본은 양자화 없이는 16GB GPU에서 평가할 수 없습니다(GGUF 백엔드 미구현). 병합 자체는 텐서 단위 스트리밍이라 가능합니다.
- `lerp search`는 객관식 로그 확률 과제(`acc`, `acc_norm`)와 greedy 디코딩 생성형 과제(`exact_match`, 예: GSM8K)를 채점합니다. 코드 실행(HumanEval, MBPP)은 지원하지 않습니다.
- 측정된 이득은 보완적인 한 쌍을 빼면 작고 대부분 노이즈 범위입니다. [측정 결과](experiments/RESULTS.md)를 보세요.
- 결과를 인용하기 전에 [비판적 검토](docs/CRITICAL_REVIEW_KO.md)와 [기술 감사](docs/V04_TECHNICAL_AUDIT.md)를 읽어 주세요. 한국어 설치 안내는 [QUICKSTART_KO.md](QUICKSTART_KO.md)에 있습니다.

## 로드맵

계획(순서대로): 부모 대비 쌍체 통계(McNemar), `lite` 엔진과 상주 평가기용 SLERP와 자체 TIES / DARE, 4B 코드·수학·한국어 LoRA 병합 실측, 후보 적용 속도 개선.
검토만 한 것(시작 전): `lerp train` 명령, GGUF / llama.cpp 평가 백엔드, 병합 전 층 제거. 하지 않을 것: Dense <-> MoE 변환, 차원 확장, 지식 편집, 자체 GPU 커널.

## 라이선스

Apache-2.0. ModelBreeder v0.4(MIT)에서 출발했으며 고지는 [NOTICE](NOTICE)에 있습니다. 예전 `modelbreeder` 명령도 별칭으로 계속 동작합니다.

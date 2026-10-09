# Lerp v0.3: 독립적이고 비판적인 기술 검토

검토 기준일: 2026-10-08. 대상: 이 ZIP에 포함된 v0.3 코드. 이 문서는 마케팅 문서가 아니라 **실패 조건까지 공개한 연구 도구 검증 기록**이다.

## 최종 판정

**연구용 알파(Alpha) / 실제 모델 성능 미검증.** 동일 계열의 작은 `safetensors` 텐서를 병합하는 수치 계산과 유전 탐색의 제어 흐름은 통과했다. 그러나 실제 PEFT/Transformers에서 생성한 어댑터를 로딩하여 추론하는 작업, 실 LLM 벤치마크를 실제로 끝까지 수행하는 작업, 다중 모델의 성능 개선 및 실서비스 운영은 검증되지 않았다. 그러므로 'AI 모델 성능 향상 엔진 완성', '새 모델을 학습했다'는 표현은 아직 부정확하다.

검증 범주 구분:
- **수치 실측:** Torch와 safetensors에서 실제 값이 들어 있는 소형 텐서를 만들고 저장/병합/재로드한 다음 계산 결과를 비교했다.
- **모의 통합:** 평가 프로그램 `lm-eval`과 `mergekit-yaml`의 외부 프로세스는 테스트에서 가짜 실행 함수로 대체했다. **실제 실행 성공을 뜻하지 않는다.**
- **기능 테스트:** 설정 검증, 불완전 결과 복구, 세대 생성, 순위 계산, 보고서 출력 등 제어 로직을 검증했다.
- **미실시:** 대형 LLM, 실제 PEFT 어댑터 추론, 실 GPU 장치, 실 벤치마크, 일반화 성능, 장기 안정성, 외부 모델 배포의 라이선스 적법성.

## 위험 기준

- **P0 / 출고 차단:** 여기서 오류가 나면 사용 자체가 깨지거나 모델 품질 주장을 신뢰할 수 없음.
- **P1 / 우선 개선:** 제한된 실험은 가능하지만 유효성, 확장성 또는 사용 편의성에 심각한 문제.
- **P2 / 개선 권장:** 운영/성능/관리 품질을 높이기 위한 과제.

상태 범주: `수정 및 테스트 완료`, `부분 검증`, `미검증`, `한계 공개`.

## A. 병합 수학 및 모델 호환성

| ID | 심각도 | 구체적인 검토 질문/공격 시나리오 | 확인 근거 및 판정 | 상태 |
|---|---|---|---|---|
| A01 | P0 | LoRA `A`와 `B`를 각각 평균 내면 교차항이 생기는가? | 단순 인자 평균을 사용하지 않고 `B'=[w1*s1*B1 ...]`, `A'=stack(A1 ...)`로 Rank를 연결하여 `B'A'=sum(wi*si*Bi*Ai)` 계산. 3부모 직접 수치 비교 `test_lora_3_parent_numerical_equivalence_per_module`. | 수정 및 테스트 완료 |
| A02 | P0 | 부모들의 Rank가 1/2/2처럼 서로 달라도 정확한가? | 입력 rank 합산 5, 출력 PEFT 설정 `r=5`, `alpha=5`, 활성화 스케일 1 사용. 서로 다른 Rank 실측 검사. | 수정 및 테스트 완료 |
| A03 | P0 | rsLoRA의 `alpha/sqrt(r)`와 일반 LoRA의 `alpha/r`가 혼동되는가? | 부모마다 별도 공식 사용. 수치 검증 테스트에 rsLoRA 부모 포함. | 수정 및 테스트 완료 |
| A04 | P0 | Attention과 MLP의 혼합 유전자가 실제 서로 다른 가중치에 적용되는가? | `tensor_group`과 레이어 인덱스를 이용해 LoRA/Full-lite 모두 실제 병합 텐서로 검증. | 수정 및 테스트 완료 |
| A05 | P0 | NaN/Inf 및 fp16 오버플로 때문에 잘못된 결과를 조용히 저장하는가? | 부모 인자에 NaN이 있으면 실패, 출력 dtype 변환 후 유한성 검사. float16 오버플로 테스트 통과. | 수정 및 테스트 완료 |
| A06 | P1 | BF16/FP16에서도 실수 연산이 정확히 일치하는가? | **아니다.** FP32로 가중 인자를 만든 뒤 출력 dtype으로 반올림하여 오차가 생김. BF16 파일 생성/유한성만 검증. 32-bit 정확성과 동일한 주장은 금지. | 부분 검증 |
| A07 | P0 | 부모 LoRA가 서로 다른 베이스 리비전에서 학습됐다면? | 어댑터가 선언한 `base_model_name_or_path`, `revision`, `target_modules`, `task_type`, key를 비교. 문자열이 같아도 가중치/리비전 동일성까지 증명할 수 없음. | 부분 검증 |
| A08 | P0 | PEFT가 생성된 어댑터를 실제 Transformer에 로드하고 토큰을 생성하는가? | PEFT+Transformers 실제 실행 환경이 없어 **로드 및 추론 미검증**. 프로덕션 출시 차단. | 미검증 |
| A09 | P1 | DoRA, aLoRA, bias, `modules_to_save`, embedding LoRA, 비균일 Rank, `alpha_pattern`을 지원하는가? | 지원 안 함. 인식 가능한 비표준 PEFT 설정과 텐서 키는 조용히 섞지 않고 거부. 모든 변형 유형의 전체 목록을 커버하는지는 미확인. | 부분 검증 |
| A10 | P1 | 부모가 6개이고 각각 Rank 64면 출력 Rank가 384로 커지는가? | **그렇다.** 압축 없는 정확한 연결이므로 디스크/VRAM/추론 지연도 증가. Rank >256 경고만 있고 예산 기반 실패 또는 Rank 압축은 없음. | 한계 공개 |
| A11 | P1 | tensor-name 분류가 Qwen/Llama 외 모든 아키텍처에서 정확한가? | 아니다. `layers/h/blocks` 및 명명 규칙 휴리스틱 사용. 무명 텐서는 `other`로, 알려지지 않은 LoRA 레이어 구조는 실패로 처리. 아키텍처별 검증 필요. | 부분 검증 |
| A12 | P1 | 일반 전체 체크포인트의 tokenizer / 구조 / 텐서 shape가 호환되는가? | 로컬 config/tokenizer 해시, safetensors header key/shape 사전 검사. 메타데이터만 동일한 서로 다른 가중치는 검출 못 함. | 부분 검증 |
| A13 | P1 | Full-lite `linear`와 `task_arithmetic`가 실제 텐서를 올바로 합치는가? | fp32 누산을 사용한 선형·베이스 상대 연산, 2/3부모/레이어별 테스트 통과. GPU 대형 샤드 및 훈련 이후 기능은 미실시. | 수정 및 테스트 완료 |
| A14 | P1 | `TIES`, `DARE`, `dare_linear` 구현을 실제 테스트했는가? | MergeKit YAML 레시피 생성 형식 검사만 통과. 실제 외부 MergeKit 실행과 품질 검증은 미실시. | 미검증 |
| A15 | P1 | task_arithmetic와 linear를 모두 자동 탐색하면 별개의 가설인가? | 정규화 가중치와 `task_scale=1`일 때 **수학적으로 동일**. `init`에서 중복 탐색 경고. `task_scale`까지 개별 유전자로 최적화하는 기능은 없음. | 부분 검증 |

## B. 평가 타당성·통계·최적화

| ID | 심각도 | 구체적인 검토 질문/공격 시나리오 | 확인 근거 및 판정 | 상태 |
|---|---|---|---|---|
| B01 | P0 | 가짜 점수를 실제 LLM 점수처럼 쓰는가? | 데모는 `SIMULATED_TOY`, `FAKE_DO_NOT_REPORT`로 표시, 기본 진화·자동 export 차단. CLI demo는 별도 `--allow-simulated` 필요. | 수정 및 테스트 완료 |
| B02 | P0 | 수동으로 `coding=1.0` 입력해 모델이 부모를 이겼다고 주장할 수 있는가? | 기본 진화에서 수동 점수 차단, `--allow-manual` 명시적 옵트인. `compare`는 `lm_eval` 점수 및 동일 설정의 부모 점수만 비교. | 수정 및 테스트 완료 |
| B03 | P0 | Screening에 3개 샘플만 평가한 모델을 최종 250샘플 모델과 동등 취급하는가? | Screening 결과는 `screen_score.json`으로 분리하고 상위 후보만 `score.json`을 받음. 실 평가 대신 모의 eval 프로세스로 흐름을 검증. | 수정 및 테스트 완료(제어 로직) |
| B04 | P0 | LoRA는 평가 시 베이스 모델만 로딩하고 어댑터를 빼먹는가? | CLI가 `pretrained=BASE,peft=ADAPTER`를 보내도록 수정. 명령어 인수만 모의 검증, PEFT 실제 적용 여부는 미확인. | 부분 검증 |
| B05 | P0 | 실제 `lm-eval run`이 설치된 환경에서 정확한 task/metric JSON을 읽는가? | 최신 CLI 문서 인터페이스와 호환되도록 작성했으나 **실 외부 프로세스 실행 미실시**. task별 key 변형 가능. | 미검증 |
| B06 | P0 | 훈련/최적화에 쓴 벤치와 최종 검증 벤치가 섞이는가? | 홀드아웃은 다른 task 이름 요구, 별도 폴더와 설정 해시로 저장, selection/leaderboard를 변경하지 않음. **문항·라벨·훈련 데이터의 실제 중복 검사는 없음.** | 부분 검증 |
| B07 | P1 | screening에서 항상 처음 N개 문항만 보면 결과가 편향되는가? | **그렇다.** `--limit N`은 무작위/층화 샘플링을 보장하지 않음. 별도 고정 시드 샘플 인덱스와 stratification 필요. | 한계 공개 |
| B08 | P1 | 같은 벤치에 100개 모델을 반복 최적화하면 우연한 고득점이 선발되는가? | **그렇다.** 다중 비교·승자 편향을 보정하지 않는다. holdout 1회와 다중 seed, CI/신뢰구간 추가 필요. | 한계 공개 |
| B09 | P1 | 벤치의 값이 모두 0..1, 클수록 좋다고 가정해도 되는가? | **아니다.** 현재 `compute_fitness`는 값 범위 [0,1], 전부 최대화 가정. Perplexity/latency(최소화), EM과 accuracy 혼합은 사전 정규화해야 함. | 한계 공개 |
| B10 | P1 | task별 중요도와 스케일이 타당한가? | YAML의 가중 평균과 최대-최소 gap penalty 사용. 가중치 선택에 대한 실증 검증 없고 데이터셋마다 점수 분산 다름. | 부분 검증 |
| B11 | P1 | Pareto가 표준 NSGA-II이며 최적해 품질을 보증하는가? | **아니다.** 비지배 정렬 + crowding 거리 + 순위 기반 부모 선택을 사용한 **NSGA-II 유사 탐색**. 표준 완전 구현/수렴 보장 아님. | 한계 공개 |
| B12 | P1 | 병합 방법 자동 선택이 모델이 스스로 최적화를 학습했다는 의미인가? | **아니다.** 현재 후보별 `method`를 부모에게서 유전하거나 25% 확률로 변경하는 단순 범주 탐색. Bayesian optimization / gradient / RL은 아님. | 한계 공개 |
| B13 | P1 | 보고서에 개선 점수가 있다면 실제 성능 향상이라고 확정할 수 있는가? | **아니다.** `compare`는 동등한 설정에서 측정한 수치적 차이만 표시. 신뢰구간·p-value·재현성·오염 방지는 추가 필요. | 부분 검증 |
| B14 | P1 | 홀드아웃 점수도 여러 차례 확인하며 튜닝하면 오염되는가? | **그렇다.** 엔진은 결과를 selection에 자동 입력하지 않을 뿐 사용자 행동을 제한하지 않음. 최종 홀드아웃은 실험 끝에 한 번만 열어야 함. | 한계 공개 |

## C. 안정성·재현성·보안·비용

| ID | 심각도 | 구체적인 검토 질문/공격 시나리오 | 확인 근거 및 판정 | 상태 |
|---|---|---|---|---|
| C01 | P0 | LoRA 모드인데 full model `config.json`을 찾느라 진입 실패하는가? | 모드별 별도 검사로 수정. synthetic PEFT adapter fixture로 `init`, `build` 검사 통과. | 수정 및 테스트 완료 |
| C02 | P0 | 평가 도중 프로세스가 죽으면 점수·모델이 반쯤 저장되는가? | `.model.partial`, `.evaluation.partial`, `.screening.partial`을 별도 작성하고 재시도 옵션으로 복구. 일부 파손 시 수동 개입 필요. | 부분 검증 |
| C03 | P0 | 세대 폴더가 완성되기 전에 state를 진행시키면 영구적으로 막히는가? | `.gen-NNN.partial` 완성 후 원자적 rename + commit 파일 + 상태 복구 테스트. 동일 파일시스템 rename 보장 전제. | 수정 및 테스트 완료 |
| C04 | P1 | 모델 병합 레시피/유전자를 누군가 수정한 뒤 원본 점수를 재사용하는가? | 설정 해시 및 레시피/재생성 결과 검증 후 build. 이미 생성한 모델 파일/외부 부모 원본 수정은 해시하지 않음. | 부분 검증 |
| C05 | P1 | 실험이 같은 모델 리비전·가중치로 재현되는가? | 실험 yaml 해시와 Python/플랫폼/seed 기록. **모델 파일 콘텐츠 SHA256, Hub commit hash, 의존성 lockfile은 없음.** | 부분 검증 |
| C06 | P1 | 동시에 `cycle` 두 개를 실행하면 state와 결과 파일이 덮이는가? | 잠금 구현 없음. **동일 run은 1개 프로세스로만 실행**해야 함. | 한계 공개 |
| C07 | P1 | 중단 재실행이 항상 완전한 exactly-once 처리를 보장하는가? | 일부 원자적 처리/복구만 구현. 외부 evaluator 완료-커밋 구간, 덮어쓰기 실패 등 모든 crash point를 시스템 수준에서 검증하지 못함. | 부분 검증 |
| C08 | P1 | 모델 파일 경로가 네트워크 다운로드/임의 코드 실행을 유발하는가? | 기본 CPU MergeKit-lite/LoRA는 로컬 safetensors만 직접 읽음. 외부 `lm-eval`/Transformers는 실행 시 모델 설정 및 `trust_remote_code` 정책을 별도로 확인해야 함. | 부분 검증 |
| C09 | P1 | 6부모 × 다세대에 대형 전체 모델을 반복 저장하는 디스크 폭발은? | `doctor`에서 대략적인 용량을 제시하지만 강제 quota, 후보 정리/압축, 중복 제거 없음. | 한계 공개 |
| C10 | P1 | 대규모 Torch CPU 병합 메모리·시간·IO는 어느 수준인가? | 1개 텐서씩 CPU 메모리 누산, 제한된 shard 크기 사용. 7B/27B/장기 실행 성능 벤치는 수행되지 않음. | 미검증 |
| C11 | P1 | Windows + Intel Arc XPU에서 GPU가속되는가? | **아니다.** 현재 Lite/LoRA는 CPU. MergeKit `--cuda`는 CUDA 환경용이며 Arc XPU 지원을 검증하지 않았음. | 한계 공개 |
| C12 | P1 | 모델 저작권과 토크나이저/가중치 재배포 조건을 검증하는가? | 라이선스 자동 검사 없음. 모델 카드에 별도 검토 명시. MIT 프로젝트 코드 라이선스가 부모 모델의 라이선스를 대체하지 않음. | 한계 공개 |
| C13 | P2 | 대시보드 HTML에 모델 이름 주입을 통한 스크립트 삽입 위험은? | 이름/텍스트 값은 HTML escape 처리. 브라우저 기반 보안 스캐너는 수행하지 않음. | 부분 검증 |
| C14 | P2 | 외부 패키지 설치 및 OS별 wheel 배포까지 통과했는가? | 소스 단위 CLI/테스트와 wheel 빌드는 검증 대상. 외부 라이브러리 설치/모델 실행은 네트워크 DNS 제한으로 실행 불가. | 부분 검증 |
| C15 | P2 | 자동화로 만든 자식 모델을 정식 신규 학습 모델로 부를 수 있는가? | **아니다.** 완성된 자식 가중치를 재훈련하거나 이어받는 진화가 아닌, 고정된 원본 부모 집합의 병합 *레시피*를 진화시킴. | 한계 공개 |

## 실제 수정한 결함과 회귀 테스트

1. LoRA용 경로에 full checkpoint validator가 잘못 적용되던 문제 → `compat.check_compatibility`; `test_lora_real_build_and_lora_model_args`.
2. `lm-eval`에 어댑터 대신 생성된 adapter 경로만 모델로 입력하는 문제 → `_eval_model_args`; 같은 테스트에서 `peft=` 확인.
3. 완료된 LoRA 결과를 full checkpoint 구조로 검사해 다시 병합하려는 문제 → `_model_complete`, `cycle`; `test_staged_lora_cycle_only_promotes_top`.
4. Screening 결과를 full evaluation과 혼동할 수 있는 문제 → 별도 JSON + promotion/score 분리; 위의 staged-cycle 테스트.
5. 여러 부모 LoRA Rank, rsLoRA, 모듈 혼합 수치 검증 부족 → `test_lora_3_parent_numerical_equivalence_per_module`, `test_lora_task_arithmetic_scales_deltas`, `test_lora_export_bfloat16_rank_sum_is_finite`.
6. full-lite 모듈별 혼합 구현 신뢰도 부족 → `test_lite_grouped_attention_vs_mlp_actual_tensors`.
7. NaN, 변환 후 overflow를 조용히 허용하던 문제 → `test_lora_rejects_nonfinite_factors`, `test_lora_cast_overflow_rejected`.
8. 중간 세대 생성 중 종료하면 다음 진화를 방해하던 문제 → `test_atomic_generation_state_recovery`, `test_generation_partial_requires_explicit_clean`.
9. 수동으로 입력한 100% 점수와 실 평가 점수를 동등하게 부모/자식 비교하는 문제 → `test_comparisons_excludes_unverified_manual_and_mismatched_protocol`, `test_manual_scores_must_be_explicitly_opted_into_evolution`.
10. Holdout 평가 결과를 탐색 모델 순위에 섞을 위험 → `test_heldout_separate_protocol_and_parent_comparison`, `test_heldout_rejects_benchmark_leakage`.

## 배포 전 Go / No-Go 체크리스트 (전부 완료해야 '실제 모델 병합 엔진' 표현 가능)

- [ ] **P0:** 임의로 만든 작은 PEFT 모델(실제 Transformers 모델 구조)에 `PeftModel.from_pretrained(base, child_adapter)`로 자식 어댑터가 오류 없이 로드되는가? (`examples/smoke_peft_load.py` 실행)
- [ ] **P0:** PEFT에서 부모 어댑터 2~3개를 적용한 직접 가중 합산 결과와 자식 PEFT 모델의 logits가 동일 출력 정밀도에서 허용 오차 이내인가?
- [ ] **P0:** 최소 0.5B급 동일 revision 베이스와 2개 실제 파인튜닝 어댑터를 이용해 자식을 만들고 **실제 prompt/token 생성**에 성공하는가?
- [ ] **P0:** `lm-eval run`을 진짜 실행해 결과 JSON, metric key, scoring과 모델 경로가 일치하는가?
- [ ] **P0:** 사용한 원본 베이스·부모 모델의 리비전(commit) 및 tokenizer가 동일한가?
- [ ] **P0:** 원본 베이스와 각 부모를 **동일한 벤치마크 구성**으로 측정한 점수와 비교하는가?
- [ ] **P0:** 최종 홀드아웃을 탐색 과정에 노출하지 않았고, 데이터 중복/유출 여부를 검토했는가?
- [ ] **P1:** 3개 이상 seed 또는 반복 평가를 수행하고 효과 크기 및 bootstrap 신뢰구간을 산출하는가?
- [ ] **P1:** 출력 Rank/CPU RAM/디스크 상한을 평가 전 예산 내로 제한하는가?
- [ ] **P1:** 서로 다른 model family/LoRA variant/텐서 이름을 오류 메시지와 함께 거부하는가?
- [ ] **P1:** 전체 merge (`mergekit-yaml`) 실제 실행과 모델 생성 후 추론 검증을 수행했는가?
- [ ] **P1:** 원본 모델 및 평가 데이터셋 재배포 라이선스를 준수하는가?
- [ ] **P1:** 완전한 패키지 wheel을 깨끗한 Windows와 Linux 가상환경에서 설치·CLI 실행하는가?
- [ ] **P2:** 동시 실행 락, 저장 공간 GC, 취소/재시작 정책, 기록/로그 보존 정책이 있는가?

## 합리적 최소 실험 설계

1. **동일한 frozen base**에서 훈련된 코드 특화 LoRA 및 추론 특화 LoRA를 준비한다. 서로 다른 모델 계열 또는 다른 베이스에서 파생된 모델은 일단 제외.
2. 0.5B~1.5B 모델 / 부모 2개 / 개체 4개 / 2개 제어점 / 2개 세대부터 시작하고 모든 아티팩트를 보관한다.
3. **정확성 단계:** FP32 merged LoRA의 `B@A`와 개별 부모 LoRA 업데이트들의 가중합을 비교한 뒤 실제 PEFT 모델 logits 비교를 실시한다.
4. **탐색 단계:** 고정된 개발용 task 세트로 screening과 final eval을 분리한다. 빠른 평가 제한(`--limit`)은 정식 성능 주장에 사용하지 않는다.
5. **비교 단계:** 원본 베이스 및 각 부모를 동일한 tokenizer/template/few-shot/평가 디바이스로 측정하고, win rate/분야별 변화/시간/메모리를 함께 기록한다.
6. **통계 단계:** 후보 선정을 마친 뒤 완전히 별도 검증 세트를 1회 실행한다. 겹치는 문항을 제거하고 bootstrap CI와 다중 비교 영향을 고려한다.
7. **출시 단계:** 모델 카드에 부모 이름/리비전/라이선스/병합 비율/방법/토큰 비용/한계/실측 결과를 명시한다.

## 외부 표준 및 유사 선행 프로젝트

- PEFT 모델 병합 개요 및 `add_weighted_adapter`: https://huggingface.co/docs/peft/main/developer_guides/model_merging
- EleutherAI lm-evaluation-harness 공식 CLI: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
- MergeKit 지원 알고리즘: https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md
- MergeKit의 기존 진화 탐색: https://github.com/arcee-ai/mergekit/blob/main/docs/evolve.md

**차별화에 대한 비판:** 진화 기반 자동 병합 탐색 자체는 새 발명이 아니다. MergeKit에도 진화 탐색이 존재한다. Lerp의 상대적 가치 후보는 (1) 일반 PEFT LoRA의 정확한 Rank 연결, (2) 모듈별 혼합 유전자, (3) 명확히 분리된 screening/holdout, (4) 실험 이력·크래시 복구·오프라인 보고서에 있다. 그렇지만 기존 프로젝트 대비 속도/품질/편의성 우위는 비교 실험이 없으므로 주장할 수 없다.

## 검증 메타데이터

- 테스트 실행 명령: `python -m pytest -q`
- 소형 실제 텐서 테스트: Torch + safetensors 사용, CPU.
- lm-eval/MergeKit 외부 통합 테스트: monkeypatch/mock. 실제 GPU/대형 LLM 실행 아님.
- 테스트의 수량은 **테스트 실행기 결과를 따르며**, 테스트 수는 품질 개선이나 추론 정확도를 증명하지 않는다.
- 이 문서에서 '수정 완료'라고 적힌 부분도 현재 소스와 테스트 환경에서 확인한 범위에 한정된다.

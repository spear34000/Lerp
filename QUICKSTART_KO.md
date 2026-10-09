# Lerp v0.5 한국어 빠른 시작

## 1. 설치

Windows PowerShell (프로젝트 폴더에서):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
```

필요한 기능별 추가 설치: 실제 CPU 전체 가중치 병합 `pip install -e ".[lite]"`, LoRA 교배 `pip install -e ".[lora]"`, MergeKit `pip install -e ".[merge]"`, 실제 성능 평가 `pip install -e ".[eval]"`.

외부 패키지는 네트워크에 따라 설치되지 않을 수 있음. LoRA 가중치 계산 테스트는 PEFT/Transformers가 설치되지 않은 환경에서도 Torch+safetensors로 수행했지만, **실제 PEFT 모델 로딩 검증은 별도 필요**.

## 2. 다운로드 없이 가짜 진화 데모

```powershell
python -m lerp init -c examples/multi_parent_demo.yaml -o runs/demo
python -m lerp simulate -r runs/demo
python -m lerp advance -r runs/demo --allow-simulated
python -m lerp simulate -r runs/demo
python -m lerp report -r runs/demo
```

`runs/demo/dashboard.html` 열기. `SIMULATED_TOY`는 실제 모델 벤치마크 점수가 **아님**. 수동 점수로 진화하려면 `advance --allow-manual`을 명시적으로 설정해야 하며 일반 실험에는 권장하지 않음.

## 3. 실제 LoRA 진화

`examples/lora_experiment.yaml`을 복사한 뒤 `base_model` 및 `parents`에 동일 베이스 모델에서 훈련한 PEFT LoRA 어댑터 2~6개를 지정. 세 파일 중 하나라도 다르면 호환성 검사를 통과하지 못할 수 있음.

```powershell
python -m lerp doctor -c my-lora.yaml
python -m lerp check -c my-lora.yaml
python -m lerp init -c my-lora.yaml -o runs/lora
python -m lerp freeze -r runs/lora --strict
python -m lerp verify-inputs -r runs/lora
python -m lerp build -r runs/lora -g 0 -i 0 --engine lora
python -m lerp baseline -r runs/lora --name all
python -m lerp cycle -r runs/lora --rounds 2 --engine lora
python -m lerp board -r runs/lora --pareto
python -m lerp report -r runs/lora
```

- `screening_limit`: 모든 후보에 적용할 저비용 평가 샘플 수
- `promote_top`: 다음 단계 평가를 받을 후보 수
- `limit`: 최종 평가 샘플 수. 이 값이 있으면 공식 벤치가 아니라 **스모크 테스트**
- `gene_groups`: Attention/MLP 등 별도 가중치 유전자. `other` 반드시 포함
- `method: auto`: 후보별 병합 방식 탐색. LoRA는 `linear` 또는 `task_arithmetic`만 지원
- `task_scale: 1`: LoRA 선형 병합과 Task Arithmetic이 수학적으로 사실상 동일하므로 탐색 의미가 작아짐

### 실제 PEFT 로딩 및 토큰 생성 P0 게이트

```powershell
python examples/smoke_peft_load.py --base ./checkpoints/base --adapter ./runs/lora/generations/gen-000/cand-000/model --device cpu
```

이 검증 스크립트는 제공하지만, 배포 환경에서 실제 PEFT 모델 로딩을 성공적으로 수행한 것은 **아님**. Torch/PEFT/Transformers를 설치하고 적합한 실 모델 파일을 별도로 준비해야 함.

## 4. 최종 독립 검증

검색에 사용하지 않은 평가셋을 설정해 검증:

```powershell
python -m lerp validate -r runs/lora -c examples/holdout_evaluation.yaml --baseline all
```

탐색 벤치와 **서로 다른 task 이름**만 사용하도록 차단했지만, 학습 데이터와의 중복까지 자동으로 검사하지는 않음. 결과는 `runs/lora/validation/`에 분리 저장하며 자동 진화에 반영하지 않음.

## 5. 결과 내보내기

```powershell
python -m lerp status -r runs/lora
python -m lerp export -r runs/lora -o runs/winner
```

CPU/디스크가 제한적이면 0.5B~1.5B 베이스로 시작하고, 부모 2~3개, 개체 4~6개, 유전자 제어점 2~4개부터 실험. **작은 가중치의 산술 검증은 완료됐지만 대형 LLM의 품질 개선 및 PEFT 실제 로딩은 미검증**.

출시 전 반드시 `CRITICAL_REVIEW_KO.md`의 남은 문제를 확인.


## v0.4: Extra verification commands

The `freeze --strict` step SHA-256 hashes every local checkpoint and rejects absent weights or mutable remote references. Run it before recording scores. `verify-inputs` checks for changes; model outputs are separately pinned. These hashes do not authenticate an untrusted file provider.

```powershell
python examples/offline_peft_equivalence.py
python -m lerp compare-samples --candidate examples/paired_candidate.jsonl --baseline examples/paired_baseline.jsonl --out runs/paired_report.json
python -m lerp audit-splits --development examples/development.jsonl --holdout examples/heldout.jsonl
```

The offline PEFT script needs actual installed `peft` and `transformers`; it was skipped in the packaged test run due to missing optional dependencies. The supplied JSONL datasets are synthetic, not real LLM scores. For more, read `V04_TECHNICAL_AUDIT.md`.

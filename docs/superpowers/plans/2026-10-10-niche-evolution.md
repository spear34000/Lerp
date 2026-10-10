# 틈새 기반 모델 진화 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 엘리트 LoRA 어댑터 보관소(틈새별)에서 보완적인 부모를 골라 병합하고 짧게 학습시켜 다음 세대의 부모로 올리는 루프(`lerp niche`)를, CPU 8GB 하한과 Arc 검증 프로필로 만든다.

**Architecture:** 순수 로직(격자, 보관소, 선택기, 단계 평가 규칙, 예산)은 torch 없이 테스트하고, 학습·병합·평가는 주입 가능한 `Operators`/`score` 함수로 감싼다. 루프는 가짜 연산자로 CPU에서 수 초 안에 통합 테스트하고, 실측은 별도 스크립트로 한다. 기존 `lerp/evolution/*`는 import만 하고 수정하지 않는다.

**Tech Stack:** Python 3.10+, pytest, torch/peft/transformers/safetensors(기존 extras `lora`), PyYAML, 기존 `lerp.statistics.mcnemar`.

**Spec:** `docs/superpowers/specs/2026-10-10-niche-evolution-design.md`

## Global Constraints

- 새 코드는 `lerp/niche/`, 테스트는 `tests/test_niche_*.py`, 실험은 `experiments/niche_*.py`에만 둔다. `lerp/evolution/*`, `experiments/restart_*`, `experiments/RESULTS.md`는 수정 금지(다른 세션 소유). 예외: `lerp/cli.py`에 `niche` 하위 명령 추가만 허용.
- 칸당 엘리트 최대 k=2. 격자: 점수 3구간(경계 1/3, 2/3), 축은 founder dev 점수 분산 상위 3개, 0세대 평가 직후 고정(최대 27칸).
- 단계 평가: 1차 스킬당 40문항(창 `(0, 40)`), 2차 스킬당 200문항(창 `(40, 240)`), 두 창은 겹치면 안 된다(assert). 최종 test 문항은 루프가 보지 않는다.
- 학습 rank 16(프로필 `cpu-8gb`는 8~16), 압축은 `compress_adapter`, 어댑터는 `out_scale=2.0`으로 저장.
- 프로필 `cpu-8gb`: Qwen2.5-0.5B, 세대당 후보 3~4, 짧은 학습 30~50스텝, batch 1~2, RSS 6GB 이하 감시. 프로필 `arc`: 후보 6~8, 100스텝, rank 16.
- 후보 실패는 그 후보만 `failed`로 기록하고 계속한다. 한 세대의 실패율이 50%를 넘으면 중단한다. 비유한 가중치는 보관소 진입 전에 거부한다.
- 성공 기준은 `lerp/niche/criteria.py`의 상수로 실행 전에 고정하고 사후 변경하지 않는다: 보관소 최고 개체가 같은 학습 스텝·데이터 구성·시작 어댑터·rank의 단일 어댑터 대조군보다 스킬 평균에서 높고 쌍체 McNemar p<0.05, 시드 3개 이상, 기존 스킬 손실 0.05 이내(참고), `cpu-8gb` 한 세대 완주.
- Arc GPU는 하나다. 실측 실행 전에 `Lerp - 모델 병합` 세션에 `SendMessage`로 알리고 그쪽 평가가 끝났는지 확인한다.
- 구현은 격리된 git worktree에서 한다(`superpowers:using-git-worktrees`). 커밋은 자기 파일만 `git add`하고, 사용자가 허락한 경우에만 한다.
- **스펙과의 차이(검토 요청):** 스펙 5절의 "문자열 변환" 스킬을 정수 정답 스킬(`count_vowels`, `digit_sum`, `max_of`)로 대체한다. 기존 `problems.verify`(첫 정수 비교), `sft_pair`, `task_definition`, `Evaluator`를 수정 없이 재사용하기 위해서다.

## Review Focus

1. 보관소가 비었을 때 1차 평가 기준선: 모두 통과해야 한다(0으로 나누기·빈 평균 금지). Task 5.
2. 밀려난 엘리트의 어댑터가 살아 있는 다른 개체의 graft `init`이면 파일을 지우면 안 된다. Task 3.
3. 후보 전부가 실패하거나 비유한일 때 세대가 죽지 않고 보관소가 변하지 않아야 한다. Task 7, 8.
4. 개체 작성 도중 프로세스가 죽어 반쯤 쓰인 어댑터 폴더가 남았을 때, 재개가 그것을 건너뛰고 다시 만들어야 한다. Task 6, 8.
5. 점수가 동점이거나 문항 수가 다른 두 개체의 비교: 동점은 교체하지 않고, 문항 정렬이 어긋나면 명확한 오류. Task 3.

---

### Task 1: 합성 스킬 3종

**Files:**
- Create: `lerp/niche/__init__.py`, `lerp/niche/skills.py`
- Test: `tests/test_niche_skills.py`

**Interfaces:**
- Consumes: `lerp.evolution.problems.FAMILIES` (dict[str, Callable[[random.Random], tuple[str, int]]]), `problems.splits`, `problems.verify`.
- Produces: `NICHE_FAMILIES: tuple[str, ...] = ("count_vowels", "digit_sum", "max_of")`. 모듈 import 시 세 생성기를 `problems.FAMILIES`에 등록한다. 이후 `problems.splits("digit_sum", ...)`가 그대로 동작한다.

- [ ] **Step 1: 실패하는 테스트.** `test_answers_are_exact`(각 가족 pool 50개에서 질문을 파싱해 정답을 독립 계산하고 `row["a"]`와 같음), `test_splits_are_disjoint`(`splits(f, 100, 40, 200, seed=1)`에서 train/dev/test 질문 집합이 서로소), `test_verify_accepts_exact_and_rejects_wrong`, `test_registration_is_idempotent`(모듈 두 번 import해도 `FAMILIES` 키 수 불변).
- [ ] **Step 2: 실패 확인.** `pytest tests/test_niche_skills.py -v` → ImportError로 FAIL.
- [ ] **Step 3: 구현.** `count_vowels`("Q: vowels in 'hello'"), `digit_sum`, `max_of`(숫자 4개) 생성기를 `_family(rng) -> tuple[str, int]` 시그니처로 작성하고 import 시 등록. 질문 문자열 형식은 기존 `_add`와 같은 스타일.
- [ ] **Step 4: 통과 확인.** 같은 명령 → PASS.
- [ ] **Step 5: 커밋.** `git add lerp/niche/__init__.py lerp/niche/skills.py tests/test_niche_skills.py` / `feat(niche): synthetic integer-answer skills`

### Task 2: 격자(축 선택과 칸 계산)

**Files:** Create `lerp/niche/grid.py`; Test `tests/test_niche_grid.py`

**Interfaces:**
- Produces: `BINS = 3`; `choose_axes(profiles: list[dict[str, float]], families: list[str], n_axes: int = 3) -> list[str]`(분산 내림차순, 동률은 가족 이름순); `cell_of(profile: dict[str, float], axes: list[str]) -> tuple[int, ...]`(각 축 `min(BINS-1, int(score*BINS))`).

- [ ] **Step 1: 테스트.** `test_choose_axes_picks_highest_variance`(분산이 큰 가족 3개가 선택되고 동률은 이름순), `test_cell_boundaries`(0.333→0, 0.334→1, 0.666→1, 0.667→2, 1.0→2, 0.0→0), `test_fewer_families_than_axes`(가족 2개면 축 2개), `test_single_profile_has_zero_variance_tie_break`.
- [ ] **Step 2: 실패 확인.** `pytest tests/test_niche_grid.py -v` → FAIL.
- [ ] **Step 3: 구현** (`statistics.pvariance` 사용).
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 커밋.** `feat(niche): niche grid axes and cells`

### Task 3: NicheArchive

**Files:** Create `lerp/niche/archive.py`; Test `tests/test_niche_archive.py`

**Interfaces:**
- Consumes: `lerp.evolution.archive.Organism`, `lerp.statistics.mcnemar(a, b) -> dict`(키 `p_value`, `a_only`, `b_only` 등. 정확한 키 이름은 구현 시 `lerp/statistics.py:123-160`에서 확인).
- Produces:
  - `@dataclass Entry: organism: Organism; cell: tuple[int, ...]; items: dict[str, list[int]]` (가족별 문항 정오).
  - `def pooled(items: dict[str, list[int]]) -> list[int]`(가족 이름순 연결).
  - `def mcnemar_better(cand: Entry, inc: Entry, p: float = 0.05) -> bool`(문항 수가 다르면 `ValueError`, 평균이 높고 p<0.05일 때만 True, 동점은 False).
  - `class NicheArchive(root: Path, k: int = 2, better: Callable[[Entry, Entry], bool] = mcnemar_better)` with `insert(entry: Entry) -> Literal["added", "replaced", "rejected"]`, `elites() -> list[Entry]`, `cells() -> dict[tuple[int, ...], list[Entry]]`, `save()/load()`(`niche_archive.json`, 원자적 쓰기).
  - 규칙: 칸에 k개 미만이면 추가. 가득 차면 가장 약한 엘리트(풀링 평균 최저)와 `better`로 비교해 이기면 교체. 밀려난 개체의 어댑터 폴더는, 살아 있는 다른 개체의 `training["init"]`이 가리키지 않을 때만 삭제.

- [ ] **Step 1: 테스트.** `test_empty_cell_adds`, `test_full_cell_replaces_weakest_only_when_better`, `test_tie_does_not_replace`(Review 5), `test_mismatched_item_counts_raises`(Review 5), `test_evicted_adapter_kept_when_referenced_as_init`(Review 2: 폴더가 남음) 및 `test_evicted_adapter_deleted_when_unreferenced`, `test_save_load_roundtrip`.
- [ ] **Step 2: 실패 확인.** `pytest tests/test_niche_archive.py -v` → FAIL.
- [ ] **Step 3: 구현.** 어댑터 경로는 `Organism.adapter`(archive root 상대).
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 커밋.** `feat(niche): niche archive with referenced-adapter-safe eviction`

### Task 4: Selector

**Files:** Create `lerp/niche/selector.py`; Test `tests/test_niche_selector.py`

**Interfaces:**
- Consumes: `Entry`, `pooled` (Task 3).
- Produces: `STALE_BONUS = 0.02`; `complementarity(a: Entry, b: Entry) -> float`(풀링 문항 중 정확히 한쪽만 맞힌 비율); `pick_parents(entries: list[Entry], last_picked: dict[str, int], generation: int, rng: random.Random) -> tuple[Entry, Entry]`(서로 다른 두 개체, 가중치 = `complementarity + STALE_BONUS * min(10, generation - last_picked.get(id, -1))`, 가중치 합 0이면 균등).

- [ ] **Step 1: 테스트.** `test_complementarity_values`(동일=0, 정반대=1), `test_pick_is_distinct_and_deterministic_for_seed`, `test_prefers_complementary_pair`(1000회 샘플에서 보완 쌍이 비보완 쌍보다 유의하게 자주), `test_stale_entry_gets_bonus`, `test_fewer_than_two_entries_raises`, `test_all_zero_weights_falls_back_to_uniform`.
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 커밋.** `feat(niche): complementarity-weighted parent selection`

### Task 5: StagedEvaluator

**Files:** Create `lerp/niche/staged.py`; Test `tests/test_niche_staged.py`

**Interfaces:**
- Produces:
  - `STAGE1 = (0, 40)`, `STAGE2 = (40, 240)`, `STAGE1_TOLERANCE = 0.05`; 모듈 import 시 두 창이 겹치면 `AssertionError`.
  - `ScoreFn = Callable[[dict[str, Path | None], tuple[int, int]], tuple[dict[str, dict[str, float]], dict[str, dict[str, list[int]]]]]` (= 기존 `evolution.orchestrator.Evaluator.score`와 같은 계약: 이름→가족→정확도, 이름→가족→문항 정오).
  - `def baseline(archive_means: list[float]) -> float | None`(비어 있으면 None).
  - `def stage1_survivors(scores: dict[str, dict[str, float]], baseline: float | None, tol: float = STAGE1_TOLERANCE) -> list[str]`(`baseline is None`이면 전부 통과, 아니면 가족 평균 ≥ `baseline - tol`).
  - `class StagedEvaluator(score: ScoreFn)` with `first(adapters) -> tuple[dict, dict]`, `second(adapters) -> tuple[dict, dict]`(각각 `STAGE1`, `STAGE2` 창으로 `score` 호출).

- [ ] **Step 1: 테스트.** `test_windows_are_disjoint`, `test_empty_archive_everyone_passes`(Review 1), `test_stage1_drops_below_baseline_minus_tol`, `test_first_and_second_use_their_windows`(가짜 `score`가 받은 창을 기록), `test_overlapping_windows_assert`(모듈 상수를 모킹해 겹치면 오류).
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 커밋.** `feat(niche): staged evaluation rules`

### Task 6: 예산, 프로필, 체크포인트

**Files:** Create `lerp/niche/budget.py`; Test `tests/test_niche_budget.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Profile: name, device, dtype, rank, candidates, child_steps, batch, accum, rss_limit_gb`; `PROFILES: dict[str, Profile]` with `"cpu-8gb"`(device cpu, dtype float32, rank 16, candidates 3, child_steps 40, batch 1, accum 4, rss_limit_gb 6.0)와 `"arc"`(xpu, bfloat16, rank 16, candidates 6, child_steps 100, batch 4, accum 2, rss_limit_gb 12.0).
  - `class BudgetExceeded(RuntimeError)`; `class MemoryGuard(limit_bytes: int, rss: Callable[[], int] = process_rss_bytes)` with `check() -> None`(초과 시 `BudgetExceeded`); `process_rss_bytes() -> int`(Windows는 `ctypes`+`GetProcessMemoryInfo`, 그 외 `resource`/`/proc`).
  - `class Checkpoint(path: Path)` with `save(state: dict) -> None`(임시 파일 후 `os.replace`), `load() -> dict | None`, `mark_done(organism_id: str)`, `is_done(organism_id: str) -> bool`.

- [ ] **Step 1: 테스트.** `test_profiles_match_spec_values`, `test_guard_raises_above_limit`(가짜 rss 주입), `test_guard_silent_below_limit`, `test_checkpoint_roundtrip_and_atomic`(저장 중 예외를 일으켜도 이전 파일이 온전), `test_partial_organism_not_marked_done`(Review 4: 어댑터 폴더는 있으나 done 표식 없음 → `is_done` False), `test_process_rss_is_positive`.
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 커밋.** `feat(niche): hardware profiles, memory guard, checkpoint`

### Task 7: Operators (병합, 학습, 압축, 검증)

**Files:** Create `lerp/niche/operators.py`; Test `tests/test_niche_operators.py`

**Interfaces:**
- Consumes: `evolution.combine.combine_adapters(parts, dest, *, out_scale=2.0)`, `graft_parts(primary, secondary, secondary_init)`, `evolution.learning.train_adapter(base, pairs, out, *, init, steps, lr, rank, seed, device, dtype, batch, accum, ordered, schedule_total, schedule_offset, state_in, state_out) -> dict`, `compress_adapter(src, dst, rank, out_scale=1.0) -> dict`, `file_sha256`.
- Produces: `class NonFiniteAdapter(ValueError)`; `def assert_finite(adapter_dir: Path) -> None`; `class Operators(base_model: str, profile: Profile)` with
  - `cross(primary: Path, secondary: Path, secondary_init: Path | None, mode: Literal["graft", "blend"], weight: float, dest: Path) -> dict`,
  - `learn(adapter: Path, pairs: Sequence[tuple[str, str]], dest: Path, *, seed: int) -> dict`(연속 학습: `init=adapter`, 프로필의 steps/batch/accum/device/dtype/rank),
  - `compress(src: Path, dest: Path) -> dict`,
  - `finalize(dest: Path) -> str`(`assert_finite` 후 SHA-256 반환).
  모든 메서드는 실패 시 `dest` 폴더를 지운 뒤 예외를 다시 던진다.

- [ ] **Step 1: 테스트.** (`tests/test_evolution_fix.py`의 `_random_adapter` 패턴 재사용, 학습은 `train_adapter`를 monkeypatch) `test_cross_graft_matches_combine`, `test_cross_blend_weights`, `test_assert_finite_rejects_nan`(Review 3), `test_failed_op_leaves_no_partial_dir`(Review 4), `test_learn_passes_profile_values`(monkeypatch된 `train_adapter`가 받은 kwargs가 프로필 값과 `init=adapter`), `test_finalize_returns_sha`.
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 커밋.** `feat(niche): operators with finite check and cleanup on failure`

### Task 8: 세대 루프와 재개

**Files:** Create `lerp/niche/loop.py`; Test `tests/test_niche_loop.py`

**Interfaces:**
- Consumes: Tasks 1–7 전부. 기존 `evolution.orchestrator.Evaluator`는 실제 실행에서만 `score` 로 주입.
- Produces:
  - `@dataclass NicheConfig: base_model: str; profile: str; families: list[str]; founders: list[str]; seed: int; generations: int; cross_mode: str = "graft"; n_train: int = 2000; mix: dict[str, float] | None = None`.
  - `def run_niche(cfg: NicheConfig, out: Path, *, operators: Operators | None = None, score: ScoreFn | None = None, make_pairs: Callable[[list[str], int], list[tuple[str, str]]] | None = None, log: Callable[[str], None] = print) -> dict`. 반환 dict: `{"generations": int, "filled_cells": int, "failed": int, "best": str, "archive": str}`.
  - 흐름은 스펙 4절 그대로: 0세대에 founder 평가 → `choose_axes` → 세대마다 `pick_parents` → `cross` → `learn`(부모 프로필에서 가장 약한 가족 우선 `make_pairs`) → `compress` → `finalize` → 1차 → 생존분 2차 → `NicheArchive.insert`. `Checkpoint`로 개체마다 저장하고 `MemoryGuard.check()`를 개체마다 호출.
  - 한 세대의 실패율>0.5면 `RuntimeError("failure rate ...")`.

- [ ] **Step 1: 테스트(전부 가짜 Operators/score, CPU 수 초).** `test_one_generation_fills_archive`, `test_deterministic_for_seed`(같은 시드 두 번 실행 시 보관소 동일), `test_resume_equals_uninterrupted`(세대 중간에 예외로 중단 후 재실행한 결과가 무중단 결과와 같음), `test_half_written_organism_is_rebuilt`(Review 4), `test_all_candidates_failing_keeps_archive_and_aborts_on_majority`(Review 3), `test_failed_candidate_is_recorded_and_loop_continues`, `test_memory_guard_exceeded_stops_cleanly`, `test_stage2_only_for_survivors`.
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.** `pytest tests/test_niche_*.py -v`
- [ ] **Step 5: 커밋.** `feat(niche): generation loop with resume and failure policy`

### Task 9: 대조군과 판정 기준

**Files:** Create `lerp/niche/criteria.py`, `lerp/niche/control.py`; Test `tests/test_niche_criteria.py`

**Interfaces:**
- Produces:
  - `CRITERIA = {"beats_control_p": 0.05, "min_seeds": 3, "retention_tolerance": 0.05}`(상수, 사후 변경 금지).
  - `def verdict(best: dict[str, list[int]], control: dict[str, list[int]], founders_best: dict[str, float], best_acc: dict[str, float]) -> dict`(풀링 평균 차이, `mcnemar` 결과, `beats_control: bool`, `retention_ok: bool`, 사람이 읽는 `text`).
  - `def seeds_verdict(per_seed: list[dict]) -> dict`(시드 수<3이면 `"insufficient_seeds"`, 모두 이겨야 `"shown"`, 일부면 `"mixed"`, 아니면 `"not_shown"`).
  - `def run_control(cfg: NicheConfig, out: Path, total_steps: int, *, train: Callable[..., dict] = train_adapter) -> Path`: 시작 어댑터=founder 0.5/0.5 blend, 같은 rank·`problems.stream` 혼합(가족 정확히 균등), 연속 학습 `total_steps`.

- [ ] **Step 1: 테스트.** `test_verdict_requires_average_win_and_p`, `test_single_skill_win_is_not_enough`, `test_retention_flag`, `test_seeds_verdict_levels`(2시드→insufficient), `test_criteria_constants_are_frozen`(spec 값과 같음), `test_control_matches_loop_total_steps_and_mix`(가짜 `train`이 받은 steps·rank·stream 구성을 확인).
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 커밋.** `feat(niche): pre-registered criteria and compute-matched control`

### Task 10: CLI, 설정 로더, 실측 스크립트

**Files:**
- Modify: `lerp/cli.py` (`niche` 하위 명령과 디스패치 추가만; 기존 `evolve` 근처 `lerp/cli.py:146`, `:333` 패턴)
- Create: `lerp/niche/config.py`, `experiments/niche_run.py`, `experiments/niche_cpu8gb.py`, `examples/niche_cpu8gb.yaml`
- Test: `tests/test_niche_cli.py`

**Interfaces:**
- Produces: `def load_config(path: Path) -> NicheConfig`(YAML, 알 수 없는 키·알 수 없는 프로필·`founders`가 `families`에 없음은 `ValueError`); `lerp niche -c CONFIG -o OUT [--control-steps N]`; `experiments/niche_run.py`는 시드 3개를 돌리고 `criteria.seeds_verdict` 결과를 `OUT/verdict.json`에 쓴다; `experiments/niche_cpu8gb.py`는 Windows Job Object로 프로세스 메모리를 8GB 미만으로 제한(`limit_process_memory(bytes) -> None`)하고 `cpu-8gb` 프로필로 한 세대를 돌려 완주 여부와 최대 RSS를 `OUT/cpu8gb.json`에 기록한다.

- [ ] **Step 1: 테스트.** `test_load_config_valid`, `test_load_config_rejects_unknown_key_profile_and_bad_founders`, `test_cli_niche_parses_and_dispatches`(`run_niche`를 monkeypatch), `test_limit_process_memory_blocks_large_alloc`(자식 프로세스에서 제한 후 큰 할당이 실패, Windows 전용 `skipif`).
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.** `pytest tests/test_niche_*.py -v` 전체 PASS, 그리고 기존 `pytest tests -q`가 이전과 같은 결과인지 확인(회귀 없음).
- [ ] **Step 5: 커밋.** `feat(niche): config loader, CLI, experiment scripts`

### Task 11: 실측과 기록 (코드 변경 없음, 결과 문서)

**Files:** Create `docs/niche_results.md`(다른 세션의 `experiments/RESULTS.md`와 분리)

- [ ] **Step 1: 조율.** 실행 전 `Lerp - 모델 병합` 세션에 `SendMessage`로 GPU 사용 시작을 알리고, 그쪽 평가가 끝났는지와 restart factorial 결론을 받는다. 결론이 연속 학습에 불리하면 `Operators.learn`의 기본값을 정한다(없으면 연속 학습).
- [ ] **Step 2: CPU-8GB 완주.** `python experiments/niche_cpu8gb.py --out runs/cpu8gb`(PYTHONUTF8=1) → `cpu8gb.json`에서 `completed: true`, `peak_rss_gb < 6.0`.
- [ ] **Step 3: Arc 3시드.** `python experiments/niche_run.py --profile arc --seeds 1 2 3 --out runs/arc`. 소요가 한 시간을 넘길 것 같으면 세대 수·후보 수를 줄여 사용자에게 먼저 알린다(짧은 실험 선호).
- [ ] **Step 4: 기록.** `docs/niche_results.md`에 시드별 표, 대조군, `verdict`를 있는 그대로 적는다. 기준을 못 넘으면 "못 이겼다"고 쓰고, 한계(소표본, 1 모델 크기, 합성 스킬)를 명시한다. 기준·상수는 수정하지 않는다.
- [ ] **Step 5: 커밋.** `docs(niche): measured results`

---

## Self-Review 결과

- **스펙 커버리지:** 1·2절(목표·기준) → Task 6, 10, 11 / 3절 구성 요소 → Task 2–7 / 4절 흐름 → Task 8 / 5절 평가 과제 → Task 1(2단계 실제 벤치마크는 1단계 기준 성립 후 별도 계획) / 6절 프로필 → Task 6, 10 / 7절 오류 처리 → Task 3, 7, 8 / 8절 테스트 → 각 Task / 9절 성공 기준 → Task 9, 11 / 10절 조율 → Global Constraints, Task 11. 빠진 항목 없음.
- **타입 일관성:** `Entry`, `pooled`, `ScoreFn`, `Profile`, `Operators`, `NicheConfig` 이름이 정의 Task와 사용 Task에서 같다.
- **분량:** 코드 본문은 시그니처와 테스트 이름 위주로 두었다.

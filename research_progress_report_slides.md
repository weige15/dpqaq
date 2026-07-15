# QAQ / DP-LLM 進度報告投影片草稿

> 用途：直接貼到簡報。每頁列出「頁面內容」與「要貼的圖片名稱」。圖片只列檔名，方便手動貼圖。
> 更新日期：2026-07-16
> 更新範圍：納入 2026-07-09 之後的 held-out quality、LODO predictor、shared-profile execution 與 real CUDA validation 結果。

## 第 1 頁：題目與目前定位

頁面內容：

- 題目：Precision-Aware Dynamic Batching for Mixed-Precision LLM Serving
- 核心目標：設計並評估能根據 request 精度需求組成 batch 的 serving 方法，改善 throughput、latency 與 memory efficiency，同時維持 model quality。
- QAQ 定位：QAQ-style query-adaptive precision routing 是目前用來提供 mixed-precision profile 的機制，不是完整研究問題本身。
- 目前進度：已完成單一 request QAQ 路徑、DP threshold guard、real trace collection、trace-driven simulation、held-out quality evaluation，以及 shared-profile execution 的 CPU 與 bounded real-CUDA validation。
- 目前結論：shared-profile path 已能正確執行，但 predictor transfer 尚未通過 preregistered gate，且目前 CUDA case 沒有證明 throughput speedup。

要貼的圖片名稱：

- 無

## 第 2 頁：研究問題

頁面內容：

- 普通 dynamic batching 通常根據 arrival time、sequence length 或 queue state 組 batch，沒有充分利用 request 之間不同的 precision demand。
- 單獨的 query-adaptive routing 能為 request 選擇不同 bit-width，但不保證這些決策適合 batched execution。
- 研究問題：能不能將 request 的 mixed-precision profile 納入 dynamic batching，在品質與 deadline 可控的前提下改善 throughput 與 latency？
- 必須分開驗證：batching policy 的效果、precision routing 的效果，以及兩者組合後是否真的有 serving benefit。
- 目前方法不是重新訓練模型，而是在 Any-Precision weights 上加 runtime precision control。

要貼的圖片名稱：

- 無

## 第 3 頁：目前整體架構

頁面內容：

```text
Any-Precision quantized weights
        |
        v
QAQRouter / held-out predecode predictor
- route id embedding
- hidden/norm feature
- optional DP estimated-error feature
- output: selected bit or group precision demand
        |
        v
QAQDPLLM_Linear
- fixed_low / fixed_high
- mlp_multibit
- dp_threshold_only
- mlp_multibit_dp_guard
- shared_profile: one supplied bit per route
        |
        v
QAQDPLLMForCausalLM
- load router / predictor artifacts
- replace quantized linears
- shared profile across prefill and decode
- collect route and safety statistics
        |
        v
Trace / quality evaluator / scheduler / CUDA benchmark
```

重點：

- QAQ router 或 held-out predictor 提供 request-level precision demand。
- `max_profile_sharing` 對實際 batch 計算 component-wise maximum，再投影到各 route 的合法 bits。
- shared execution 會固定整個 batch 的 route map；router、confidence fallback、DP guard 在純 shared path 中不重複執行。
- Route-safety audit 與 task-level quality 是兩種不同的證據，不能互相替代。

要貼的圖片名稱：

- 無

## 第 4 頁：Router 訓練與標籤來源

頁面內容：

- Router 標籤不是 random label，也不是 mock label。
- 對每個 captured activation `x`，比較 reference bit 和 candidate bit 的輸出誤差。
- 標籤定義：

```text
rel_error_b = ||W_ref x - W_b x|| / (||W_ref x|| + eps)
label = 最小的安全 bit b，使 rel_error_b <= error_threshold
```

- Binary mode：只允許兩個 bit，例如 3-bit / 6-bit。
- Multibit mode：在 3、4、5、6 bit 中選最小安全 bit。
- Checkpoint 會保存 router state dict、candidate bits、route map、label mode、threshold、training stats。
- 最新 quality evaluator 仍使用真實 activation、reference-bit output error、finite-logit 檢查與 held-out token NLL / perplexity。

要貼的圖片名稱：

- 無

## 第 5 頁：Runtime 模式與比較對象

頁面內容：

目前支援的 runtime modes：

| Mode | 作用 |
|---|---|
| `fixed_low` | 固定使用最低可用 precision |
| `fixed_high` | 固定使用最高 precision，作為保守 baseline |
| `mlp_multibit / qaq` | 使用 QAQ router 預測 precision |
| `dp_threshold_only` | 只使用 DP-LLM threshold decision |
| `mlp_multibit_dp_guard` | QAQ router + DP threshold guard |
| `max_profile_sharing` | 使用 held-out group demand 的 batch shared profile |

比較邏輯：

- `fixed_low` / `fixed_high` 建立低精度與高精度 baseline。
- `qaq` / `dp_threshold_only` / `mlp_multibit_dp_guard` 分離 routing 與 guard 的效果。
- `max_profile_sharing` 驗證 scheduler-supplied profile 能否真正穿過 prefill 與 decode execution path。
- Predictor transfer 尚未建立，因此部署決策仍保留 conservative fixed-high fallback。

要貼的圖片名稱：

- 無

## 第 6 頁：Serving-aware dynamic batching 架構

頁面內容：

```text
Prompt set / request stream
        |
        v
Real QAQ trace + prompt-only predictors
- per-request selected bits / group demand
- uncertainty and conservative fallback lane
        |
        v
Profile extraction
- scalar bit budget
- coarse block precision profile
- workload type / arrival time / prompt length
        |
        v
Scheduler
- ordinary dynamic batching
- scalar budget batching
- block profile batching
- max / quantile profile sharing
        |
        v
Executor
- fixed-high or grouped QAQ baseline
- shared route profile across prefill and decode
        |
        v
Quality audit + CUDA-synchronized benchmark
```

目前定位：

- Trace 是 real QAQ generation；simulator 是 policy selection tool，不是 performance claim。
- v2 shared-profile execution 已完成，且 bounded real-CUDA run 已通過。
- Predictor 的 cross-dataset transfer gate 尚未通過，因此 scheduler integration 尚未被視為可部署的 adaptive decision。

要貼的圖片名稱：

- 無

## 第 7 頁：截至目前已完成項目

頁面內容：

- 已完成 QAQ runtime wrapper：`QAQDPLLMForCausalLM.py`
- 已完成 runtime quantized linear：`QAQDPLLM_Linear.py`
- 已完成 train / inference / benchmark / trace / simulation scripts。
- 已加入 DP threshold guard：`mlp_multibit_dp_guard`
- 已完成真實 held-out quality evaluation：WikiText-2 與 C4，各 16 windows、context length 512、8,176 scored target tokens。
- 已完成 strict LODO predictor evaluation：四個 source、三個 seeds、support-matched rerun。
- 已完成 `shared_profile(...)` execution contract；targeted CPU gate 33 tests、full `tests/router` 93 tests passed。
- 已完成 bounded real-CUDA shared-profile comparison：RTX 3090、3 repeats、8 requests、最大 batch size 4。

目前 evidence：

- `doc/qaq-profile-batching-benchmark.md`
- `doc/qaq-lodo-three-dataset-results.md`
- `doc/performance-profile.md`
- `figures/gen_fig_qaq_current_progress.py`
- `artifacts/qaq_current_progress_20260716/figures/figure_manifest.json`
- `/tmp/qaq-task-quality-wikitext2-current.json`、`/tmp/qaq-task-quality-c4-current.json`
- `/tmp/qaq-shared-profile-lower-demand-gpu4.json` 與其 separate quality audit

備註：`/tmp` benchmark JSON 沒有加入 repository；它們是可重現命令產生的 validation artifacts。

要貼的圖片名稱：

- 無

## 第 8 頁：Real held-out quality 結果

頁面內容：

設定：

- Any-Precision Llama 3.1 8B、candidate bits：3、4、5、6。
- WikiText-2 test 與 C4 validation，各 16 個 held-out windows。
- Context length：512；每個 dataset 8,176 scored target tokens。
- Metric：teacher-forced perplexity；finite logits 均為 true。
- 這是 QAQ quality baseline（產生於 shared-profile v2 之前），不等同於 shared-profile 的 task-level quality。

結果：

| Dataset | Mode | Perplexity | Delta vs fixed-high | Effective bits | Fallback | DP guard |
|---|---|---:|---:|---:|---:|---:|
| WikiText-2 | `fixed_low` | 12.7770 | +3.2600 | 3.0000 | 0.000% | 0.000% |
| WikiText-2 | `dp_threshold_only` | 9.8755 | +0.3586 | 4.4103 | 0.000% | 0.000% |
| WikiText-2 | `mlp_multibit` | 9.6590 | +0.1420 | 5.1828 | 42.053% | 0.000% |
| WikiText-2 | `mlp_multibit_dp_guard` | 9.6225 | +0.1055 | 5.1958 | 42.053% | 1.206% |
| WikiText-2 | `fixed_high` | 9.5170 | +0.0000 | 6.0000 | 0.000% | 0.000% |
| C4 | `fixed_low` | 15.0702 | +3.1779 | 3.0000 | 0.000% | 0.000% |
| C4 | `dp_threshold_only` | 12.3132 | +0.4209 | 4.4490 | 0.000% | 0.000% |
| C4 | `mlp_multibit` | 12.0831 | +0.1908 | 5.1148 | 37.261% | 0.000% |
| C4 | `mlp_multibit_dp_guard` | 12.0455 | +0.1533 | 5.1287 | 37.261% | 1.496% |
| C4 | `fixed_high` | 11.8923 | +0.0000 | 6.0000 | 0.000% | 0.000% |

解讀：

- DP guard 在這個 held-out evaluator 上略微降低 perplexity gap，但沒有回到 fixed-high quality。
- QAQ 達成約 5.1--5.2 effective bits，而不是 3-bit baseline 的成本；fallback rate 仍很高。
- 這頁建立 quality evidence，但尚未證明 dynamic batching 的 task-level quality。

要貼的圖片名稱：

- `artifacts/qaq_current_progress_20260716/figures/fig_heldout_quality_perplexity.png`（quality / perplexity comparison）

## 第 9 頁：單一 request runtime baseline（歷史結果）

頁面內容：

設定：

- GPU：NVIDIA RTX 3090；Any-Precision Llama 3.1 8B packed checkpoint。
- Candidate bits：3、4、5、6；prompt count：1；max new tokens：16。
- Warmup：1；repeat：3；CUDA path 可執行且 logits finite。

結果：

| Mode | Mean latency | Tokens/s | Effective bits | Fallback | DP guard |
|---|---:|---:|---:|---:|---:|
| `fixed_low` | 0.501 s | 31.96 | 4.27 | 0.00 | 0.00 |
| `fixed_high` | 0.497 s | 32.18 | 6.00 | 0.00 | 0.00 |
| `qaq` | 2.992 s | 5.35 | 5.47 | 0.176 | 0.00 |
| `dp_threshold_only` | 2.017 s | 7.93 | 5.09 | 0.00 | 0.00 |
| `mlp_multibit_dp_guard` | 3.911 s | 4.09 | 5.48 | 0.176 | 0.008 |

解讀：

- Router / estimator / grouping overhead 在單一 request path 上仍然明顯。
- 這是 runtime integration baseline，不是 current shared-profile performance result。
- 因此後續 benchmark 必須把 routing overhead、shared execution 與 model-loading cost 分開量測。

要貼的圖片名稱：

- 無

## 第 10 頁：Real trace 結果：不同 workload 的 selected bits

頁面內容：

設定：

- 200 requests，共 4 類 workload：`chat`、`code`、`math`、`summarization`，各 50 筆。
- Router mode：`mlp_multibit_dp_guard`；max new tokens：8。

結果：

| Workload | Avg selected bit mean | Effective bits mean | Fallback total | DP guard triggers |
|---|---:|---:|---:|---:|
| chat | 5.7465 | 5.7175 | 0 | 1607 |
| code | 5.7921 | 5.7593 | 0 | 1557 |
| math | 5.8037 | 5.7782 | 0 | 1688 |
| summarization | 5.9879 | 5.9861 | 0 | 1049 |
| overall | 5.8326 | 5.8103 | 0 | 5901 |

解讀：

- Real QAQ trace 能收集每個 request 的 bit profile。
- `summarization` 比其他 workload 更接近 6-bit，顯示 workload-dependent precision demand。
- 這是 descriptive result，不是品質或速度 claim；summarization completion 長度混合，不能直接做 latency comparison。

要貼的圖片名稱：

- `fig_bits_by_workload.png`

## 第 11 頁：Profile diversity 結果

頁面內容：

結果：

- Total trace records：200
- Routes per record：224
- Unique coarse profiles：18
- Scalar bucket 0.25-bit unique：3
- Scalar bucket 0.10-bit unique：5
- Overall coarse profile fragmentation ratio：0.09
- Mean pairwise route expected-bit distance：0.1312

Workload profile diversity：

| Workload | Coarse profiles | Scalar 0.25-bit buckets |
|---|---:|---:|
| chat | 9 | 2 |
| code | 11 | 2 |
| math | 11 | 2 |
| summarization | 1 | 2 |

解讀：

- Request profile 不是完全隨機，存在可重複的 profile。
- Scalar bucket 比 coarse profile 更粗，適合作為第一版 batching signal 的候選。
- 這個結果支持「precision profile 可以作為 batching signal」的研究方向，但不代表 predictor 已能跨 dataset 泛化。

要貼的圖片名稱：

- `fig_profile_fragmentation_by_workload.png`
- `fig_coarse_profile_occupancy_heatmap.png`

## 第 12 頁：Prompt-only predictor 的 transfer 結果

頁面內容：

設定：

- 最終 support-matched LODO rerun：1,280 requests，640 development、160 calibration、480 test。
- Dataset：WikiText-2、C4、FineWeb-Edu、HellaSwag；seeds：17、29、43。
- Predictor 只使用 prompt / prefill features；continuation、observed route 與 quality signal 僅作為 targets。

Preregistered endpoint gates：

| Held-out dataset | Safe bit | Effective bits | Group profile |
|---|---:|---:|---:|
| WikiText-2 | 0/3 seeds pass | 0/3 pass | 0/3 pass |
| C4 | 3/3 pass | 3/3 pass | 3/3 pass |
| FineWeb-Edu | 3/3 pass | 3/3 pass | 0/3 pass |
| HellaSwag | 3/3 pass | 0/3 pass | 0/3 pass |

解讀：

- Overall `predictability_established: false`。
- HellaSwag support-matched rerun 的 effective-bit R² 從約 -4.91 改善到 -1.55，但仍為 negative，不能算通過。
- 因此 scheduler integration 暫時 disabled；保守 fixed-high fallback 仍是有效 deployment policy。

要貼的圖片名稱：

- `artifacts/qaq_current_progress_20260716/figures/fig_predictor_lodo_gates.png`（LODO gate heatmap）

## 第 13 頁：Simulator 結果與 policy selection

頁面內容：

同一份 200-request trace 上比較多個 batching policy：

| Policy | Batches | Mean batch size | Simulated p95 latency | Simulated req/s |
|---|---:|---:|---:|---:|
| ordinary dynamic batching | 50 | 4.00 | 89742 ms | 2.136 |
| block profile batching | 50 | 4.00 | 89742 ms | 2.136 |
| max profile sharing | 50 | 4.00 | 92489 ms | 2.075 |
| quantile profile sharing | 50 | 4.00 | 92489 ms | 2.075 |
| scalar budget batching | 51 | 3.92 | 80862 ms | 2.356 |

解讀：

- Simulator 將 `scalar_budget_batching` 選為候選，但 simulator 本身不是 throughput evidence。
- 實作驗證目前先完成 `max_profile_sharing`，因為它可直接把完整 route profile 套用到 prefill 與每個 decode step。
- `quantile profile sharing` 仍未完成；simulator 的 policy ranking 不應直接外推成 GPU ranking。

要貼的圖片名稱：

- `fig_simulated_policy_tradeoff.png`
- `fig_lane_occupancy_by_policy.png`

## 第 14 頁：Shared-profile execution 的最新 real-CUDA 結果

頁面內容：

設定：

- GPU：NVIDIA RTX 3090；real Any-Precision CUDA kernels；commit：`7fd62b1`。
- WikiText-2 held-out request stream：8 requests；max batch size：4；max wait：50 ms。
- Candidate bits：3、4、5、6；warmup：1；measured repeats：3。
- `fixed_high` 與 `max_profile_sharing` 使用相同 request stream；timing 有 CUDA synchronization。

結果：

| Policy | Latency p50 / p95 | Generated tokens/s | Effective bits | Shared rows | Profile exact |
|---|---:|---:|---:|---:|---:|
| `fixed_high` | 905.337 / 1146.733 ms | 33.4677 | 6.000 | 0% | — |
| `max_profile_sharing` | 916.279 / 1170.720 ms | 33.0366 | 6.000 | 100% | 5,376 / 5,376 |

Separate route-safety audit：

- 241,920 real low-bit/reference-bit decisions。
- Underprecision：0（0.000%）；exact：25,050（10.355%）；over-precision：216,870（89.645%）。
- Shared execution fallback：0；DP guard：0；profile padding：0。

解讀：

- Execution contract 已真正穿過 prefill 與 decode，且 scheduler-profile accounting 全部 exact。
- 目前沒有 speedup：相對 `fixed_high`，p50 latency 約 +1.2%，generated tokens/s 約 -1.3%。
- 96 個 WikiText-2 held-out requests 的 group demand 都高於 5（約 5.4675--5.5362），投影到合法 bits 3/4/5/6 後全部選到 6-bit；這次主要驗證 execution correctness，不是 adaptive low-bit benefit。

要貼的圖片名稱：

- `artifacts/qaq_current_progress_20260716/figures/fig_shared_profile_cuda.png`（real-CUDA latency / throughput comparison）
- `artifacts/qaq_current_progress_20260716/figures/fig_shared_profile_safety.png`（route-safety audit）

## 第 15 頁：目前已知限制與研究判讀

頁面內容：

已經證明：

- QAQ router、DP guard、real trace、quality evaluator 與 shared-profile execution 都有真實資料路徑。
- QAQ held-out quality table 顯示 precision / perplexity trade-off；shared-profile route audit 沒有 underprecision violation。
- v2 shared-profile CUDA path 可重現執行，並且統計反映實際執行 bit，而非 padding estimate。

尚未證明：

- 尚未證明 dynamic batching 提升 end-to-end throughput、P99 latency 或 memory efficiency。
- Predictor cross-dataset transfer gate 未通過，不能把 held-out profile prediction 當成可部署的泛化能力。
- Shared-profile quality audit 是 route-level output-error safety，不是 task accuracy、perplexity 或 generated-answer quality。
- v2 timing 目前只有 8 requests、單一 RTX 3090、3 repeats，而且 demand 全部投影為 6-bit。
- 尚未完成 lower-demand multi-rate sweep、quantile sharing、post-load phase-isolated profiling、transfer/HBM/kernel-switch accounting。

要貼的圖片名稱：

- 無

## 第 16 頁：下一步計畫與目前 take-home message

頁面內容：

短期下一步：

1. 校準與重新評估 group-demand predictor；在所有 LODO gates 通過前，維持 fixed-high fallback。
2. 建立 lower-demand、更多 request、固定 output length、較高 arrival rate 的 v2 GPU sweep，至少 10 repeats。
3. 比較 `fixed_high`、`fcfs`、`max_profile_sharing`、`scalar_predicted` 與 `quantile_profile_sharing`，並加入 task-level quality / route audit。
4. 做 post-load phase-isolated CUDA profiling，分開測 predictor、scheduler、shared execution、dequantized matmul 與 model-loading cost。
5. 最後形成 quality、effective bits、p50/p95/p99、TTFT/TPOT、tokens/s、fallback、guard、profile padding 與 per-layer histogram 的完整 ablation table。

目前 take-home message：

> 研究主線已從 QAQ routing 接到真正的 shared-profile execution，但目前最重要的瓶頸是 predictor calibration 與 quality-aware serving validation；尚未有 dynamic batching speedup claim。

要貼的圖片名稱：

- 無

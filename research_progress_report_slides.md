# QAQ / DP-LLM 進度報告投影片草稿

> 用途：直接貼到簡報。每頁列出「頁面內容」與「要貼的圖片名稱」。圖片只列檔名，方便手動貼圖。
> 日期：2026-07-09

## 第 1 頁：題目與目前定位

頁面內容：

- 題目：QAQ-style Query-Adaptive Precision Routing on Any-Precision LLM
- 核心目標：根據不同 query / request 的精度需求，動態選擇推論 bit-width。
- 目前進度：已完成單一 request QAQ 路徑、DP threshold guard、real trace collection、trace-driven simulation、初步 GPU batched replay。
- 報告重點：目前架構、目前結果、尚未驗證的限制、下一步實驗。

要貼的圖片名稱：

- 無

## 第 2 頁：研究問題

頁面內容：

- 現有 fixed precision inference 對所有 request 使用同一精度。
- 問題：簡單 request 可能不需要高精度，困難 request 又可能在低精度下品質下降。
- 研究問題：能不能用 query-aware routing 在品質可控的前提下降低 effective bits，並讓 serving batching 更有效？
- 目前方法不是重新訓練模型，而是在 Any-Precision weights 上加 runtime precision control。

要貼的圖片名稱：

- 無

## 第 3 頁：目前整體架構

頁面內容：

```text
Any-Precision quantized weights
        |
        v
QAQRouter
- route id embedding
- hidden/norm feature
- optional DP estimated-error feature
- output: selected precision bit
        |
        v
QAQDPLLM_Linear
- fixed_low / fixed_high
- mlp_multibit
- dp_threshold_only
- mlp_multibit_dp_guard
- group rows by selected bit
        |
        v
QAQDPLLMForCausalLM
- load router checkpoint
- load estimator artifacts
- replace quantized linears
- collect router stats
        |
        v
Trace / benchmark / dynamic batching simulator
```

重點：

- QAQ router 決定每個 linear route 使用的 bit。
- DP guard 使用 `T_d.pt` threshold，避免 router 選得太低。
- Runtime stats 會記錄 average selected bit、effective bits、fallback、DP guard trigger、per-layer histogram。

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

要貼的圖片名稱：

- 無

## 第 5 頁：Runtime 模式與比較對象

頁面內容：

目前支援的 runtime modes：

| Mode | 作用 |
|---|---|
| `fixed_low` | 固定使用最低可用 precision |
| `fixed_high` | 固定使用最高 precision，作為保守 baseline |
| `mlp_multibit` / `qaq` | 使用 QAQ router 預測 precision |
| `dp_threshold_only` | 只使用 DP-LLM threshold decision |
| `mlp_multibit_dp_guard` | QAQ router + DP threshold guard |

比較邏輯：

- `fixed_low` / `fixed_high` 建立低精度與高精度 baseline。
- `qaq` 檢查 learned router 是否能降低 bit budget。
- `dp_threshold_only` 檢查 DP-LLM threshold 本身的行為。
- `mlp_multibit_dp_guard` 檢查 router 加 guard 後是否更保守。

要貼的圖片名稱：

- 無

## 第 6 頁：Serving-aware dynamic batching 架構

頁面內容：

```text
Prompt set / request stream
        |
        v
Real QAQ trace collection
- per-request selected bits
- effective bits
- per-layer bit histogram
- fallback / DP guard count
        |
        v
Profile extraction
- scalar bit budget
- coarse block precision profile
- workload type
        |
        v
Trace-driven simulator
- ordinary dynamic batching
- scalar budget batching
- block profile batching
- max / quantile profile sharing
        |
        v
GPU batched replay
- replay selected policy candidates
- measure CUDA synchronized batch runtime
```

目前定位：

- Trace 是 real QAQ generation。
- Simulator 是 policy selection tool，不是 performance claim。
- GPU replay 是初步 validation artifact，還沒有完整 shared-profile override。

要貼的圖片名稱：

- 無

## 第 7 頁：目前完成項目

頁面內容：

- 已完成 QAQ runtime wrapper：`QAQDPLLMForCausalLM.py`
- 已完成 runtime quantized linear：`QAQDPLLM_Linear.py`
- 已完成 train / inference / benchmark / trace / simulation scripts。
- 已加入 DP threshold guard：`mlp_multibit_dp_guard`
- 已完成 200-request mixed workload trace。
- 已產生 profile diversity plots 與 simulator plots。
- 已做初步 real GPU batched replay。

目前 artifact：

- 單一 request stabilization：`artifacts/qaq_single_request_stabilization_20260709_035238/`
- 200-request mixed trace：`artifacts/qaq_mixed_trace_20260709_050923/`
- DP guard benchmark：`artifacts/qaq_dp_guard_benchmark_20260708_230117/`

要貼的圖片名稱：

- 無

## 第 8 頁：單一 request benchmark 結果

頁面內容：

設定：

- GPU：NVIDIA RTX 3090
- Model：Any-Precision Llama 3.1 8B packed checkpoint
- Candidate bits：3、4、5、6
- Prompt count：1
- Max new tokens：16
- Warmup：1
- Repeat：3

結果：

| Mode | Mean latency | Tokens/s | Effective bits | Fallback | DP guard |
|---|---:|---:|---:|---:|---:|
| `fixed_low` | 0.501 s | 31.96 | 4.27 | 0.00 | 0.00 |
| `fixed_high` | 0.497 s | 32.18 | 6.00 | 0.00 | 0.00 |
| `qaq` | 2.992 s | 5.35 | 5.47 | 0.176 | 0.00 |
| `dp_threshold_only` | 2.017 s | 7.93 | 5.09 | 0.00 | 0.00 |
| `mlp_multibit_dp_guard` | 3.911 s | 4.09 | 5.48 | 0.176 | 0.008 |

解讀：

- 多模式可在真實 GPU / CUDA path 上執行，logits finite。
- QAQ / DP guard 目前有明顯 routing overhead，還不能宣稱速度優勢。
- 這頁主要證明 runtime path 已接起來，並指出 overhead 是下一步優化重點。

要貼的圖片名稱：

- 無

## 第 9 頁：Real trace 結果：不同 workload 的 selected bits

頁面內容：

設定：

- 200 requests，共 4 類 workload。
- `chat`、`code`、`math`、`summarization` 各 50 筆。
- Router mode：`mlp_multibit_dp_guard`
- Max new tokens：8

結果：

| Workload | Avg selected bit mean | Effective bits mean | Fallback total | DP guard triggers |
|---|---:|---:|---:|---:|
| chat | 5.7465 | 5.7175 | 0 | 1607 |
| code | 5.7921 | 5.7593 | 0 | 1557 |
| math | 5.8037 | 5.7782 | 0 | 1688 |
| summarization | 5.9879 | 5.9861 | 0 | 1049 |
| overall | 5.8326 | 5.8103 | 0 | 5901 |

解讀：

- Real QAQ trace 已能收集每個 request 的 bit profile。
- `summarization` 幾乎都接近 6-bit，比其他 workload 更保守。
- DP guard 有大量 trigger，表示 threshold guard 確實參與 precision decision。

要貼的圖片名稱：

- `fig_bits_by_workload.png`

## 第 10 頁：Profile diversity 結果

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
- Scalar bucket 比 coarse profile 更粗，可能適合作為第一版 batching signal。
- 這是 descriptive result，不是品質或速度 claim。

要貼的圖片名稱：

- `fig_profile_fragmentation_by_workload.png`
- `fig_coarse_profile_occupancy_heatmap.png`

## 第 11 頁：Simulator 結果：選出下一個 GPU 驗證目標

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

- Simulator 選出 `scalar_budget_batching` 作為下一個 GPU replay target。
- 這只是 simulator result，不能當真實 throughput speedup。
- 它的價值是幫我們縮小下一步 GPU 實驗範圍。

要貼的圖片名稱：

- `fig_simulated_policy_tradeoff.png`
- `fig_lane_occupancy_by_policy.png`

## 第 12 頁：初步 GPU batched replay 結果

頁面內容：

設定：

- Replay simulator batch membership。
- Warmup batches：1
- Repeat：1
- Status：`REAL_GPU_BATCHED_REPLAY_NO_SHARED_PROFILE_OVERRIDES`

結果：

| Policy | Batches | Requests | Mean batch GPU ms | P95 batch GPU ms | Token slots/s |
|---|---:|---:|---:|---:|---:|
| ordinary dynamic batching | 50 | 200 | 2132.13 | 2398.15 | 14.746 |
| scalar budget batching | 51 | 200 | 1868.20 | 2328.68 | 14.956 |

觀察：

- `scalar_budget_batching` 在這次 replay 的 mean batch GPU time 較低。
- 但只跑 1 repeat，且沒有 queue delay replay，也沒有 shared-profile override。
- 因此這是 preliminary validation artifact，不是穩定 speedup claim。

要貼的圖片名稱：

- `fig_gpu_replay_policy_comparison.png`

## 第 13 頁：目前限制

頁面內容：

尚未完成或尚未驗證：

- 還沒有正式 quality metric，例如 perplexity、exact match、pass@1 或 task accuracy。
- 還沒有 under-precision / over-precision label。
- 還沒有 transfer bytes、HBM bytes、kernel switches、CUDA graph reuse 等系統指標。
- GPU batched replay 目前只有 1 repeat。
- GPU replay 尚未實作真正 shared-profile override。
- Summarization trace 有 1-token 和 8-token completion 混合，latency 比較需要依 output length 分層。
- 目前不能宣稱 dynamic batching 真正提升 throughput，只能說已完成初步 trace、simulation、replay pipeline。

要貼的圖片名稱：

- 無

## 第 14 頁：下一步計畫

頁面內容：

短期下一步：

1. 重跑 GPU batched replay，增加 repeat 數並固定 output length 條件。
2. 實作 shared-profile override，讓 scalar / block profile 真正影響 batch execution。
3. 加入 quality evaluator，至少先做 perplexity 或 task-level metric。
4. 記錄 under-precision / over-precision labels，檢查 batching 是否犧牲品質。
5. 分析 QAQ / DP guard overhead，區分 router compute、estimator、kernel grouping、CUDA launch overhead。

中期目標：

- 建立完整 main result table：quality、effective bits、latency、throughput、fallback、DP guard、per-layer bit histogram。
- 做 ablation：`fixed_low`、`fixed_high`、`dp_threshold_only`、`mlp_multibit`、`mlp_multibit_dp_guard`、dynamic batching variants。

要貼的圖片名稱：

- 無

## 第 15 頁：總結

頁面內容：

- 目前已從單一 request routing 推進到 serving-aware trace / simulation / replay pipeline。
- 真實 QAQ trace 顯示不同 workload 有不同 selected-bit pattern。
- Profile diversity 結果支持「precision profile 可以作為 batching signal」這個方向。
- Simulator 建議先驗證 `scalar_budget_batching`。
- 初步 GPU replay 已完成，但還不能當 performance claim。
- 下一步要補上 quality、shared-profile execution、更多 repeats，以及真正可發表的 end-to-end benchmark。

Take-home message：

> 目前研究主線已接起來，但真正的 paper claim 需要 quality-aware、CUDA-synchronized、shared-profile dynamic batching validation。

要貼的圖片名稱：

- 無


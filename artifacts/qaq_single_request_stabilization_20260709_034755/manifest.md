# QAQ Single-Request Stabilization Run

- Date: 2026-07-09T03:47:55+08:00
- Git commit: 59c2a040de85314e40f73a4fabfd07ebccc83416
- Host: basic-1
- AP_MODEL_PATH: /nfs/home/s314511048/dpqaq/cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512
- ROUTER_CHECKPOINT: UNVALIDATED
- ESTIMATOR_RESULTS: UNVALIDATED

## Git Status
 M any_precision/modules/QAQDPLLMForCausalLM.py
 M tests/router/test_qaq_dp_guard.py
?? artifacts/qaq_single_request_stabilization_20260709_034755/

## GPU
Thu Jul  9 03:47:55 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.03             Driver Version: 580.159.03     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        On  |   00000000:1B:00.0 Off |                  N/A |
| 30%   27C    P8             24W /  300W |       1MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        On  |   00000000:1C:00.0 Off |                  N/A |
| 30%   29C    P8             31W /  300W |       1MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 3090        On  |   00000000:1D:00.0 Off |                  N/A |
| 30%   29C    P8             22W /  300W |       1MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 3090        On  |   00000000:1E:00.0 Off |                  N/A |
| 30%   28C    P8             27W /  300W |       1MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA GeForce RTX 3090        On  |   00000000:89:00.0 Off |                  N/A |
| 30%   29C    P8             19W /  300W |       1MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA GeForce RTX 3090        On  |   00000000:8A:00.0 Off |                  N/A |
| 30%   29C    P8             18W /  300W |       1MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA GeForce RTX 3090        On  |   00000000:8B:00.0 Off |                  N/A |
| 30%   29C    P8             16W /  300W |       1MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA GeForce RTX 3090        On  |   00000000:8C:00.0 Off |                  N/A |
| 30%   27C    P8             18W /  300W |       1MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

## Artifact
- artifacts/qaq_single_request_stabilization_20260709_034755/qaq_inference_stats.json
- artifacts/qaq_single_request_stabilization_20260709_034755/summary.json

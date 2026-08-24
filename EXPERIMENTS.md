<!-- TỰ SINH bởi scripts/gen_experiment_log.py từ runs/*.json. Không sửa tay. -->
# Nhật ký thực nghiệm

35 run được log trong `runs/`. Xem [RESEARCH.md](RESEARCH.md) để biết mục tiêu/metric/dataset.

## Tổng hợp

| thời gian | loại run | attack | manifest | steps | crafted/skipped | ASR TB (target) | mAP-drop % TB (target) | log |
|---|---|---|---|---|---|---|---|---|
| 20260821T154422Z | run_attack | osfd | dev_50.json | 100 | 49/1 | 74.0% | 77.2% | `runs/run_attack_osfd_dev_50_20260821T154422Z.json` |
| 20260821T155439Z | run_attack | mi_fgsm | dev_50.json | 100 | 49/1 | 39.2% | 44.7% | `runs/run_attack_mi_fgsm_dev_50_20260821T155439Z.json` |
| 20260823T072302Z | run_attack | e3_k1_norrb | dev_50.json | 100 | 49/1 | 44.3% | 45.5% | `runs/run_attack_osfd_e3_k1_norrb_dev_50_20260823T072302Z.json` |
| 20260823T073141Z | run_attack | e3_k1_rrb | dev_50.json | 100 | 49/1 | 60.3% | 59.0% | `runs/run_attack_osfd_e3_k1_rrb_dev_50_20260823T073141Z.json` |
| 20260823T073642Z | run_attack | e3_k3_norrb | dev_50.json | 100 | 49/1 | 48.5% | 48.6% | `runs/run_attack_osfd_e3_k3_norrb_dev_50_20260823T073642Z.json` |
| 20260823T074516Z | run_attack | e3_k3_rrb | dev_50.json | 100 | 49/1 | 71.8% | 72.0% | `runs/run_attack_osfd_e3_k3_rrb_dev_50_20260823T074516Z.json` |
| 20260823T091448Z | run_attack | m_osfd | dev_50.json | 100 | 20/0 | 73.5% | 67.0% | `runs/run_attack_osfd_mvc_osfd_dev_50_n20_20260823T091448Z.json` |
| 20260823T092121Z | run_attack | m_mvc_avg | dev_50.json | 100 | 20/0 | 74.4% | 67.4% | `runs/run_attack_osfd_mvc_mvc_avg_dev_50_n20_20260823T092121Z.json` |
| 20260823T092808Z | run_attack | m_mvc_cons | dev_50.json | 100 | 20/0 | 58.9% | 53.3% | `runs/run_attack_osfd_mvc_mvc_cons_dev_50_n20_20260823T092808Z.json` |
| 20260823T095013Z | run_attack | m_mvc_cons_lam10 | dev_50.json | 100 | 20/0 | 60.9% | 60.9% | `runs/run_attack_osfd_mvc_mvc_cons_lam10_dev_50_n20_20260823T095013Z.json` |
| 20260823T101823Z | run_attack | m_osfd | dev_50.json | 100 | 20/0 | 58.2% | 53.7% | `runs/run_attack_osfd_rcg_osfd_dev_50_n20_20260823T101823Z.json` |
| 20260823T102607Z | run_attack | m_rcg_avg | dev_50.json | 100 | 20/0 | 67.5% | 62.6% | `runs/run_attack_osfd_rcg_rcg_avg_dev_50_n20_20260823T102607Z.json` |
| 20260823T103349Z | run_attack | m_rcg_gate | dev_50.json | 100 | 20/0 | 68.9% | 64.8% | `runs/run_attack_osfd_rcg_rcg_gate_dev_50_n20_20260823T103349Z.json` |
| 20260823T144110Z | run_attack | n2b_osfd | dev_50.json | 100 | 20/0 | 63.5% | 61.6% | `runs/run_attack_osfd_n2b_osfd_dev_50_n20_20260823T144110Z.json` |
| 20260823T144846Z | run_attack | n2b_statnorm | dev_50.json | 100 | 20/0 | 56.5% | 50.2% | `runs/run_attack_osfd_n2b_statnorm_dev_50_n20_20260823T144846Z.json` |
| 20260823T144849Z | run_attack | n2a_robust | dev_50.json | 100 | 20/0 | 4.1% | 5.9% | `runs/run_attack_osfd_n2a_robust_dev_50_n20_20260823T144849Z.json` |
| 20260823T152554Z | run_attack | n5_osfd | dev_50.json | 100 | 20/0 | 63.5% | 61.4% | `runs/run_attack_osfd_n5_osfd_dev_50_n20_20260823T152554Z.json` |
| 20260823T152923Z | run_attack | n5_dob_easy | dev_50.json | 100 | 20/0 | 64.2% | 60.2% | `runs/run_attack_osfd_n5_dob_easy_dev_50_n20_20260823T152923Z.json` |
| 20260823T153247Z | run_attack | n5_dob_hard | dev_50.json | 100 | 20/0 | 63.9% | 60.9% | `runs/run_attack_osfd_n5_dob_hard_dev_50_n20_20260823T153247Z.json` |
| 20260823T154306Z | run_attack | n5b_osfd | dev_50.json | 100 | 20/0 | 64.2% | 59.3% | `runs/run_attack_osfd_n5b_osfd_dev_50_n20_20260823T154306Z.json` |
| 20260823T154639Z | run_attack | n5b_dob_easy_v1 | dev_50.json | 100 | 20/0 | 64.2% | 62.3% | `runs/run_attack_osfd_n5b_dob_easy_v1_dev_50_n20_20260823T154639Z.json` |
| 20260823T155013Z | run_attack | n5b_dob_hard_v1 | dev_50.json | 100 | 20/0 | 63.5% | 61.2% | `runs/run_attack_osfd_n5b_dob_hard_v1_dev_50_n20_20260823T155013Z.json` |
| 20260823T160459Z | run_attack | n6a_osfd_k1 | dev_50.json | 5 | 2/0 | — | — | `runs/run_attack_osfd_n6a_osfd_k1_dev_50_n2_20260823T160459Z.json` |
| 20260823T160501Z | run_attack | n6a_rrb_avg_k3 | dev_50.json | 5 | 2/0 | — | — | `runs/run_attack_osfd_n6a_rrb_avg_k3_dev_50_n2_20260823T160501Z.json` |
| 20260823T160502Z | run_attack | n6a_rrb_cr_k3 | dev_50.json | 5 | 2/0 | — | — | `runs/run_attack_osfd_n6a_rrb_cr_k3_dev_50_n2_20260823T160502Z.json` |
| 20260823T160943Z | run_attack | n6a_osfd_k1 | dev_50.json | 5 | 2/0 | — | — | `runs/run_attack_osfd_n6a_osfd_k1_dev_50_n2_20260823T160943Z.json` |
| 20260823T160944Z | run_attack | n6a_rrb_avg_k3 | dev_50.json | 5 | 2/0 | — | — | `runs/run_attack_osfd_n6a_rrb_avg_k3_dev_50_n2_20260823T160944Z.json` |
| 20260823T160945Z | run_attack | n6a_rrb_cr_k3 | dev_50.json | 5 | 2/0 | — | — | `runs/run_attack_osfd_n6a_rrb_cr_k3_dev_50_n2_20260823T160945Z.json` |
| 20260824T144540Z | run_attack | n6a_osfd_k1 | dev_50.json | 100 | 20/0 | 65.9% | 60.1% | `runs/run_attack_osfd_n6a_osfd_k1_dev_50_n20_20260824T144540Z.json` |
| 20260824T144614Z | run_attack | n6a_rrb_avg_k3 | dev_50.json | 100 | 20/0 | 67.5% | 65.6% | `runs/run_attack_osfd_n6a_rrb_avg_k3_dev_50_n20_20260824T144614Z.json` |
| 20260824T144648Z | run_attack | n6a_rrb_cr_k3 | dev_50.json | 100 | 20/0 | 67.9% | 66.2% | `runs/run_attack_osfd_n6a_rrb_cr_k3_dev_50_n20_20260824T144648Z.json` |
| 20260824T152203Z | run_attack | n6b_osfd_local | dev_50.json | 100 | 20/0 | 66.5% | 63.2% | `runs/run_attack_osfd_n6b_osfd_local_dev_50_n20_20260824T152203Z.json` |
| 20260824T152236Z | run_attack | n6b_path_m3 | dev_50.json | 100 | 20/0 | 71.6% | 70.3% | `runs/run_attack_osfd_n6b_path_m3_dev_50_n20_20260824T152236Z.json` |
| 20260824T180319Z | run_attack | n6b_osfd_local | dev_300.json | 100 | 296/4 | 75.1% | 77.9% | `runs/run_attack_osfd_n6b_osfd_local_dev_300_n300_20260824T180319Z.json` |
| 20260824T180958Z | run_attack | n6b_path_m3 | dev_300.json | 100 | 296/4 | 76.9% | 79.9% | `runs/run_attack_osfd_n6b_path_m3_dev_300_n300_20260824T180958Z.json` |

## Chi tiết từng run

### 20260824T180958Z — `run_attack_osfd_n6b_path_m3_dev_300_n300_20260824T180958Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_300.json`
- attack: `n6b_path_m3`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, m_lambda=3
- crafted=296 skipped=4 craft=8622.2s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6b_path_m3 | faster_rcnn_r50 | — | 0.4372 | 0.0014 | 99.7% | 0.6321 | 0.0016 | 98.8% |
| n6b_path_m3 | fcos_r50 | A | 0.4311 | 0.0077 | 98.2% | 0.6167 | 0.0131 | 97.7% |
| n6b_path_m3 | deformable_detr | A | 0.4965 | 0.0016 | 99.7% | 0.6725 | 0.0023 | 98.0% |
| n6b_path_m3 | yolov3_d53 | B | 0.3955 | 0.0312 | 92.1% | 0.6352 | 0.0590 | 84.6% |
| n6b_path_m3 | yolox_l | B | 0.5680 | 0.1389 | 75.6% | 0.7467 | 0.1911 | 70.9% |
| n6b_path_m3 | mask_rcnn_swin_t | C | 0.4791 | 0.0917 | 80.9% | 0.6972 | 0.1511 | 71.1% |
| n6b_path_m3 | dino_swin_l | C | 0.6087 | 0.4066 | 33.2% | 0.7801 | 0.5498 | 39.1% |

### 20260824T180319Z — `run_attack_osfd_n6b_osfd_local_dev_300_n300_20260824T180319Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_300.json`
- attack: `n6b_osfd_local`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, m_lambda=3
- crafted=296 skipped=4 craft=8622.2s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6b_osfd_local | faster_rcnn_r50 | — | 0.4372 | 0.0010 | 99.8% | 0.6321 | 0.0011 | 99.2% |
| n6b_osfd_local | fcos_r50 | A | 0.4311 | 0.0059 | 98.6% | 0.6167 | 0.0098 | 98.1% |
| n6b_osfd_local | deformable_detr | A | 0.4965 | 0.0011 | 99.8% | 0.6725 | 0.0014 | 98.5% |
| n6b_osfd_local | yolov3_d53 | B | 0.3955 | 0.0365 | 90.8% | 0.6352 | 0.0673 | 83.9% |
| n6b_osfd_local | yolox_l | B | 0.5680 | 0.1474 | 74.0% | 0.7467 | 0.2069 | 69.2% |
| n6b_osfd_local | mask_rcnn_swin_t | C | 0.4791 | 0.1133 | 76.4% | 0.6972 | 0.1812 | 67.8% |
| n6b_osfd_local | dino_swin_l | C | 0.6087 | 0.4392 | 27.8% | 0.7801 | 0.5842 | 33.0% |

### 20260824T152236Z — `run_attack_osfd_n6b_path_m3_dev_50_n20_20260824T152236Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6b_path_m3`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, m_lambda=3
- crafted=20 skipped=0 craft=568.2s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6b_path_m3 | faster_rcnn_r50 | — | 0.4519 | 0.0000 | 100.0% | 0.6062 | 0.0000 | 98.9% |
| n6b_path_m3 | yolox_l | B | 0.5782 | 0.0493 | 91.5% | 0.6989 | 0.1088 | 87.1% |
| n6b_path_m3 | mask_rcnn_swin_t | C | 0.4815 | 0.0813 | 83.1% | 0.6337 | 0.1332 | 85.6% |
| n6b_path_m3 | dino_swin_l | C | 0.6613 | 0.4215 | 36.3% | 0.8039 | 0.5772 | 42.0% |

### 20260824T152203Z — `run_attack_osfd_n6b_osfd_local_dev_50_n20_20260824T152203Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6b_osfd_local`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, m_lambda=3
- crafted=20 skipped=0 craft=568.2s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6b_osfd_local | faster_rcnn_r50 | — | 0.4519 | 0.0002 | 100.0% | 0.6062 | 0.0004 | 100.0% |
| n6b_osfd_local | yolox_l | B | 0.5782 | 0.1093 | 81.1% | 0.6989 | 0.1776 | 86.1% |
| n6b_osfd_local | mask_rcnn_swin_t | C | 0.4815 | 0.0894 | 81.4% | 0.6337 | 0.1461 | 82.5% |
| n6b_osfd_local | dino_swin_l | C | 0.6613 | 0.4818 | 27.2% | 0.8039 | 0.6307 | 31.0% |

### 20260824T144648Z — `run_attack_osfd_n6a_rrb_cr_k3_dev_50_n20_20260824T144648Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6a_rrb_cr_k3`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=20 skipped=0 craft=1028.4s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6a_rrb_cr_k3 | faster_rcnn_r50 | — | 0.4519 | 0.0000 | 100.0% | 0.6062 | 0.0000 | 100.0% |
| n6a_rrb_cr_k3 | yolox_l | B | 0.5782 | 0.0607 | 89.5% | 0.6989 | 0.1293 | 85.1% |
| n6a_rrb_cr_k3 | mask_rcnn_swin_t | C | 0.4815 | 0.1116 | 76.8% | 0.6337 | 0.1850 | 84.5% |
| n6a_rrb_cr_k3 | dino_swin_l | C | 0.6613 | 0.4479 | 32.3% | 0.8039 | 0.5636 | 34.0% |

### 20260824T144614Z — `run_attack_osfd_n6a_rrb_avg_k3_dev_50_n20_20260824T144614Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6a_rrb_avg_k3`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=20 skipped=0 craft=1028.4s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6a_rrb_avg_k3 | faster_rcnn_r50 | — | 0.4519 | 0.0000 | 100.0% | 0.6062 | 0.0000 | 100.0% |
| n6a_rrb_avg_k3 | yolox_l | B | 0.5782 | 0.0563 | 90.3% | 0.6989 | 0.1114 | 86.1% |
| n6a_rrb_avg_k3 | mask_rcnn_swin_t | C | 0.4815 | 0.1206 | 75.0% | 0.6337 | 0.2008 | 82.5% |
| n6a_rrb_avg_k3 | dino_swin_l | C | 0.6613 | 0.4517 | 31.7% | 0.8039 | 0.5783 | 34.0% |

### 20260824T144540Z — `run_attack_osfd_n6a_osfd_k1_dev_50_n20_20260824T144540Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6a_osfd_k1`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=20 skipped=0 craft=1028.4s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6a_osfd_k1 | faster_rcnn_r50 | — | 0.4519 | 0.0002 | 99.9% | 0.6062 | 0.0005 | 100.0% |
| n6a_osfd_k1 | yolox_l | B | 0.5782 | 0.1240 | 78.6% | 0.6989 | 0.1947 | 84.2% |
| n6a_osfd_k1 | mask_rcnn_swin_t | C | 0.4815 | 0.1213 | 74.8% | 0.6337 | 0.1844 | 82.5% |
| n6a_osfd_k1 | dino_swin_l | C | 0.6613 | 0.4833 | 26.9% | 0.8039 | 0.6046 | 31.0% |

### 20260823T160945Z — `run_attack_osfd_n6a_rrb_cr_k3_dev_50_n2_20260823T160945Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6a_rrb_cr_k3`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=5, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=2 skipped=0 craft=6.0s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6a_rrb_cr_k3 | faster_rcnn_r50 | — | 0.5840 | 0.1945 | 66.7% | 0.7010 | 0.3422 | 83.3% |

### 20260823T160944Z — `run_attack_osfd_n6a_rrb_avg_k3_dev_50_n2_20260823T160944Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6a_rrb_avg_k3`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=5, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=2 skipped=0 craft=6.0s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6a_rrb_avg_k3 | faster_rcnn_r50 | — | 0.5840 | 0.1945 | 66.7% | 0.7010 | 0.3422 | 83.3% |

### 20260823T160943Z — `run_attack_osfd_n6a_osfd_k1_dev_50_n2_20260823T160943Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6a_osfd_k1`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=5, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=2 skipped=0 craft=6.0s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6a_osfd_k1 | faster_rcnn_r50 | — | 0.5840 | 0.4210 | 27.9% | 0.7010 | 0.5683 | 50.0% |

### 20260823T160502Z — `run_attack_osfd_n6a_rrb_cr_k3_dev_50_n2_20260823T160502Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6a_rrb_cr_k3`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=5, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=2 skipped=0 craft=6.2s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6a_rrb_cr_k3 | faster_rcnn_r50 | — | 0.5840 | 0.1945 | 66.7% | 0.7010 | 0.3422 | 83.3% |

### 20260823T160501Z — `run_attack_osfd_n6a_rrb_avg_k3_dev_50_n2_20260823T160501Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6a_rrb_avg_k3`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=5, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=2 skipped=0 craft=6.2s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6a_rrb_avg_k3 | faster_rcnn_r50 | — | 0.5840 | 0.1945 | 66.7% | 0.7010 | 0.3422 | 83.3% |

### 20260823T160459Z — `run_attack_osfd_n6a_osfd_k1_dev_50_n2_20260823T160459Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n6a_osfd_k1`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=5, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=2 skipped=0 craft=6.2s eval=Nones

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n6a_osfd_k1 | faster_rcnn_r50 | — | 0.5840 | 0.4210 | 27.9% | 0.7010 | 0.5683 | 50.0% |

### 20260823T155013Z — `run_attack_osfd_n5b_dob_hard_v1_dev_50_n20_20260823T155013Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n5b_dob_hard_v1`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=dob_hard_v1, direction=hard, beta=1.0, w_min=0.5, w_max=1.5
- crafted=20 skipped=0 craft=178.0s eval=33.7s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| n5b_dob_hard_v1 | faster_rcnn_r50 | — | 0.4519 | 0.0459 | 89.8% | 0.6062 | 0.0658 | 96.8% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| n5b_dob_hard_v1 | yolox_l | B | 0.5782 | 0.0932 | 83.9% | 0.6989 | 0.1488 | 81.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| n5b_dob_hard_v1 | mask_rcnn_swin_t | C | 0.4815 | 0.1363 | 71.7% | 0.6337 | 0.2139 | 79.4% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| n5b_dob_hard_v1 | dino_swin_l | C | 0.6613 | 0.4768 | 27.9% | 0.8039 | 0.6017 | 30.0% |

### 20260823T154639Z — `run_attack_osfd_n5b_dob_easy_v1_dev_50_n20_20260823T154639Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n5b_dob_easy_v1`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=dob_easy_v1, direction=easy, beta=1.0, w_min=0.5, w_max=1.5
- crafted=20 skipped=0 craft=178.5s eval=33.7s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| n5b_dob_easy_v1 | faster_rcnn_r50 | — | 0.4519 | 0.0459 | 89.8% | 0.6062 | 0.0658 | 96.8% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| n5b_dob_easy_v1 | yolox_l | B | 0.5782 | 0.0764 | 86.8% | 0.6989 | 0.1415 | 82.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| n5b_dob_easy_v1 | mask_rcnn_swin_t | C | 0.4815 | 0.1244 | 74.2% | 0.6337 | 0.2117 | 79.4% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| n5b_dob_easy_v1 | dino_swin_l | C | 0.6613 | 0.4888 | 26.1% | 0.8039 | 0.6276 | 31.0% |

### 20260823T154306Z — `run_attack_osfd_n5b_osfd_dev_50_n20_20260823T154306Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n5b_osfd`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=osfd, direction=None, beta=1.0, w_min=0.5, w_max=1.5
- crafted=20 skipped=0 craft=177.7s eval=43.3s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| n5b_osfd | faster_rcnn_r50 | — | 0.4519 | 0.0524 | 88.4% | 0.6062 | 0.0659 | 96.8% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| n5b_osfd | yolox_l | B | 0.5782 | 0.0862 | 85.1% | 0.6989 | 0.1475 | 82.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| n5b_osfd | mask_rcnn_swin_t | C | 0.4815 | 0.1663 | 65.5% | 0.6337 | 0.2713 | 80.4% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| n5b_osfd | dino_swin_l | C | 0.6613 | 0.4810 | 27.3% | 0.8039 | 0.6415 | 30.0% |

### 20260823T153247Z — `run_attack_osfd_n5_dob_hard_dev_50_n20_20260823T153247Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n5_dob_hard`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=dob_hard, direction=hard, beta=1.0, w_min=0.5, w_max=1.5
- crafted=20 skipped=0 craft=172.8s eval=29.9s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| n5_dob_hard | faster_rcnn_r50 | — | 0.4519 | 0.0427 | 90.5% | 0.6062 | 0.0658 | 96.8% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| n5_dob_hard | yolox_l | B | 0.5782 | 0.0779 | 86.5% | 0.6989 | 0.1401 | 82.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| n5_dob_hard | mask_rcnn_swin_t | C | 0.4815 | 0.1493 | 69.0% | 0.6337 | 0.2274 | 79.4% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| n5_dob_hard | dino_swin_l | C | 0.6613 | 0.4816 | 27.2% | 0.8039 | 0.6293 | 30.0% |

### 20260823T152923Z — `run_attack_osfd_n5_dob_easy_dev_50_n20_20260823T152923Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n5_dob_easy`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=dob_easy, direction=easy, beta=1.0, w_min=0.5, w_max=1.5
- crafted=20 skipped=0 craft=172.6s eval=34.4s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| n5_dob_easy | faster_rcnn_r50 | — | 0.4519 | 0.0459 | 89.8% | 0.6062 | 0.0658 | 96.8% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| n5_dob_easy | yolox_l | B | 0.5782 | 0.0883 | 84.7% | 0.6989 | 0.1405 | 82.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| n5_dob_easy | mask_rcnn_swin_t | C | 0.4815 | 0.1390 | 71.1% | 0.6337 | 0.2197 | 79.4% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| n5_dob_easy | dino_swin_l | C | 0.6613 | 0.4976 | 24.8% | 0.8039 | 0.6454 | 31.0% |

### 20260823T152554Z — `run_attack_osfd_n5_osfd_dev_50_n20_20260823T152554Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n5_osfd`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=osfd, direction=None, beta=1.0, w_min=0.5, w_max=1.5
- crafted=20 skipped=0 craft=172.1s eval=41.8s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| n5_osfd | faster_rcnn_r50 | — | 0.4519 | 0.0459 | 89.8% | 0.6062 | 0.0659 | 96.8% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| n5_osfd | yolox_l | B | 0.5782 | 0.0873 | 84.9% | 0.6989 | 0.1460 | 82.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| n5_osfd | mask_rcnn_swin_t | C | 0.4815 | 0.1417 | 70.6% | 0.6337 | 0.2244 | 79.4% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| n5_osfd | dino_swin_l | C | 0.6613 | 0.4714 | 28.7% | 0.8039 | 0.6240 | 29.0% |

### 20260823T144849Z — `run_attack_osfd_n2a_robust_dev_50_n20_20260823T144849Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n2a_robust`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, robust_ckpt=/tmp/claude-0/-workspace/db032472-a127-4e0f-ac2c-3d65c9b9e5a5/scratchpad/robust_ckpt/imagenet_linf_4.pt
- crafted=20 skipped=0 craft=376.9s eval=69.0s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| n2a_robust | faster_rcnn_r50 | — | 0.4519 | 0.4067 | 10.0% | 0.6062 | 0.5990 | 11.7% |
| n2a_robust | yolox_l | B | 0.5782 | 0.5383 | 6.9% | 0.6989 | 0.6724 | 0.0% |
| n2a_robust | mask_rcnn_swin_t | C | 0.4815 | 0.4490 | 6.8% | 0.6337 | 0.6074 | 8.2% |
| n2a_robust | dino_swin_l | C | 0.6613 | 0.6339 | 4.1% | 0.8039 | 0.7584 | 4.0% |

### 20260823T144846Z — `run_attack_osfd_n2b_statnorm_dev_50_n20_20260823T144846Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n2b_statnorm`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=statnorm, standardize=True
- crafted=20 skipped=0 craft=400.6s eval=53.7s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| n2b_statnorm | faster_rcnn_r50 | — | 0.4519 | 0.0525 | 88.4% | 0.6062 | 0.0661 | 95.7% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| n2b_statnorm | yolox_l | B | 0.5782 | 0.1786 | 69.1% | 0.6989 | 0.2618 | 73.3% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| n2b_statnorm | mask_rcnn_swin_t | C | 0.4815 | 0.2176 | 54.8% | 0.6337 | 0.3299 | 70.1% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| n2b_statnorm | dino_swin_l | C | 0.6613 | 0.4858 | 26.5% | 0.8039 | 0.6406 | 26.0% |

### 20260823T144110Z — `run_attack_osfd_n2b_osfd_dev_50_n20_20260823T144110Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `n2b_osfd`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=osfd, standardize=False
- crafted=20 skipped=0 craft=174.1s eval=52.2s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| n2b_osfd | faster_rcnn_r50 | — | 0.4519 | 0.0524 | 88.4% | 0.6062 | 0.0659 | 96.8% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| n2b_osfd | yolox_l | B | 0.5782 | 0.0853 | 85.2% | 0.6989 | 0.1455 | 81.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| n2b_osfd | mask_rcnn_swin_t | C | 0.4815 | 0.1381 | 71.3% | 0.6337 | 0.2171 | 79.4% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| n2b_osfd | dino_swin_l | C | 0.6613 | 0.4744 | 28.3% | 0.8039 | 0.6220 | 30.0% |

### 20260823T103349Z — `run_attack_osfd_rcg_rcg_gate_dev_50_n20_20260823T103349Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `m_rcg_gate`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=rcg_gate, k_draws=3, use_gate=True
- crafted=20 skipped=0 craft=427.0s eval=33.3s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| m_rcg_gate | faster_rcnn_r50 | — | 0.4519 | 0.0000 | 100.0% | 0.6062 | 0.0000 | 98.9% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| m_rcg_gate | yolox_l | B | 0.5782 | 0.0850 | 85.3% | 0.6989 | 0.1562 | 85.1% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| m_rcg_gate | mask_rcnn_swin_t | C | 0.4815 | 0.0905 | 81.2% | 0.6337 | 0.1550 | 82.5% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| m_rcg_gate | dino_swin_l | C | 0.6613 | 0.4767 | 27.9% | 0.8039 | 0.6080 | 39.0% |

### 20260823T102607Z — `run_attack_osfd_rcg_rcg_avg_dev_50_n20_20260823T102607Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `m_rcg_avg`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=rcg_avg, k_draws=3, use_gate=False
- crafted=20 skipped=0 craft=426.4s eval=35.6s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| m_rcg_avg | faster_rcnn_r50 | — | 0.4519 | 0.0000 | 100.0% | 0.6062 | 0.0000 | 100.0% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| m_rcg_avg | yolox_l | B | 0.5782 | 0.0703 | 87.8% | 0.6989 | 0.1342 | 86.1% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| m_rcg_avg | mask_rcnn_swin_t | C | 0.4815 | 0.1271 | 73.6% | 0.6337 | 0.2058 | 80.4% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| m_rcg_avg | dino_swin_l | C | 0.6613 | 0.4863 | 26.5% | 0.8039 | 0.6139 | 36.0% |

### 20260823T101823Z — `run_attack_osfd_rcg_osfd_dev_50_n20_20260823T101823Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `m_osfd`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=osfd, k_draws=1, use_gate=False
- crafted=20 skipped=0 craft=143.7s eval=23.5s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| m_osfd | faster_rcnn_r50 | — | 0.4519 | 0.0739 | 83.7% | 0.6062 | 0.1007 | 92.6% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| m_osfd | yolox_l | B | 0.5782 | 0.1732 | 70.0% | 0.6989 | 0.2274 | 75.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| m_osfd | mask_rcnn_swin_t | C | 0.4815 | 0.1655 | 65.6% | 0.6337 | 0.2395 | 74.2% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| m_osfd | dino_swin_l | C | 0.6613 | 0.4942 | 25.3% | 0.8039 | 0.6494 | 25.0% |

### 20260823T095013Z — `run_attack_osfd_mvc_mvc_cons_lam10_dev_50_n20_20260823T095013Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `m_mvc_cons_lam10`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=mvc_cons_lam10, n_variants=2, use_consistency=True, keep_prob=0.9, lambda_cons=10.0
- crafted=20 skipped=0 craft=357.2s eval=32.1s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| m_mvc_cons_lam10 | yolox_l | B | 0.5782 | 0.1247 | 78.4% | 0.6989 | 0.2269 | 75.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| m_mvc_cons_lam10 | mask_rcnn_swin_t | C | 0.4815 | 0.1209 | 74.9% | 0.6337 | 0.2176 | 80.4% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| m_mvc_cons_lam10 | dino_swin_l | C | 0.6613 | 0.4665 | 29.5% | 0.8039 | 0.5952 | 27.0% |

### 20260823T092808Z — `run_attack_osfd_mvc_mvc_cons_dev_50_n20_20260823T092808Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `m_mvc_cons`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=mvc_cons, n_variants=2, use_consistency=True, keep_prob=0.9, lambda_cons=100.0
- crafted=20 skipped=0 craft=356.5s eval=48.4s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| m_mvc_cons | faster_rcnn_r50 | — | 0.4519 | 0.1052 | 76.7% | 0.6062 | 0.1329 | 90.4% |
| clean | fcos_r50 | A | — | 0.4470 | — | — | 0.5987 | — |
| m_mvc_cons | fcos_r50 | A | 0.4470 | 0.1333 | 70.2% | 0.5987 | 0.1937 | 80.3% |
| clean | deformable_detr | A | — | 0.5535 | — | — | 0.6902 | — |
| m_mvc_cons | deformable_detr | A | 0.5535 | 0.1257 | 77.3% | 0.6902 | 0.1507 | 87.5% |
| clean | yolov3_d53 | B | — | 0.4158 | — | — | 0.6419 | — |
| m_mvc_cons | yolov3_d53 | B | 0.4158 | 0.1368 | 67.1% | 0.6419 | 0.2387 | 72.6% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| m_mvc_cons | yolox_l | B | 0.5782 | 0.3383 | 41.5% | 0.6989 | 0.4563 | 45.5% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| m_mvc_cons | mask_rcnn_swin_t | C | 0.4815 | 0.2774 | 42.4% | 0.6337 | 0.4172 | 49.5% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| m_mvc_cons | dino_swin_l | C | 0.6613 | 0.5200 | 21.4% | 0.8039 | 0.7068 | 18.0% |

### 20260823T092121Z — `run_attack_osfd_mvc_mvc_avg_dev_50_n20_20260823T092121Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `m_mvc_avg`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=mvc_avg, n_variants=2, use_consistency=False, keep_prob=0.9, lambda_cons=100.0
- crafted=20 skipped=0 craft=340.9s eval=49.8s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| m_mvc_avg | faster_rcnn_r50 | — | 0.4519 | 0.0800 | 82.3% | 0.6062 | 0.1003 | 92.6% |
| clean | fcos_r50 | A | — | 0.4470 | — | — | 0.5987 | — |
| m_mvc_avg | fcos_r50 | A | 0.4470 | 0.0919 | 79.4% | 0.5987 | 0.1059 | 88.5% |
| clean | deformable_detr | A | — | 0.5535 | — | — | 0.6902 | — |
| m_mvc_avg | deformable_detr | A | 0.5535 | 0.0958 | 82.7% | 0.6902 | 0.1056 | 90.9% |
| clean | yolov3_d53 | B | — | 0.4158 | — | — | 0.6419 | — |
| m_mvc_avg | yolov3_d53 | B | 0.4158 | 0.0855 | 79.4% | 0.6419 | 0.1314 | 85.7% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| m_mvc_avg | yolox_l | B | 0.5782 | 0.1664 | 71.2% | 0.6989 | 0.2264 | 76.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| m_mvc_avg | mask_rcnn_swin_t | C | 0.4815 | 0.1682 | 65.1% | 0.6337 | 0.2644 | 73.2% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| m_mvc_avg | dino_swin_l | C | 0.6613 | 0.4845 | 26.7% | 0.8039 | 0.6414 | 32.0% |

### 20260823T091448Z — `run_attack_osfd_mvc_osfd_dev_50_n20_20260823T091448Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `m_osfd`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800, tag=osfd, n_variants=1, use_consistency=False, keep_prob=0.9, lambda_cons=100.0
- crafted=20 skipped=0 craft=172.8s eval=62.7s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4519 | — | — | 0.6062 | — |
| m_osfd | faster_rcnn_r50 | — | 0.4519 | 0.0739 | 83.7% | 0.6062 | 0.1007 | 92.6% |
| clean | fcos_r50 | A | — | 0.4470 | — | — | 0.5987 | — |
| m_osfd | fcos_r50 | A | 0.4470 | 0.0947 | 78.8% | 0.5987 | 0.1027 | 88.5% |
| clean | deformable_detr | A | — | 0.5535 | — | — | 0.6902 | — |
| m_osfd | deformable_detr | A | 0.5535 | 0.0762 | 86.2% | 0.6902 | 0.1067 | 90.9% |
| clean | yolov3_d53 | B | — | 0.4158 | — | — | 0.6419 | — |
| m_osfd | yolov3_d53 | B | 0.4158 | 0.1005 | 75.8% | 0.6419 | 0.1498 | 86.9% |
| clean | yolox_l | B | — | 0.5782 | — | — | 0.6989 | — |
| m_osfd | yolox_l | B | 0.5782 | 0.1732 | 70.0% | 0.6989 | 0.2274 | 75.2% |
| clean | mask_rcnn_swin_t | C | — | 0.4815 | — | — | 0.6337 | — |
| m_osfd | mask_rcnn_swin_t | C | 0.4815 | 0.1655 | 65.6% | 0.6337 | 0.2395 | 74.2% |
| clean | dino_swin_l | C | — | 0.6613 | — | — | 0.8039 | — |
| m_osfd | dino_swin_l | C | 0.6613 | 0.4942 | 25.3% | 0.8039 | 0.6494 | 25.0% |

### 20260823T074516Z — `run_attack_osfd_e3_k3_rrb_dev_50_20260823T074516Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `e3_k3_rrb`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=49 skipped=1 craft=425.5s eval=87.7s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4610 | — | — | 0.6466 | — |
| e3_k3_rrb | faster_rcnn_r50 | — | 0.4611 | 0.0336 | 92.7% | 0.6468 | 0.0472 | 96.4% |
| clean | fcos_r50 | A | — | 0.4530 | — | — | 0.6506 | — |
| e3_k3_rrb | fcos_r50 | A | 0.4531 | 0.0579 | 87.2% | 0.6507 | 0.0717 | 93.8% |
| clean | deformable_detr | A | — | 0.5066 | — | — | 0.6628 | — |
| e3_k3_rrb | deformable_detr | A | 0.5150 | 0.0355 | 93.1% | 0.6722 | 0.0513 | 93.9% |
| clean | yolov3_d53 | B | — | 0.4411 | — | — | 0.6626 | — |
| e3_k3_rrb | yolov3_d53 | B | 0.4411 | 0.0723 | 83.6% | 0.6627 | 0.1219 | 79.7% |
| clean | yolox_l | B | — | 0.6067 | — | — | 0.7536 | — |
| e3_k3_rrb | yolox_l | B | 0.6068 | 0.1494 | 75.4% | 0.7538 | 0.2099 | 68.5% |
| clean | mask_rcnn_swin_t | C | — | 0.5052 | — | — | 0.6952 | — |
| e3_k3_rrb | mask_rcnn_swin_t | C | 0.5052 | 0.1515 | 70.0% | 0.6957 | 0.2226 | 67.4% |
| clean | dino_swin_l | C | — | 0.6547 | — | — | 0.7998 | — |
| e3_k3_rrb | dino_swin_l | C | 0.6548 | 0.5072 | 22.5% | 0.7998 | 0.6797 | 27.5% |

### 20260823T073642Z — `run_attack_osfd_e3_k3_norrb_dev_50_20260823T073642Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `e3_k3_norrb`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, use_rrb=False, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=49 skipped=1 craft=214.1s eval=85.2s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4610 | — | — | 0.6466 | — |
| e3_k3_norrb | faster_rcnn_r50 | — | 0.4611 | 0.0000 | 100.0% | 0.6468 | 0.0000 | 100.0% |
| clean | fcos_r50 | A | — | 0.4530 | — | — | 0.6506 | — |
| e3_k3_norrb | fcos_r50 | A | 0.4531 | 0.0844 | 81.4% | 0.6507 | 0.1312 | 87.7% |
| clean | deformable_detr | A | — | 0.5066 | — | — | 0.6628 | — |
| e3_k3_norrb | deformable_detr | A | 0.5150 | 0.0073 | 98.6% | 0.6722 | 0.0107 | 98.7% |
| clean | yolov3_d53 | B | — | 0.4411 | — | — | 0.6626 | — |
| e3_k3_norrb | yolov3_d53 | B | 0.4411 | 0.2041 | 53.7% | 0.6627 | 0.3792 | 49.8% |
| clean | yolox_l | B | — | 0.6067 | — | — | 0.7536 | — |
| e3_k3_norrb | yolox_l | B | 0.6068 | 0.4340 | 28.5% | 0.7538 | 0.5902 | 24.3% |
| clean | mask_rcnn_swin_t | C | — | 0.5052 | — | — | 0.6952 | — |
| e3_k3_norrb | mask_rcnn_swin_t | C | 0.5052 | 0.3938 | 22.1% | 0.6957 | 0.5650 | 22.3% |
| clean | dino_swin_l | C | — | 0.6547 | — | — | 0.7998 | — |
| e3_k3_norrb | dino_swin_l | C | 0.6548 | 0.6057 | 7.5% | 0.7998 | 0.7555 | 8.4% |

### 20260823T073141Z — `run_attack_osfd_e3_k1_rrb_dev_50_20260823T073141Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `e3_k1_rrb`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=1.0, use_rrb=True, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=49 skipped=1 craft=425.9s eval=91.5s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4610 | — | — | 0.6466 | — |
| e3_k1_rrb | faster_rcnn_r50 | — | 0.4611 | 0.0583 | 87.4% | 0.6468 | 0.0783 | 85.6% |
| clean | fcos_r50 | A | — | 0.4530 | — | — | 0.6506 | — |
| e3_k1_rrb | fcos_r50 | A | 0.4531 | 0.0938 | 79.3% | 0.6507 | 0.1274 | 83.3% |
| clean | deformable_detr | A | — | 0.5066 | — | — | 0.6628 | — |
| e3_k1_rrb | deformable_detr | A | 0.5150 | 0.0798 | 84.5% | 0.6722 | 0.1003 | 87.8% |
| clean | yolov3_d53 | B | — | 0.4411 | — | — | 0.6626 | — |
| e3_k1_rrb | yolov3_d53 | B | 0.4411 | 0.1257 | 71.5% | 0.6627 | 0.2277 | 70.5% |
| clean | yolox_l | B | — | 0.6067 | — | — | 0.7536 | — |
| e3_k1_rrb | yolox_l | B | 0.6068 | 0.2650 | 56.3% | 0.7538 | 0.3694 | 53.0% |
| clean | mask_rcnn_swin_t | C | — | 0.5052 | — | — | 0.6952 | — |
| e3_k1_rrb | mask_rcnn_swin_t | C | 0.5052 | 0.2683 | 46.9% | 0.6957 | 0.3850 | 45.9% |
| clean | dino_swin_l | C | — | 0.6547 | — | — | 0.7998 | — |
| e3_k1_rrb | dino_swin_l | C | 0.6548 | 0.5531 | 15.5% | 0.7998 | 0.6864 | 21.1% |

### 20260823T072302Z — `run_attack_osfd_e3_k1_norrb_dev_50_20260823T072302Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `e3_k1_norrb`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=1.0, use_rrb=False, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=49 skipped=1 craft=205.6s eval=138.8s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4610 | — | — | 0.6466 | — |
| e3_k1_norrb | faster_rcnn_r50 | — | 0.4611 | 0.0020 | 99.6% | 0.6468 | 0.0055 | 97.3% |
| clean | fcos_r50 | A | — | 0.4530 | — | — | 0.6506 | — |
| e3_k1_norrb | fcos_r50 | A | 0.4531 | 0.1121 | 75.2% | 0.6507 | 0.1822 | 77.2% |
| clean | deformable_detr | A | — | 0.5066 | — | — | 0.6628 | — |
| e3_k1_norrb | deformable_detr | A | 0.5150 | 0.0088 | 98.3% | 0.6722 | 0.0119 | 96.5% |
| clean | yolov3_d53 | B | — | 0.4411 | — | — | 0.6626 | — |
| e3_k1_norrb | yolov3_d53 | B | 0.4411 | 0.2422 | 45.1% | 0.6627 | 0.4160 | 39.1% |
| clean | yolox_l | B | — | 0.6067 | — | — | 0.7536 | — |
| e3_k1_norrb | yolox_l | B | 0.6068 | 0.4502 | 25.8% | 0.7538 | 0.5989 | 27.1% |
| clean | mask_rcnn_swin_t | C | — | 0.5052 | — | — | 0.6952 | — |
| e3_k1_norrb | mask_rcnn_swin_t | C | 0.5052 | 0.4008 | 20.7% | 0.6957 | 0.5852 | 18.6% |
| clean | dino_swin_l | C | — | 0.6547 | — | — | 0.7998 | — |
| e3_k1_norrb | dino_swin_l | C | 0.6548 | 0.6028 | 7.9% | 0.7998 | 0.7597 | 7.6% |

### 20260821T155439Z — `run_attack_mi_fgsm_dev_50_20260821T155439Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `mi_fgsm`
- config: attack_type=mi_fgsm, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=49 skipped=1 craft=491.4s eval=99.6s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4610 | — | — | 0.6466 | — |
| mi_fgsm | faster_rcnn_r50 | — | 0.4611 | 0.0000 | 100.0% | 0.6468 | 0.0000 | 99.5% |
| clean | fcos_r50 | A | — | 0.4530 | — | — | 0.6506 | — |
| mi_fgsm | fcos_r50 | A | 0.4531 | 0.1175 | 74.1% | 0.6507 | 0.1958 | 60.5% |
| clean | deformable_detr | A | — | 0.5066 | — | — | 0.6628 | — |
| mi_fgsm | deformable_detr | A | 0.5150 | 0.0051 | 99.0% | 0.6722 | 0.0090 | 93.5% |
| clean | yolov3_d53 | B | — | 0.4411 | — | — | 0.6626 | — |
| mi_fgsm | yolov3_d53 | B | 0.4411 | 0.2714 | 38.5% | 0.6627 | 0.4705 | 32.9% |
| clean | yolox_l | B | — | 0.6067 | — | — | 0.7536 | — |
| mi_fgsm | yolox_l | B | 0.6068 | 0.4399 | 27.5% | 0.7538 | 0.5987 | 16.7% |
| clean | mask_rcnn_swin_t | C | — | 0.5052 | — | — | 0.6952 | — |
| mi_fgsm | mask_rcnn_swin_t | C | 0.5052 | 0.4112 | 18.6% | 0.6957 | 0.5836 | 20.2% |
| clean | dino_swin_l | C | — | 0.6547 | — | — | 0.7998 | — |
| mi_fgsm | dino_swin_l | C | 0.6548 | 0.5858 | 10.5% | 0.7998 | 0.7334 | 11.6% |

### 20260821T154422Z — `run_attack_osfd_dev_50_20260821T154422Z.json`

- loại run: `run_attack`
- manifest: `data/manifests/dev_50.json`
- attack: `osfd`
- config: attack_type=osfd, epsilon=5.0, alpha=1.0, steps=100, mu=1.0, k=3.0, theta=7.0, l_s=10, rho=0.8, s_max=1.1, sigma=6.0, canvas=800
- crafted=49 skipped=1 craft=423.3s eval=102.7s

| attack | model | nhóm | mAP_clean | mAP_adv | mAP_drop % | AP50_clean | AP50_adv | ASR |
|---|---|---|---|---|---|---|---|---|
| clean | faster_rcnn_r50 | — | — | 0.4610 | — | — | 0.6466 | — |
| osfd | faster_rcnn_r50 | — | 0.4611 | 0.0002 | 100.0% | 0.6468 | 0.0009 | 99.1% |
| clean | fcos_r50 | A | — | 0.4530 | — | — | 0.6506 | — |
| osfd | fcos_r50 | A | 0.4531 | 0.0147 | 96.7% | 0.6507 | 0.0241 | 98.1% |
| clean | deformable_detr | A | — | 0.5066 | — | — | 0.6628 | — |
| osfd | deformable_detr | A | 0.5150 | 0.0025 | 99.5% | 0.6722 | 0.0038 | 97.4% |
| clean | yolov3_d53 | B | — | 0.4411 | — | — | 0.6626 | — |
| osfd | yolov3_d53 | B | 0.4411 | 0.0468 | 89.4% | 0.6627 | 0.1029 | 81.6% |
| clean | yolox_l | B | — | 0.6067 | — | — | 0.7536 | — |
| osfd | yolox_l | B | 0.6068 | 0.1152 | 81.0% | 0.7538 | 0.1841 | 71.3% |
| clean | mask_rcnn_swin_t | C | — | 0.5052 | — | — | 0.6952 | — |
| osfd | mask_rcnn_swin_t | C | 0.5052 | 0.1402 | 72.2% | 0.6957 | 0.2180 | 69.4% |
| clean | dino_swin_l | C | — | 0.6547 | — | — | 0.7998 | — |
| osfd | dino_swin_l | C | 0.6548 | 0.4968 | 24.1% | 0.7998 | 0.6642 | 26.3% |


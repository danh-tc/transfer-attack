# Nhật ký nghiên cứu: Tái hiện transfer-attack OSFD

## 1. Mục tiêu

**Mục tiêu cuối cùng của project này KHÔNG phải là tái hiện lại OSFD, mà là đề xuất một
phương pháp transfer-attack mới (novel method) có khả năng transfer tốt hơn OSFD** —
đặc biệt là thu hẹp gap transferability sang **nhóm C (backbone Swin Transformer)**, nơi
OSFD hiện đang yếu nhất.

Cơ sở cho mục tiêu này: **OSFD** ("Transferable Adversarial Attacks for Object Detection
Using Object-Aware Significant Feature Distortion", Ding et al., AAAI-24 —
`papers/06244-AAAI24.DingX.pdf`, code gốc tại `reference-repo/OSFD/`) claim rằng tấn công
vào **backbone feature** của detector — triệt tiêu (suppress) significant feature tại
vùng object và khuếch đại (amplify) vicinal feature xung quanh biên object, qua loss ở
Eq. 2 kết hợp augmentation RRB — giúp adversarial example untargeted black-box transfer
tốt hơn baseline tấn công task/output loss (MI-FGSM), across mọi kiến trúc detector
(one-/two-stage, anchor-based/free, CNN/Transformer).

Run baseline đầu tiên (`dev_50`, 100 step — xem [EXPERIMENTS.md](EXPERIMENTS.md)) đã xác
nhận claim này **đúng nhưng không đều**: OSFD vượt trội MI-FGSM rõ rệt ở nhóm A/B (ResNet,
CNN non-ResNet), nhưng khoảng cách thu hẹp mạnh ở nhóm C —
`dino_swin_l`: ASR OSFD 26.3% vs MI-FGSM 11.6% (chênh ~15 điểm, so với ~40-50 điểm ở nhóm
A/B), mAP-drop tương tự (24.1% vs 10.5%). Đây chính là **GAP** mà project sẽ tập trung
khai thác: **OSFD vẫn transfer kém sang backbone Transformer khác biệt (nhóm C)**, gợi ý
rằng cơ chế suppress/amplify significant-vs-vicinal feature của OSFD — vốn được thiết kế
dựa trên spatial consistency của backbone CNN — có thể không khai thác đúng đặc tính của
self-attention trong Swin Transformer.

Vì vậy project trả lời 2 câu hỏi, theo 2 giai đoạn:

1. **(Đã làm — baseline)** Khi craft trên một surrogate Faster R-CNN (ResNet-50) duy
   nhất, OSFD có làm giảm accuracy các target black-box nhiều hơn MI-FGSM không, và gap
   đó lớn/nhỏ thế nào theo từng nhóm backbone (A/B/C)?
2. **(Đang làm — mục tiêu chính)** Có thể thiết kế một phương pháp mới, kế thừa ý tưởng
   feature-level attack của OSFD nhưng sửa/thay cơ chế suppress-amplify (hoặc augmentation
   RRB, hoặc cả hai) để **thu hẹp gap ở nhóm C**, mà không đánh đổi hiệu quả ở nhóm A/B,
   hay không?

Việc tái hiện OSFD + baseline MI-FGSM ở §2-§7 dưới đây là bước đệm bắt buộc (để có số đo
gap đáng tin cậy và một baseline so sánh công bằng), **không phải đích đến**. Đây cũng là
bản tái hiện có thu hẹp phạm vi (scoped), không phải replicate 1:1 — xem
[§5 Điểm khác biệt](#5-điểm-khác-biệt-so-với-paper) để biết những gì khác so với setup
thực nghiệm gốc của paper và lý do.

## 2. Độ đo (metrics)

Hai attack được so sánh dưới cùng ngân sách nhiễu (perturbation budget) và cùng số step:

- **`osfd`** — `transfer_attack/losses.py::osfd_loss`, loss MSE trên backbone feature
  (Eq. 2 của paper), tối ưu qua augmentation RRB (`transfer_attack/augment.py`).
- **`mi_fgsm`** — `transfer_attack/losses.py::detector_task_loss`, chính loss huấn luyện
  của detector (tổng tất cả term `*loss*` từ `model.loss()`), không dùng augmentation.
  Đóng vai trò baseline "tấn công output, không tấn công feature" mà paper cho rằng kém
  hiệu quả hơn về khả năng transfer.

Cả hai dùng chung vòng lặp crafting I-FGSM + MI-momentum (`transfer_attack/attack.py`),
nên biến độc lập duy nhất là *cái gì đang bị tối ưu*, không phải *cách tối ưu*.

Trên mỗi target model, với cả input clean và adversarial (`transfer_attack/eval_metrics.py`):

- **mAP** (độ đo chính thức của COCO — AP trung bình trên IoU 0.5:0.95, bước 0.05) và
  **AP50**, tính qua `pycocotools.COCOeval`. Báo cáo dưới dạng
  `mAP_drop_pct = 100 * (mAP_clean - mAP_adv) / mAP_clean` trên **cùng** tập ảnh mà noise
  adversarial thực sự bao phủ (xem ghi chú về cache trong
  `evaluate.py::evaluate_one_model`).
- **ASR (evasion success rate)** — **không có trong paper gốc**, được thêm vào ở đây vì
  mAP-drop gộp chung hai hiệu ứng (mất detection thật vs. sinh thêm false positive) nên
  khó suy luận per-image hơn. Định nghĩa (`eval_metrics.py::compute_asr_for_image`): với
  mỗi GT box được detect đúng trên ảnh *clean* (greedy match IoU≥0.5 + score≥0.3 + đúng
  class), kiểm tra xem prediction tương ứng còn tồn tại trên ảnh *adversarial* hay không;
  ASR = tỷ lệ không còn match, tính gộp trên toàn dataset (không average theo từng ảnh).

Cả mAP-drop và ASR đều được tính riêng cho từng target model và từng attack; Table 1 của
paper chỉ báo cáo mAP, nên ASR là phần bổ sung riêng của project này để phân tích chi
tiết hơn.

## 3. Dataset

**COCO val2017** (không phải VOC2012 — xem [§5](#5-điểm-khác-biệt-so-với-paper)), các
manifest cố định nằm ở `data/manifests/` (shuffle có seed, `seed=42`, sinh bởi
`setup_env.sh` bước 15):

| manifest | số ảnh | vai trò |
|---|---|---|
| `dev_30.json` | 30 | smoke test nhanh |
| `dev_50.json` | 50 | các run baseline hiện tại (tài liệu này) |
| `dev_300.json` | 300 | dev set chính — **held-out khỏi `val_100`**, dùng để iterate hyperparameter |
| `val_100.json` | 100 | held-out — **không đụng vào cho đến khi config đã chốt**; số liệu báo cáo cuối cùng nên lấy từ split này |

Tất cả ảnh được resize giữ tỷ lệ + zero-pad bottom/right về canvas cố định `800×800`
(`transfer_attack/data.py::load_canvas_image`, `CANVAS=800` trong `constants.py`) để mọi
model — surrogate và cả 6 target — nhìn thấy perturbation giống hệt nhau về mặt bit, không
lệch resize giữa các model.

Ảnh không có GT box hợp lệ nào (không phải crowd, không degenerate) sẽ bị skip khi crafting
(`craft.py`/`run_attack.py` log số này ở `n_skipped`).

## 4. Model

Một surrogate, sáu target black-box chia 3 nhóm theo họ backbone
(`transfer_attack/models.py::MODEL_REGISTRY`), tất cả dùng checkpoint model-zoo mmdet
3.3.0:

| vai trò | nhóm | tên | config | backbone / họ detector |
|---|---|---|---|---|
| surrogate | — | `faster_rcnn_r50` | `faster-rcnn_r50_fpn_1x_coco` | ResNet-50, two-stage |
| target | A | `fcos_r50` | `fcos_r50-caffe_fpn_gn-head_1x_coco` | ResNet-50, anchor-free one-stage |
| target | A | `deformable_detr` | `deformable-detr_r50_16xb2-50e_coco` | ResNet-50, transformer detector |
| target | B | `yolov3_d53` | `yolov3_d53_mstrain-608_273e_coco` | Darknet-53 (CNN non-ResNet) |
| target | B | `yolox_l` | `yolox_l_8x8_300e_coco` | CSPDarknet (CNN non-ResNet) |
| target | C | `mask_rcnn_swin_t` | `mask-rcnn_swin-t-p4-w7_fpn_1x_coco` | Swin-T, two-stage |
| target | C | `dino_swin_l` | `dino-5scale_swin-l_8xb2-12e_coco` | Swin-L, transformer detector |

Nhóm A cô lập trục "cùng họ backbone với surrogate, khác head/kiến trúc"; nhóm B cô lập
trục "khác họ backbone CNN"; nhóm C cô lập trục "backbone Transformer" — ba trục mà claim
về khả năng transfer của paper cần phải đúng across.

## 5. Điểm khác biệt so với paper

Ghi lại rõ để bất kỳ khoảng cách nào giữa số liệu của mình và Table 1 của paper đều có
nguyên nhân đã biết, chứ không phải chỉ là "tái hiện không khớp":

| paper | project này | lý do |
|---|---|---|
| Surrogate: 4 model (YOLOv3, VFNet, FasterRCNN-R101, MaskRCNN-Swin) | 1 surrogate (Faster R-CNN R50) | Thu hẹp phạm vi — trả lời "OSFD có transfer tốt hơn MI-FGSM từ *một* surrogate không" thì không cần sweep cả surrogate; có thể thêm sau như một trục thực nghiệm riêng. |
| Dataset: 2000 ảnh từ VOC2012 trainval | COCO val2017, manifest 30–300 ảnh | Tất cả checkpoint target model đã được train/eval sẵn trên COCO (mmdet model zoo), tránh phải map category VOC→COCO; manifest nhỏ hơn để iterate nhanh, `dev_300`/`val_100` được sizing cho một lần kiểm tra held-out cuối, không phải full 2000 ảnh như paper. |
| RRB: rotation và resizing là hai nhánh **song song** trên cùng input adversarial (Fig. 2) | rotation và resizing là **tuần tự** (`branch2 = resize(branch1)`) | Port đúng nguyên văn từ `reference-repo/OSFD/attack/base/RRB.py`'s `forward()` thực tế, vốn **không khớp với hình minh họa trong paper** — xem comment đầu file `transfer_attack/augment.py`. Đây là **sai khác đã biết giữa mô tả trong paper và code do chính tác giả paper release**; mình theo code vì đó là thứ tạo ra số liệu họ báo cáo. |
| Evaluation: chỉ mAP (Table 1) | mAP, AP50, **và ASR** | ASR được thêm vào để có tín hiệu evasion per-image dễ diễn giải hơn — xem [§2](#2-độ-đo-metrics). |
200 attack step (mặc định của paper, cũng là `constants.STEPS`), 2000 ảnh | các run hiện tại dùng **50 ảnh (`dev_50`) / 100 step** | **Chủ đích, không phải tạm thời**: vì mục tiêu chính là *tìm phương pháp mới* (§1), cần vòng lặp thử-nghiệm nhanh — 50 ảnh/100 step giữ mỗi lần craft+eval trong vài phút để so sánh nhiều biến thể phương pháp liên tiếp. Cấu hình rẻ này là chuẩn cho giai đoạn khai phá (exploration); chỉ khi đã chọn được ứng viên tốt nhất mới chạy full 200-step trên `dev_300`, rồi xác nhận trên `val_100`. |

## 6. Cách chạy lại một experiment

```bash
source /workspace/evasion-venv/bin/activate
cd /workspace/transfer-attack

# craft + evaluate một attack end-to-end, ghi ra 1 file run-log JSON vào runs/
python scripts/run_attack.py --attack osfd    --manifest data/manifests/dev_50.json --steps 100
python scripts/run_attack.py --attack mi_fgsm --manifest data/manifests/dev_50.json --steps 100

# sinh lại bảng experiment log (file đi kèm với tài liệu này, xem bên dưới)
python scripts/gen_experiment_log.py
```

`scripts/check_env.py` (cũng được gọi ở cuối `setup_env.sh`) kiểm tra checkpoint/config/
manifest/COCO annotation resolve được trước khi chạy các lệnh trên. `scripts/sanity_check.py`
vẽ box dự đoán lên vài ảnh cho mỗi model — bước kiểm tra thủ công bằng mắt xem việc xử lý
mean/std/bgr_to_rgb theo từng model trong `normalize.py` có thực sự đúng không, vì phần này
được suy ra bằng cách đối xứng với reference repo chứ không được paper mô tả trực tiếp.

## 7. Nhật ký thực nghiệm

Xem **[EXPERIMENTS.md](EXPERIMENTS.md)** — tự sinh từ mọi file `runs/*.json` bằng
`scripts/gen_experiment_log.py`. Sinh lại file này sau mỗi lần chạy `run_attack.py`/
`evaluate.py` mới; không sửa tay.

## 8. Chuỗi diagnostic E1→E3: cơ chế gap nhóm C

Năm thực nghiệm chẩn đoán (không phải thử phương pháp mới) chạy trên chính noise OSFD
`dev_50`/100 step ở baseline, nhằm trả lời: **tại sao OSFD transfer kém hơn hẳn sang nhóm C
(Swin Transformer) so với nhóm A/B?** Script tương ứng:
`scripts/e1_feature_damage.py`, `scripts/e2_pipeline_attenuation.py`,
`scripts/e2b_proposal_selection.py`, `scripts/e2c_roi_classification_collapse.py`,
`scripts/e3_osfd_rrb_factorial.py`.

**E1 — Backbone feature damage không giải thích gap.** Cosine distance backbone feature
clean/adv cho cả 7 model (`results/e1_feature_damage_summary.csv`): cả hai target Swin bị
phá ít hơn nhóm A/B (cos_dist 0.14–0.22 vs 0.45–0.65), nhưng **giữa hai model Swin, thứ tự
đảo ngược so với mAP_drop** — `mask_rcnn_swin_t` bị phá backbone *ít hơn* `dino_swin_l`
(0.141 < 0.225) nhưng mAP_drop lại *cao hơn nhiều* (72.2% > 24.1%). → backbone-feature-
mismatch không phải bottleneck chính.

**E2 — Pipeline attenuation (Mask-RCNN-SwinT vs DINO-SwinL).** Đo distortion tại từng
checkpoint dọc pipeline mỗi model (`results/e2_pipeline_attenuation.csv`). Mask-RCNN's
RoI-head-output (đo tại proposal cố định từ ảnh clean) chỉ đổi 0.078 — quá thấp để giải
thích mAP_drop 72.2%, cho thấy phép đo "tại proposal cố định" đang bỏ sót đúng cơ chế tấn
công thật.

**E2b — Proposal-selection instability: một phần, không đủ.** Overlap giữa tập proposal
RPN chọn trên ảnh clean vs adv của `mask_rcnn_swin_t` (`results/e2b_proposal_selection.csv`):
proposal *identity* đổi mạnh (Jaccard=0.24, chỉ 38% proposal clean còn "sống" trong tập
adv) — xác nhận NMS/top-K khuếch đại rời rạc một lệch RPN-score rất nhỏ (E2: cos_dist=
0.050). Nhưng **GT recall chỉ giảm 14.1%** (95.3%→81.9%) — RPN vẫn định vị đúng phần lớn
object thật trên ảnh adv → không đủ để một mình giải thích 72.2% mAP_drop.

**E2c — RoI classification collapse: mảnh còn thiếu.** Trên đúng các GT box vẫn được cả
hai proposal-set (clean và adv) định vị đúng (71.2% của 337 GT box,
`results/e2c_roi_classification_collapse.csv`): correct-class score giảm 0.726→0.248
(−65.9%), tỷ lệ misclassify tăng 22.9%→77.5%, tỷ lệ rơi dưới score_thr=0.3 tăng
18.3%→67.5% — **gần khớp ASR quan sát được ở baseline (69.4%)**. RoI classification
collapse giải thích **phần lớn** attack success quan sát được trên `mask_rcnn_swin_t`
trong setup evaluation này (không phải toàn bộ — vẫn còn effect từ localization/post-
processing chưa tách bạch hết).

**E3 — RRB là driver chính; k=3 chỉ phát huy khi đi cùng RRB (interaction effect).**
Factorial 2×2 `k∈{1,3} × RRB∈{off,on}`, giữ nguyên mọi thứ khác
(`results/e3_osfd_rrb_factorial_summary.csv`, `runs/run_attack_osfd_e3_*`). Trên nhóm B/C:
bật RRB (giữ k=1) tăng ASR **+13 đến +31 điểm**; tăng k lên 3 (không RRB) chỉ đổi ASR
**−2.8 đến +3.7 điểm** (nhiễu); nhưng **interaction dương mạnh** đúng ở nhóm khó transfer
nhất — `mask_rcnn_swin_t` +17.8, `yolox_l` +18.3, `dino_swin_l` +5.6 điểm ASR vượt mức
cộng dồn tuyến tính của hai effect riêng lẻ.

→ **Kết luận có kỷ luật**: RRB là điều kiện cần tạo transferability; k=3 gần như vô dụng
khi đứng một mình, nhưng khuếch đại thêm đáng kể khi có RRB — đây là quan hệ **synergy**
(interaction thật), không phải hai đóng góp độc lập cộng tuyến tính. Diễn giải cơ chế: k=3
áp dụng lên các *augmented view* do RRB tạo ra, không phải ảnh gốc — nên "chỗ" để
amplification phát huy chỉ xuất hiện khi RRB đã tạo ra biến thể input.

**Câu hỏi mở tiếp theo (chưa chạy, đang cân nhắc)**: *RRB đang tạo ra thay đổi gì cụ thể
khiến k=3 "bật" lên?* Hai hướng đề xuất:
- **Feature displacement dưới RRB**: RRB có làm significant/vicinal region lệch đủ để k=3
  tạo gradient khác biệt hay không?
- **Gradient behavior**: so cosine-similarity của gradient giữa cấu hình k=1 và k=3,
  có/không RRB — nếu `cos(g_{k=1}, g_{k=3}) ≈ 1` khi không có RRB nhưng giảm rõ khi bật
  RRB, đó là bằng chứng trực tiếp cho cơ chế synergy vừa quan sát ở E3.

## 9. Phase M: Method Discovery (candidate mechanisms mới)

Sau khi E1-E3b đóng lại chuỗi diagnostic (§8), mục tiêu chuyển sang đúng §1: tìm một
mechanism **mới**, tốt hơn OSFD trên nhóm B/C, thay vì tiếp tục giải thích OSFD.

**M1 — Design constraints** (chốt trước khi generate candidate):
ưu tiên B/C (đặc biệt YOLOX-L / `mask_rcnn_swin_t` / `dino_swin_l`); không chỉ "thêm
augmentation"; không dựa vào instantaneous gradient alignment (E3b không support hypothesis
này); nên nhắm vào representation/model generalization hoặc trajectory-level robustness (vì
RRB — input-space view consistency — là driver transfer chính theo E3).

**M2/M3 — 3 candidate mechanism + novelty scan** (không code, chỉ specify + search literature):

| Candidate | Ý tưởng | Novelty | Fit với E1-E3b | Cost | Verdict |
|---|---|---|---|---|---|
| 1. CRA (Cross-Representation Alignment) | Fine-tune surrogate backbone để align feature với 1 witness model khác họ (kiểu SAA), rồi attack như bình thường | Thấp — gần trùng cơ chế **SAA** (NeurIPS 2025, arXiv:2501.01015: align surrogate-witness feature, tấn công common representation) | Trung bình (chỉ khớp E1) | Trung bình-cao (cần witness + training loop mới) | Không chọn — overlap quá cao |
| 2. TCR (Trajectory-Consistency Regularization) | Regularize *phương sai của update direction* qua cửa sổ K step (dưới các RRB view khác nhau), thay MI-momentum đơn thuần | Cao nhất — không tìm thấy publish trực tiếp; gần nhất là CSA/SVRE nhưng hoạt động *giữa các model/checkpoint*, không phải *trong nội bộ 1 trajectory qua augmented view* | Yếu nhất — E3b chỉ *loại trừ* 2 hypothesis đơn giản, chưa *xác nhận* trajectory-effect | Trung bình (cơ chế chưa đặc tả xong lúc đề xuất) | Candidate dự phòng — đang đặc tả (xem dưới) |
| 3. MVC (Model-space View Consistency) | Đối xứng với E3: ép feature suppress nhất quán qua N biến thể nhẹ của surrogate (channel-masked backbone), không chỉ ensemble gradient | Trung bình — không gian đông (T-SEA CVPR23, RaPA, FUGEA 2026) nhưng khung "explicit consistency loss qua model-variant" chưa thấy publish trực tiếp | Cao nhất lúc đề xuất (đối xứng trực tiếp với E3) | Trung bình (cần tự viết hook) | **Đã pilot — NO-GO** (xem M4) |

**M4 — Pilot Candidate 3 (MVC): NO-GO.** Script `scripts/m_mvc_pilot.py`, kết quả
`results/m_mvc_pilot_summary.csv`, log `runs/run_attack_osfd_mvc_*`. Thiết kế: 3 variant cùng
budget (k=3, RRB on) — `osfd` (baseline), `mvc_avg` (N=2 biến thể channel-masked backbone,
loss = trung bình, naive ensemble), `mvc_cons` (`mvc_avg` + penalty phạt disagreement giữa 2
biến thể).

- **Bug bắt được trước khi chạy full**: bản đầu tiên mask *sau khi* backbone đã forward xong
  (post-hoc, rẻ vì chỉ 1 forward pass) — nhưng về toán học, 2 "biến thể" khi đó chỉ là 2 view
  khác nhau của **cùng một tensor**, nên tại các channel cả 2 mask cùng giữ, giá trị *y hệt
  nhau tuyệt đối* → `disagree_mse = 0.000000` ở mọi stage, không phải do lambda nhỏ mà do
  term này **vô nghĩa theo construction**. Sửa đúng: forward hook giữa backbone (tại
  `layer2`), mỗi biến thể forward `backbone(aug)` **độc lập thật sự** để `layer3`/`layer4`
  tính lại riêng — disagree khi đó > 0 thật ở các stage sau điểm rẽ nhánh.
- **Sanity λ=100** (ước lượng scale sơ bộ, chưa tune): `mvc_cons` **collapse đồng loạt trên
  mọi model**, mạnh nhất đúng ở B/C (`yolox_l` −30.7 điểm ASR, `mask_rcnn_swin_t` −23.7) —
  do λ quá lớn so với scale `osfd_loss` (~300-350x), lấn át objective ascend chính.
- **Sanity λ=10** (N=20, chỉ 3 hard target để tiết kiệm): collapse biến mất, nhưng gain rõ
  (`CONS > AVG` ≥3-5 điểm) chỉ đạt ở **1/3 target** (`mask_rcnn_swin_t`: +7.2 vs avg, +6.2 vs
  osfd — đúng target mà E2c xác định có bottleneck RoI-classification-collapse); `yolox_l`
  gần như không đổi (−1.0), `dino_swin_l` giảm nhẹ (−5.0).
- **Verdict: NO-GO**, theo đúng tiêu chí đã chốt trước khi chạy (`ΔASR_cons vs avg ≥ 3-5
  điểm trên ≥2/3 hard target`) — chỉ đạt 1/3. Chủ động **không tiếp tục tune λ** để cứu riêng
  `mask_rcnn_swin_t` (tránh post-hoc fitting làm yếu câu chuyện nghiên cứu). Tín hiệu dương ở
  `mask_rcnn_swin_t` được ghi nhận như một lead khả dĩ cho tương lai, không phải noise thuần
  túy, nhưng không theo đuổi tiếp trong Phase M này.

**M5 — Candidate 2 (TCR), thu hẹp sau novelty scan lần 2.** Scan sâu hơn vào đúng cơ chế
"dùng lịch sử gradient qua nhiều iteration để ổn định update" cho thấy đây là **một trong
những góc đông nhất** của literature (MI-FGSM gốc, NI-FGSM, VMI-FGSM, Direction Tuning
arXiv:2303.15109, Decaying Steps for Sign-based Optimizers 2026, Staircase Sign Method đều
làm biến thể của chính ý tưởng này). Thu hẹp lại đúng niche còn trống: không phải variance
qua **thời gian** (đã đông), mà variance từ **RRB resampling tại cùng một trạng thái noise**
— đổi tên nội bộ thành **RCG (RRB-resampling Consistency Gate)**.

**RCG v0**: 3 variant cùng budget — `osfd` (K=1 draw/step, baseline) — `rcg_avg` (K=3 draw
RRB độc lập/step tại cùng noise, gradient = trung bình, KHÔNG gate — control bắt buộc để
tách bạch "gain từ đâu") — `rcg_gate` (`rcg_avg` + gate per-pixel = `|mean_i sign(g_i)|`
nhân vào step, không cần lambda vì gate tự nhiên bound [0,1]). Script
`scripts/m_rcg_pilot.py`, kết quả `results/m_rcg_pilot_summary.csv`, log
`runs/run_attack_osfd_rcg_*`.

Kết quả (N=20, 100 step, surrogate + YOLOX-L/Mask-SwinT/DINO-SwinL), ASR:

| model | osfd | rcg_avg | rcg_gate | avg−osfd | gate−avg |
|---|---|---|---|---|---|
| faster_rcnn_r50 (surrogate) | 92.6 | 100.0 | 98.9 | +7.4 | −1.1 |
| yolox_l | 75.2 | 86.1 | 85.1 | +10.9 | −1.0 |
| mask_rcnn_swin_t | 74.2 | 80.4 | 82.5 | +6.2 | +2.1 |
| dino_swin_l | 25.0 | 36.0 | 39.0 | +11.0 | +3.0 |

Diagnostics: `mean_gate(rcg_gate)=0.41` (gate thực sự selective, không trivially ≈1);
`frac_saturated` giảm rõ dưới gate (0.86→0.72) — xác nhận gate có làm giảm effective step
size như lo ngại trước khi chạy.

**Verdict: NO-GO cho phần gate** — `gate−avg` chỉ đạt ngưỡng ≥3-5 điểm ở 0/3 hard target rõ
ràng (`dino_swin_l` chạm đúng biên +3.0, hai target còn lại flat/âm nhẹ), không đạt tiêu chí
≥2/3 đã chốt trước khi chạy.

**Finding quan trọng hơn nằm ở cột `avg−osfd`, không phải `gate−avg`**: chỉ tăng K=1→3 draw
RRB/step (không cần gate) đã tăng ASR +6 đến +11 điểm ở **mọi** model, kể cả surrogate. Đây
là dạng Monte-Carlo variance-reduction qua nhiều augmented-view sample/step — về bản chất là
mở rộng trực tiếp của chính finding E3 (nhiều RRB view hơn → transfer tốt hơn), **không phải
mechanism mới** theo đúng constraint M1 ("không chỉ là thêm augmentation"). Có tín hiệu dương
thật, nhưng không đủ tư cách làm contribution chính.

## 10. Kết luận Phase M — đóng lại, mở Phase N

Ba candidate (CRA, MVC, TCR→RCG) đều bị loại hoặc NO-GO sau khi pilot/scan kỹ:

- **CRA**: loại trước pilot vì overlap cơ chế quá cao với SAA (NeurIPS 2025).
- **MVC**: pilot đầy đủ cả λ=100 và λ=10 — explicit model-space consistency không
  generalize ổn định (chỉ 1/3 hard target có gain rõ, không đạt ≥2/3).
- **RCG**: consistency gate (temporal→resampling-narrowed) không tạo gain vượt naive
  multi-view averaging (`rcg_avg`) — cũng không đạt ≥2/3.

**Finding tổng hợp đáng giữ lại** (khung cho Phase N, không phải một candidate riêng):

> Transfer gain hiện tại của OSFD đến chủ yếu từ **stochastic multi-view averaging** (RRB ở
> input-space theo E3; tăng K draw/step theo RCG-AVG) — trong khi **mọi** explicit
> consistency constraint đã thử (model-space qua MVC, gradient-agreement qua RCG) đều không
> tạo thêm lợi ích đáng tin cậy vượt naive averaging. "Consistency/agreement" và "thêm
> stochastic view" đã bị loại như trục tìm kiếm chính cho candidate tiếp theo.

**Phase N — rethink mechanism từ bottleneck còn chưa giải quyết**, hai hướng ưu tiên ban đầu:
- Khai thác **cross-representation vulnerability mà không cần witness model** (khác CRA —
  CRA cần witness, đây là tìm cách đạt hiệu ứng tương tự mà không cần grey-box access).
- Tìm **objective agnostic với detector pipeline** — tác động mạnh đồng thời cả YOLOX
  (CNN, one-stage), Mask-Swin (two-stage, RoI-classification-collapse theo E2c), và DINO
  (transformer, continuous decoder) — thay vì tối ưu sâu thêm quanh RRB.

## 11. Phase N1: Spectral vulnerability diagnostic (E4/E4b) — NO-GO

Sau khi bổ sung scan literature classification→OD (Scholar Gateway, đến 2026), hướng
**frequency-domain/spectral perturbation** nổi lên là chưa bị Phase M đụng tới và có tiền lệ
khá mạnh ở CNN↔ViT (SAA cho witness-alignment; GRVT/NRSM cho gradient-neighborhood-variance;
Wang et al. 2022 và FUSE 2026 cho frequency-domain). Trước khi thiết kế attack mới, chạy
diagnostic rẻ trước — đúng discipline đã dùng suốt Phase M.

**E4 — Spectral decomposition of existing OSFD noise (raw, không craft lại).** Script
`scripts/e4_spectral_decomposition.py`, kết quả `results/e4_spectral_decomposition.csv`.
Decompose noise OSFD `dev_50` (49 ảnh, đã crafted) bằng 2D FFT radial-band (low ≤1/3, mid
≤2/3, high >2/3 bán kính tần số chuẩn hóa; reconstruction check max abs err ~1e-5, chỉ residual
float). Inject riêng từng band (không craft lại) vào surrogate + YOLOX-L + Mask-SwinT +
DINO-SwinL:

| model | full | low (61% energy) | mid (29% energy) | high (10% energy) |
|---|---|---|---|---|
| faster_rcnn_r50 (surrogate) | 99.1% | 93.2% | 21.6% | 0.5% |
| yolox_l | 70.9% | 29.1% | 5.2% | 0.0% |
| mask_rcnn_swin_t | 68.6% | 50.0% | 5.4% | 1.2% |
| dino_swin_l | 27.5% | 13.5% | 2.0% | 1.6% |

Low band giữ tỷ lệ ASR cao hơn hẳn tỷ lệ energy của nó ở mọi model; high band gần như vô
dụng **kể cả trên chính surrogate/CNN** (bác bỏ phần giả thuyết ban đầu "high-freq chủ yếu
phá CNN").

**E4b — L∞-normalized band-wise re-evaluation** (tách frequency-effect khỏi amplitude-effect,
vẫn không craft lại — mỗi band rescale về đúng `epsilon` L∞ budget của attack gốc):

| model | low_raw→low_norm | mid_raw→mid_norm | high_raw→high_norm | low_norm/low_raw |
|---|---|---|---|---|
| faster_rcnn_r50 | 93.2→31.5 | 21.6→5.9 | 0.5→0.5 | 34% |
| yolox_l | 29.1→4.8 | 5.2→0.0 | 0.0→0.0 | 16% |
| mask_rcnn_swin_t | 50.0→9.9 | 5.4→2.5 | 1.2→1.2 | 20% |
| dino_swin_l | 13.5→4.4 | 2.0→1.2 | 1.6→1.2 | 33% |

Thứ tự `low > mid > high` vẫn giữ nguyên sau normalize, nhưng **mọi** band — kể cả low —
sụt hiệu quả mạnh (giữ chỉ 16-34% so với raw), và tỷ lệ giữ lại **không có signature riêng
cho cross-architecture transfer** (surrogate 34% và `dino_swin_l` 33% gần bằng nhau;
`yolox_l`/`mask_rcnn_swin_t` 16-20% còn thấp hơn cả surrogate).

**Kết luận (đúng câu chữ đã chốt)**:

> Existing OSFD perturbations are spectrally concentrated in low frequencies, but band-wise
> normalization removes most of their attack efficacy without revealing a cross-architecture-
> specific advantage; thus spectral composition of already-crafted perturbations is not
> sufficient evidence of an intrinsic shared vulnerable subspace.

**Phạm vi bị loại vs còn mở**:
- `Spectral decomposition of noise đã crafted (post-hoc)` → **NO-GO**.
- `Spectral-*constrained* optimization` (tối ưu trực tiếp dưới projection/filter tần số
  trong vòng lặp attack, không phải decompose-rồi-rescale noise có sẵn — đúng cách Wang 2022/
  FUSE 2026 làm) → **OPEN / parked, chưa bị falsify** — là một optimization dynamics khác hẳn,
  cần craft lại nên đắt hơn hẳn diagnostic vừa chạy; không ưu tiên trừ khi có thêm lý do
  mechanistic mạnh hơn.

## 12. Phase N2: STAT-NORM & Robust Surrogate pilots — NO-GO

Sau khi scan classification→OD literature bổ sung (Scholar Gateway, đến 2026), shortlist 3
hướng: N2-B (statistical/normalization-invariant features — main), N2-A (robust surrogate —
control phụ, chạy song song), N2-C (flatness/sharpness — parked, rủi ro overlap RCG).

**N2-B — per-channel spatial standardization trước `osfd_loss`.** Motivation (cố ý viết
thận trọng, không claim causality từ E1):

> OSFD may exploit feature magnitude and channel-statistic patterns that are specific to the
> surrogate representation; removing affine/statistical information before feature distortion
> may force the attack toward structural feature changes that survive architecture changes
> better.

Mechanism: `F_hat = (F - mean_HW(F)) / (std_HW(F) + eps)` áp cho cả clean và adv feature mỗi
stage, mọi thứ khác giữ nguyên (k=3, RRB on, budget). Script
`scripts/n2b_statnorm_pilot.py`, kết quả `results/n2b_statnorm_pilot_summary.csv`. N=20,
100 step:

| model | OSFD | STAT-NORM | delta |
|---|---|---|---|
| faster_rcnn_r50 (surrogate) | 96.8 | 95.7 | −1.1 |
| yolox_l | 81.2 | 73.3 | −7.9 |
| mask_rcnn_swin_t | 79.4 | 70.1 | −9.3 |
| dino_swin_l | 30.0 | 26.0 | −4.0 |

**Verdict: NO-GO, và pattern đi ngược hẳn hypothesis** — không chỉ mọi model giảm (không đạt
ngưỡng +5), mà đúng 2 hard target (`mask_rcnn_swin_t` −9.3, `yolox_l` −7.9) giảm **nhiều hơn**
surrogate (−1.1) — ngược hẳn chữ ký kỳ vọng `Δ_DINO/Mask > Δ_surrogate`.

**N2-A — robust ImageNet surrogate (control phụ, không phải contribution chính).** Checkpoint
Salman et al. 2020 (Linf eps=4/255, `github.com/MadryLab/robustness`), verify khớp 318/318
key backbone ResNet-50 của mmdet (0 missing). Chỉ backbone dùng để craft (`osfd` không chạm
neck/head); evaluation surrogate vẫn dùng checkpoint chuẩn — không có Frankenstein
backbone/head mismatch. Script `scripts/n2a_robust_surrogate_pilot.py`, kết quả
`results/n2a_robust_surrogate_summary.csv`. N=20, 100 step, so với baseline OSFD chuẩn (N2-B):

| model | ASR (robust-crafted) | OSFD baseline | delta |
|---|---|---|---|
| faster_rcnn_r50 (surrogate) | 11.7 | 96.8 | −85.1 |
| yolox_l | 0.0 | 81.2 | −81.2 |
| mask_rcnn_swin_t | 8.2 | 79.4 | −71.2 |
| dino_swin_l | 4.0 | 30.0 | −26.0 |

Collapse toàn diện mọi model. **Verdict (viết thận trọng vì có confound budget chưa loại
trừ)**:

> Robust-surrogate crafting collapses under the standard OSFD budget; because adversarial
> training changes optimization geometry, this result does not cleanly falsify robust-
> surrogate transferability, but it provides no positive evidence under the current attack
> regime.

Không theo đuổi thêm (tăng step/epsilon để "fair" đòi hỏi match attack strength trước — dễ
biến thành nhánh tuning dài cho 1 idea novelty thấp, không đáng effort).

**Kết luận Phase N2**: `N2-B` → NO-GO sạch (không confound). `N2-A` → NO-GO thực dụng /
inconclusive về mechanism. `N2-C` → vẫn parked, ưu tiên còn thấp hơn sau kết quả N2-A/RCG.

## 13. Phase N3: Success-vs-Failure diagnostic trên DINO transfer

Sau chuỗi NO-GO liên tiếp ở mức **aggregate/average** (E1 feature damage, E3b instantaneous
gradient, MVC/RCG consistency, E4/E4b spectral, N2-B stat-norm) — đổi góc nhìn: `dino_swin_l`
tự nhiên tạo ra 2 population rõ rệt trên cùng 1 surrogate/attack/budget (ASR ~25-30%, nghĩa là
phần lớn ảnh **thất bại**, một phần **thành công**) — một natural controlled experiment để hỏi
"cái gì phân biệt ảnh transfer-thành-công khỏi thất-bại" thay vì tiếp tục đổi method rồi nhìn
ASR trung bình.

**Căn cứ**: chủ yếu suy ra từ chính data của project (rất mạnh); literature chỉ support gián
tiếp (DetectSec: backbone ảnh hưởng transferability, kết luận classification không tự động
chuyển sang detection; FDA/frequency-feature: transfer đến từ shared abstract features;
transformation-invariance/feature-attention/gradient-neighborhood: property của perturbation
liên quan tới transfer) — **chưa thấy work nào đến 2026 phân loại success-vs-failure cho đúng
bài toán OD CNN→Transformer**. Novelty tiềm năng cao, risk chase noise cũng cao hơn vì chưa có
positive mechanism sẵn.

**Kế hoạch (diagnostic rẻ, không craft lại)**: chia 49 ảnh `dev_50` thành S (DINO attack
thành công) / F (thất bại) dựa trên noise OSFD đã có, so sánh property không cần craft lại:
object size/count, surrogate loss cuối, backbone distortion (cos_dist kiểu E1), perturbation
L1/L2/saturation ratio, spectral energy ratio theo band (kiểu E4), spatial concentration
(energy trong GT box vs background). Nếu không property nào tách rõ S/F → kill hướng trong 1
experiment. Nếu có → mới scan literature quanh property đó và thiết kế method.

**E5 — kết quả và bài học quan trọng hơn cả kết quả.** Script
`scripts/e5_success_vs_failure.py`, kết quả `results/e5_success_vs_failure.csv` (per-image)
và `results/e5_success_vs_failure_summary.csv`. 49 ảnh có "attack opportunity" trên
`dino_swin_l` (≥1 GT box detect đúng trên clean), 26 success / 23 failure (`success` = ≥1 box
evaded, cùng định nghĩa "clean-correct" mà ASR pooled dùng).

Effect size ban đầu (Cohen's d, binary S vs F) rất mạnh cho `n_gt` (số GT box): **d=1.184**
— ảnh success có trung bình 10.0 GT box vs 3.35 ở ảnh failure. Nhưng đây là **statistical
artifact**: định nghĩa "success" nhị phân (≥1 evaded, không chuẩn hóa theo số cơ hội) khiến
ảnh có nhiều GT box hơn tự nhiên có xác suất "≥1 evaded" cao hơn thuần túy vì có nhiều cơ hội
hơn (giống tung nhiều đồng xu) — không liên quan gì đến cơ chế transfer thật. Kiểm tra lại
bằng Pearson correlation với **evasion rate liên tục** (`n_evaded/n_clean_correct`, loại bỏ
đúng confound này): `n_gt` rơi từ d=1.18 xuống corr=+0.20 — gần như nhiễu. **Không property
nào trong 10 property đã test (object size/count, backbone cos_dist, noise L1/L2/saturation,
spectral band ratio, spatial concentration trong GT) vượt \|r\|=0.2** với evasion rate.

**Kết luận (2 finding tách bạch, không gộp)**:
1. Binary `≥1 evaded` bị confound mạnh bởi `n_gt` — **không dùng metric này để suy luận
   mechanism** trong bất kỳ diagnostic per-image tương lai nào.
2. Với continuous image-level evasion rate, mọi property global đã test đều rất yếu
   (\|r\|≤0.20) — **không có evidence rằng simple global image/perturbation property giải
   thích DINO transfer**, ở mức phân tích **per-image**.

**Đóng Phase N3 ở mức per-image simple-property** — chủ động không thêm property ngẫu nhiên
tiếp (tránh fishing). Bài học định hướng quan trọng hơn kết quả: **đơn vị phân tích có thể
sai level** — attack success của detector xảy ra ở cấp **object/detection**, không phải cấp
ảnh; gộp thành 1 con số evasion-rate/ảnh có thể che mất mechanism per-object. Việc này khớp
lại với E2b/E2c (pipeline failure đã quan sát được xảy ra ở cấp object/proposal, không phải
cấp ảnh).

**Hướng mở kế tiếp (chưa chạy)**: per-object diagnostic thay vì per-image — mỗi GT box
clean-correct là 1 sample nhị phân `y_j = evaded hay không`, so sánh: object area/scale,
class, clean confidence, clean box quality, perturbation energy trong object + vành đai lân
cận (vicinal ring), surrogate feature distortion tại vùng không gian tương ứng, overlap giữa
perturbation và object boundary/texture.

## 14. N4/N4b: Object-level diagnostic — finding dương đầu tiên

Sau chuỗi NO-GO liên tiếp (E1, E3b, MVC/RCG, E4/E4b, N2-B, N3 per-image), N4 đổi đúng đơn vị
phân tích xuống cấp **object** (khớp lại với E2b/E2c: pipeline failure quan sát được ở cấp
object/proposal, không phải cấp ảnh) — và cho ra **finding dương đầu tiên đủ mạnh và ổn định
để đáng thiết kế method**.

**N4 — per-object predictor của evasion trên DINO.** Script
`scripts/n4_object_level_diagnostic.py`, kết quả `results/n4_object_level_diagnostic.csv`,
`results/n4_object_level_summary.csv`. 251 object clean-correct (49 ảnh `dev_50`, noise OSFD
đã có, không craft lại), evaded=69 (27.5%, khớp ASR baseline). AUC (Mann-Whitney) + bootstrap
cluster theo ảnh (2000 draw, resample ảnh không phải object — tôn trọng correlation giữa các
object cùng ảnh):

| property | AUC | 95% CI | same-side frac |
|---|---|---|---|
| `clean_confidence` | 0.210 | [0.142, 0.272] | 1.000 |
| `iou_quality` | 0.270 | [0.194, 0.338] | 1.000 |
| `object_area` | 0.335 | [0.234, 0.424] | 1.000 |
| `pert_energy_ring` | 0.559 | [0.472, 0.670] | 0.882 |
| `surrogate_local_cos_dist` | 0.538 | [0.429, 0.623] | 0.788 |
| `pert_energy_object` | 0.517 | [0.442, 0.617] | 0.638 |

3 property đầu đạt GO criterion (AUC rõ lệch 0.5, CI không chạm 0.5, giữ dấu 100% qua
bootstrap); 3 property sau (đúng những property gắn trực tiếp với cơ chế OSFD — vicinal-ring
theo paper, local backbone distortion) **không** đạt ngưỡng ổn định.

**N4b — marginal response theo cường độ perturbation, tách theo difficulty.** Script
`scripts/n4b_strength_response.py`, kết quả `results/n4b_strength_response.csv`. Chia 251
object theo median-split `clean_confidence` (easy/hard, n≈125/126 mỗi nhóm), scale lại đúng
noise đã có (0.5×/0.75×/1.0×, không craft lại), đo evasion rate mỗi nhóm mỗi scale:

| scale | easy (low confidence) | hard (high confidence) |
|---|---|---|
| 0.5× | 14.4% | 0.0% |
| 0.75× | 29.6% | 1.6% |
| 1.0× | 46.4% | 8.7% |
| slope marginal (Δrate/Δscale, 0.5→1.0) | **+0.640** | **+0.175** |

Easy-group **chưa bão hòa** ở budget hiện tại (vẫn tăng dốc nhất ngay tại 1.0×, dốc gấp ~3.7×
hard-group); hard-group gần như trơ với việc tăng cường độ tuyến tính.

**Kết luận (viết đúng mức, tránh claim quá rộng)**:

> Object-level transfer to DINO is strongly associated with clean detection difficulty — low
> confidence, poor localization quality, and small size — while measured OSFD-local
> perturbation properties (energy in-object/in-ring, local backbone distortion) show weak
> predictive power. Under the current OSFD budget and DINO target, low-confidence objects
> exhibit substantially higher marginal evasion response to perturbation scaling than
> high-confidence objects.

Không claim "hard object miễn nhiễm" (chỉ 1 target/1 attack/N=49, chưa đủ để generalize) —
chỉ claim marginal-response chênh lệch rõ dưới đúng setup đã đo.

**Hypothesis method tiếp theo (đặt tên tạm: Difficulty-Aware Object Budgeting — DOB, chưa
code)**: phân bổ lại "chỗ" trong ràng buộc `‖δ‖∞ ≤ epsilon` — không đơn giản chia scalar budget
vì đây là ràng buộc theo từng pixel, cần định nghĩa spatial weighting/update rule cụ thể (vd
tăng alpha effective ở vùng easy-object, giảm ở vùng hard-object) sao cho vẫn giữ đúng
constraint gốc. Đây là hướng đầu tiên sau nhiều vòng NO-GO có đủ positive evidence (2 diagnostic
độc lập, nối logic với nhau) để đáng thiết kế pilot thật sự.

## 15. DOB v0/v1: Difficulty-Aware pilot — NO-GO

**DOB-v0 (step-size weighting) — implementation NO-GO / inconclusive, không phải falsify
hypothesis.** Thiết kế: `δ_{t+1} = clip(δ_t + alpha·W(x)⊙sign(g_mom), -eps, eps)`, W xây từ
surrogate clean detections (`clean_confidence` matched GT, giống N4), `w_j = clip(1 + sign·β·
(d_j-mean_d), 0.5, 1.5)`, overlap lấy max. Script `scripts/n5_dob_pilot.py`, kết quả
`results/n5_dob_pilot_summary.csv`. N=20, 100 step:

| model | OSFD | DOB-EASY | DOB-HARD | easy−osfd | easy−hard |
|---|---|---|---|---|---|
| faster_rcnn_r50 | 96.8 | 96.8 | 96.8 | +0.0 | +0.0 |
| yolox_l | 82.2 | 82.2 | 82.2 | +0.0 | +0.0 |
| mask_rcnn_swin_t | 79.4 | 79.4 | 79.4 | +0.0 | +0.0 |
| dino_swin_l | 29.0 | 31.0 | 30.0 | +2.0 | +1.0 |

Không đạt GO criterion (+2.0 << +5 trên DINO). Nhưng 3/4 model cho kết quả **y hệt tuyệt
đối** giữa cả 3 variant — kiểm tra `saturation fraction` cuối cùng của noise: osfd=0.858,
dob_easy=0.858, dob_hard=0.859 — **gần như giống hệt nhau**. Nguyên nhân: với
`epsilon=5, alpha=1, w_min=0.5`, pixel bị down-weight nhiều nhất vẫn chỉ cần tối đa
`5/(1×0.5)=10` step để bão hòa `±epsilon`; với 100 step tổng, ~86% pixel đã bão hòa từ rất
sớm bất kể W — W chỉ đổi **tốc độ tiến tới biên**, không đổi **đích đến**, nên khi đủ step để
hầu hết pixel bão hòa, DOB-v0 hội tụ về gần đúng noise giống hệt OSFD gốc. **Đây là lỗi thiết
kế thực thi (v0), không phải bằng chứng phủ định hypothesis difficulty-aware.**

**DOB-v1 (objective/loss weighting, loại bỏ confound saturation) — NO-GO sạch hơn.** Thay vì
weight step size, weight trực tiếp trong loss: `L_DOB(stage) = Σ_p W_l(p)·D_p(stage) /
Σ_p W_l(p)` (W resize theo từng stage bằng area-interpolation, D_p vẫn chính xác là OSFD
distortion). Update rule/epsilon/alpha/steps hoàn toàn không đổi so với OSFD gốc — chỉ
`osfd_loss` bị thay. Script `scripts/n5b_dob_v1_pilot.py`, kết quả
`results/n5b_dob_v1_pilot_summary.csv`. N=20, 100 step:

| model | OSFD | DOB-EASY-v1 | DOB-HARD-v1 | easy−osfd | easy−hard |
|---|---|---|---|---|---|
| faster_rcnn_r50 | 96.8 | 96.8 | 96.8 | +0.0 | +0.0 |
| yolox_l | 82.2 | 82.2 | 81.2 | +0.0 | +1.0 |
| dino_swin_l | 30.0 | 31.0 | 30.0 | +1.0 | +1.0 |
| mask_rcnn_swin_t | 80.4 | 79.4 | 79.4 | −1.0 | +0.0 |

Không đạt GO criterion (+1.0 << +5 trên DINO), yếu hơn cả v0 (+2.0), âm nhẹ trên
`mask_rcnn_swin_t`. Lần này **không còn confound kỹ thuật đã biết** để giải thích — v1 chỉ đổi
objective, giữ nguyên hoàn toàn update rule/epsilon/alpha/steps.

**Kết luận: đóng hẳn DOB (cả v0 và v1) — NO-GO.** N4/N4b vẫn là finding thật (object dễ evade
có marginal response cao hơn khi tăng cường độ *toàn cục*), nhưng việc chuyển hoá insight đó
thành ưu tiên không gian trong loss/update — theo cả 2 cách đã thử — không tạo gain thực tế.

## 16. Phase N6: Brainstorm có chủ đích từ evidence thực nghiệm + novelty scan

Sau khi DOB đóng, brainstorm lại **chỉ dựa trên finding đã có thật** (không đoán mù), rồi scan
literature 2023-2026 để lọc novelty. Không còn ưu tiên "thêm feature loss mới" chung chung.
3 hướng giữ lại:

**N6-A — RRB Gradient Conflict Resolution (ưu tiên #1).** Xuất phát từ 2 finding đã có:
`RRB ≫ no-RRB` và `K=3 averaging > K=1` (E3, RCG-AVG) đều mạnh trên mọi hard target, nhưng
RCG cho thấy agreement-*gating* không vượt naive averaging. Hypothesis mới: khi nhiều RRB view
cho gradient xung đột, **naive averaging đang cancel những direction có transfer value**, thay
vì "agreement thấp = bỏ" (RCG) nên là "conflict = reconcile/retain useful component". Literature
2026 có papers đúng niche (resolve gradient conflict trong multi-input-transformation attack,
báo cáo hoạt động cả CNN/Transformer — classification, chưa OD; AugTrans 2026, OD augmentation
mạnh nhất hiện có cho OD, không xử lý gradient-conflict). Pilot dự kiến: `OSFD vs RRB-AVG vs
RRB-ConflictResolved`, cùng K=3, cùng compute. Novelty medium-high, mechanistic fit rất cao
(bám đúng 2 finding dương duy nhất — RRB, K=3-AVG — chưa bị NO-GO), risk medium.

**N6-B — Integrated/Path-Averaged Gradient cho OSFD (ưu tiên #2).** MIG (ICCV 2023): dùng
integrated gradient (tích phân dọc path clean→adversarial) thay gradient tức thời, quan sát
IG ổn định hơn giữa CNN/ViT. Khớp với E3b (instantaneous gradient không giải thích synergy) và
RCG (local agreement không đủ) — gợi ý "gradient tại 1 điểm" quá model-specific, gradient tích
phân dọc trajectory có thể ổn định hơn. Khác RRB (sample transformation-space) và khác
flatness (không jitter quanh điểm hiện tại, mà tích phân có cấu trúc dọc path). Weakness: MIG
là mechanism đã established ở classification — nếu chỉ "MIG + OSFD" thì novelty chỉ nằm ở domain
application, cần cho thấy path-integrated gradient đặc biệt giải cross-architecture detector
gap thì mới thành contribution mạnh. Novelty medium, fit high, risk low-medium.

**N6-C — Cross-Layer Relational Feature Distortion (ưu tiên #3, cần diagnostic trước, chưa
code).** E1 cho thấy absolute feature distortion tại 1 layer/stride không giải thích tốt DINO;
STAT-NORM (N2-B) cũng NO-GO — "bỏ mean/std rồi phá activation" chưa tìm ra representation
invariant. Hướng khác: attack **relation giữa nhiều layer/scale** (`R_{l,m}=sim(F_l,F_m)`)
thay vì raw feature value — motivation từ MFDA (multi-layer feature transfer tốt hơn
single-layer), feature-distribution-attack literature (model-specificity tăng dần theo layer),
ATT/TGR cho ViT (interaction across layer quan trọng, xử lý từng layer độc lập dễ overfit
surrogate). Rủi ro cao nhất trong 3 (design space lớn, dễ biến thành "1 feature loss khác" nếu
không có diagnostic support) — **không code ngay**, cần 1 diagnostic rẻ trước: liệu cross-layer
relation distortion (clean→adv) có correlate với DINO object-level evasion tốt hơn raw local
cosine (đã NO-GO ở N4) hay không.

**Hướng loại khỏi shortlist**: augmentation/structure-frequency thêm (AugTrans 2026 đã phủ OD
augmentation mạnh, đã tự NO-GO ở E4/RCG-AVG); token-specific ViT attack (PNA/TGR/ATT cần
surrogate ViT, đổi surrogate sẽ đổi hẳn research question); feature-distance đơn thuần (OSFD
chính nó đã thuộc family này, literature đã rất dày — DFAA 2025 và nhiều attack khác).

**Quyết định**: đi **N6-A trước** — hướng đầu tiên sau nhiều vòng NO-GO vừa (1) bám đúng finding
dương thật của chính project (RRB, K=3-AVG), (2) không lặp lại cơ chế RCG đã NO-GO, (3) có
external evidence 2026, (4) chưa thấy OD work giải đúng bottleneck optimization này. Bước tiếp
theo: đặc tả conflict-resolution rule tối giản (control `AVG vs conflict-resolved`, không thêm
hyperparameter lớn) — chưa code.

**N6-A v0 pilot — NO-GO.** Script `scripts/n6a_gcr_pilot.py`, kết quả
`results/n6a_gcr_pilot_summary.csv`, log `runs/run_attack_osfd_n6a_*`. Mechanism: per pixel,
chiếu bỏ phần gradient của mỗi view (trong K=3 draw RRB/step) *xung đột* (dot<0) với
leave-one-out consensus của K-1 view còn lại (PCGrad-style), rồi lấy trung bình — so với
`rrb_avg_k3` (K=3 naive average, cùng budget) và `osfd_k1` (K=1, reference). `rrb_avg_k3`/
`rrb_cr_k3` crafted **lockstep** (chia sẻ RNG snapshot quanh mỗi K=3 draw để cùng tham số
augmentation ngẫu nhiên, loại bỏ confound "AVG gặp may hơn CR"). N=20, 100 step, ASR:

| model | group | osfd_k1 | rrb_avg_k3 | rrb_cr_k3 | CR−AVG |
|---|---|---|---|---|---|
| faster_rcnn_r50 (surrogate) | — | 100.0 | 100.0 | 100.0 | +0.0 |
| yolox_l | B | 84.2 | 86.1 | 85.1 | −1.0 |
| mask_rcnn_swin_t | C | 82.5 | 82.5 | 84.5 | +2.1 |
| dino_swin_l | C | 31.0 | 34.0 | 34.0 | +0.0 |

GO criterion đã pre-register (CR−AVG ≥+3 trên ≥2/3 của {yolox_l, mask_rcnn_swin_t,
dino_swin_l}, không target nào giảm >3 điểm) — đạt **0/3**.

Diagnostics giải thích cơ chế NO-GO (không phải "unclear", mà falsify có cơ sở):
`conflict_pixel_ratio=0.495` (gần một nửa pixel có ít nhất 1 view dot<0 với leave-one-out
consensus — conflict hình học tồn tại thật) nhưng `correction_ratio=0.00013` (phần gradient bị
chiếu bỏ nhỏ không đáng kể so với norm gốc) và `cos_cr_avg≈0.9999999998` (hướng gradient sau
resolve gần như trùng khít hướng trung bình thô). → Conflict theo định nghĩa `dot<0` xảy ra phổ
biến về mặt hình học, nhưng biên độ phần bị coi là "xung đột" quá nhỏ để việc chiếu bỏ nó đổi
được `sign(gradient)` — thứ duy nhất update rule (I-FGSM/MI) thực sự dùng. Đây là bằng chứng
khá vững cho việc bác bỏ đúng hypothesis N6-A ở dạng v0 này (per-pixel leave-one-out
sign-conflict projection), không phải do bug hay chưa đủ mạnh.

**Kết luận N6-A v0**: đóng, NO-GO. Không tiếp tục tune (không có hyperparameter lớn để tune —
`eta` chỉ là numerical guard, đúng như thiết kế ban đầu). **Quyết định: bỏ hẳn N6-A v1**
(sign-conflict) — lý do: conflict theo raw dot-product đã falsify khá sạch rằng conflict không
đủ biên độ để matter; đổi định nghĩa sang `sign(gradient)` có nguy cơ "ép" conflict trở nên quan
trọng bằng định nghĩa mới thay vì theo đúng bằng chứng hình học vừa đo được, và hướng
sign-disagreement còn quá gần cơ chế RCG (disagreement → xử lý update) đã NO-GO ở Phase M.
Chuyển hẳn sang **N6-B trước, N6-C sau** (xem §16-B).

### N6-B0: Path-integrated gradient diagnostic (rẻ, trước khi pilot)

Motivation: MIG (Ma et al., ICCV 2023) và MuMoDIG (AAAI 2025) báo cáo path/integrated gradient
ổn định hơn instantaneous gradient giữa CNN và ViT cho classification transfer. Điều này khớp
đúng 2 finding đã có của project theo hai cách khác nhau: E3b (cosine/norm gradient tức thời
không giải thích synergy RRB×k) và N6-A (local RRB conflict tồn tại nhiều về mặt hình học nhưng
biên độ correction vô nghĩa) — cả hai đều gợi ý **gradient tại một điểm không phải nơi chứa tín
hiệu transfer hữu ích; có thể là gradient tích hợp dọc theo một path mới là nơi đó**. Trước khi
code lại toàn bộ craft loop theo path-gradient (rủi ro lặp lại đúng bài học N6-A: code full pilot
rồi mới phát hiện operator gần như không đổi gì), chạy diagnostic rẻ trước, đúng kỷ luật đã dùng
suốt Phase M/N.

**Thiết kế** (script `scripts/n6b0_path_gradient_diagnostic.py`, không craft lại — tái dùng noise
`n6a_osfd_k1` đã có, tức OSFD chuẩn K=1/RRB-on/100-step/epsilon=5, làm `delta_t`):

- `g_local = ∇_δ L_OSFD(x+δ_t)` — gradient tức thời tại state đã crafted (không cần tính riêng,
  chính là điểm cuối `λ=1` của path).
- `g_path = mean_{m=1..M} ∇_δ L_OSFD(x+λ_m·δ_t)`, `λ_m=m/M`, `M=10` — right-Riemann-sum path
  average từ gần 0 đến 1, bao gồm cả điểm `λ=1` (=`g_local`).
- Mọi `λ_m` dùng chung 1 RNG snapshot cho `rrb_forward` (lockstep, giống n6a) — cô lập hiệu ứng
  scale-theo-λ khỏi randomness của augmentation.
- Đo 1 (rẻ): `cos(g_path, g_local)` + tỷ lệ đồng dấu — trả lời "path-averaging có thực sự tạo
  gradient khác biệt hay không" trước khi tốn gì thêm.
- Đo 2 (quan trọng hơn): one-step controlled probe — từ `x_t=clamp(x+δ_t,0,255)`, đi thêm đúng 1
  step `alpha·sign(g_local)` vs `alpha·sign(g_path)`, đo incremental evasion (dùng đúng
  `compute_asr_for_image`/định nghĩa ASR chuẩn của cả project) trên `yolox_l`/`mask_rcnn_swin_t`/
  `dino_swin_l` so với `x_t` một mình. **Sai khác có chủ đích so với budget attack thật**: probe
  step chỉ clamp về `[0,255]` pixel-range, KHÔNG re-clamp về epsilon-ball — vì N4b đã đo ~86%
  pixel của noise 100-step/epsilon=5 đã bão hòa ở biên; nếu re-clamp về đúng epsilon, local và
  path sẽ hội tụ về gần như cùng kết quả bất kể hướng nào tốt hơn (đúng confound đã hạ DOB-v0).

**Kết quả** (N=20, M=10, `dev_50`):

- `avg cos(g_path, g_local) = 0.699`, `avg sign_agree_ratio = 0.817` — **path-averaging tạo
  gradient khác biệt thật sự** (không phải no-op như correction của N6-A, vốn có cos≈0.9999999998).

| model | group | ASR(x_t) | ASR(local) | ASR(path) | incr_local | incr_path | path−local |
|---|---|---|---|---|---|---|---|
| faster_rcnn_r50 (surrogate) | — | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 | +0.0 |
| yolox_l | B | 84.2 | 90.1 | 90.1 | +5.9 | +5.9 | +0.0 |
| mask_rcnn_swin_t | C | 82.5 | 84.5 | 87.6 | +2.1 | +5.2 | **+3.1** |
| dino_swin_l | C | 31.0 | 35.0 | 39.0 | +4.0 | +8.0 | **+4.0** |

**Đọc kết quả**: surrogate đã bão hòa ASR=100% nên không có "chỗ" để phân biệt (kỳ vọng, không
phải vấn đề). Trên `yolox_l`, local và path đạt đúng cùng mức incremental (+5.9) — hai hướng có
vẻ đã chạm cùng một "trần" easy-to-evade ở model này, không phải path kém hơn. Trên **2/3 hard
target** — đúng `mask_rcnn_swin_t` và `dino_swin_l`, hai model nhóm C mà project đang nhắm tới —
path direction cho incremental evasion cao hơn rõ rệt so với local (+3.1 và +4.0 điểm), và mức
tăng tuyệt đối lớn nhất rơi đúng vào `dino_swin_l` — target khó transfer nhất trong toàn bộ
project.

**Verdict: GO signal cho N6-B** (không phải NO-GO như N6-A/DOB/RCG/MVC/STAT-NORM). Cả hai điều
kiện đã đặt trước khi chạy đều đạt: (1) `cos(g_path,g_local)` khác 1 rõ rệt (0.699, không phải
≈1), xác nhận path-averaging tạo ra thứ mới thật sự chứ không phải no-op; (2) path direction cho
incremental evasion cao hơn local trên ≥2/3 hard target ({yolox_l, mask_rcnn_swin_t, dino_swin_l})
— đạt 2/3, target còn lại (yolox_l) hòa chứ không âm. Đây là finding dương thứ hai của project sau
N4/N4b, và là finding dương **đầu tiên** trực tiếp support một phương hướng craft-loop mới (khác
N4/N4b — vốn dẫn tới DOB, đã NO-GO).

**Bước tiếp theo (chưa làm)**: thiết kế N6-B pilot thật — path-integrated gradient OSFD attack
đầy đủ 100-step, so với OSFD chuẩn cùng budget, N=20 rồi N=300 nếu dương. Cần quyết định thêm:
cách tích hợp path-averaging vào MỖI step của vòng lặp 100-step (M=10 forward/backward mỗi step
sẽ đắt hơn hẳn — cần cân nhắc M nhỏ hơn hoặc amortize), và path nên tính dọc theo state nào (từ
`0` đến state-tại-step-hiện-tại, hay một window ngắn hơn) — chưa chốt.

### N6-B v0 pilot: full path-averaged OSFD attack — **GO**

Quyết định trước khi code (không sweep sau khi thấy kết quả): path tính từ **clean → current
state** (không dùng local window quanh `x+δ_t` — window sẽ đổi hẳn hypothesis sang
"local-neighborhood smoothing", chồng lấn VMI-FGSM/flatness, thứ project đã chủ động parked ở
N2-C), **M=3 pre-registered, không sweep nếu fail**. Script `scripts/n6b_path_pilot.py`, kết quả
`results/n6b_path_pilot_summary.csv`, log `runs/run_attack_osfd_n6b_*`.

Mechanism mỗi step `t` (giữ nguyên epsilon/alpha/mu/k/RRB so với OSFD chuẩn, chỉ đổi cách tính
gradient ascent):

```
λ_m = m/M,  m=1..M=3
g_m       = ∇_δ L_OSFD(x + λ_m·δ_t)
g_path,t  = mean_m g_m
m_t       = μ·m_{t-1} + g_path,t / mean|g_path,t|
δ_{t+1}   = Π_ε(δ_t + α·sign(m_t))
```

`osfd_local` dùng đúng update rule này với chỉ term `m=M` (λ=1) — chính là OSFD chuẩn đã dùng
xuyên suốt project. Hai trajectory crafted **lockstep** (share RNG snapshot mỗi step, giống
n6a_gcr_pilot.py) để loại bỏ confound "path gặp may hơn local". Diagnostic `cos(g_path, g_m=M)` +
sign-disagree lấy "miễn phí" từ chính M draw của path (term `m=M` đã là gradient tức thời tại
đúng điểm đó, không cần forward/backward thêm).

**Kết quả** (N=20, 100 step, ASR %):

| model | group | osfd_local | path_m3 | path−local |
|---|---|---|---|---|
| faster_rcnn_r50 (surrogate) | — | 100.0 | 98.9 | −1.1 |
| yolox_l | B | 86.1 | 87.1 | +1.0 |
| mask_rcnn_swin_t | C | 82.5 | 85.6 | **+3.1** |
| dino_swin_l | C | 31.0 | 42.0 | **+11.0** |

GO criterion đã pre-register (path−local ≥+3 trên ≥2/3 của {yolox_l, mask_rcnn_swin_t,
dino_swin_l}, không target nào giảm >3 điểm): **đạt 2/3** (mask_rcnn_swin_t +3.1, dino_swin_l
+11.0), `yolox_l` dương nhẹ (+1.0) chứ không âm, surrogate chỉ giảm nhẹ (−1.1, trong biên).
`dino_swin_l` +11.0 vượt xa ngưỡng "+5 đáng chú ý" đã đặt trước — đây là mức tăng ASR lớn nhất
trên `dino_swin_l` trong toàn bộ project tính đến hiện tại.

Diagnostic theo step (mean over 20 ảnh) trả lời đúng câu hỏi "effect có chỉ ở early-stage không":

| step | cos(g_path, g_local-tại-cùng-điểm) | sign_disagree |
|---|---|---|
| 1 | 0.983 | 0.049 |
| 25 | 0.834 | 0.134 |
| 50 | 0.811 | 0.140 |
| 75 | 0.802 | 0.143 |
| 100 | 0.798 | 0.144 |

Cosine giảm dần rồi ổn định quanh **~0.80** từ step 25 trở đi, KHÔNG trôi ngược về 1 —
path-averaging tiếp tục tạo gradient khác biệt **xuyên suốt cả trajectory**, không phải hiệu ứng
chỉ mạnh lúc đầu rồi biến mất. Đây là bằng chứng cho một mechanism thật, bền vững theo thời gian,
không phải nhiễu early-step.

**Verdict: GO.** Đây là finding dương mạnh nhất của project tính đến hiện tại — vượt rõ mọi
candidate trước đó (MVC/RCG/STAT-NORM/DOB đều NO-GO; N6-A NO-GO; ngay cả RCG-AVG, kết quả dương
nhất trước đây, chỉ đạt +11.0 trên DINO nhờ mở rộng K=3-draw/step chứ không phải mechanism mới —
còn ở đây gain đến từ path-averaging thật, một trục hoàn toàn khác K-draw). Path-integrated OSFD
gradient (M=3, clean→current) là **candidate mạnh nhất hiện tại của project**.

**Bước tiếp theo (chưa làm)**: xác nhận N=20 không phải may mắn do sample nhỏ — chạy lại trên
`dev_300` (N lớn hơn, có thể cần giảm cost bằng cách không log diagnostic mỗi step); nếu vẫn giữ
gain, xác nhận held-out trên `val_100` theo đúng quy trình đã định ở §17. Cân nhắc thêm: pilot
non-hard targets (fcos_r50, deformable_detr, yolov3_d53 — nhóm A/B còn thiếu) để xác nhận không
đánh đổi hiệu quả ở nhóm dễ transfer.

### N6-B v0 xác nhận trên `dev_300` — **CONFIRMED CANDIDATE**

Chạy lại đúng nguyên v0 (không đổi hyperparameter nào — cùng `osfd_local` vs `path_m3`, 100
step, cùng epsilon/alpha/mu/RRB, M=3, path clean→current), N=300 (`data/manifests/dev_300.json`),
mở rộng eval full 7 model (thêm `fcos_r50`, `deformable_detr`, `yolov3_d53` so với pilot N=20),
cộng **paired image-cluster bootstrap 95% CI** (2000 draw, resample ảnh — cùng convention với N4)
cho `ASR(path_m3) − ASR(osfd_local)`. Cùng script `scripts/n6b_path_pilot.py`, log
`runs/run_attack_osfd_n6b_*_dev_300_*`.

Tiêu chí xác nhận đã pre-register trước khi chạy: DINO gain vẫn ≥+5 ASR; Mask-Swin vẫn ≥+3 hoặc
ít nhất dương ổn định; không model nào giảm >3; surrogate drop nhỏ chấp nhận được; và — điều
kiện quyết định coi là "confirmed" thay vì "chỉ pilot" — CI của DINO không được cắt 0.

**Kết quả (N=300, ASR %, kèm 95% bootstrap CI cho delta)**:

| model | group | osfd_local | path_m3 | path−local | 95% CI | same_side_frac |
|---|---|---|---|---|---|---|
| faster_rcnn_r50 (surrogate) | — | 99.2 | 98.8 | −0.3 | [−0.8, +0.1] | 0.92 |
| deformable_detr | A | 98.5 | 98.0 | −0.5 | [−1.1, +0.0] | 0.96 |
| fcos_r50 | A | 98.1 | 97.7 | −0.3 | [−1.0, +0.4] | 0.79 |
| yolov3_d53 | B | 83.9 | 84.6 | +0.7 | [−0.3, +1.7] | 0.93 |
| yolox_l | B | 69.2 | 70.9 | +1.7 | [+0.4, +3.1] | 0.994 |
| mask_rcnn_swin_t | C | 67.8 | 71.1 | **+3.2** | **[+1.9, +4.6]** | 1.000 |
| dino_swin_l | C | 33.0 | 39.1 | **+6.2** | **[+4.6, +7.8]** | 1.000 |

**Mọi tiêu chí đã pre-register đều đạt**: DINO +6.2 (≥+5), CI không cắt 0; Mask-Swin +3.2 (≥+3),
CI không cắt 0; không model nào giảm quá −0.5 (<<3); surrogate chỉ −0.3. Nhóm A (`fcos_r50`,
`deformable_detr`) gần như flat, CI cắt 0 ở cả hai — nhất quán với cách đọc "đã sát ceiling ASR
(~98%) từ OSFD chuẩn, không còn chỗ để cải thiện", và quan trọng hơn: **không có đánh đổi** (mức
giảm nhỏ, trong biên chấp nhận được). `yolov3_d53` dương nhẹ nhưng CI cắt 0 (không ý nghĩa thống
kê rõ ràng); `yolox_l` dương và CI không cắt 0 dù dưới ngưỡng +3 GO ban đầu.

Diagnostic `cos(g_path, g_local)` theo step tái lập gần như y hệt N=20 (step 100: 0.795 vs 0.798
trước đó; `sign_disagree`: 0.145 vs 0.144) — xác nhận effect không phải artifact của N nhỏ, và
mechanism ổn định qua cỡ mẫu khác nhau.

**Verdict: CONFIRMED CANDIDATE** (đúng lời đã chốt trước khi chạy: "Nếu DINO vẫn +5 trở lên và CI
không cắt 0, lúc đó mình sẽ xem N6-B là confirmed candidate, không còn chỉ pilot signal"). Đây là
lần đầu tiên trong project một candidate vượt qua cả hai cửa N=20 pilot và N=300 confirm với CI
thống kê vững, trên đúng 2 hard target nhóm C mà toàn bộ project nhắm tới.

**Ghi chú về novelty** (đã thảo luận, không đổi bởi kết quả confirm): effect size xác nhận không
làm tăng novelty của cơ chế path-gradient (MIG/MuMoDIG đã có cho classification); contribution
đáng bảo vệ là **path-average trong đúng feature-distortion loss của OD, đóng góp cụ thể vào gap
CNN→Transformer trong object detection** — cần đối chiếu kỹ với paper 2026 dùng MIG-initialization
cho OD (đã tìm thấy lúc scan literature N6) trước khi viết claim contribution, để chắc chắn không
trùng.

**Bước tiếp theo**: xác nhận held-out trên `val_100`, **không đổi hyperparameter nào** giữa
`dev_300` và `val_100` (M=3, budget, mọi thứ giữ nguyên) — theo đúng quy trình đã định ở §17.

### N6-B v0 xác nhận trên `val_100` (held-out thật) — **KẾT QUẢ HỖN HỢP (mixed), không phải
full-confirm**

Chạy đúng nguyên v0 (M=3, clean→current, cùng update rule, không đổi hyperparameter nào so với
`dev_300`), N=98/100 (2 ảnh skip do 0 valid GT, giống hành vi thường thấy), full 7 model, cùng
paired bootstrap 95% CI. Script/log giống hệt cách chạy `dev_300` (`scripts/n6b_path_pilot.py
--manifest data/manifests/val_100.json --n-images 100`).

**Lưu ý kỹ thuật xảy ra giữa chừng (không phải confound của kết quả)**: lần chạy đầu crash ở pha
eval với `KeyError` — bug có sẵn trong `scripts/evaluate.py` (2 chỗ gọi
`to_identity_coco_results()` không lọc dict cache dự đoán theo đúng `image_ids`/`common_ids` của
run hiện tại trước khi build COCO results; cache dự đoán dùng chung thư mục
`results/_n6b_predictions` giữa `dev_50`/`dev_300`/`val_100` nên lẫn ID ảnh không có trong
`scale_by_image_id`). Bug lộ ra lần đầu đúng lúc này vì `val_100` là split đầu tiên hoàn toàn
disjoint với `dev_300`/`dev_50` (đúng thiết kế held-out). Đã sửa 2 điểm filter dict tại
[scripts/evaluate.py:177,199], verify bằng smoke test N=3 trước khi chạy lại full — không đổi
logic craft/eval nào khác, không phải một xác nhận "được nới lỏng" để pass.

**Kết quả (N=98, ASR %, so với `dev_300` N=296)**:

| model | group | osfd_local | path_m3 | Δ val_100 | 95% CI val_100 | Δ dev_300 | 95% CI dev_300 |
|---|---|---|---|---|---|---|---|
| faster_rcnn_r50 (surrogate) | — | 98.7 | 97.6 | −1.1 | [−2.3, 0.0] | −0.3 | [−0.8, +0.1] |
| deformable_detr | A | 98.6 | 97.3 | **−1.2** | **[−2.5, −0.2]** | −0.5 | [−1.1, 0.0] |
| fcos_r50 | A | 97.0 | 96.6 | −0.3 | [−1.1, 0.0] | −0.3 | [−1.0, +0.4] |
| yolov3_d53 | B | 81.9 | 83.3 | +1.4 | [−0.9, +3.6] | +0.7 | [−0.3, +1.7] |
| yolox_l | B | 66.2 | 68.1 | +1.8 | [−0.4, +4.0] | +1.7 | [+0.4, +3.1] |
| mask_rcnn_swin_t | C | 65.7 | 67.4 | **+1.7** | **[−0.4, +3.6]** | +3.2 | [+1.9, +4.6] |
| dino_swin_l | C | 32.7 | 39.2 | **+6.5** | **[+2.9, +10.3]** | +6.2 | [+4.6, +7.8] |

**Đọc kết quả, tách rõ 2 claim thay vì gộp chung "GO/NO-GO"**:

1. **DINO — CONFIRMED, mạnh hơn cả `dev_300`**: +6.5 ASR (≥+5), CI [+2.9,+10.3] không cắt 0, còn
   cao hơn điểm ước lượng của `dev_300` (+6.2). Đây là claim chính của N6-B và **giữ vững hoàn
   toàn** trên held-out thật.
2. **Mask-Swin-T — KHÔNG đạt tiêu chí đã dùng cho `dev_300`**: point estimate vẫn dương (+1.7,
   same_side_frac=0.959 — 95.9% bootstrap draw vẫn dương) nhưng **CI cắt 0** lần đầu tiên
   ([−0.4,+3.6]), khác hẳn `dev_300` (+3.2, CI [+1.9,+4.6] không cắt 0). Một phần do N nhỏ hơn
   (98 vs 296, CI theo lý thuyết rộng hơn ~√3≈1.75× nếu variance không đổi — độ rộng CI thực tế
   val_100/dev_300 ≈ 4.0/2.7 ≈ 1.48×, cùng bậc độ lớn) — **không kết luận được** "hiệu ứng biến
   mất" hay "hiệu ứng vẫn còn nhưng underpowered ở N=98"; đây là kết quả **inconclusive**, không
   phải falsify.
3. **Pattern "DINO gain > Mask gain > CNN gain" (câu hỏi đặt ra trước khi chạy) — KHÔNG giữ
   được rõ ràng trên `val_100`**: point estimate Mask-Swin (+1.7) thực ra **thấp hơn** cả
   `yolox_l` (+1.8, nhóm B/CNN) và gần bằng `yolov3_d53` (+1.4) — trên split này, gain của
   Mask-Swin không tách biệt được khỏi gain của nhóm CNN B về mặt thống kê (CI của cả 3 đều cắt
   0). Ở `dev_300`, Mask-Swin tách biệt rõ khỏi nhóm B (CI không cắt 0, point estimate cao hơn
   hẳn yolox_l +1.7). Trên `val_100` ranh giới này mờ đi.
4. **`deformable_detr` (nhóm A) — tín hiệu âm nhẹ nhưng CI không cắt 0** (−1.2, CI
   [−2.5,−0.2]), rõ hơn `dev_300` (−0.5, CI chạm 0). Biên độ rất nhỏ (~1 điểm ASR trên nền
   ASR~97-98%, gần ceiling) — cách đọc hợp lý nhất là hiệu ứng ceiling/variance ở vùng ASR gần
   bão hòa (surrogate cũng cùng dấu −1.1), không phải bằng chứng cho một trade-off cơ chế thật,
   nhưng đáng ghi lại vì đây là CI đầu tiên trong toàn bộ N6-B không cắt 0 theo hướng bất lợi.

**Kết luận (viết đúng mức, không spin)**: theo đúng tiêu chí held-out đã đặt trước khi chạy
("DINO gain lớn > Mask gain > CNN gain nhỏ", "Mask dương rõ" để coi effectiveness "gần như
khóa") — **DINO đạt, Mask KHÔNG đạt rõ ràng, pattern thứ tự DINO>Mask>CNN không giữ được sạch**.
N6-B **không nên coi là "effectiveness đã khóa toàn diện"** ở bước này. Điều có thể khóa: **DINO
gain là robust, tái lập qua 3 lần đo độc lập (pilot N=20, confirm N=300, held-out N=98) với CI
không cắt 0 cả 3 lần** — đây là phần contribution chắc chắn nhất. Điều chưa thể khóa: liệu N6-B
có thật sự cải thiện **cả 2** hard target nhóm C hay chỉ 1 (DINO), với Mask-Swin cần thêm bằng
chứng (N lớn hơn, hoặc chấp nhận claim yếu hơn "positive trend, not yet significant at N=98") để
kết luận.

## 17. Trạng thái / bước tiếp theo

- [x] Environment + checkpoint + COCO val2017 + manifest đã setup xong (`setup_env.sh`).
- [x] Run baseline: `osfd` vs `mi_fgsm`, `dev_50`, 100 step — xác nhận OSFD transfer tốt
      hơn MI-FGSM ở mọi nhóm, nhưng **gap thu hẹp mạnh ở nhóm C** (Swin Transformer).
- [x] **Chuỗi diagnostic E1→E3 hoàn tất** (xem §8 ở trên). Tóm tắt: gap nhóm C không phải
      do backbone-feature-mismatch (E1); một phần do RPN proposal-selection instability
      nhưng chủ yếu do RoI classification collapse ở `mask_rcnn_swin_t` (E2/E2b/E2c); và
      phần lớn transfer gain của OSFD đến từ RRB, `k=3` chỉ phát huy qua interaction với
      RRB, không có standalone benefit đáng kể (E3).
- [ ] (mở, chưa quyết định) Gradient/feature-displacement diagnostic để giải thích *cơ chế*
      của synergy RRB×k vừa phát hiện ở E3 — xem câu hỏi mở cuối §8.
- [x] **Phase M — Method Discovery, đã đóng** (xem §9-10): 3 candidate (CRA/MVC/TCR→RCG)
      đều bị loại hoặc NO-GO — CRA (overlap SAA quá cao, không pilot), MVC (model-space
      consistency, NO-GO cả λ=100 và λ=10), RCG (gradient-agreement gate, NO-GO — không vượt
      naive multi-view averaging `rcg_avg`, dù `rcg_avg` tự nó cho gain dương rõ +6 đến +11
      điểm ASR mọi model — nhưng chỉ là mở rộng của E3, không phải mechanism mới). Finding
      giữ lại: "consistency/agreement" và "thêm stochastic view" bị loại khỏi trục tìm kiếm.
- [x] **Phase N1 — Spectral vulnerability diagnostic (E4/E4b), đã đóng** (xem §11): decompose
      noise OSFD đã crafted thành radial frequency band, inject riêng từng band không craft
      lại. Raw: low band (61% energy) giữ tỷ lệ ASR cao hơn hẳn tỷ lệ energy ở mọi model.
      Nhưng sau L∞-normalize (E4b, tách frequency-effect khỏi amplitude-effect): mọi band —
      kể cả low — sụt hiệu quả mạnh (còn 16-34%), và không có signature riêng cho
      cross-architecture transfer. **NO-GO cho "spectral decomposition of existing noise"**;
      "spectral-*constrained* optimization" (khác hẳn về mặt kỹ thuật) để **OPEN/parked**,
      chưa bị falsify, không ưu tiên trừ khi có thêm lý do mechanistic mạnh hơn.
- [x] **Phase N2 — STAT-NORM & Robust Surrogate, đã đóng** (xem §12): N2-B (per-channel
      spatial standardization trước `osfd_loss`) — NO-GO sạch, pattern ngược hẳn hypothesis
      (hard target giảm nhiều hơn surrogate). N2-A (robust ImageNet surrogate, Salman et al.
      2020) — collapse toàn diện, NO-GO thực dụng/inconclusive về mechanism (confound budget
      chưa loại trừ, không theo đuổi thêm). N2-C (flatness) — vẫn parked, ưu tiên thấp nhất.
- [x] **Phase N3 — Success-vs-Failure per-image trên DINO, đã đóng** (xem §13): script
      `scripts/e5_success_vs_failure.py`. Effect size ban đầu mạnh (n_gt, Cohen's d=1.18) hoá
      ra là statistical artifact của định nghĩa "success" nhị phân — sau khi sửa bằng evasion
      rate liên tục, **mọi property global đã test đều rất yếu (\|r\|≤0.20)**. NO-GO ở mức
      per-image; bài học giữ lại: đơn vị phân tích có thể sai level — nên chuyển sang
      **per-object diagnostic** (mỗi GT box clean-correct là 1 sample nhị phân evaded/not),
      khớp lại với E2b/E2c (pipeline failure đã quan sát ở cấp object/proposal).
- [x] **N4/N4b — Object-level diagnostic, finding dương đầu tiên** (xem §14): `clean_confidence`/
      `iou_quality`/`object_area` dự đoán evasion trên DINO tốt và ổn định (AUC lệch rõ 0.5,
      giữ dấu 100% qua bootstrap cluster-theo-ảnh); property gắn với cơ chế OSFD (perturbation
      energy, local backbone distortion) thì không. N4b: object dễ evade (low confidence) có
      marginal response với cường độ perturbation cao hơn hẳn (~3.7×) object khó evade — chưa
      bão hòa ở budget hiện tại. → hypothesis method **Difficulty-Aware Object Budgeting
      (DOB)**, chưa code.
- [x] **DOB v0 + v1 — pilot đầy đủ, đóng NO-GO** (xem §14-15): v0 (step-size weighting)
      implementation NO-GO do confound saturation (~86% pixel bão hòa bất kể W, do đủ 100
      step); v1 (objective/loss weighting, loại bỏ confound) vẫn NO-GO, yếu hơn cả v0
      (+1.0 điểm ASR trên DINO, cần +5). Đóng hẳn hypothesis "spatial difficulty weighting
      trong loss/update" — không tiếp tục tune thêm.
- [x] **Phase N6 — brainstorm có chủ đích + novelty scan, đã shortlist** (xem §16): 3 hướng
      giữ lại từ evidence thực nghiệm thật (không đoán mù) — N6-A (RRB gradient conflict
      resolution, ưu tiên #1, bám đúng finding RRB≫no-RRB và K=3-AVG>K=1 chưa bị NO-GO),
      N6-B (integrated/path gradient cho OSFD, ưu tiên #2), N6-C (cross-layer relational
      feature distortion, ưu tiên #3, cần diagnostic trước khi code).
- [x] **N6-A v0 pilot — đã chạy, NO-GO, đóng hẳn (không làm v1)** (xem §16): per-pixel
      leave-one-out gradient-conflict projection (PCGrad-style) trên K=3 RRB view không vượt
      naive averaging (0/3 hard target đạt ngưỡng +3). Diagnostics cho thấy lý do: conflict hình
      học phổ biến (`conflict_pixel_ratio=0.495`) nhưng biên độ phần bị chiếu bỏ quá nhỏ
      (`correction_ratio=0.00013`, `cos_cr_avg≈1`) để đổi được `sign(gradient)` mà update rule
      thực sự dùng. Quyết định bỏ v1 (sign-conflict): tránh "ép" conflict quan trọng bằng định
      nghĩa mới trái ngược bằng chứng hình học, và tránh lặp lại cơ chế RCG (disagreement→xử lý
      update) đã NO-GO.
- [x] **N6-B0 path-gradient diagnostic — đã chạy, GO signal** (xem §16): path-integrated OSFD
      gradient (M=10, right-Riemann trên noise `n6a_osfd_k1` có sẵn, không craft lại) khác biệt
      thật với gradient tức thời (`cos=0.699`, không phải no-op), và one-step probe theo hướng
      path cho incremental evasion cao hơn local trên 2/3 hard target (`mask_rcnn_swin_t` +3.1,
      `dino_swin_l` +4.0; `yolox_l` hòa +0.0). Finding dương thứ hai của project, đầu tiên hỗ trợ
      trực tiếp một craft-loop mới.
- [x] **N6-B v0 pilot — đã chạy, GO** (xem §16): path-averaged OSFD gradient (M=3 pre-registered,
      clean→current, lockstep pairing với osfd_local) đạt GO criterion (path−local ≥+3 trên 2/3
      hard target: `mask_rcnn_swin_t` +3.1, `dino_swin_l` +11.0; `yolox_l` +1.0 không âm; không
      target nào giảm >3 điểm). Diagnostic `cos(g_path,g_local)` ổn định ~0.80 xuyên suốt 100
      step (không trôi về 1) — mechanism bền vững, không phải hiệu ứng early-stage. **Candidate
      mạnh nhất của project tính đến hiện tại.**
- [x] **N6-B v0 xác nhận trên `dev_300` — CONFIRMED CANDIDATE** (xem §16): N=300, full 7 model,
      paired bootstrap CI. DINO +6.2 (≥+5, CI=[+4.6,+7.8] không cắt 0), Mask-Swin +3.2
      (CI=[+1.9,+4.6] không cắt 0), không model nào giảm >0.5, nhóm A/B không bị đánh đổi.
      Mechanism tái lập ổn định qua cỡ mẫu (cosine step-100 ~0.795 cả N=20 lẫn N=300).
- [x] **Xác nhận held-out trên `val_100` — KẾT QUẢ HỖN HỢP, không phải full-confirm** (xem
      §16-B "N6-B v0 xác nhận trên `val_100`"): DINO +6.5 ASR, CI [+2.9,+10.3] không cắt 0 —
      **CONFIRMED, robust qua 3 lần đo độc lập** (pilot N=20, dev_300 N=300, held-out N=98).
      Mask-Swin +1.7, CI [−0.4,+3.6] **cắt 0** (khác `dev_300`'s +3.2 CI không cắt 0) —
      inconclusive ở N=98, không tách biệt rõ khỏi gain nhóm CNN-B trên split này. Pattern
      "DINO>Mask>CNN" đặt ra trước khi chạy **không giữ sạch**. `deformable_detr` có CI âm nhẹ
      không cắt 0 lần đầu tiên (−1.2, [−2.5,−0.2]), biên độ nhỏ, đọc như ceiling-effect.
      **Không nên tuyên bố "effectiveness đã khóa toàn diện"** — chỉ DINO gain là chắc chắn.
- [ ] N6-C (cross-layer relational feature distortion) hạ xuống phương án dự phòng — chỉ quay
      lại nếu N6-B không giữ được gain ở `dev_300`/`val_100`.
- [ ] Sau khi có ứng viên tốt nhất: chạy full 200 step trên `dev_300` để confirm không
      phải nhiễu do sample size nhỏ.
- [ ] Xác nhận held-out trên `val_100` cho phương pháp thắng cuộc (chỉ làm khi
      hyperparameter đã chốt từ kết quả `dev_300` — không iterate dựa vào split này).
- ~~Ablation RRB của OSFD gốc (Table 2 paper: bỏ rotation/resizing/blur riêng lẻ)~~ — phần
  lớn câu hỏi này đã được trả lời chính xác hơn bởi E3 (factorial k×RRB thay vì chỉ bật/tắt
  RRB toàn bộ); tách riêng rotation vs resizing vẫn là việc còn lại nếu cần độ chi tiết cao
  hơn.
- ~~Sensitivity của `k`~~ — đã trả lời bởi E3: `k=3` gần như vô nghĩa nếu không có RRB.

## 18. Prior-art verification cho N6-B: HIFA (2025) & IJCNN (2024)

Sau khi N6-B v0 confirmed trên `dev_300` (§16), bước tiếp theo trong roadmap "prove" là verify
prior-art trước khi tiếp tục đầu tư vào mechanism-proof/novelty-control, theo đúng 2 paper được
chỉ định check: **IEEE IJCNN 2024** (`ieeexplore.ieee.org/document/10651486`) và **HIFA, Journal
of Supercomputing 2025** (`doi.org/10.1007/s11227-025-07225-7`).

**Giới hạn đã biết trước khi đọc**: cả hai bài đều closed-access (IEEE trả HTTP 418 khi fetch;
Springer yêu cầu institutional login qua `idp.springer.com`) và không có bản open-access/preprint
tìm được qua search. Không có full-text — chỉ có title/authors/abstract/TLDR qua Semantic Scholar.
Quyết định (đã thống nhất): không theo đuổi route ngoài luồng để lấy full-text; dùng đúng phần
metadata sẵn có, ghi rõ câu nào trong 4 câu hỏi kỹ thuật đã đặt ra **không trả lời được** do
paywall, để không lẫn "đã verify" với "chưa verify".

### 18-A. IJCNN 2024 — Wei, Gao, Quan, Luo, *"A Transferable Adversarial Attack against Object
Detection Networks"* — **full-text đã đọc (user cung cấp PDF), CLOSED, không phải chỉ abstract**

Physical adversarial **patch** dán lên nắp ca-pô xe (digital + in ra dán vật lý), tối ưu bằng
Adam optimizer thường (lr=0.03, 500 epoch) trên loss tổng hợp
`L_total = α·L_cls + β·L_IoU + λ·L_NPS + γ·L_TV` (Eq. 10) — `L_cls` (classification-probability,
Eq. 4) và `L_IoU` (Eq. 3) là 2 core loss tấn công output/task, `L_NPS`/`L_TV` chỉ để patch in
được/mượt màu, không liên quan cơ chế tấn công. Bổ sung robustness qua perspective/affine
transform + light/blur augmentation của chính patch (không phải augmentation của toàn ảnh kiểu
RRB). Surrogate/target chính: YOLOv3 (pretrained COCO); transferability test sang YOLOv2/YOLOv4/
YOLOv7 (physical) — toàn bộ đều là **họ YOLO/CNN**, không có DETR/Swin/ViT nào. Domain hẹp hơn
hẳn OSFD/N6-B: chỉ 3 category (car/dog/fire-hydrant), không phải toàn bộ 80 class COCO, và chỉ
tấn công 1 object/ảnh (object có patch), không phải toàn ảnh dưới ràng buộc L∞.

| câu hỏi đã đặt trước | trả lời (full-text, đã confirm) |
|---|---|
| IG dùng cho attribution hay update direction? | **Không dùng ở đâu cả** — không có bất kỳ đề cập integrated-gradient/path-gradient nào trong toàn bài; optimizer là Adam chuẩn trên patch pixel, không có khái niệm "path" nào trong update rule |
| Path recompute mỗi iteration hay cố định? | N/A — không có path-gradient mechanism trong bài |
| Path baseline→input hay clean→current? | N/A — không áp dụng |
| Objective: task loss hay feature-distortion? | **Task loss xác nhận 100%** (Eq. 3/4/10: IoU + classification-probability), không chạm intermediate feature nào |
| Có test CNN→Transformer/Swin không? | **Không, xác nhận** — YOLOv2/v3/v4/v7 only, toàn CNN one-stage |

**Đánh giá (đã chốt, không còn "sơ bộ")**: **zero overlap cơ chế** với N6-B — khác hoàn toàn về
attack surface (patch cục bộ 1 object vs L∞ toàn ảnh), objective (task loss vs feature-distortion),
optimization (Adam trên patch vs I-FGSM/MI-momentum + path-integrated gradient trên toàn bộ
`δ`), và scope test (CNN-only YOLO family vs CNN→Transformer). Bài này **không phải prior-art
cần lo cho N6-B** — có thể đóng hẳn câu hỏi này, không cần theo dõi thêm.

### 18-B. HIFA — Ding, Sun, Mao, Dai, Ding (2025), *"Improving the transferability of adversarial
examples via the high-level interpretable features for object detection"*

TLDR (Semantic Scholar): *"A High-level Interpretable Features Attack method that can effectively
attack various object detection models... significantly enhancing the cross-model transferability
of adversarial examples."* Search snippet bổ sung: mục đích đề cập rõ là **"address the issue of
gradient saturation during backpropagation in existing methods"** qua **feature-level attack**
(interfering với intermediate features model).

| câu hỏi đã đặt trước | trả lời được từ abstract/TLDR? |
|---|---|
| IG dùng cho attribution hay update direction? | **Không xác nhận được** — "gradient saturation" là đúng vấn đề mà integrated-gradient (Sundararajan et al.) giải quyết trong literature classification, nhưng abstract không nói rõ HIFA có dùng IG hay cơ chế khác (vd warm-restart, feature-normalization, multi-scale selection) |
| Path recompute mỗi iteration hay cố định? | Không trả lời được |
| Path baseline→input hay clean→current? | Không trả lời được |
| Objective: task loss hay feature-distortion? | **Feature-level** — xác nhận được (giống OSFD/N6-B, khác IJCNN 2024) |
| Có test CNN→Transformer/Swin không? | Model list nêu trong TLDR: Faster R-CNN, SSD, RetinaNet, YOLOv5, YOLOv8 — **toàn CNN, không thấy Swin/ViT/DETR** nào. Tín hiệu gián tiếp (không phải xác nhận đầy đủ, vì TLDR có thể lược bớt model) |

**Đánh giá sơ bộ (rủi ro overlap cao hơn IJCNN, nhưng chưa kết luận được)**: HIFA cùng family
"feature-level attack cho OD" như OSFD/N6-B, và cụm từ "gradient saturation" trùng đúng
motivation mà N6-B dùng để giải thích tại sao path-integrated gradient nên tốt hơn instantaneous
gradient (§16-B0: "gradient tại một điểm không phải nơi chứa tín hiệu transfer hữu ích"). Đây là
overlap-risk cao nhất trong 2 bài, **nhưng không đủ căn cứ để kết luận trùng cơ chế** — "gradient
saturation" có nhiều cách giải quyết không phải path/integrated-gradient (ví dụ: chuẩn hoá
feature-magnitude, chọn layer/scale khác, warm-up step size). Không có bằng chứng HIFA test trên
backbone Transformer — nếu đúng (cần full-text xác nhận), đây vẫn là điểm khác biệt bảo vệ được
cho N6-B (đúng gap CNN→Transformer mà HIFA không chạm tới).

**Kết luận phần 18 (cập nhật sau khi có full-text IJCNN 2024)**: **IJCNN 2024 — CLOSED**, zero
overlap xác nhận qua full-text (§18-A), không cần theo dõi thêm. **HIFA — vẫn OPEN**, vẫn chỉ có
abstract/TLDR, là rủi ro cần theo dõi cao nhất nếu sau này có được full-text (chưa có route lấy
được, theo lựa chọn đã thống nhất). Quyết định giữ nguyên: **không block mechanism-proof (bước 3)
chờ full-text HIFA** — tiếp tục sang diagnostic `cos(g_local, g_target)` vs `cos(g_path,
g_target)`, và khi viết claim contribution cuối cùng, đặt câu "HIFA full-text chưa verify" như
một risk chưa đóng, không phải đã loại trừ.

### 18-C. Fresh literature scan (sau held-out `val_100`) — trước khi thiết kế mechanism-proof

Scan lại có chủ đích quanh 4 trục: path/integrated-gradient trong transferable OD, CNN→Transformer
detector transfer, "recompute path mỗi iteration" (đặc thù cách N6-B làm), feature-loss
path-averaging. Mục tiêu: khóa novelty scope trước khi tốn compute cho mechanism diagnostic.

**HIFA — thêm chi tiết cơ chế (vẫn chỉ từ search snippet, chưa phải full-text, nhưng rõ hơn TLDR
trước)**: dùng **"Diversity-Enhanced Integrated Gradients"** để *xác định key/high-level feature
mà nhiều model cùng dựa vào* ("assess key features that different models rely on in common,
providing guidance for generating adversarial examples") — "diversity" đến từ **augment input**
(motion blur, salt-and-pepper noise) để làm IG ổn định hơn qua nhiều view, giống input-ensemble
IG hơn là path theo trajectory tấn công. **Đọc khác biệt cấu trúc quan trọng so với N6-B**: HIFA
dùng IG như **attribution/feature-selection** (chọn/định vị feature nào để tấn công mạnh hơn),
còn N6-B dùng path-gradient để **thay thế trực tiếp toàn bộ ascent gradient mỗi step**
(`g_path,t` là thứ duy nhất update rule dùng, không phải một feature-importance map phụ trợ);
path của N6-B đi theo **trajectory `clean→current-state` đang tiến hóa cùng `δ_t` qua 100 step**,
trong khi "diversity" của HIFA nghe giống một **tập input-view cố định** (augment ảnh) hơn là một
path phụ thuộc trạng thái δ đang được tối ưu. → **Hạ mức risk overlap** so với đánh giá "cao nhất,
cần theo dõi" ở §18-B ban đầu — vẫn OPEN (chưa full-text), nhưng bằng chứng gián tiếp hiện có
nghiêng về "khác cơ chế" nhiều hơn là "trùng cơ chế".

**MuMoDIG xác nhận = arXiv:2412.18844 (Ren et al., AAAI 2025)** — đúng paper project đã biết
(nhắc ở §16-B0 làm motivation cho N6-B0). Refine integration path theo 3 trục
multiplicity/monotonicity/diversity, domain **classification** (CNN+ViT), không phải OD. Không
phải risk mới, chỉ xác nhận lại danh tính paper đã cite.

**TAIG (arXiv:2205.13152)** — tổ tiên trực tiếp của dòng MIG/MuMoDIG, IG classification attack
với 2 biến thể path (straight-line vs random piecewise-linear). Cùng family đã biết, domain
classification, không phải OD — không phải risk mới.

**DMFAA** (distillation-based surrogate, feature-based, cross-architecture: CNN/Mamba/Transformer)
— cần bước train surrogate qua distillation từ model khác họ trước khi attack. **Cùng họ với
candidate CRA đã loại ở Phase M** (§9, overlap cao với SAA) vì đều cần witness/distillation.
Không đe dọa N6-B (N6-B không cần witness model, không train lại surrogate).

**Context hỗ trợ motivation (không phải prior-art cạnh tranh)**: benchmark 2026
(`arXiv:2602.16494`, "Benchmarking Adversarial Robustness and Adversarial Training Strategies for
Object Detection") xác nhận độc lập, ngoài chính project: *"modern adversarial attacks against
object detection models show a significant lack of transferability to transformer-based
architectures"* — đúng đúng gap CNN→Transformer mà toàn bộ project theo đuổi từ đầu (RESEARCH.md
§1), đáng trích dẫn khi viết motivation/related-work, không phải điều cần lo về novelty.

**Kết luận 18-C**: không phát hiện thêm paper nào khớp đúng cơ chế "path-integrated gradient thay
thế ascent gradient mỗi step, path clean→current-state theo δ đang tối ưu, cho feature-distortion
loss trong OD, nhắm CNN→Transformer gap" — đây vẫn là niche N6-B đang giữ. HIFA là closest-still-
open risk nhưng bằng chứng gián tiếp cho thấy vai trò IG khác nhau về cấu trúc. **Đủ tin tưởng để
tiếp tục sang mechanism-proof (bước kế tiếp) mà không cần chặn lại chờ full-text HIFA.**

## 19. Mechanism-proof: gradient-alignment diagnostic — GO criterion KHÔNG đạt, nhưng có tín hiệu
hướng nhất quán

Câu hỏi trực tiếp: liệu path-averaged gradient của surrogate có **align tốt hơn** với hướng
ascent thật của chính target (`g_target` = gradient của target's own feature-distortion loss tại
đúng state đó, không RRB) so với instantaneous gradient hay không — nếu có, đó là bằng chứng cơ
chế thật cho "path tìm ra hướng transfer tốt hơn", không chỉ "path tạo step lớn hơn". Script mới
`scripts/n6b_alignment_diagnostic.py`, không recraft (dùng noise đã crafted của `dev_300`), N=100,
4 target: `yolox_l`, `mask_rcnn_swin_t`, `dino_swin_l`, `deformable_detr`.

Bug gặp giữa chừng: DINO-Swin-L dùng gradient checkpointing nội bộ, không tương thích
`torch.autograd.grad(loss, inputs=...)` (`RuntimeError: Checkpointing is not compatible with
.grad()...`). Sửa bằng `.backward()` + đọc `.grad` trực tiếp thay vì `torch.autograd.grad`, verify
qua smoke test N=5 cả 4 target trước khi chạy full — không đổi logic đo đạc.

**GO criterion đã pre-register trước khi chạy**: `mean_cos_path − mean_cos_local ≥ +0.03` với 95%
CI không cắt 0, trên `dino_swin_l` và/hoặc `mask_rcnn_swin_t`, mới tính là bằng chứng cơ chế trực
tiếp. Kết quả (N=100):

| target | group | mean_cos_local | mean_cos_path | Δ | 95% CI | same_side_frac |
|---|---|---|---|---|---|---|
| dino_swin_l | C | 0.0061 | 0.0096 | **+0.0034** | [+0.0024, +0.0045] | 1.000 |
| mask_rcnn_swin_t | C | 0.0693 | 0.0755 | **+0.0062** | [+0.0037, +0.0087] | 1.000 |
| yolox_l | B | 0.0487 | 0.0470 | −0.0017 | [−0.0044, +0.0011] cắt 0 | 0.892 |
| deformable_detr | A | 0.0251 | 0.0219 | **−0.0033** | [−0.0049, −0.0015] | 1.000 |

**Đọc đúng mức, không spin**: **GO criterion KHÔNG đạt trên cả 2 target** — cả `dino_swin_l`
(+0.0034) và `mask_rcnn_swin_t` (+0.0062) đều có CI không cắt 0 (thống kê thật, N=100,
same_side_frac=1.000 cả hai) nhưng **biên độ chỉ bằng ~11-20% ngưỡng đã đặt trước (+0.03)**. Theo
đúng chữ đã pre-register, đây là **KHÔNG ĐẠT** — "path tạo alignment tốt hơn rõ rệt với true
gradient của target" **không được chứng minh** ở quy mô này.

**Tín hiệu đáng ghi lại dù không đạt GO — dấu của Δalignment khớp đúng dấu của ΔASR trên cả 4
target đã test** (so với bảng ASR ở §16-B/held-out):

| target | Δalignment (dấu) | ΔASR (dấu, dev_300/held-out) |
|---|---|---|
| dino_swin_l | + (statistically real) | + (mạnh, confirmed 3 lần đo) |
| mask_rcnn_swin_t | + (statistically real) | + (dev_300), gần 0/cắt CI (held-out) |
| yolox_l | ~0 (CI cắt 0) | + nhỏ, CI cắt 0 cả 2 lần đo |
| deformable_detr | − (statistically real) | − (CI không cắt 0 ở held-out, biên độ nhỏ) |

Không phải trùng hợp ngẫu nhiên hoàn toàn (4/4 model đúng dấu), nhưng **biên độ alignment quá nhỏ
để một mình giải thích biên độ ASR gain quan sát được** (vd +0.0034 cosine trên DINO vs +6.5 điểm
ASR) — cosine similarity ở một điểm là proxy khá gián tiếp/nhiễu cho hiệu ứng tích lũy qua 100
step, nên một lợi thế alignment nhỏ mỗi step *có thể* cộng dồn thành khác biệt lớn theo thời gian,
nhưng **diagnostic này (đo tại 1 điểm cuối trajectory) không đủ để khẳng định hay bác bỏ khả năng
đó** — cần đo alignment dọc theo cả trajectory (không chỉ tại step cuối) mới trả lời được câu hỏi
tích lũy.

**Kết luận (đúng kỷ luật đã dùng suốt project)**: **"alignment tốt hơn" không phải lời giải thích
đầy đủ/đủ mạnh cho ASR gain của N6-B** ở dạng đo hiện tại — đây là **NO-GO cho phiên bản
mechanism-proof này theo đúng tiêu chí đã đặt**, dù hướng dấu nhất quán (không random) là một manh
mối nhỏ đáng giữ lại, không phải noise thuần túy. Diễn giải thận trọng nhất: **ASR gain của N6-B
nhiều khả năng đến từ một cơ chế khác alignment-tại-1-điểm** — ứng viên hợp lý nhất theo đúng
hypothesis đã nêu trong docstring của script (chưa test): **hiệu ứng tích lũy/quỹ đạo** (nhiều
alignment-nhỏ cộng dồn qua 100 step) hoặc **effective step size / saturation dynamics** (path có
thể tránh bão hòa `sign()` sớm hơn local, tương tự cơ chế từng nghi ngờ ở DOB — dù DOB chính nó đã
NO-GO cho weighting theo difficulty, saturation dynamics theo path lại là câu hỏi khác, chưa test
trực tiếp).

## 20. Novelty control: `G_osfd` vs `G_det` — pilot N=20, tín hiệu đúng hướng nhưng chưa đạt
significance

Sau khi alignment mechanism-proof (§19) không đạt GO criterion như cơ chế chính, câu hỏi novelty
còn lại: N6-B thắng vì path-averaging **generically tốt cho mọi objective**, hay path-averaging
có **interaction đặc biệt với đúng OSFD's object-aware feature-distortion loss**? Script mới
`scripts/n6b_novelty_control.py`, 4 variant `det_local`/`det_path`/`osfd_local`/`osfd_path` (bảng
thiết kế xem docstring script), N=20, `dev_50`, craft=1183.7s.

**Lưu ý quan trọng về tính so sánh được**: `det_*` KHÔNG dùng RRB (đúng baseline `mi_fgsm` sẵn có
của project, chưa từng dùng RRB), `osfd_*` có RRB — đây là **objective control**, không phải
compute-identical ablation (đã note rõ trong docstring, không phải oversight).

**Kết quả (N=20, ASR %)**:

| model | group | ASR(det_local) | ASR(det_path) | **G_det** | ASR(osfd_local) | ASR(osfd_path) | **G_osfd** | interaction | 95% CI | same_side |
|---|---|---|---|---|---|---|---|---|---|---|
| faster_rcnn_r50 | — | 100.0 | 100.0 | 0.0 (ceiling) | 100.0 | 98.9 | −1.1 | −1.1 | [−3.9, 0.0] | 0.626 |
| yolox_l | B | 16.8 | 21.8 | **+5.0** | 86.1 | 87.1 | +1.0 | −4.0 | [−10.9, +7.1] cắt 0 | 0.774 |
| mask_rcnn_swin_t | C | 27.8 | 27.8 | **0.0 (flat)** | 81.4 | 86.6 | **+5.2** | +5.2 | [−2.4, +13.2] cắt 0 | 0.915 |
| dino_swin_l | C | 11.0 | 11.0 | **0.0 (flat)** | 31.0 | 40.0 | **+9.0** | +9.0 | [−1.5, +16.1] cắt 0 (sát) | 0.964 |

**Đọc đúng mức**: pattern đúng hướng hypothesis của bạn rất rõ về mặt **điểm ước lượng** — trên cả
2 hard target nhóm C, `det_path` cho **ĐÚNG BẰNG 0 cải thiện** so với `det_local` (không phải chỉ
nhỏ — flat tuyệt đối, 27.8→27.8 và 11.0→11.0), trong khi `osfd_path` cho gain rõ (+5.2, +9.0).
Đây là khác biệt định tính (path giúp = 0 vs path giúp > 0), không chỉ khác biệt về độ lớn. Tuy
nhiên **CI của interaction chưa cắt được 0 ở cả 3 model** — `dino_swin_l` sát nhất (CI
[−1.5,+16.1], same_side_frac=0.964, tức 96.4% bootstrap draw dương) nhưng chưa đạt 95%
significance ở N=20. Đây là pilot-stage signal, chưa phải confirmed.

**1 confound cần nêu rõ, không giấu**: `yolox_l` đi NGƯỢC pattern (G_det=+5.0 > G_osfd=+1.0) —
nhưng đọc kỹ thấy có **ceiling/floor confound**: `osfd_local` đã ở ASR=86.1% trên `yolox_l` (ít
chỗ để tăng thêm), còn `det_local` chỉ ở 16.8% (nhiều chỗ để tăng) — so sánh Δ điểm-ASR trực tiếp
giữa 2 objective có baseline khác xa nhau vốn thiên vị cho bên có baseline thấp hơn. Không thể kết
luận "det thắng osfd trên yolox_l" một cách sạch từ số liệu này.

Điểm nhất quán bổ sung với toàn bộ project: `ASR(det_local)` trên 2 hard target nhóm C rất thấp
(27.8%, 11.0%) so với `ASR(osfd_local)` (81.4%, 31.0%) — tái xác nhận finding gốc của project
(OSFD transfer tốt hơn hẳn task-loss baseline `mi_fgsm`/`det` sang hard target), độc lập với câu
hỏi path-averaging.

**Kết luận pilot N=20 (đúng vị trí, chưa confirmed lúc đó)**: tín hiệu đủ mạnh để không loại bỏ
hypothesis, nhưng N=20 chưa đủ power. User quyết định KHÔNG scale N=100 full 4-target (chi phí
~1.5-2h không đáng so với info gain) — thay vào đó chạy **DINO-only, N=50** (rẻ hơn, tập trung
đúng nơi story mạnh nhất, đúng ưu tiên information-gain/compute-cost).

### DINO-only N=49 (gần full `dev_50`) — interaction **CONFIRMED, CI không cắt 0 lần đầu**

Script/config giữ nguyên (`scripts/n6b_novelty_control.py`, chỉ đổi `--targets dino_swin_l`),
N=49/50 (1 skip do 0 GT hợp lệ, đúng hành vi thường thấy của `dev_50`), craft=2965.1s.

| model | ASR(det_local) | ASR(det_path) | G_det | ASR(osfd_local) | ASR(osfd_path) | G_osfd | interaction | 95% CI | same_side |
|---|---|---|---|---|---|---|---|---|---|
| dino_swin_l | 9.6% | 10.0% | +0.4 | 28.3% | 34.7% | **+6.4** | **+6.0** | **[+0.9, +10.5]** | 0.987 |
| faster_rcnn_r50 (surrogate) | 99.1% | 98.6% | −0.5 | 98.2% | 97.3% | −0.9 | −0.5 | [−2.4, +1.6] cắt 0 | 0.603 |

**Đọc kết quả**: trên `dino_swin_l`, `det_path` gần như không cải thiện gì so với `det_local`
(+0.4, ASR đã rất thấp 9.6%→10.0%, task-loss baseline transfer rất kém sang DINO — tái xác nhận
finding gốc project), trong khi `osfd_path` cho gain rõ (+6.4, khớp đúng tầm với N6-B's confirmed
DINO gain ở §16-B: +6.2 dev_300, +6.5 held-out). **Interaction = +6.0, CI=[+0.9,+10.5] KHÔNG cắt
0** — lần đầu tiên trong toàn bộ novelty-control đạt statistical significance. Surrogate (đã
ceiling ~98-99% cả 2 objective) không có gì để phân biệt, đúng kỳ vọng.

**Kết luận (đủ căn cứ để phát biểu, nhưng đọc đúng phạm vi)**: trên `dino_swin_l` cụ thể,
**path-averaging có interaction thật với đúng OSFD objective — không phải hiệu ứng generic của
path-averaging trên mọi loss** (nếu path generic tốt, `det_path` đã phải cho gain tương tự trên
DINO, nhưng không — gần như flat). Đây là bằng chứng novelty tốt nhất hiện có, **nhưng phạm vi kết
luận chỉ giới hạn ở DINO** (1 model, 1 hard target) — chưa lặp lại trên `mask_rcnn_swin_t` ở N đủ
lớn (pilot N=20 cho tín hiệu tương tự +5.2 nhưng CI vẫn cắt 0, chưa re-run riêng Mask ở N lớn hơn
theo cùng cách đã làm cho DINO). Không suy rộng thành "novelty story đã khóa cho mọi hard target"
— chỉ khóa được cho DINO.

## 21. Breadth test: `dino_r50` — tách được decoder-DINO khỏi backbone-Swin, kết luận rõ ràng

Câu hỏi: gain bất thường của N6-B trên `dino_swin_l` (+6.2 dev_300, +6.5 held-out, CI không cắt 0
cả 2 lần) đến từ **decoder DINO** (denoising anchor box, contrastive query selection) hay từ
**backbone Swin**? `dino_swin_l` gộp cả 2 biến số nên không tách được. Thêm target mới
**`dino_r50`** vào `MODEL_REGISTRY` (group mới "D", không ép vào A/B/C có sẵn vì đây là trục khác)
— cùng decoder DINO, backbone **ResNet-50** (CNN) thay vì Swin. Checkpoint tải công khai từ
`download.openmmlab.com` (`dino-4scale_r50_8xb2-12e_coco`), verify tích hợp đầy đủ (config
resolve tự động qua `.mim/configs`, checkpoint load sạch không warning, backbone forward 3 stage
đúng shape ResNet-FPN, `predict_canvas` end-to-end ra detection hợp lệ) trước khi dùng.

**Không cần craft lại gì** — script mới `scripts/n6b_breadth_eval.py` tái dùng trực tiếp noise đã
crafted của `dev_300` (N=296, đúng run đã CONFIRMED `dino_swin_l` +6.2 ở §16-B), chỉ chạy
inference cho model mới. Miễn phí về craft compute, N=296 (mạnh hơn hẳn 1 pilot mới).

**Kết quả**:

| model | decoder | backbone | ASR(osfd_local) | ASR(path_m3) | Δ | 95% CI |
|---|---|---|---|---|---|---|
| `dino_swin_l` | DINO | Swin (Transformer) | 33.0% | 39.1% | **+6.2** | **[+4.6,+7.8]** không cắt 0 |
| `dino_r50` | DINO (giống hệt) | ResNet-50 (CNN) | **99.2%** | 98.8% | −0.3 | [−0.9,+0.2] cắt 0 |

**Đọc rõ, không mơ hồ**: `dino_r50` đã bão hòa ASR ~99% **ngay từ `osfd_local`** — giống hệt
pattern của mọi target backbone-CNN khác trong project (`fcos_r50` 97-98%, `deformable_detr`
97-99%, surrogate 98-99%), khác hẳn `dino_swin_l` (chỉ 33%, target khó nhất project). Path không
còn gì để cải thiện thêm (đã ceiling, đúng logic "ít chỗ để tăng" đã thấy ở mọi CNN-backbone
target khác) — delta âm nhẹ, CI cắt 0, giống hệt `fcos_r50`/`deformable_detr`.

**Kết luận — ĐÃ SỬA LẠI để không overclaim (bản đầu tiên đi quá xa, xem lý do bên dưới)**. Cần
tách rõ 2 hypothesis khác nhau mà kết quả `dino_r50` KHÔNG có sức nặng như nhau để trả lời:

- **H1 — nguồn gốc transfer difficulty**: `Swin/representation gap → OSFD transfer khó`.
  `dino_r50` **support H1 khá mạnh**: thay Swin-L bằng R50, giữ nguyên kiến trúc decoder DINO, thì
  `ASR_local` từ 33.0% (Swin) nhảy lên 99.2% (R50) — loại bỏ gần như hoàn toàn "hard transfer
  behavior" trong khi giữ nguyên decoder. Câu đúng mức để viết: *"Replacing the Swin backbone of
  DINO with ResNet-50 eliminates the severe transfer difficulty observed on DINO-Swin-L, while
  keeping the DINO detector architecture. This implicates the backbone/representation family,
  rather than the DINO decoder alone, as the major source of transfer difficulty."* Câu này được
  data support tốt, có thể dùng.
- **H2 — tại sao N6-B (path-averaging) giúp**: `Swin/representation gap → path-averaging hữu ích`.
  **`dino_r50` KHÔNG trả lời được câu này**, vì một confound lớn: `ASR_local` đã ~99.2% — bão hòa
  ceiling gần như tuyệt đối, gần như không còn detection nào để "phá" thêm, nên **không quan sát
  được liệu path operator có giúp gì hay không trên kiến trúc này** — delta âm nhẹ (−0.3, CI cắt
  0) phản ánh "hết chỗ để đo", không phải "path không giúp CNN backbone" theo nghĩa mechanism.

**Không viết** "Swin backbone là biến số quyết định path gain" ở dạng dứt điểm — H2 chưa được
`dino_r50` xác nhận (chỉ mới H1). Bằng chứng hiện có cho H2 (path gain theo backbone gia đình)
vẫn chỉ mang tính **supportive, chưa khóa**:

| target | backbone | path gain | trạng thái |
|---|---|---|---|
| dino_swin_l | Swin | **+6.2 / +6.5** | confirmed, CI không cắt 0 cả 2 lần đo |
| mask_rcnn_swin_t | Swin | +3.2 (dev_300) / +1.7 (held-out) | **partial-support**, held-out CI cắt 0 |
| dino_r50 | R50 (CNN) | −0.3 nhưng **ceiling** | không đo được (không phải bằng chứng "path không giúp") |
| yolox_l | CNN (Darknet) | +1-2 | yếu, CI thường cắt 0 |
| fcos_r50 / deformable_detr | CNN (R50) | ≈0 / âm nhẹ | ceiling tương tự |

DINO robust cho H2, Mask chỉ partial. **Cần một matched-pair thứ hai không bị ceiling để test H2
sạch** — xem kế hoạch phiên sau bên dưới (`HANDOFF.md`): thêm `mask_rcnn_r50` (Mask R-CNN +
ResNet-50-FPN, cùng family detector với `mask_rcnn_swin_t` nhưng khác backbone), tái dùng noise
`dev_300` có sẵn (không cần craft lại, giống hệt cách đã làm cho `dino_r50`). Nếu
`mask_rcnn_r50` cũng gần-ceiling (ASR_local cao, ít room) trong khi `mask_rcnn_swin_t` thấp
(67.8%, path gain +3.2) — đó sẽ là **2 matched-pair độc lập cùng chỉ 1 hướng**, bằng chứng cho H2
mạnh hơn hẳn 1 pair đơn lẻ. Nếu `mask_rcnn_r50` KHÔNG ceiling (còn room để đo) mà path vẫn không
giúp nhiều — đó lại là bằng chứng chống lại H2 theo cách sạch hơn `dino_r50` (vì lần này loại được
confound ceiling).

### Matched-pair #2: `mask_rcnn_r50` — cùng pattern với DINO pair, củng cố H1 rõ rệt

Thêm `mask_rcnn_r50` (Mask R-CNN + ResNet-50-FPN, checkpoint công khai `mask-rcnn_r50_fpn_1x_coco`
từ `download.openmmlab.com`) vào `MODEL_REGISTRY` (group "D", cùng nhóm với `dino_r50`), verify
tích hợp (backbone 4-stage ResNet-FPN đúng shape, `predict_canvas` hoạt động), rồi eval bằng
`scripts/n6b_breadth_eval.py` trên noise `dev_300` có sẵn — **không craft lại**, N=296, gần như
free về compute (chỉ inference).

**Kết quả, đặt cạnh cả 2 matched-pair**:

| pair | head/decoder | backbone | ASR(local) | ASR(path) | Δ | 95% CI |
|---|---|---|---|---|---|---|
| `dino_swin_l` | DINO | Swin-L | 33.0% | 39.1% | **+6.2** | [+4.6,+7.8] không cắt 0 |
| `dino_r50` | DINO (giống hệt) | R50 | 99.2% | 98.8% | −0.3 | [−0.9,+0.2] cắt 0 |
| `mask_rcnn_swin_t` | Mask R-CNN | Swin-T | 67.8% | 71.1% | **+3.2** | [+1.9,+4.6] không cắt 0 |
| `mask_rcnn_r50` | Mask R-CNN (giống hệt) | R50 | **99.3%** | 98.6% | **−0.7** | **[−1.22,−0.22]** không cắt 0 |

**Đọc kết quả — H1 được củng cố mạnh bởi pair độc lập thứ hai**: y hệt pattern `dino_r50`, đổi
Swin→R50 mà giữ nguyên detector head (Mask R-CNN) làm `ASR_local` nhảy từ 67.8%→99.3% — cùng
hướng, cùng biên độ lớn, trên một kiến trúc detector hoàn toàn khác (two-stage RPN+RoI, không
phải DETR-style decoder). **2 matched-pair độc lập (DINO decoder, Mask R-CNN head) giờ cùng chỉ
1 hướng** — bằng chứng cho H1 ("Swin/representation gap là nguồn gốc transfer difficulty, không
phải đặc thù của riêng 1 decoder/head nào") mạnh hơn hẳn so với chỉ 1 pair.

**Về H2 — khác `dino_r50` một điểm đáng chú ý**: `mask_rcnn_r50` có CI **không cắt 0** (khác
`dino_r50` CI cắt 0) nhưng theo hướng **âm** (−0.7, CI=[−1.22,−0.22]) — path làm giảm nhẹ ASR trên
target CNN gần-ceiling này, có ý nghĩa thống kê thật (không phải nhiễu). Không mâu thuẫn với H1
— khớp đúng pattern đã thấy ở mọi CNN-backbone gần-ceiling khác trong project (`fcos_r50`,
`deformable_detr`, surrogate đều âm nhẹ) — đọc hợp lý nhất: khi đã bão hòa ASR, path-averaging
không "lãng phí" ngân sách một cách vô hại mà có xu hướng nhẹ về hướng kém tối ưu hơn local thuần
(một dạng biến thể của saturation dynamics, không phải bằng chứng path "hại" theo nghĩa cơ chế).
**Không đổi kết luận H1**; với H2 vẫn giữ nguyên vị trí đã chốt — chỉ `dino_swin_l` có interaction
CI không cắt 0 dương rõ (xem §20 DINO-only N=49), `mask_rcnn_swin_t` mới chỉ có ASR-gain riêng
confirmed (§16-B) chứ chưa có interaction-vs-det_path confirmed cùng chuẩn.

**Kết luận cập nhật (thay thế phần "chưa chốt" ở trên)**: H1 giờ **support bởi 2 matched-pair độc
lập**, đủ mạnh để phát biểu rộng hơn: *"Across two independent detector-head families (DINO's
transformer decoder and Mask R-CNN's two-stage RPN+RoI head), replacing a Swin backbone with
ResNet-50 while keeping the head fixed eliminates the transfer difficulty almost entirely (ASR
jumps to ~99% in both cases) — strong convergent evidence that backbone/representation family,
not detector-head architecture, is the primary source of the CNN→Transformer transfer gap this
project targets."* H2 (path-averaging specifically amplified by Swin) vẫn giữ nguyên trạng thái
đã chốt trước đó — confirmed cho DINO, chưa confirmed cho Mask-Swin-T ở cùng chuẩn interaction
test.

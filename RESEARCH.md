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
luận chỉ giới hạn ở DINO** (1 model, 1 hard target) — đã kiểm tra thêm ở N=50 cho
`mask_rcnn_swin_t` để xem có generalize hay không (xem ngay dưới). Không suy rộng thành "novelty
story đã khóa cho mọi hard target" — chỉ khóa được cho DINO.

### Mask-Swin-T N=50 — interaction KHÔNG generalize, đóng H2-broad

Chạy lại đúng script/config (`scripts/n6b_novelty_control.py --targets mask_rcnn_swin_t
--n-images 50`), N=50 (đủ 50 ảnh, không skip), 4 variant trên cả surrogate + target, log
`runs/run_attack_n6bctl_{det_local,det_path,osfd_local,osfd_path}_dev_50_n50_20260829T*.json`.

**Kết quả (N=50, ASR %, 95% bootstrap CI cho interaction)**:

| model | ASR(det_local) | ASR(det_path) | G_det | ASR(osfd_local) | ASR(osfd_path) | G_osfd | interaction | 95% CI | same_side |
|---|---|---|---|---|---|---|---|---|---|
| faster_rcnn_r50 (surrogate) | 99.1% | 98.6% | −0.45 | 98.2% | 97.3% | −0.90 | −0.45 | [−2.58, +1.55] cắt 0 | 0.597 |
| mask_rcnn_swin_t | 19.4% | 21.9% | +2.48 | 71.1% | 76.0% | +4.96 | **+2.48** | **[−2.96, +8.0]** cắt 0 | 0.820 |

**So trực tiếp với pilot N=20 (bảng ở trên)**:

| N | G_det | G_osfd | interaction | 95% CI | same_side |
|---|---|---|---|---|---|
| N=20 (pilot) | 0.0 (flat tuyệt đối) | +5.2 | +5.2 | [−2.4, +13.2] cắt 0 | 0.915 |
| N=50 | +2.48 | +4.96 | +2.48 | [−2.96, +8.0] cắt 0 | 0.820 |

Tăng N từ 20 lên 50 **không siết CI về phía dương** như kỳ vọng nếu đây thuần túy là vấn đề thiếu
power (điều đã xảy ra đúng như vậy với DINO: N=20 CI cắt 0 sát → N=49 CI [+0.9,+10.5] không cắt
0) — ngược lại, point estimate của interaction **giảm** (+5.2→+2.48) và `same_side_frac` **giảm**
(0.915→0.820). Nguyên nhân cụ thể quan sát được: pattern định tính "det_path ≈ det_local (flat
tuyệt đối)" — chính điều làm interaction ở DINO thuyết phục (generic task-loss path không giúp gì,
chỉ OSFD objective mới được path khuếch đại) — **không tái lập** ở N=50: `det_path` giờ đã tăng
thật so với `det_local` (+2.48), không còn flat.

**Kết luận (đóng câu hỏi, không chạy thêm N lớn hơn)**: H2 ở dạng "path-averaging có interaction
đặc biệt với OSFD objective, khuếch đại bởi backbone Swin nói chung" **không được support cho
`mask_rcnn_swin_t`** — không phải do thiếu power (đã loại trừ, vì signal đi ngược hướng kỳ vọng
khi tăng N thay vì siết chặt về phía dương), mà là bằng chứng thực sự yếu đi. Story chính xác nhất
hiện có: **OSFD-path interaction là hiện tượng đặc thù cho `dino_swin_l`/decoder DINO, không phải
property chung của "mọi target có backbone Swin"**. Tách bạch rõ với H1 (§21 dưới đây), vốn vẫn
được support mạnh bởi 2 matched-pair độc lập rằng backbone/representation family giải thích
transfer difficulty tốt trên cả DINO lẫn Mask R-CNN — **H1 (nguồn gốc transfer difficulty) và H2
(tại sao path-averaging giúp) là hai câu hỏi tách biệt**: dữ liệu hiện có xác nhận H1 rộng (2
matched-pair), nhưng cho thấy H2 hẹp hơn ban đầu nghĩ — chỉ đúng cho DINO, không generalize sang
Mask-Swin-T dù cùng họ backbone Swin. Không chạy thêm Mask novelty-control ở N lớn hơn — câu hỏi
đã đủ dữ liệu để đóng.

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
**Không đổi kết luận H1**; với H2, `dino_swin_l` có interaction CI không cắt 0 dương rõ (xem §20
DINO-only N=49), còn `mask_rcnn_swin_t` có ASR-gain riêng confirmed (§16-B) nhưng interaction-vs-
det_path đã được test riêng ở N=50 (§20, "Mask-Swin-T N=50") và **không generalize** — CI vẫn cắt
0, signal yếu đi khi tăng N thay vì siết chặt.

**Kết luận cập nhật (thay thế phần "chưa chốt" ở trên)**: H1 giờ **support bởi 2 matched-pair độc
lập**, đủ mạnh để phát biểu rộng hơn: *"Across two independent detector-head families (DINO's
transformer decoder and Mask R-CNN's two-stage RPN+RoI head), replacing a Swin backbone with
ResNet-50 while keeping the head fixed eliminates the transfer difficulty almost entirely (ASR
jumps to ~99% in both cases) — strong convergent evidence that backbone/representation family,
not detector-head architecture, is the primary source of the CNN→Transformer transfer gap this
project targets."* H2 (path-averaging specifically amplified by Swin) đã được test ở cùng chuẩn
interaction cho cả 2 hard target nhóm C: **confirmed cho DINO** (interaction +6.0, CI=[+0.9,+10.5]
không cắt 0), **không generalize sang Mask-Swin-T** (§20, N=50: interaction +2.48, CI=[−2.96,+8.0]
cắt 0, signal yếu đi khi N tăng từ 20→50 chứ không siết chặt — loại trừ khả năng chỉ do thiếu
power). Kết luận cuối cho H2: **OSFD-path interaction là hiện tượng đặc thù cho DINO/decoder DINO,
không phải property chung của mọi target backbone Swin** — tách bạch rõ khỏi H1 (nguồn gốc transfer
difficulty), vốn vẫn support rộng trên cả 2 pair.

## 22. Compute-matched control: path-M3 so với naive K=3 RRB averaging — GO cho DINO

Câu hỏi objection mạnh nhất còn mở cho N6-B: `path_m3` transfer tốt hơn `osfd_local` vì **cấu trúc
path clean→current**, hay chỉ vì dùng **nhiều gradient evaluation hơn** (M=3 forward/backward mỗi
step thay vì 1) — tức là một dạng Monte-Carlo variance reduction thuần túy, không đặc thù path?
Project đã có 2 datapoint cũ liên quan (`rcg_avg` — Phase M, `rrb_avg_k3` — N6-A) đóng vai trò
"K=3 naive averaging, cùng compute" nhưng **không đồng nhất với nhau** trên DINO (rcg_avg +11.0 vs
rrb_avg_k3 +3.0, cùng 20 ảnh, cùng config danh nghĩa) — không đủ sạch để dùng làm evidence, vì cả
hai là 2 script riêng biệt, không lockstep-pair trực tiếp với `path_m3`.

**Thiết kế mới** (script `scripts/n6cm_compute_matched_pilot.py`): 3-way lockstep trên cùng 1 ảnh —
`osfd_local` (1 draw/step, λ=1), `rrb_avg_k3` (K=3 draw độc lập/step, λ=1, gradient = trung bình —
đúng compute 3x như path), `path_m3` (M=3 draw tại λ=1/3,2/3,1, chia sẻ đúng 1 augmentation draw
qua cả 3 λ, theo đúng thiết kế `n6b_path_pilot.py`). Một RNG snapshot được chụp mỗi step và phục
hồi riêng trước draw của `osfd_local`, trước draw đầu của `rrb_avg_k3`, và trước MỖI draw của
`path_m3` — khiến `osfd_local`, draw-đầu của `rrb_avg_k3`, và cả 3 draw của `path_m3` cùng thấy
**đúng 1 augmentation instance** ở mỗi step; chỉ draw thứ 2-3 của `rrb_avg_k3` là random-draw độc
lập thêm (không thể loại bỏ mà không phá vỡ chính cơ chế đang test). Đây là pairing chặt hơn cả 2
tiền lệ cũ (N6-A's `osfd_k1` chỉ khớp seed ở step 1, không lockstep-restore mỗi step; so 2 script
riêng biệt như `rcg_avg` vs `rrb_avg_k3` thì không pairing gì cả).

Primary quantity pre-registered: `delta_path_vs_avg = ASR(path_m3) - ASR(rrb_avg_k3)` trên
`dino_swin_l`, với paired image-cluster bootstrap 95% CI (quy ước giống N4/N6-B). GO criterion
(chốt trước khi chạy N=49): point estimate ≥ +3 **và** CI không cắt 0; CI chạm/cắt 0 dù point
estimate dương → inconclusive, dừng, không scale N=300 để rescue; effect <+3 hoặc đổi dấu → NO-GO.
`yolox_l`/`mask_rcnn_swin_t` chỉ secondary, không dùng để cứu quyết định trên DINO.

**Pilot N=20** (`results/n6cm_compute_matched_pilot_summary.csv`): tín hiệu đúng hướng nhưng chưa
decisive.

| model | osfd_local | rrb_avg_k3 | path_m3 | path−avg | 95% CI | same_side |
|---|---|---|---|---|---|---|
| faster_rcnn_r50 (surrogate) | 96.81 | 97.87 | 96.81 | −1.06 | [−3.88,0.0] cắt 0 | 0.627 |
| yolox_l | 82.18 | 78.22 | 85.15 | +6.93 | [+1.80,+12.86] không cắt 0 | 1.000 |
| dino_swin_l | 27.0 | 30.0 | 38.0 | **+8.0** | [0.0,+16.3] chạm biên 0 | 0.981 |
| mask_rcnn_swin_t | 80.41 | 81.44 | 82.47 | +1.03 | [−5.06,+7.53] cắt 0 | 0.687 |

Điểm quan trọng nhất ở N=20: DINO point estimate +8.0 — vượt xa ngưỡng +3, nhưng CI chạm đúng 0.0 —
"promising nhưng underpowered", không đạt GO sạch. `yolox_l` ở N=20 cho tín hiệu sạch nhất
(+6.93, CI không cắt 0) — sẽ không bền khi lên N=49 (xem bên dưới).

**Self-check xác nhận implementation đúng**: `path_m3 − osfd_local` trên DINO ở N=20 = **+11.0**,
khớp tuyệt đối với con số gốc `n6b_path_pilot.py` đã báo cáo ở §16-B (N=20, cùng manifest). Đồng
thời `rrb_avg_k3 − osfd_local` trên DINO = **+3.0**, khớp với con số gốc N6-A (§16), **không khớp**
`rcg_avg`'s +11.0 (Phase M, §9) — củng cố nghi ngờ `rcg_avg` là artifact của một implementation
riêng biệt, không đáng tin làm compute-matched evidence; từ đây **loại `rcg_avg` khỏi evidence
chính** cho câu hỏi compute-matched.

**Confirmation N=49** (`results/n6cm_compute_matched_pilot_n50.csv`, cùng script/config, không đổi
gì):

| model | osfd_local | rrb_avg_k3 | path_m3 | path−avg | 95% CI | same_side |
|---|---|---|---|---|---|---|
| faster_rcnn_r50 (surrogate) | 96.85 | 98.20 | 96.40 | −1.80 | [−3.85,−0.40] không cắt 0 (âm) | 0.986 |
| yolox_l | 72.51 | 70.52 | 73.71 | +3.19 | [−0.72,+6.90] cắt 0 | 0.960 |
| **dino_swin_l** | 26.69 | 27.49 | 34.26 | **+6.77** | **[+2.67,+11.15]** KHÔNG cắt 0 | 0.999 |
| mask_rcnn_swin_t | 70.66 | 71.07 | 74.38 | +3.31 | [−0.47,+7.66] cắt 0 (sát biên) | 0.965 |

**Đọc kết quả, theo đúng khung pre-registered**:

- **DINO — GO**: point estimate +6.77 (≥+3), CI [+2.67,+11.15] **không cắt 0**. Tín hiệu N=20
  "promising nhưng underpowered" (+8.0, CI chạm 0) đã siết chặt về phía dương khi tăng N — đúng
  pattern đã thấy với DINO ở novelty-control (§20: N=20 CI cắt 0 → N=49 CI không cắt 0).
- **yolox_l — đảo chiều đáng lưu ý**: N=20 từng là tín hiệu sạch nhất (+6.93, CI không cắt 0), N=49
  co lại còn +3.19 và **CI cắt 0** — tín hiệu N=20 không bền, minh chứng cho lý do không nên kết
  luận sớm từ N nhỏ (đối lập với DINO, nơi N=20→N=49 đi đúng hướng củng cố).
- **mask_rcnn_swin_t**: +3.31 (đạt ngưỡng biên độ) nhưng CI cắt 0 sát mép — secondary, không gate
  quyết định theo thỏa thuận trước khi chạy.
- **surrogate**: −1.80, CI không cắt 0 (âm) — khớp pattern "gần ceiling → path hơi kém hơn avg" đã
  thấy xuyên suốt project ở mọi model gần ceiling (không phải bằng chứng chống lại mechanism).

**Kết luận (đạt GO theo đúng tiêu chí đã chốt trước khi chạy)**:

> Under matched gradient-computation budget, path-integrated OSFD outperforms naive multi-view
> gradient averaging on the hardest DINO-Swin target (+6.77 ASR, 95% CI [+2.67,+11.15]),
> indicating that its gain cannot be explained solely by additional gradient evaluations — the
> clean→current path structure contributes beyond naive K=3 RRB averaging.

Theo đúng stopping rule đã đặt trước ("nếu N≈50 confirm thì không cần N=300 trừ khi final table cần
precision cao hơn") — **dừng compute ở đây cho control này**. Đây là mảnh evidence mạnh thứ 3 cho
N6-B (sau DINO held-out confirm ở §16-B và DINO novelty-control interaction ở §20), và là mảnh duy
nhất trực tiếp loại trừ "compute nhiều hơn" như lời giải thích thay thế.

## 23. E6: Backbone Adversarial Response Coupling — mechanism candidate cho H1, STRONG GO

Sau §21 (H1 confirmed bởi 2 matched-pair độc lập: đổi Swin→R50 giữ nguyên head làm ASR nhảy lên
~99% cả 2 lần), câu hỏi còn để ngỏ là **tại sao** backbone/representation family lại là nguồn gốc
gap, không phải chỉ **rằng** nó là nguồn gốc. Hai lời giải thích đơn giản nhất đã bị loại: `‖ΔF‖`
tại backbone (E1, §8) không giải thích được transfer ordering (giữa 2 target Swin, model bị phá
backbone *ít hơn* lại có mAP_drop *cao hơn*); và instantaneous gradient cosine với true gradient
của target (§19) có tín hiệu đúng dấu nhưng biên độ chỉ bằng 11-20% ngưỡng mechanism đã đặt.

**Hypothesis mới** (không phải "dùng CKA/feature-alignment/subspace" — đó chỉ là công cụ đo,
không phải novelty): transferability phụ thuộc vào việc hai backbone có **phản ứng cùng hình dạng**
trước cùng một perturbation hay không — gọi là **Backbone Adversarial Response Coupling**. Novelty
nằm ở việc **disentangle và định lượng một mechanism của cross-backbone transfer trong object
detection dưới matched-head controls** (khác `‖ΔF‖`/gradient-cosine — cả hai đo "phá được bao
nhiêu"/"đúng hướng đến đâu" tại backbone của chính target; response coupling đo "phản ứng của
target có giống phản ứng của surrogate hay không", so sánh **shape** chứ không phải magnitude).

**Thiết kế** (script mới `scripts/e6_response_coupling.py`, chỉ forward pass, không backward):
với mỗi model `m`, mỗi ảnh `i`: `ΔF_m,i = F_m(x_i+δ_i) - F_m(x_i)`, mỗi backbone stage được
adaptive-avg-pool về lưới không gian cố định (`--pool-size 7`) rồi concat qua mọi stage thành 1
vector/ảnh, L2-normalize theo từng ảnh (`ΔF_hat = ΔF/(‖ΔF‖+ε)`, để một vài ảnh có magnitude bất
thường không lấn át). Similarity giữa surrogate và target đo bằng **linear CKA trong Gram
(sample×sample) space** (Kornblith et al. 2019) — chọn đúng vì Gram matrix chỉ cần cùng N ảnh,
không cần cùng channel-width/spatial-size/số-stage giữa ResNet/Swin/Darknet, nên so sánh được
xuyên kiến trúc mà không cần project về chung 1 chiều. `C_response(s,t) = CKA(Gram(ΔF_hat_s),
Gram(ΔF_hat_t))` là quantity chính; `S_clean(s,t) = CKA(Gram(F_hat_s(x)), Gram(F_hat_t(x)))` (clean
feature, không nhiễu) là control để tách "hai backbone vốn giống nhau sẵn" khỏi "hai backbone phản
ứng giống nhau trước đúng nhiễu này". Hai matched-pair từ §21 (`dino_r50` vs `dino_swin_l`,
`mask_rcnn_r50` vs `mask_rcnn_swin_t`) là test quan trọng nhất vì giữ cố định head/decoder, cô lập
đúng biến số backbone.

**Lưu ý về noise**: môi trường chạy lại không còn giữ noise gốc đã dùng cho §16-22 (`results/`
gitignored, không commit) — noise cho §23 được **recraft mới cùng config y hệt** (đọc từ
`runs/*.json`: epsilon=5, alpha=1, steps=100, mu=1, k=3, RRB on) trên `faster_rcnn_r50`, không
phải cùng 1 file tensor. Verify: ASR/mAP_drop trên noise mới khớp noise cũ trong biên ~1 điểm (vd
`dino_swin_l` 33.6% mới vs 33.0-33.2% cũ, `mask_rcnn_swin_t` 68.1% mới vs 67.8% cũ) — đủ để tin
noise mới đại diện đúng cho cùng attack, không phải một biến thể khác.

**Bug bắt được trước khi tin CI**: bootstrap resample-with-replacement chuẩn (cùng quy ước với
`paired_bootstrap_asr_delta` ở N6-B/N4) áp trực tiếp lên thống kê Gram/CKA cho CI **thiên lệch có
hệ thống** — verify bằng synthetic test (data tương quan giả lập, CKA=0.894): with-replacement
bootstrap đặt chính point estimate **ngoài** CI 95% của nó ([0.902,0.936]), vì ảnh trùng lặp do
resample có similarity=1 tuyệt đối với chính nó, thổi phồng Gram entries một cách hệ thống chứ
không phải nhiễu. Sửa bằng **subsampling không hoàn lại** (m=80% N/draw, chuẩn cho kernel/
U-statistic dưới resampling) — cùng synthetic test, CI phục hồi chứa đúng point estimate
([0.892,0.912]). Toàn bộ CI báo cáo dưới đây dùng phương án đã sửa. Đồng thời tối ưu `center_gram`
từ O(n³) (`H@K@H` matmul tường minh) sang O(n²) (công thức row/col/grand-mean, verify numerically
identical, diff ~1e-14) — không đổi kết quả, chỉ tránh CI-computation trở thành cost chính ở N lớn.

**GO criteria đã pre-register**: Strong GO cần đồng thời (1) cả 2 matched-pair đúng hướng
`C_response(R50) > C_response(Swin)`; (2) bootstrap CI của delta không cắt 0 cho cả 2 pair; (3)
`corr(C_response, ASR) > 0` trên toàn bộ target; (4) `corr(C_response,ASR) > corr(S_clean,ASR)`
(response coupling giải thích transfer tốt hơn hẳn việc hai model chỉ đơn thuần có clean
representation giống nhau). Weak GO: cả 2 pair đúng hướng nhưng (2)-(4) chưa sạch. NO-GO: một pair
sai hướng, hoặc `corr(C_response,ASR)≈0`, hoặc `S_clean` giải thích transfer tốt ngang/hơn
`C_response` — nếu NO-GO, **không đổi sang biến thể CKA/subspace khác để cứu hypothesis**.

**Kết quả — chạy ở 3 quy mô N (49→120→296, tái dùng `dev_50`/`dev_300`) và 2 cách pool
(mean/RMS) làm sensitivity check trước khi cam kết compute lớn** (đề nghị của user, thay vì scale
thẳng lên `dev_300`): tín hiệu không những giữ dấu mà **mạnh dần theo N** — đúng chữ ký của effect
thật, khác hẳn mọi lần trước trong project mà tín hiệu pilot thường yếu đi hoặc dao động khi tăng N
(vd Mask-Swin-T novelty-control ở §20, N6-B0's `mask_rcnn_swin_t`/`yolox_l` ở §22).

| N (ảnh) | pool | Δ C_response DINO (R50−Swin) | 95% CI | Δ Mask-RCNN | 95% CI | corr(C_response,ASR) | corr(S_clean,ASR) |
|---|---|---|---|---|---|---|---|
| 49 | mean | +0.0068 | [+0.0001,+0.0104] | +0.0254 | [+0.0185,+0.0240] | +0.438 | +0.115 |
| 120 | mean | +0.0218 | [+0.0130,+0.0234] | +0.0447 | [+0.0370,+0.0424] | +0.515 | −0.039 |
| **296** | **mean** | **+0.0362** | **[+0.0254,+0.0372]** | **+0.0645** | **[+0.0574,+0.0632]** | **+0.504** | **−0.051** |
| 49 | rms | +0.0505 | [+0.0250,+0.0601] | +0.0603 | [+0.0478,+0.0608] | +0.822 | +0.398 |
| 120 | rms | +0.0531 | [+0.0372,+0.0594] | +0.1033 | [+0.0846,+0.1073] | +0.765 | +0.203 |
| **296** | **rms** | **+0.1055** | **[+0.0854,+0.1098]** | **+0.1302** | **[+0.1147,+0.1364]** | **+0.836** | **+0.193** |

Sign của cả 2 matched-pair **không đổi một lần nào** qua mọi N, mọi pool mode, mọi
`bootstrap_frac` (0.7/0.8/0.9) đã thử (xem `results/e6_response_coupling*.csv` cho số đầy đủ từng
lần chạy).

**Bảng chính (N=296, mean-pool, `results/e6_response_coupling_dev300.csv`)**:

| target | group | S_clean | C_response | ASR | mAP_drop_pct |
|---|---|---|---|---|---|
| fcos_r50 | A | 0.963 | 0.926 | 97.5 | 98.7 |
| deformable_detr | A | 0.896 | 0.883 | 99.1 | 99.8 |
| yolov3_d53 | B | 0.920 | 0.915 | 83.3 | 90.8 |
| yolox_l | B | 0.931 | 0.912 | 69.3 | 76.1 |
| mask_rcnn_swin_t | C | 0.976 | 0.929 | 68.1 | 77.8 |
| dino_swin_l | C | 0.909 | 0.815 | 33.6 | 27.4 |
| dino_r50 | D | 0.825 | 0.851 | 99.7 | 99.8 |
| mask_rcnn_r50 | D | 0.998 | 0.993 | 99.1 | 99.8 |

**Đọc kết quả**: cả 4 tiêu chí Strong GO đạt ở N=296, cả 2 pool mode. Đáng chú ý nhất là hướng đi
của `corr(S_clean,ASR)` theo N — từ +0.115 (N=49) tụt xuống **âm nhẹ** (−0.051 ở N=296, mean-pool)
— nghĩa là ở quy mô đủ lớn, việc hai backbone có clean representation giống nhau **không còn dự
đoán được gì** về transfer (thậm chí hơi ngược), trong khi `C_response` giữ nguyên tín hiệu dương
ổn định xuyên suốt. Đây là bằng chứng khá sạch rằng cái quan trọng không phải "hai backbone vốn đã
giống nhau" mà là "hai backbone phản ứng giống nhau trước đúng perturbation này" — đúng tinh thần
hypothesis đặt ra, không phải một proxy gián tiếp của representation similarity thông thường.

**Audit bổ sung: subsampling CI (m<n) có đáng tin ở đúng N=296 không?** Raw percentile CI từ
m-out-of-n subsampling (m=80% N) ước lượng sampling distribution của thống kê ở cỡ mẫu **m**, không
phải **n** đầy đủ — thiếu hệ số rescale `sqrt(m/n)` theo lý thuyết subsampling (Politis & Romano),
nên không nên gọi thẳng là CI cho thống kê ở N=296. Sửa bằng **delete-1 jackknife** trực tiếp trên
statistic `θ = C_response(R50) − C_response(Swin)` (tái dùng đúng Gram matrix đã cache, không
forward lại): `θ_(-i)` với mỗi ảnh bị bỏ, `SE_jack = sqrt((n-1)/n · Σ(θ_(-i)−θ̄)²)`,
`CI = θ ± 1.96·SE_jack`. Script thêm `jackknife_matched_pair_delta()` +
`--save-gram-npz` (lưu Gram matrix ra `.npz` để audit sau không cần re-extract).

| pool | pair | θ (point) | SE_jack | 95% CI (θ±1.96·SE) | bias_jack |
|---|---|---|---|---|---|
| mean | dino | +0.0362 | 0.0066 | [+0.0233,+0.0491] | −0.0218 |
| mean | mask_rcnn | +0.0645 | 0.0032 | [+0.0582,+0.0708] | −0.0179 |
| rms | dino | +0.1055 | 0.0131 | [+0.0799,+0.1312] | −0.0345 |
| rms | mask_rcnn | +0.1302 | 0.0117 | [+0.1073,+0.1531] | −0.0196 |

Cả 4 trường hợp CI loại trừ 0 rõ ràng ở N=296 — kết luận GO **giữ vững** dưới phương pháp CI chặt
hơn. Nhưng làm audit tương tự ở **N=49** cho kết quả khác: jackknife CI của DINO **cắt 0**
([−0.0051,+0.0186]), trong khi subsampling ở N=49 báo excl-0 rất sát mép (+0.0001 đến +0.0104 tùy
frac) — xác nhận đúng lo ngại rằng CI subsampling lạc quan hơn thực tế ở N nhỏ; **jackknife chỉ
đồng thuận GO ở N=296** (quy mô confirm), không phải N=49 (pilot). Đây là narrative sạch: pilot chỉ
"promising", confirm thật sự đến từ N lớn.

**Giới hạn cần ghi nhận, không dùng để đảo kết luận**: `|bias_jack|/SE_jack` khá lớn ở cả 4 trường
hợp (1.7×–5.6×) — theo lý thuyết jackknife, tỷ lệ này lớn gợi ý CKA (statistic phi tuyến do có
centering+normalization) không hoàn toàn "smooth" cho xấp xỉ delete-1, nên `SE_jack` bản thân có
thể chưa hoàn toàn chính xác. Không đổi hướng kết luận vì bias luôn **âm** ở cả 4 trường hợp (nghĩa
là θ̄ trung bình các fold thấp hơn θ_full) — nếu bias-correct thì point estimate sẽ dịch xa 0 hơn,
không lại gần 0. Ghi nhận như limitation của inference cho nonlinear-kernel statistic, không phải
bằng chứng chống lại effect direction.

**Kết luận — đóng E6, viết ở mức thận trọng (không causal, không claim CI "perfectly calibrated")**:

> Backbone adversarial response coupling is strongly associated with cross-backbone
> transferability in object detection, and this relationship is more informative than clean
> representation similarity under matched-head controls. The pattern holds across N=49→120→296,
> mean/RMS pooling, and multiple subsampling fractions; at N=296, a delete-1 jackknife computed
> directly on the difference statistic (avoiding the m-out-of-n subsampling rescale question)
> confirms both matched pairs remain positive with the CI excluding zero. This is the first
> mechanism candidate in this project's diagnostic chain (after ‖ΔF‖ in E1 and gradient cosine in
> §19 both failed) that survives a full replication ladder — including a stricter CI method —
> without weakening. No causal claim is made; C_response is measured, not intervened upon.

**Việc đã làm khác quy trình thường lệ của project (đáng ghi lại)**: (1) lần đầu tiên project chạy
**sensitivity check 2 trục** (pool mode + bootstrap_frac) *trước khi* scale N, theo đề nghị của
user — thay vì pilot N nhỏ rồi confirm N lớn như mọi candidate trước (MVC/RCG/DOB/N6-A/N6-B). Cách
này phát hiện được ngay ở N=49 rằng tín hiệu DINO/mean-pool mấp mé CI=0 (đáng ngờ), nhưng đồng thời
cho thấy sign không đổi qua pool mode/frac — đúng tín hiệu "promising, đáng scale" trước khi tốn
compute N=300. (2) Bắt được 1 bug thật (bootstrap-with-replacement bias cho Gram/CKA) mà nếu chạy
thẳng N=300 rồi mới nhìn CI mới phát hiện, đắt hơn nhiều. (3) Audit lại chính phương pháp CI bằng
jackknife sau khi có kết quả N=296 — phát hiện subsampling CI ở N=49 lạc quan hơn thực tế, dù không
đảo kết luận cuối cùng ở N=296.

**Đóng E6** — không chạy thêm sensitivity nào khác cho candidate này. `C_response` là **mechanism
candidate**, chưa phải method design: cần forward pass qua target thật để tính, dùng được cho
diagnostic nhưng không dùng được trong vòng lặp attack black-box thật. Bước tiếp theo chuyển sang
E7 (xem §24) — câu hỏi mới: `C_response` giải thích rất tốt **hướng** transfer trong mỗi matched-pair
(giữ cố định head), nhưng không tự nó giải thích được vì sao **mức nền ASR khác nhau nhiều đến vậy
giữa 2 detector head khác nhau khi cùng gắn Swin backbone** — `dino_swin_l` (C_response=0.815,
ASR=33.6%) so với `mask_rcnn_swin_t` (C_response=0.929, ASR=68.1%): cả 2 chỉ số của Mask đều cao
hơn DINO, nhưng biên độ chênh lệch ASR (68.1−33.6=34.5 điểm) lớn hơn nhiều so với biên độ chênh
lệch C_response (0.929−0.815=0.114) — nói cách khác, quan hệ "C_response → ASR" dường như có
**hệ số/intercept khác nhau tùy detector head**, không phải một hàm số chung cho mọi kiến trúc.
Đây chính là chỗ downstream pipeline (neck/proposal/decoder riêng của từng head) có thể đóng vai
trò **khuếch đại hoặc suy giảm** backbone-level coupling theo cách khác nhau giữa DINO và Mask
R-CNN — câu hỏi E7 muốn trả lời.

**Bước tiếp theo (chưa làm, cần cân nhắc trước khi code)**: đây là **mechanism candidate**, chưa
phải method design — khác biệt quan trọng với N4/N4b (§14, cũng là finding dương nhưng chuyển hoá
thành DOB rồi NO-GO). Rào cản cụ thể: `C_response` như định nghĩa hiện tại cần forward pass qua
chính **target** model để đo — dùng được cho phân tích/diagnostic (đúng vai trò ở đây), nhưng
**không thể tính trong vòng lặp attack thật** (black-box, không có gradient/feature của target).
Bất kỳ method nào muốn khai thác finding này cần một proxy chỉ dùng phía surrogate (vd một tập
surrogate-variant ước lượng "expected response coupling" mà không cần chạm target) — đây chính là
rào cản mà candidate CRA (§16, loại vì overlap SAA) và hướng "cross-representation vulnerability mà
không cần witness model" (§10, Phase N mở đầu) đã né tránh. Chưa quyết định thiết kế cụ thể; cần
scan lại xem có cách nào ước lượng response-coupling chỉ từ phía surrogate (vd qua tập backbone-
variant nội bộ, giống hướng MVC đã thử — nhưng MVC đã NO-GO ở dạng model-space consistency đơn
giản, §9) trước khi code pilot.

## 24. E7: Downstream Amplification — NO-GO

**Câu hỏi**: `C_response` (§23) giải thích tốt **hướng** transfer trong mỗi matched-pair (giữ cố
định head, đổi backbone), nhưng không giải thích được vì sao **mức nền ASR khác nhau nhiều đến vậy
giữa 2 detector head khi cùng gắn Swin**: `dino_swin_l` (C_response=0.815, ASR=33.6%) vs
`mask_rcnn_swin_t` (C_response=0.929, ASR=68.1%) — biên độ chênh ASR (34.5 điểm) không tỷ lệ thuận
đơn giản với biên độ chênh C_response (0.114). Hypothesis: downstream pipeline (neck/proposal/
decoder riêng mỗi head) khuếch đại hoặc suy giảm backbone-level disturbance khác nhau tùy kiến
trúc — `Transfer ≈ Backbone Response Coupling + Downstream Amplification`.

**Phát hiện quan trọng trước khi thiết kế: không cần xây instrumentation mới.** `scripts/
e2_pipeline_attenuation.py` (đã viết từ chuỗi diagnostic E1-E3, §8) **đã có sẵn** đúng phép đo
per-stage cần cho cả 2 kiến trúc — `measure_mask_rcnn()` (backbone→FPN neck→RPN raw→RoI pooled
feat→RoI head output, xử lý đúng confound "proposal selection khác nhau giữa clean/adv" bằng cách
tái dùng proposal chọn trên ảnh clean cho cả 2 forward RoI) và `measure_dino()` (backbone→neck/
input-proj→encoder memory→decoder hidden-states cuối, dùng trực tiếp `model.pre_transformer`/
`forward_encoder`/`pre_decoder`/`forward_decoder` — các method nội bộ mmdet's `DINO`/
`DeformableDETR`, không cần hook mới). Script hiện chỉ chạy cho `mask_rcnn_swin_t` và `dino_swin_l`
(2 model duy nhất tồn tại lúc viết) — **việc còn lại của E7 chủ yếu là thêm `mask_rcnn_r50` và
`dino_r50` vào `MEASURERS`**, chạy lại trên `dev_300` (N=296, tái dùng noise đã crafted của §23,
không craft lại — free compute), rồi định nghĩa + so sánh một quantity "downstream amplification"
giữa R50 và Swin trong mỗi matched-pair.

**Quantity đề xuất** (chưa chốt, cần bàn trước khi code): `A(model) = mean_rel_l2(final_stage) /
mean_rel_l2(backbone_stage)` — tỷ lệ relative-L2 distortion giữa stage cuối cùng đo được (RoI head
output cho Mask-RCNN, decoder hidden-states cuối cho DINO) và stage backbone. `A>1` nghĩa là
downstream pipeline khuếch đại thêm disturbance đã có ở backbone; `A<1` nghĩa là suy giảm. So sánh
`A(mask_rcnn_r50)` vs `A(mask_rcnn_swin_t)`, và `A(dino_r50)` vs `A(dino_swin_l)` — nếu backbone
Swin đi kèm `A` lớn hơn rõ rệt so với R50 (cùng head), đó là bằng chứng downstream amplification là
một biến số thật, độc lập với response-coupling đã đo ở §23.

**Giới hạn đã biết trước khi chạy (ghi lại từ chính docstring gốc của E2, không phải phát hiện
mới)**: so sánh giữa 2 HỌ detector khác nhau (Mask-RCNN 5-stage vs DINO 4-stage, đơn vị/dimension
mỗi stage khác hẳn nhau) chỉ có ý nghĩa **trong nội bộ mỗi họ** (R50 vs Swin, giữ cố định pipeline
structure) — không nên so sánh trực tiếp `A(mask_rcnn_*)` với `A(dino_*)` như hai số cùng thang đo,
chỉ so sánh **độ chênh lệch R50-vs-Swin trong từng họ** với nhau. Decoder checkpoint của DINO cũng
đã được chính E2 ghi chú là so sánh "mềm" hơn RoI (mỗi query slot có thể attend vùng khác nhau giữa
clean/adv qua top-k reference points input-dependent) — không phải strict same-region comparison
như RoI-fix đã làm cho Mask-RCNN.

**Việc cần làm (theo đúng thứ tự, không nhảy cóc)**:
1. Thêm `mask_rcnn_r50: measure_mask_rcnn`, `dino_r50: measure_dino` vào `MEASURERS` trong
   `scripts/e2_pipeline_attenuation.py` (không đổi logic đo, chỉ thêm registry entry).
2. Chạy lại script trên `dev_300` (N=296, noise đã có từ §23 — không craft lại) cho cả 4 model.
3. Tính `A(model)` cho cả 4, so sánh chênh lệch trong mỗi matched-pair.
4. Đối chiếu chênh lệch `A` với chênh lệch `C_response` (§23) và chênh lệch ASR thật — xem
   `Transfer ≈ f(C_response) + g(A)` có giải thích tốt hơn chỉ `C_response` một mình hay không
   (ít nhất ở mức định tính/bảng số, chưa cần fit mô hình formal).
5. GO/NO-GO criterion (cần chốt trước khi chạy, theo đúng discipline project): nếu chênh lệch `A`
   giữa R50/Swin **không nhất quán chiều** giữa 2 matched-pair, hoặc quá nhỏ so với chênh lệch ASR
   quan sát được, đây là NO-GO cho "downstream amplification" như tầng giải thích thứ hai — quay
   lại đọc `C_response` một mình là đủ, không thêm tầng mới.

**Kết quả — NO-GO cho `A` như đã định nghĩa.** Chạy N=296 (`results/e7_pipeline_attenuation_matched.csv`),
`A(model) = mean_rel_l2(final_stage) / mean_rel_l2(backbone_stage)`:

| model | backbone rel_l2 | final-stage rel_l2 | A (rel_l2) | A (cos_dist) | ASR |
|---|---|---|---|---|---|
| mask_rcnn_swin_t | 0.475 | 0.403 (RoI head) | 0.848 | 0.683 | 68.1% |
| mask_rcnn_r50 | 0.987 | 0.996 (RoI head) | **1.010** | 1.318 | 99.1% |
| dino_swin_l | 0.545 | 0.661 (decoder) | **1.212** | 1.499 | 33.6% |
| dino_r50 | 1.175 | 0.943 (decoder) | 0.803 | 0.781 | 99.7% |

Δ`A`(R50−Swin) rel_l2: Mask-RCNN **+0.162** (R50 giữ/khuếch đại disturbance qua pipeline nhiều hơn
Swin-T) nhưng DINO **−0.410** (ngược lại — Swin-L khuếch đại nhiều hơn R50). Cùng chiều đảo ngược
lặp lại khi dùng `cos_dist` thay `rel_l2` (Mask: R50 1.318 > Swin-T 0.683; DINO: Swin-L 1.499 >
R50 0.781) — không phải artifact của một cách đo cụ thể. **Đúng tiêu chí đã pre-register: chiều
không nhất quán giữa 2 matched-pair → NO-GO**, không đổi sang định nghĩa `A` khác để cứu hypothesis.

**Đọc thêm (không đổi verdict, chỉ để hiểu tại sao)**: với Mask-RCNN, *hình dạng* quỹ đạo per-stage
(neck tăng → RPN giảm mạnh → RoI-pooled tăng → RoI-head giảm) giống nhau về mặt tương đối giữa R50
và Swin-T — khác biệt chủ yếu nằm ở **biên độ tuyệt đối** tại backbone (Swin-T thấp hơn R50 nhiều,
0.475 vs 0.987, đúng như E1 đã thấy), không phải ở cách downstream pipeline xử lý nó khác nhau về
*chất*. Với DINO, backbone rel_l2 của R50 (1.175) cao hơn Swin-L (0.545) — ngược hướng so với cặp
Mask-RCNN — nên "R50 nào cũng bị phá backbone nhiều hơn Swin" **không phải pattern chung** (đã biết
từ E1, §8: `mask_rcnn_swin_t` bị phá backbone *ít hơn* `dino_swin_l` dù mAP_drop cao hơn nhiều).
Kết hợp lại: downstream-amplification-ratio đơn giản không phải mảnh giải thích còn thiếu; puzzle
"ASR gap giữa 2 head cùng Swin backbone không tỷ lệ với gap C_response" (mở đầu §24) **vẫn để
ngỏ**, chưa có lời giải từ hướng này.

**Kết luận E7 (đóng, không mở rộng thêm biến thể)**:

> A simple end-to-end amplification ratio (final-stage vs. backbone-stage relative distortion)
> does not show a consistent R50-vs-Swin direction across the two matched-head pairs (Mask R-CNN:
> R50 preserves more disturbance than Swin-T; DINO: Swin-L amplifies more than R50) — under both
> rel_l2 and cos_dist. This specific "downstream amplification" formalization is NOT a viable
> second decomposition term alongside backbone response coupling (§23); per this project's
> discipline, no alternative metric was tried to rescue it. The question of why ASR gaps between
> detector heads on the same Swin backbone don't scale proportionally with their C_response gaps
> remains open.

Không theo đuổi thêm biến thể của `A` (vd chuẩn hoá theo per-stage baseline khác, hay lấy log-ratio
thay vì ratio thô) trong phiên này — nếu muốn tiếp tục câu hỏi decomposition, cần một hướng khác
hẳn cách đo pipeline-attenuation kiểu E2 (vốn đã tự nhận trong docstring gốc là chỉ so sánh nội bộ
từng họ detector, không cùng thang đo giữa 2 họ) chứ không phải sửa công thức `A`.

## 25. E8: Task-Relevant Response Alignment — NO-GO

**Câu hỏi**: sau E6 (GO) và E7 (NO-GO), ba lời giải thích dạng "kích thước disturbance vô hướng"
đều đã thất bại — `‖ΔF‖` thô (E1), gradient cosine tức thời surrogate-vs-target (§19), và tỷ lệ
khuếch đại pipeline vô hướng (§24/E7). Hypothesis mới: không phải **độ lớn** của response mà là
**hướng** — phần `ΔF` nào align với direction mà chính detector đó thực sự nhạy cảm (task-sensitive
direction), bất kể magnitude tổng. Formal: `g_t^l = ∂L_det,t/∂F_t^l` (gradient loss detection của
CHÍNH target, tại backbone feature của chính nó, đo tại ảnh clean), rồi
`P_t^l = |⟨ΔF_t^l, g_t^l⟩| / (‖ΔF_t^l‖·‖g_t^l‖)`. Prediction đã pre-register: `P(mask_rcnn_swin_t) >
P(dino_swin_l)` (khớp ASR 68.1%>33.6%), và quan trọng hơn — `P` phải giải thích transfer ordering
tốt hơn `‖ΔF‖` (E1) và tỷ lệ amplification (E7).

**Engineering — 4 bug thật bắt được qua smoke-test trước khi chạy N=296** (script mới
`scripts/e8_task_relevant_alignment.py`, tái dùng `transfer_attack/losses.py::detector_task_loss`/
`build_gt_data_sample`, vốn trước giờ **chỉ từng chạy trên surrogate cho mi_fgsm**, chưa từng chạy
trên bất kỳ target nào — mọi bug dưới đây là lần đầu code path này chạm 1 target thật):

1. **DINO's `pre_decoder` chỉ tạo `enc_outputs_class`/`enc_outputs_coord`/`dn_meta`** (3 argument
   bắt buộc của `DINOHead.loss()`) **khi `self.training=True`** (mmdet `dino.py:191-213`, gate hoàn
   toàn độc lập với BatchNorm) — ở `model.eval()` (trạng thái chuẩn của cả project) các key này bị
   bỏ hẳn, gây `TypeError`. Sửa bằng `_set_training_keep_norm_frozen()`: bật `model.train()` (để
   `self.training` đúng cho logic denoising) rồi **ngay lập tức ép mọi `BatchNorm`/`Dropout` submodule
   về lại `.eval()`** — cần thiết vì `dino_r50`/`mask_rcnn_r50` dùng BatchNorm **không frozen**
   (`norm_cfg requires_grad=True/False` nhưng vẫn là real BN, không phải GN/LN như 2 biến thể Swin),
   nên `model.train()` trần sẽ khiến BN dùng batch-statistics của **đúng 1 ảnh** thay vì running-stats
   đã calibrate — sẽ tính gradient qua một forward pass khác hẳn (và không ổn định) so với forward
   pass thật đang được dùng để đo ASR/mAP ở mọi nơi khác trong project.
2. **`gt_boxes` chưa `.to(device)`** trước khi đưa vào `build_gt_data_sample` (khác `attack.py`'s
   `craft_one_image`, vốn làm việc này ngay đầu hàm) — gây `RuntimeError` device-mismatch sâu bên
   trong `CdnQueryGenerator` của DINO (denoising query embedding lookup).
3. **`mask_rcnn_r50`'s `out_indices=(0,1,2,3)` bao gồm stage bị `frozen_stages=1` đóng băng** (layer1,
   `requires_grad=False` hoàn toàn vì cả path lẫn input đều không cần grad) — `retain_grad()` trên
   tensor đó crash. Sửa: chỉ `retain_grad()` tensor nào `requires_grad=True`; stage bị freeze coi
   như đóng góp gradient = 0 vào phép đo alignment (đúng về ngữ nghĩa — không có tín hiệu task-gradient
   nào từ một nhánh đã đóng băng, không phải giá trị đoán mò).
4. **Mask R-CNN's `mask_head.loss()` cần `gt_instances.masks`** (segmentation GT) mà
   `build_gt_data_sample` chưa bao giờ set (project chỉ đánh giá box mAP/ASR, không segmentation) —
   `AttributeError`. Sửa: tạm gỡ `roi_head.mask_head` (đặt `None`) trước khi gọi `detector_task_loss`,
   khôi phục lại sau — đúng phạm vi "chỉ tính box-detection task loss", không phải workaround để né
   thiếu data.

Sau khi sửa cả 4, smoke-test N=3 trên toàn bộ 8 model (bao gồm cả `dino_swin_l`, model rủi ro nhất
vì backbone có `with_cp=True`) cho giá trị hợp lệ (không NaN, không None) trước khi chạy N=296.

**Kết quả (N=296, `results/e8_task_relevant_alignment.csv`)**:

| target | group | mean_P | ASR |
|---|---|---|---|
| fcos_r50 | A | 0.0079 | 97.5 |
| deformable_detr | A | 0.0062 | 99.1 |
| yolov3_d53 | B | 0.0100 | 83.3 |
| yolox_l | B | 0.0084 | 69.3 |
| mask_rcnn_swin_t | C | 0.0142 | 68.1 |
| dino_swin_l | C | 0.0067 | 33.6 |
| dino_r50 | D | 0.0070 | 99.7 |
| mask_rcnn_r50 | D | 0.0136 | 99.1 |

**Matched-pair**: DINO delta = P(R50)−P(SwinL) = 0.0070−0.0067 = **+0.0003** (CI hai bên chồng lấn
gần như hoàn toàn: [0.0067,0.0074] vs [0.0064,0.0070]); Mask-RCNN delta = P(R50)−P(SwinT) =
0.0136−0.0142 = **−0.0005** (CI cũng chồng lấn: [0.0130,0.0143] vs [0.0134,0.0149]). Cả hai delta
**không tách biệt khỏi 0** trong phạm vi CI, dù ASR giữa 2 vế mỗi pair chênh nhau rất lớn (66 điểm
cho DINO, 31 điểm cho Mask-RCNN). **Cross-target**: `corr(mean_P, ASR)` trên 8 target = **+0.054** —
gần như không có quan hệ tuyến tính nào (yếu hơn cả `corr(S_clean,ASR)` NO-GO của E6, §23).

**Chẩn đoán thêm — độ lớn tuyệt đối của P gần sát "sàn nhiễu ngẫu nhiên"**: với dimension đã pool
(concat mọi backbone stage, pool 7×7) từ ~70,560 (`mask_rcnn_swin_t`) đến ~188,160 (`mask_rcnn_r50`),
cosine kỳ vọng giữa **2 vector ngẫu nhiên không liên quan** trong không gian chiều `D` là
`~1/√D ≈ 0.0023–0.0038`. Giá trị `P` quan sát được (0.0062–0.0142) chỉ cao hơn sàn này **2-6 lần** —
có một tín hiệu khác-0 thật (không phải thuần túy random), nhưng biên độ quá nhỏ so với sàn nhiễu
để mang đủ thông tin phân biệt matched-pair hay tương quan với ASR. Đây là gợi ý kỹ thuật quan trọng
cho bất kỳ ai muốn thử lại quantity dạng này: cosine-alignment thô trên vector pool+concat toàn bộ
backbone (chiều cực cao) có thể bị "curse of dimensionality" che khuất bất kỳ tín hiệu thật nào —
nếu tồn tại, tín hiệu đó nhiều khả năng nằm trong một **subspace hạng thấp** (như chính hypothesis
gốc đã gợi ý "hoặc energy projected onto a low-rank task-sensitive subspace"), không phải trải đều
trên toàn bộ chiều pooled.

**Kết luận E8 (đóng, không đổi metric để cứu hypothesis)**:

> A raw cosine-alignment between the observed backbone response difference and the target's own
> detection-loss gradient (both pooled+concatenated across all backbone stages) does not
> distinguish R50 from Swin within either matched-head pair (both deltas statistically
> indistinguishable from zero despite 31-66 point ASR gaps), and does not correlate with ASR across
> the 8 targets (r=0.054). The observed magnitudes (0.006-0.014) are only 2-6x the theoretical
> random-vector cosine floor (~1/sqrt(D) for the ~70k-190k pooled dimensionality used here) --
> consistent with a real but very weak signal that this specific high-dimensional raw-cosine
> formalization cannot usefully extract. Per this project's discipline, no alternative metric
> (e.g. low-rank subspace projection) was tried within E8 itself to rescue it; that would be a
> separate, newly pre-registered experiment, not a patch to this one.

**Trạng thái sau E6→E7→E8**: `C_response` (§23) vẫn là mechanism candidate duy nhất sống sót qua
audit. Ba hướng giải thích "phần transfer mà C_response chưa giải thích hết" đều NO-GO: downstream
amplification vô hướng (E7), task-relevant alignment thô (E8). Câu hỏi mở đầu §24 (tại sao ASR gap
giữa 2 head cùng Swin backbone không tỷ lệ với C_response gap) **vẫn để ngỏ** sau cả 2 lần thử.

## 26. E9: Response-Coupling Intervention — NO-GO

**Chuyển từ correlation sang intervention**: sau E6 (GO)→E7 (NO-GO)→E8 (NO-GO), thay vì tiếp tục
tìm thêm biến giải thích tương quan với `C_response`, câu hỏi chuyển thành causal: nếu chủ động
craft một perturbation làm tăng `C_response`, ASR có tăng theo không — đặc biệt trên target
cross-backbone khó (Swin)? Ràng buộc nền tảng: `C_response` cần forward qua target thật để tính
(§23), nên trong vòng lặp crafting black-box thật (chỉ chạm surrogate `faster_rcnn_r50`, đúng
protocol xuyên suốt project), **không thể tối ưu trực tiếp `C_response`** — chỉ có thể tối ưu một
**proxy phía surrogate**, rồi ĐO `C_response`/ASR thật trên target sau khi crafting xong (đo, không
phải objective).

**Proxy đã chọn (xác nhận với user trước khi code)**: spectral/low-frequency bias — hồi sinh hướng
đã parked ở §11 ("spectral-*constrained* optimization... OPEN, chưa bị falsify") với motivation mới
từ E6 (cấu trúc tần số thấp có thể là thứ CNN và Transformer backbone cùng dựa vào tương tự nhau
hơn; E4 cũng đã thấy low band giữ tỷ lệ ASR cao hơn tỷ lệ energy của nó). Không đổi tên/hồi sinh cơ
chế MVC đã NO-GO (§9) — proxy này khác hẳn về bản chất (spectral projection lên chính `δ`, không
phải ép đồng thuận đa biến thể model).

**Thiết kế** (script mới `scripts/n9_response_coupling_intervention.py`): 2 trajectory lockstep
(cùng RNG snapshot mỗi step, cùng quy ước `n6b_path_pilot.py`) trên cùng ảnh — `delta_base` (OSFD
chuẩn) và `delta_coupling` (OSFD chuẩn, nhưng sau mỗi step, `δ` bị chiếu qua low-pass radial FFT
mask, chỉ giữ băng tần **low** — đúng ngưỡng `radius_frac ≤ 1/3` mà E4 đã định nghĩa sẵn, không phát
minh hyperparameter mới). Compute-matched theo construction (cùng số step, cùng gradient computation
cho cả 2 trajectory; overhead thêm chỉ là 1 FFT+IFFT/step, không đáng kể).

**GO/NO-GO đã pre-register**: GO cần `C_response(coupling) > C_response(base)` trên target chưa
thấy VÀ ASR/mAP-drop tăng cùng chiều VÀ effect mạnh hơn ở target cross-backbone. NO-GO nếu
`C_response` tăng mà ASR không tăng, hoặc ngược lại, hoặc chỉ cải thiện ở target cùng họ CNN.

**Kết quả (pilot N=20, 100 step, cả 8 target, `results/n9_response_coupling_pilot_n20.csv`)**:

| target | group | ΔASR | 95% CI | ΔC_response |
|---|---|---|---|---|
| fcos_r50 | A | −8.20 | [−12.20,−3.85] | −0.0090 |
| deformable_detr | A | −4.55 | [−6.78,−1.47] | +0.0032 |
| yolov3_d53 | B | −35.71 | [−45.31,−25.49] | −0.0074 |
| yolox_l | B | **−47.52** | [−55.26,−40.35] | −0.0091 |
| mask_rcnn_swin_t | C | −27.84 | [−34.85,−18.06] | −0.0071 |
| dino_swin_l | C | −7.00 | [−9.72,−2.63] | +0.0092 |
| dino_r50 | D | −3.19 | [−4.76,0.00] | +0.0033 |
| mask_rcnn_r50 | D | −4.44 | [−6.56,−2.60] | +0.0001 |

**Đọc kết quả — NO-GO rõ ràng, cả 2 vế đều fail, không cần scale N**: `ΔASR` **âm ở cả 8/8 target**
(CI hầu hết không cắt 0), lớn nhất đúng ở nhóm CNN dễ transfer (`yolox_l` −47.5, `yolov3_d53`
−35.7) — ngược hẳn hypothesis (kỳ vọng cross-backbone Swin mới là nơi cải thiện). `ΔC_response`
dao động rất nhỏ quanh 0 (−0.009 đến +0.009), không có xu hướng tăng nhất quán — proxy spectral
low-pass **không hề làm tăng response coupling một cách đáng tin cậy**, và ngay cả ở 3 target có
`ΔC_response` dương nhẹ (`deformable_detr`, `dino_swin_l`, `dino_r50`), `ΔASR` vẫn âm — hai đại
lượng không đi cùng chiều ở bất kỳ target nào theo hướng GO cần. Pattern đồng nhất chiều trên toàn
bộ 8 target (không phải noise ngẫu nhiên) — không cần chạy N=296 để biết thêm.

**Diễn giải cơ chế thất bại (không phải để cứu hypothesis, chỉ để hiểu)**: chiếu `δ` về đúng 1 băng
tần thấp **ở MỌI step trong suốt 100 step** rất có thể giới hạn nghiêm trọng không gian tìm kiếm của
chính optimization — khớp với phát hiện E4b (sau L∞-renormalize, ngay cả band low cũng mất phần lớn
hiệu quả tấn công, "not sufficient evidence of an intrinsic shared vulnerable subspace"). Việc ép
cứng suốt cả trajectory khác hẳn E4's post-hoc decompose-rồi-inject (chỉ áp dụng 1 lần lên noise đã
crafted xong) — ở đây constraint tác động vào chính quá trình ascent, nhiều khả năng làm giảm hiệu
quả ascent nói chung (giải thích ASR giảm đều ở MỌI target, kể cả surrogate-gần-họ) chứ không phải
chọn lọc "chỉ mất phần disruption CNN-specific".

**Kết luận E9 (đóng, không đổi proxy để cứu hypothesis)**:

> A hard spectral low-pass projection applied to the perturbation at every optimization step
> (reviving S11's parked spectral-constrained-optimization direction) decreases ASR on all 8
> targets (largest drops on same-family CNN targets, not the hard cross-backbone ones the
> hypothesis targeted) while leaving C_response essentially unchanged (|delta| < 0.01, no
> consistent sign). Neither half of the causal hypothesis holds: the surrogate-only proxy neither
> reliably increases C_response nor, where it marginally does, does ASR follow. Per this project's
> discipline, this closes the "explanation-by-correlation to intervention" line for THIS specific
> proxy without trying alternative formulations (e.g. a soft frequency-domain regularizer instead
> of a hard per-step projection, or a different radius) within the same experiment — any such
> variant would need to be a separately pre-registered follow-up.

**Trạng thái sau E6→E7→E8→E9**: `C_response` vẫn là mechanism candidate duy nhất đứng vững của
project (§23). Ba hướng mở rộng nó — giải thích residual gap (E7, E8) và can thiệp trực tiếp lên nó
(E9) — đều NO-GO. Không có proxy phía surrogate nào đã thử làm tăng được `C_response` một cách đáng
tin cậy; đây là rào cản cụ thể (không phải giả thuyết) cho bất kỳ ai muốn biến `C_response` thành
một method tấn công thật, không chỉ một diagnostic.

## 27. E10: Cross-Architecture Semantic-Region Relational Geometry — STRONG GO (pilot, N=49)

Sau chuỗi E6→E9 (candidate mechanism `C_response` đứng vững nhưng không tìm được cách biến thành
objective tấn công), E10 đổi hẳn trục đo lường thay vì tiếp tục mở rộng `C_response`: mọi diagnostic
từ E1 đến E9 đều so sánh **raw feature similarity** (cosine/L2 giữa toàn bộ feature map hoặc giữa
Gram matrix của nó) — E10 giả thuyết rằng cái thật sự sống sót qua đổi kiến trúc backbone không phải
độ lớn/hướng của feature tại một điểm, mà **quan hệ tương đối giữa các vùng ngữ nghĩa của cùng một
object** (interior/boundary/near-background/far-background) — một đại lượng không so sánh vector
feature trực tiếp giữa các model (vốn sống ở không gian khác nhau), nên "architecture-agnostic by
construction" thay vì phải suy luận gián tiếp qua Gram/CKA như E6.

**Thiết kế** (script mới `scripts/e10_relational_geometry.py`, theo đúng đặc tả người dùng đưa ra
trước khi chạy — không đổi sau khi thấy kết quả): với mỗi GT box, định nghĩa 4 vùng bằng
shrink/expand hình chữ nhật của chính box đó — **O** (interior, shrink 12.5%/cạnh), **E** (boundary,
`box - O`), **B_n** (near-background, `expand(box, 25%) - box`), **B_f** (far-background, phần ảnh
còn lại) — **B_n/B_f loại trừ tường minh vùng chồng lấn với MỌI GT box khác trong ảnh** (đúng yêu
cầu tránh "background" lẫn object khác). Pool feature mỗi vùng bằng weighted-mean qua
`F.interpolate(mode="area")` resize mask xuống đúng kích thước từng backbone stage (không threshold
cứng — giữ nguyên tỷ lệ phủ phân số), bỏ qua (object, stage) nếu vùng nào phủ dưới 1 stage-cell.
Relational signature `r = [d(O,E), d(O,Bn), d(O,Bf), d(E,Bn), d(E,Bf), d(Bn,Bf)]`, `d = 1 - cosine`.
Toàn bộ phân tích dùng **stage cuối cùng** của mỗi backbone (tương đối theo số stage riêng của từng
model, không phải index tuyệt đối — mọi model trong registry đều có 4 stage nên không có vấn đề so
sánh lệch độ sâu).

6 model: `faster_rcnn_r50`, `dino_r50`, `mask_rcnn_r50` (family "R50"), `yolox_l` (family "CSP", chỉ
1 model nên không tính được within-family consistency — cột `CSP_consistency` = NaN có chủ đích),
`mask_rcnn_swin_t`, `dino_swin_l` (family "Swin"). Dataset `dev_50` (N=49 sau khi 1 ảnh bị skip do
0 GT box hợp lệ, giống mọi run khác trên manifest này). Noise: `osfd` mặc định (k=3, RRB on, recraft
mới trên checkout này — xem HANDOFF.md) cho Q1/Q2, cộng 2 arm factorial của E3 tái tạo đúng
hyperparameter cũ (`e3_k1_norrb`: k=1 RRB off, `e3_k1_rrb`: k=1 RRB on) cho Q3 — cả 3 bộ đều craft
mới trên `dev_50` vì `results/` gitignored trống hoàn toàn trên checkout này (đã verify ASR khớp
biên ~1-2 điểm so với log lịch sử tương ứng, xem bảng Q2/Q3 dưới). Ngưỡng pre-register (chốt trước
khi chạy, không đổi sau khi thấy số): GO-A cần ≥2/6 relation có cả `global` VÀ `cross_family`
Spearman rho > 0.30; GO-B/GO-C đọc dấu + độ khớp hướng giữa disruption và ASR, không có ngưỡng số
cứng (theo đúng đặc tả gốc — "discovery", không phải confirmatory test).

**Table A — clean relational invariance** (`results/e10_table_A_clean_invariance.csv`; Spearman rho
của từng relation qua các cặp model, N=194 object có dữ liệu hợp lệ ở cả 6 model / 337 object khả
dụng tổng cộng):

| relation | R50 (3 cặp) | CSP | Swin (1 cặp) | cross-family | global |
|---|---:|---:|---:|---:|---:|
| O↔E | 0.893 | NaN | 0.556 | 0.768 | 0.779 |
| O↔nearBG | 0.832 | NaN | 0.525 | 0.641 | 0.672 |
| O↔farBG | 0.541 | NaN | 0.088 | 0.304 | 0.337 |
| E↔nearBG | 0.886 | NaN | 0.681 | 0.734 | 0.761 |
| E↔farBG | 0.594 | NaN | 0.323 | 0.466 | 0.482 |
| nearBG↔farBG | 0.755 | NaN | 0.645 | 0.687 | 0.698 |

**GO-A: PASS rõ ràng, 6/6 relation đạt ngưỡng** (cần ≥2/6) — kể cả `O↔farBG`, relation yếu nhất,
vẫn qua ngưỡng (`cross_family=0.304`). Đáng chú ý: cross-family consistency (R50 so với Swin/CSP,
đúng phép so sánh mà GO-1 criterion yêu cầu — "không chỉ R50↔R50") **cao gần bằng within-R50-family**
ở phần lớn relation (`O↔E`: 0.768 vs 0.893; `E↔nearBG`: 0.734 vs 0.886) — quan hệ tương đối giữa các
vùng ngữ nghĩa thực sự là một tín hiệu per-object tái lập được across ResNet/CSPDarknet/Swin, không
chỉ trùng hợp trong 1 họ backbone. `Swin_consistency` (chỉ 1 cặp `mask_rcnn_swin_t` vs `dino_swin_l`)
nhìn chung thấp hơn R50 (kỳ vọng — ít cặp hơn, ước lượng nhiễu hơn; `O↔farBG` Swin=0.088 gần 0 là
điểm yếu nhất của bảng, cần đọc thận trọng ở N nhỏ này).

**Table B — OSFD attack disruption, giới hạn vào 6 relation shared** (Q1 xác nhận cả 6 đều shared
nên `disrupt_shared == disrupt_all6` ở pilot này; `results/e10_table_B_attack_disruption.csv`):

| model | family | ASR | disrupt_shared |
|---|---|---:|---:|
| faster_rcnn_r50 | R50 (surrogate) | 99.1 | 0.4040 |
| dino_r50 | R50 | 98.3 | 0.3459 |
| mask_rcnn_r50 | R50 | 99.1 | 0.3862 |
| yolox_l | CSP | 70.9 | 0.3369 |
| mask_rcnn_swin_t | Swin | 69.4 | 0.0388 |
| dino_swin_l | Swin | 28.3 | 0.1918 |

`corr(ASR, disrupt_shared)` qua 6 target = **+0.621**. Matched-pair (cùng head/decoder, chỉ đổi
backbone — cặp evidence chính xuyên suốt project từ §21):

- **DINO**: `disrupt_shared(dino_r50)=0.346 > disrupt_shared(dino_swin_l)=0.192`, cùng chiều với
  `ASR(dino_r50)=98.3 > ASR(dino_swin_l)=28.3` → **khớp hướng**.
- **Mask R-CNN**: `disrupt_shared(mask_rcnn_r50)=0.386 > disrupt_shared(mask_rcnn_swin_t)=0.039`
  (gấp ~10 lần, chênh lệch lớn nhất trong toàn bảng), cùng chiều với
  `ASR(mask_rcnn_r50)=99.1 > ASR(mask_rcnn_swin_t)=69.4` → **khớp hướng**.

**GO-B: PASS** — cả correlation tổng lẫn cả 2 matched-pair đều đúng hướng, không có trường hợp
ngược. `mask_rcnn_swin_t` có `disrupt_shared` thấp bất thường (0.039, thấp hơn cả `dino_swin_l`) dù
ASR của nó cao hơn `dino_swin_l` (69.4 > 28.3) — ngoại lệ duy nhất trong bảng, đáng ghi nhận nhưng
không đủ để đảo verdict (matched-pair so sánh cùng-head là bằng chứng chính, không phải so sánh
chéo-head).

**Table C — RRB mechanism, disruption giới hạn vào 6 relation shared**
(`results/e10_table_C_rrb_mechanism.csv`; ASR fetch mới trên noise recraft, không lấy từ log E3 cũ
vì E3 cũ chưa có `dino_r50`/`mask_rcnn_r50` trong registry — số ASR ở đây cho `mask_rcnn_swin_t`
(noRRB=19.0, RRB=46.7) khớp biên ~vài điểm với con số lịch sử §8's E3 factorial, xác nhận recraft
đúng config):

| model | family | Δ disruption (RRB−noRRB) | Δ ASR (RRB−noRRB) | khớp dấu? |
|---|---|---:|---:|---|
| faster_rcnn_r50 | R50 | −0.176 | −9.9 | ✅ |
| dino_r50 | R50 | −0.020 | −8.4 | ✅ |
| mask_rcnn_r50 | R50 | −0.155 | −11.7 | ✅ |
| yolox_l | CSP | +0.043 | +28.7 | ✅ |
| mask_rcnn_swin_t | Swin | +0.017 | +27.7 | ✅ |
| dino_swin_l | Swin | +0.066 | +14.3 | ✅ |

**GO-C: PASS, 6/6 model khớp dấu** — và đáng chú ý hơn cả tỷ lệ 6/6: đây là khớp dấu **theo cả 2
hướng**, không phải cùng một chiều dương giả tạo. Trên 3 target họ R50 (đã gần ceiling ASR ~99% dù
không có RRB, k=1), thêm RRB làm **giảm cả disruption lẫn ASR** — RRB "tốn" một phần hiệu quả vào
việc tạo augmented view thay vì tối đa hoá tấn công lên 1 view duy nhất khi target đã gần bão hoà.
Trên 3 target khó (CSP/Swin, ASR còn nhiều dư địa), thêm RRB làm **tăng cả hai** — đúng như E3 (§8)
đã quan sát ở mức ASR thô. Cùng MỘT đại lượng vô hướng (disruption của đúng 6 relation đã xác định
là shared ở Q1) giải thích được ASR di chuyển **cả lên lẫn xuống** tuỳ target — đây là bằng chứng
mạnh hơn nhiều so với việc chỉ khớp dấu một chiều, vì loại trừ khả năng "khớp dấu" chỉ là trùng hợp
do hầu hết delta cùng dấu ngẫu nhiên.

**Kết luận E10 (cả 3 tầng pass, mạnh và sạch — không cần rescue/metric-shop)**:

> A 6-dimensional relational signature between an object's interior, boundary, near-background and
> far-background — built purely from within-image region comparisons, never comparing raw feature
> vectors across architectures — is a reproducible per-object signal across ResNet, CSPDarknet and
> Swin backbones (GO-A, 6/6 relations, cross-family rho up to 0.77). OSFD's clean→adversarial
> disruption of exactly this shared signature tracks ASR both in aggregate (r=0.62 across 6 targets)
> and in both same-head/different-backbone matched pairs (GO-B). RRB's effect on this same shared-
> relation disruption metric moves in the same direction as its effect on ASR for all 6 models,
> including the sign flip between near-ceiling R50 targets (RRB slightly hurts both) and harder
> CSP/Swin targets (RRB helps both) (GO-C). This is the first mechanism in the project's diagnostic
> history (E1 through E9) where a single scalar explains attack success moving in BOTH directions
> across a manipulation (RRB on/off), not just a one-directional correlation.

**Giới hạn cần đọc kèm trước khi scale**: (1) N=49 (`dev_50`) là pilot đúng như kế hoạch, chưa phải
confirm — Swin-family trong Table A chỉ có 1 cặp model nên ước lượng nhiễu hơn hẳn R50 (3 cặp); (2)
`O↔farBG` là relation yếu nhất (cross_family=0.304, sát ngưỡng) — nếu scale N=300 cho kết quả tụt
dưới 0.30 thì shared-set sẽ còn 5/6 thay vì 6/6, cần re-tính Table B/C với shared-set mới chứ không
giữ nguyên kết luận cũ; (3) `mask_rcnn_swin_t` có `disrupt_shared` thấp bất thường relative đến ASR
của nó trong Table B — chưa có lời giải, không ảnh hưởng verdict nhưng đáng theo dõi ở N lớn hơn;
(4) toàn bộ 3 GO đều dựa trên **stage cuối** duy nhất — chưa kiểm tra các stage nông hơn có cho
pattern khác không (dữ liệu đã có sẵn trong `results/e10_relational_clean.csv`/`_delta.csv`, chỉ
chưa phân tích).

**Bước tiếp theo (chưa làm, cần xác nhận với user trước khi chạy — compute cost đáng kể, ~6x pilot
này tức nhiều giờ)**: đúng kế hoạch pre-registered ban đầu, GO ở pilot N=49 nghĩa là scale lên
`dev_300` để confirm trước khi bắt đầu thiết kế attack objective mới dựa trên relational signature
này.

## 28. E10 dev_300 confirmation — STRONG GO giữ vững (N=296), 1 relation rớt khỏi shared-set đúng như dự đoán

Theo đúng điều kiện user đặt ra trước khi scale: **không đổi bất kỳ định nghĩa/metric/hypothesis
nào của E10** (FROZEN SPEC ở đầu `scripts/e10_relational_geometry.py` giữ nguyên nguyên văn), chỉ
được tối ưu engineering và phải chứng minh tương đương số học trước khi chạy `dev_300`.

**Bước 0 — Equivalence check.** Script hiện tại (`scripts/e10_relational_geometry.py`) hoá ra đã
sẵn 2 optimization mà "Engineering notes" của `HANDOFF.md` liệt kê là "chưa sửa" (mask chỉ build 1
lần/ảnh qua dict `image_data`, và `pool_regions_batched` gộp interpolate cho cả 4 vùng thay vì gọi
riêng) — nhiều khả năng phiên trước đã sửa xong ngay trước khi commit nhưng quên cập nhật câu chữ
"chưa sửa" trong `HANDOFF.md`. Vì không có reference CSV nào của bản chưa-tối-ưu được lưu lại
(`results/` gitignored, chưa từng tồn tại `results/e10_pilot_reference/` mà FROZEN SPEC comment
nhắc tới), việc "verify equivalence" được làm bằng cách viết một harness riêng
(`scripts/e10_equivalence_check.py`) dựng lại đúng code path naive cũ (`pool_regions_one_stage`
gọi riêng lẻ từng vùng, build mask lặp lại) và so trực tiếp với code path hiện tại trên cùng
feature đã load (không qua noise/RNG) — **kết quả: `max_abs_diff = 0.0` tuyệt đối** trên 17,838 giá
trị relation, 3 backbone family (R50/Swin-two-stage/Swin-DINO-decoder). Hai optimization là
bit-exact, không phải "gần đúng trong tolerance".

**Bước 1 — Reproduce pilot trên `dev_50` bằng code đã tối ưu** (noise recraft mới, seed=42 mặc
định của `craft.py`, không phải file tensor cũ vì `results/` trống trên checkout này). Table A
(không phụ thuộc RNG) khớp pilot tới 3-4 số thập phân (`O↔E` global 0.7788 vs pilot 0.779,
`E↔nearBG` 0.7610 vs 0.761, ...). Table B: `corr(ASR, disrupt_shared)` rerun = 0.625 (pilot 0.621),
matched-pair giữ đúng hướng và tỷ lệ (Mask R50/Swin-T gap ~10.6x, pilot ~9.9x). Table C: 6/6 khớp
dấu, đúng pattern đảo chiều cũ. **Kết luận bước 1: optimized script reproduce đúng pilot, an toàn
để chạy `dev_300`.** Thời gian chạy full 6-model trên `dev_50`: **5 phút 49 giây** — nhanh hơn ~8
lần so với con số ~48 phút `HANDOFF.md` ghi cho bản trước khi tối ưu.

**Bước 2 — Craft `dev_300` (3 noise variant, 296/300 ảnh mỗi bộ, 4 ảnh skip do 0 GT box hợp lệ,
giống mọi run khác trên manifest này)**: `osfd` (k=3, RRB on) 27m28s, `e3_k1_norrb` (k=1, no-RRB)
13m33s, `e3_k1_rrb` (k=1, RRB on) 27m35s — tổng ~68m36s.

**Bước 3 — Chạy `e10_relational_geometry.py` trên `dev_300`**: **32 phút 59 giây** (so với ước tính
ngoại suy tuyến tính ~35 phút từ `dev_50` — khớp gần đúng, không có chi phí phi tuyến bất ngờ khi
scale 6x). N=1206 object có dữ liệu hợp lệ ở cả 6 model tại last-stage (so với 194 ở pilot,
đúng tỷ lệ ~6.2x).

**Table A — clean relational invariance (`dev_300`, N=1206 object)**:

| relation | R50 (3 cặp) | Swin (1 cặp) | cross-family | global | pilot global (N=49) |
|---|---:|---:|---:|---:|---:|
| O↔E | 0.859 | 0.559 | 0.742 | 0.753 | 0.779 |
| O↔nearBG | 0.834 | 0.484 | 0.649 | 0.675 | 0.672 |
| O↔farBG | 0.530 | 0.161 | **0.276** | 0.319 | 0.337 |
| E↔nearBG | 0.875 | 0.729 | 0.763 | 0.783 | 0.761 |
| E↔farBG | 0.550 | 0.417 | 0.461 | 0.476 | 0.482 |
| nearBG↔farBG | 0.764 | 0.680 | 0.720 | 0.726 | 0.698 |

**GO-A: PASS nhưng shared-set co lại còn 5/6** — `O↔farBG`'s `cross_family_consistency` tụt từ
0.304 (pilot, sát ngưỡng) xuống **0.276 (dưới `GO1_CONSISTENCY_THR=0.30`)**, nên relation này
**không còn nằm trong shared-set** dùng cho Table B/C (5 relation còn lại: `O↔E`, `O↔nearBG`,
`E↔nearBG`, `E↔farBG`, `nearBG↔farBG`). Đây **đúng chính xác** kịch bản đã cảnh báo trước ở giới
hạn #2 của §27 ("nếu scale N=300 cho kết quả tụt dưới 0.30 thì shared-set sẽ còn 5/6") — không phải
một bất ngờ hay một lần rescue post-hoc, mà là điều đã được pre-register là có thể xảy ra. Điều
kiện GO-A gốc (`≥2/6 relation qua ngưỡng`, `GO1_MIN_RELATIONS=2`) vẫn PASS rất thoải mái (5/6 ≫ 2).

**Table B — attack disruption, giới hạn vào 5 relation shared mới**:

| model | family | ASR | disrupt_shared (5-rel) | pilot disrupt_shared (6-rel, N=49) |
|---|---|---:|---:|---:|
| faster_rcnn_r50 | R50 (surrogate) | 99.37 | 0.3135 | 0.4040 |
| dino_r50 | R50 | 99.45 | 0.3043 | 0.3459 |
| mask_rcnn_r50 | R50 | 99.03 | 0.3083 | 0.3862 |
| yolox_l | CSP | 67.88 | 0.2598 | 0.3369 |
| mask_rcnn_swin_t | Swin | 68.62 | 0.0291 | 0.0388 |
| dino_swin_l | Swin | 32.34 | 0.1504 | 0.1918 |

(Giá trị tuyệt đối `disrupt_shared` giảm nhẹ so với pilot vì loại `O↔farBG` — relation có magnitude
delta lớn nhất — khỏi tổng L2, không phải vì disruption thực sự yếu đi; so sánh đúng phải là
tương quan/thứ tự, không phải giá trị tuyệt đối.)

`corr(ASR, disrupt_shared)` qua 6 target = **+0.660** (pilot N=49: +0.621; rerun `dev_50`: +0.625)
— **tăng nhẹ khi scale lên N lớn hơn**, không suy yếu. Matched-pair vẫn khớp hướng cả 2 cặp:

- **DINO**: `disrupt_shared(dino_r50)=0.304 > disrupt_shared(dino_swin_l)=0.150`, cùng chiều
  `ASR(dino_r50)=99.45 > ASR(dino_swin_l)=32.34` → khớp.
- **Mask R-CNN**: `disrupt_shared(mask_rcnn_r50)=0.308 > disrupt_shared(mask_rcnn_swin_t)=0.029`
  (~10.6x, pilot ~9.9x — gap giữ nguyên độ lớn) cùng chiều `ASR(mask_rcnn_r50)=99.03 >
  ASR(mask_rcnn_swin_t)=68.62` → khớp.

**GO-B: PASS, mạnh hơn pilot chứ không yếu đi.** `mask_rcnn_swin_t`'s `disrupt_shared` thấp bất
thường so với ASR của nó (điểm ngoại lệ đã ghi nhận ở §27) **vẫn còn nguyên ở N=296** (0.029, thấp
hơn cả `dino_swin_l` dù ASR cao hơn) — chưa có lời giải, không ảnh hưởng verdict nhưng vẫn là câu
hỏi mở.

**Table C — RRB mechanism, giới hạn vào 5 relation shared mới**:

| model | family | Δ disruption (RRB−noRRB) | Δ ASR (RRB−noRRB) | khớp dấu? | pilot Δ disruption (N=49) | pilot Δ ASR |
|---|---|---:|---:|---|---:|---:|
| faster_rcnn_r50 | R50 | −0.115 | −10.7 | ✅ | −0.176 | −9.9 |
| dino_r50 | R50 | −0.022 | −8.3 | ✅ | −0.020 | −8.4 |
| mask_rcnn_r50 | R50 | −0.098 | −10.9 | ✅ | −0.155 | −11.7 |
| yolox_l | CSP | +0.034 | +22.2 | ✅ | +0.043 | +28.7 |
| mask_rcnn_swin_t | Swin | +0.015 | +26.4 | ✅ | +0.017 | +27.7 |
| dino_swin_l | Swin | +0.043 | +13.6 | ✅ | +0.066 | +14.3 |

**GO-C: PASS, 6/6 khớp dấu — reproduce chính xác pattern đảo chiều đã thấy ở pilot**: 3 target họ
R50 (đã gần ceiling ASR ~97-99% dù RRB off) — thêm RRB làm giảm cả disruption lẫn ASR; 3 target khó
(CSP/Swin, còn nhiều dư địa ASR) — thêm RRB làm tăng cả hai. Magnitude ở N=296 gần với pilot N=49,
không có target nào đổi dấu.

**Kết luận E10 dev_300 (confirmation, không phải discovery mới)**:

> Scaling E10 from N=49 to N=296 confirms GO-B (correlation strengthens from 0.621 to 0.660,
> both matched pairs hold with unchanged effect sizes) and GO-C (6/6 sign match reproduced exactly,
> including the R50-saturates/CSP-Swin-improves flip) with no weakening. GO-A downgrades from 6/6
> to 5/6 shared relations exactly as flagged as a risk before scaling (`O↔farBG`'s cross-family
> consistency crosses below the pre-registered 0.30 threshold, 0.304→0.276) — this is an honest
> shrinkage of the invariant set, not a falsification: the GO-A criterion itself (≥2/6) still
> passes comfortably, and the two relations the user flagged as most interesting going in
> (`O↔E`=0.742, `E↔nearBG`=0.763 cross-family) are among the strongest survivors, unchanged in
> rank. The relational-geometry mechanism is confirmed at scale, with the `O↔farBG` component
> specifically identified as not architecture-invariant enough to trust going forward.

**Điểm đáng chú ý nhất cho bước tiếp theo (method derivation)**: đúng như user đã lưu ý trước khi
chạy — 2 relation mạnh nhất và ổn định nhất qua cả 2 N (`O↔E` và `E↔nearBG`, cả hai đều liên quan
trực tiếp đến **boundary E**) tiếp tục đứng đầu bảng ở `dev_300` (cross-family 0.742 và 0.763,
cao nhất trong 5 relation còn lại), trong khi `O↔farBG` (không liên quan boundary, so interior với
background xa) là relation duy nhất bị loại. Đây là bằng chứng bổ sung (không phải xác nhận
thống kê chính thức, chỉ là quan sát nhất quán qua 2 N) cho hướng user đề xuất: invariant thật có
thể tập trung vào **transition O→E→Bn** (foreground-to-background gần), không phải quan hệ chung
chung với mọi vùng bao gồm cả far-background.

**Trạng thái**: E10 đã đóng ở cả 2 tầng pilot (N=49) và confirmation (N=296) — **STRONG GO xác
nhận, sẵn sàng cho bước tiếp theo là derive attack method mới từ relational invariant** (không phải
OSFD + L_rel combine — theo đúng quyết định user đã nêu). Bước này CHƯA làm, cần quay lại với user
trước khi bắt đầu (thiết kế method mới là quyết định lớn, không phải confirmation run thuần tuý).

## 29. TGA — Semantic Transition Geometry Attack: method-derivation từ E10, NO-GO (3 biến thể)

Bước method-derivation mà §27-28 để ngỏ: derive một attack objective trực tiếp từ relational
invariant O↔E/O↔nearBG/E↔nearBG (không phải OSFD + L_rel combine, mà từ đầu, bỏ hẳn cơ chế
suppress-object/amplify-vicinal của OSFD). Đặt tên **TGA (Transition-Geometry Attack)**: thay vì tấn
công độ lớn/hướng của raw feature tại object, tấn công trực tiếp **quan hệ tương đối** giữa 3 vùng
O (interior), E (boundary), Bn (near-background) — đúng 3 vùng trong "transition" O→E→Bn mà E10 xác
định là mạnh và ổn định nhất qua cả N=49 và N=296 (§28's "2 relation mạnh nhất... đều liên quan trực
tiếp đến boundary E"). **Bỏ hẳn far-background (Bf)**: `O↔farBG` là relation DUY NHẤT rớt khỏi
shared-set khi scale N=49→296 (§28), nên v0 không đưa Bf vào method để giữ novelty story sạch và
tránh dùng đúng phần yếu nhất của invariant.

**Thiết kế v0 (chốt trước khi chạy, theo đúng đặc tả user đưa ra)**: RRB OFF cho mọi biến thể (cô
lập xem bản thân relational objective có tín hiệu hay không, trước khi trộn augmentation vào); region
definition verbatim từ FROZEN SPEC của E10 (`shrink_frac=0.125`, `expand_frac=0.25`,
`MASK_MIN_STAGE_CELLS=1.0` — code mới `transfer_attack/regions.py`, copy tách biệt để không đụng vào
script E10 đã frozen/verified). Hai công thức loss, test cô lập từng cái trước khi nghĩ tới combine:

- **Rank (v0 chính)** — đọc ordering `s_ij = sign(r_clean_i - r_clean_j)` từ chính ảnh clean cho mỗi
  cặp trong 3 relation `{OE, OBn, EBn}`, ascend `L_rank = Σ softplus(-s_ij·(r_adv_i - r_adv_j)/T)`
  (T=1.0, không tune) để đảo ngược ordering đó (`transfer_attack/losses.py::tga_rank_loss`).
- **Geom** — `L_geom = ‖r̂_adv - r̂_clean‖²` với `r̂ = r/(‖r‖₂+eps)` (magnitude-based, thử sau khi rank
  NO-GO — xem chẩn đoán bên dưới), `tga_geom_loss`.
- Cả 2 average qua **mọi backbone stage** (mặc định) hoặc riêng **last-stage-only** (cờ
  `tga_last_stage_only`, phạm vi mà E10 thực sự đã confirm invariant).

Code: `transfer_attack/regions.py` (mask O/E/Bn/Bf + pooling, differentiable), `transfer_attack/
losses.py` (`tga_relational_vector`, `tga_rank_loss`, `tga_geom_loss`, `tga_loss`), tích hợp
`attack_type="tga"` vào `transfer_attack/attack.py::craft_one_image` (tái dùng nguyên vòng lặp
I-FGSM+MI-momentum, không viết loop riêng). Pilot script mới `scripts/tga_v0_pilot.py`.

**GO criterion (pre-register trước khi chạy)**: Strong GO nếu TGA thắng OSFD-noRRB ≥+5 điểm ASR
trên ≥2/3 hard target (`yolox_l`, `mask_rcnn_swin_t`, `dino_swin_l`) và không collapse >20 điểm trên
target còn lại; Weak GO nếu chỉ thắng rõ trên `dino_swin_l`; còn lại NO-GO.

**Kết quả (N=20, 100 step, `dev_50`, 6 model — đúng model set E10 dùng,
`results/tga_v0_pilot_summary.csv` + `results/tga_v0_lastonly_summary.csv` +
`results/tga_v0_geom_summary.csv`)**:

| model | group | osfd_norrb | mi_fgsm | tga_rank (all-stage) | tga_rank (last-stage) | tga_geom (all-stage) |
|---|---|---:|---:|---:|---:|---:|
| faster_rcnn_r50 (white-box surrogate) | — | 95.7 | 95.7 | 66.0 | 73.4 | 56.4 |
| yolox_l | B | 22.8 | 18.8 | 5.9 | 5.9 | 4.0 |
| dino_swin_l | C | 9.0 | 14.0 | 5.0 | 3.0 | 4.0 |
| mask_rcnn_swin_t | C | 23.7 | 21.6 | 6.2 | 8.2 | 6.2 |
| dino_r50 | D | 100.0 | 92.6 | 61.7 | 61.7 | 44.7 |
| mask_rcnn_r50 | D | 100.0 | 98.9 | 68.1 | 65.9 | 44.0 |

**NO-GO rõ ràng, cả 3 biến thể, không có ngoại lệ trên bất kỳ model nào** — kể cả `dino_swin_l`
(Weak-GO criterion cũng fail: cả 3 biến thể đều ÂM trên chính target Swin khó nhất, target mà TGA
được kỳ vọng cải thiện nhất). Đáng chú ý nhất: TGA thua ngay cả ở **white-box surrogate** (−22 đến
−39 điểm so với OSFD-noRRB, và thua cả MI-FGSM) — đây không phải câu chuyện "không transfer", mà là
bản thân loss objective tạo ra adversarial perturbation yếu hơn hẳn, kể cả tấn công chính model dùng
để craft.

**Tuần tự chẩn đoán (mỗi bước đều pre-register trước khi chạy tiếp, không phải retune sau khi thấy
số xấu)**:
1. `tga_rank` (all-stage) NO-GO → nghi ngờ multi-stage average pha loãng tín hiệu (E10 chỉ confirm
   invariant ở last-stage, chưa test stage nông hơn — limitation đã ghi ở §27).
2. `tga_rank` (last-stage-only, đúng phạm vi E10 confirm) → **pattern gần như y hệt** (surrogate
   66.0→73.4, các target khác không đổi hoặc còn tệ hơn) — loại trừ giả thuyết "sai stage". Vấn đề
   nằm ở chính công thức loss, không phải phạm vi stage.
3. Chẩn đoán kỹ thuật từ chính công thức: `softplus(-s_ij·diff/T)` có đuôi giảm theo hàm mũ khi
   `diff` đã lệch xa đúng chiều clean ordering (`z≪0` → `softplus(z)≈exp(z)≈0`, gradient cũng ≈0) —
   nghĩa là mọi cặp relation mà ảnh clean vốn đã tách biệt rõ gần như không đóng góp gradient suốt
   phần lớn quá trình optimize, bất kể chọn stage nào.
4. Thử `tga_geom` (magnitude-based, không có failure mode vanishing-gradient này theo construction)
   → **vẫn NO-GO, thậm chí tệ hơn rank trên nhóm R50** (`dino_r50` −55.3 so với rank's −38.3,
   `mask_rcnn_r50` −56.0 so với −31.9), chỉ xấp xỉ ngang rank trên 3 hard target. Loại trừ luôn giả
   thuyết "vanishing gradient là nguyên nhân chính" — đổi công thức không sửa được vấn đề.

**Kết luận cuối (đóng, không thử `rank_geom` combine hay tune thêm — tổ hợp tuyến tính của 2 thành
phần đều đã NO-GO độc lập khó có khả năng đột biến thành mạnh, và đã test đủ 2 công thức hợp lý ×
2 phạm vi stage cho rank để kết luận có kỷ luật)**:

> Attacking the RELATIVE ordering or magnitude of distances between an object's interior (O),
> boundary (E) and near-background (Bn) — the exact relational signature E10 confirmed as a
> reproducible cross-architecture invariant — fails as a direct optimization objective. All 3
> variants tested (rank-based, all backbone stages; rank-based, last stage only; magnitude-based,
> all stages) lose to OSFD-noRRB on EVERY model tested, including a 22-39 ASR point loss on the
> WHITE-BOX surrogate itself — not merely a transfer failure but a fundamentally weak ascent
> objective. Restricting to E10's actually-confirmed last stage does not fix the rank variant
> (identical failure pattern), and switching from a rank-inversion loss (diagnosed as having a
> vanishing gradient for well-separated clean relations) to a magnitude-based loss does not fix it
> either (uniformly equal or worse). The likely root cause: a 3-scalar-per-object relational
> signature is far too sparse a gradient signal compared to OSFD's dense per-channel backbone-
> feature MSE, regardless of how that signature is turned into a loss. E10's discovery (a
> confirmed, reproducible diagnostic invariant) does not transfer into a competitive attack
> objective by attacking it directly — this closes the "attack the E10 invariant directly" line of
> method derivation.

**Việc KHÔNG làm (quyết định rõ, tránh redo)**: không thử `rank_geom` (tổ hợp tuyến tính của L_rank
+ λ·L_geom) — cả 2 thành phần đã NO-GO độc lập, đúng kỷ luật "không tune/combine để cứu candidate"
áp dụng xuyên suốt project (MVC §9, RCG §9, N2-B §12, DOB §15, E9 §26). Không thử thêm biến thể
temperature/shrink_frac/expand_frac khác — đây là hyperparameter search để cứu kết quả xấu, không
phải chẩn đoán cơ chế. Không thêm RRB vào TGA (bước 9 trong plan gốc, chỉ có ý nghĩa nếu v0-noRRB đã
có tín hiệu dương — không áp dụng nữa vì cả 3 v0 đều NO-GO).

**Trạng thái sau §27-29 (một phần được §30 dưới đây làm rõ thêm)**: E10 vẫn là mechanism/diagnostic
discovery mạnh nhất và sạch nhất của toàn project (relational invariant tồn tại thật, confirmed ở
N=296) — TGA (rank/geom, pooled-scalar) không tự động chuyển thành attack objective cạnh tranh được
với OSFD. Nhưng §30 cho thấy nguyên nhân cụ thể hơn dự đoán ban đầu: không phải bản thân invariant vô
dụng, mà là cách **pool về 1 scalar/object** làm mất quá nhiều tín hiệu — giữ lại cấu trúc không gian
(DBTA) cải thiện đáng kể (từ NO-GO trắng trợn sang cạnh tranh thật ở white-box). Roadmap "prove" cho
N6-B (§16-22, method mạnh nhất đã xác nhận của project trước E10) không bị ảnh hưởng bởi các kết quả
E10-derivation này.

## 30. DBTA — Dense Boundary-Transition Attack: cải thiện rõ so với TGA nhưng RRB không thu hẹp gap

Đề xuất ngay sau TGA đóng (§29), dựa đúng chẩn đoán thất bại của TGA: 3 scalar/object (pooled mean
của O/E/Bn) là tín hiệu quá thưa để làm attack objective, bất kể rank hay magnitude. **DBTA giữ
nguyên margin `shrink_frac=0.125`/`expand_frac=0.25`** (đúng frozen spec E10/TGA) nhưng **không pool
region về 1 vector** — thay vào đó, sample một trường điểm DÀY dọc theo **chính perimeter của GT
box** (mỗi điểm `p` trên biên box, tạo 3 điểm đồng tâm: `p_O` dịch vào trong theo `shrink_frac`, `p_E
= p`, `p_B` dịch ra ngoài theo `expand_frac`), đọc feature tại từng điểm qua `F.grid_sample`
(differentiable, stage-agnostic vì tọa độ chuẩn hóa theo H/W của chính stage đó). Loss ascend MSE
thô (không rank, không normalize — học từ TGA rằng vấn đề là độ thưa chứ không phải công thức) giữa
transition vector `T(p) = [f_E(p)-f_O(p) ; f_B(p)-f_E(p)]` của adv so với clean, trung bình qua mọi
điểm × object × stage. Code: `transfer_attack/dbta.py`, tích hợp `attack_type="dbta"` vào
`transfer_attack/attack.py`. Pilot: `scripts/dbta_v0_pilot.py`.

**Thiết kế pre-registered**: KHÔNG nhảy thẳng vào so sánh transfer — kiểm tra **white-box sanity**
trước (DBTA có phải một objective ascent cạnh tranh trên chính surrogate dùng để craft hay không),
vì TGA thất bại ngay ở mức này (thua cả white-box), không phải ở transfer. Ngưỡng: FAIL nếu ASR
white-box thua OSFD-noRRB > `WB_COLLAPSE_THR=20` điểm (giống mức TGA đã fail); chỉ đọc hard-target
delta nếu PASS.

**Round 1 (RRB OFF, N=20/100-step, `dev_50`, `results/dbta_v0_pilot_summary.csv`)**:

| model | group | osfd_norrb | dbta_norrb | Δ |
|---|---|---:|---:|---:|
| faster_rcnn_r50 (white-box) | — | 95.7 | 90.4 | **−5.3** |
| yolox_l | B | 22.8 | 20.8 | −2.0 |
| dino_swin_l | C | 9.0 | 9.0 | +0.0 |
| mask_rcnn_swin_t | C | 22.7 | 11.3 | −11.3 |
| dino_r50 | D | 100.0 | 93.6 | −6.4 |
| mask_rcnn_r50 | D | 100.0 | 94.5 | −5.5 |

**WHITE-BOX SANITY PASS rõ ràng** — lệch chỉ −5.3 điểm (so với TGA's −22 đến −39 trên MỌI model),
xác nhận đúng chẩn đoán "sparsity là nguyên nhân chính, không phải rank-vs-magnitude". Trên hard
target: `yolox_l` gần hòa (−2.0), `dino_swin_l` **hòa tuyệt đối** (+0.0, đúng target khó nhất project
muốn cải thiện), `mask_rcnn_swin_t` vẫn kém rõ (−11.3, điểm yếu nhất, khớp với E10's ghi nhận
`mask_rcnn_swin_t` có `disrupt_shared` bất thường thấp so với ASR của nó — câu hỏi mở chưa giải thích
được xuyên suốt project). Chưa đạt GO (`≥+5` trên `≥2/3` hard target) nhưng khác hẳn TGA: đây là
"sanity PASS, transfer chưa thắng" — trạng thái đáng đầu tư thêm, không phải NO-GO ngay.

**Round 2 — thêm RRB (lockstep với hướng đã dùng project-wide để tăng transfer)**: lần thử đầu tiên
**tái dùng nguyên `boundary_contour_points` tính 1 lần trên ảnh clean** cho cả 2 augmented view của
RRB — **collapse mạnh, quay lại đúng cấp độ thất bại của TGA** (white-box −38.3, mọi hard target âm
sâu). **Chẩn đoán đúng nguyên nhân (không phải objective yếu)**: `rrb_forward` xoay+resize TOÀN BỘ
ảnh trước khi forward qua backbone; OSFD không bị ảnh hưởng vì loss của nó là MSE feature-map không
phụ thuộc vị trí cụ thể, nhưng DBTA sample tại **tọa độ canvas cố định** (tính từ GT box gốc, chưa
xoay) — sau khi ảnh bị xoay, object thật đã "di chuyển" sang vị trí khác, DBTA vẫn sample tọa độ cũ
→ đọc nhầm feature của background/vùng không liên quan, nặng nhất ở các điểm biên (xa tâm xoay nhất).

**Sửa đúng**: viết `rrb_forward_with_params` (tái tạo độc lập đúng phân phối random của
`transfer_attack/augment.py`'s `random_axis_rotation`/`adaptive_random_resizing` — KHÔNG sửa/gọi lại
code gốc, để noise OSFD hiện có không bị ảnh hưởng) trả về thêm transform params (góc xoay, tâm xoay,
resize scale/pad) mỗi lần draw; `rotate_points`/`resize_pad_resize_points` áp lại đúng transform đó
lên tọa độ sample point trước khi `grid_sample` trên feature map đã augment. **Cả 2 công thức
point-transform được verify thực nghiệm** (không suy từ tài liệu/đoán dấu) bằng ảnh synthetic có 1
pixel sáng đã biết tọa độ, so khớp với hành vi thật của `torchvision.transforms.functional.rotate`
và `F.interpolate(align_corners=True)` — phát hiện công thức rotation "đoán ngây thơ" (ma trận xoay
chuẩn CCW) sai dấu so với convention thật của `torchvision`.

**Bug thứ hai bắt được trong quá trình fix (methodology, không phải DBTA)**: `evaluate.py` cache
adversarial prediction theo `(model, attack_tag)` **không theo nội dung noise**. Pilot script dùng
`predictions_dir` cố định (không timestamp) — sau khi sửa code craft và re-run cùng attack tag, eval
âm thầm tái dùng prediction cũ từ noise HỎNG trước khi fix (bắt được qua log "0 computed + cached, N
from cache" bất thường ở lần smoke-test đầu sau fix). Đã sửa `scripts/dbta_v0_pilot.py` dùng
`predictions_dir` timestamp riêng mỗi lần chạy script (giống convention `run_attack.py` vốn đã làm
đúng việc này) — không có bug này trong `scripts/tga_v0_pilot.py` (mỗi TGA-variant dùng tag riêng
biệt, không bao giờ redo cùng tag với code khác nhau).

**Round 2 sau khi fix (N=20/100-step, `results/dbta_v0_rrb_summary.csv`)**:

| model | group | osfd_norrb | dbta_norrb | osfd (RRB) | dbta_rrb | Δ norrb | Δ RRB |
|---|---|---:|---:|---:|---:|---:|---:|
| faster_rcnn_r50 (white-box) | — | 95.7 | 90.4 | 98.9 | 90.4 | −5.3 | −8.5 |
| yolox_l | B | 22.8 | 20.8 | 81.2 | 46.5 | −2.0 | **−34.7** |
| dino_swin_l | C | 9.0 | 9.0 | 34.0 | 25.0 | +0.0 | **−9.0** |
| mask_rcnn_swin_t | C | 22.7 | 11.3 | 82.5 | 33.0 | −11.3 | **−49.5** |
| dino_r50 | D | 100.0 | 93.6 | 100.0 | 89.4 | −6.4 | −10.6 |
| mask_rcnn_r50 | D | 100.0 | 94.5 | 98.9 | 90.1 | −5.5 | −8.8 |

**WHITE-BOX SANITY PASS sau fix** (−8.5, khớp lại đúng biên độ round 1, xác nhận fix hình học đúng —
không còn collapse kiểu TGA). Nhưng **hard-target gap RỘNG RA thay vì thu hẹp**: DBTA tự nó CÓ tăng
tuyệt đối khi thêm RRB (`yolox_l` +25.7, `mask_rcnn_swin_t` +21.7, `dino_swin_l` +16.0 so với
`dbta_norrb`) — RRB không vô dụng với DBTA — nhưng OSFD tăng MẠNH HƠN NHIỀU cùng lúc (`yolox_l`
+58.4, `mask_rcnn_swin_t` +59.8, `dino_swin_l` +25.0), nên khoảng cách tương đối giữa 2 method nới
rộng thay vì thu hẹp, rõ nhất ở `mask_rcnn_swin_t` (−11.3 → −49.5) và `yolox_l` (−2.0 → −34.7).

**Kết luận (theo quyết định user, không chạy thêm steps/tune để cứu hướng RRB)**:

> DBTA's dense per-point boundary-transition objective is a competitive white-box ascent target
> (unlike TGA's pooled-scalar rank/geom losses, which collapsed even against their own surrogate) --
> confirming that signal SPARSITY, not loss shape, was TGA's root failure. However, adding RRB (this
> project's single largest known transfer driver, +13 to +31 ASR points for OSFD per E3) does not
> close the gap to OSFD -- it widens it, because OSFD's global feature-map MSE benefits far more from
> RRB's multi-view averaging than DBTA's localized per-point objective does. RRB integration required
> a real geometric fix first (naive reuse of clean-frame sample points under RRB's rotate+resize
> collapsed identically to TGA, -38 ASR points on the white-box surrogate) -- after fixing it
> (tracking and re-applying the exact RRB transform to boundary points), white-box sanity is restored
> but the RRB extension itself does not help DBTA's transfer story. The RRB line of extension is
> closed; DBTA-noRRB (round 1, roughly tied with OSFD-noRRB on 2/3 hard targets, particularly
> dino_swin_l) remains the candidate's best and current result.

**Trạng thái**: DBTA chưa đạt GO (chưa thắng OSFD trên bất kỳ cấu hình nào đã thử) nhưng cũng không
NO-GO sạch như TGA — nó là candidate đầu tiên sau TGA sống sót qua white-box sanity và chỉ thua sát
nút trên 2/3 hard target ở cấu hình tốt nhất (no-RRB). **Việc CHƯA làm, cần bàn với user trước khi
tiếp tục** (không tự chọn để tránh lặp lại việc mở rộng không xin phép): tăng `steps`/`n_points` cho
riêng nhánh no-RRB (chưa thử, khác với việc "thêm RRB" đã đóng ở trên), điều tra vì sao
`mask_rcnn_swin_t` luôn là điểm yếu nhất qua cả TGA và DBTA (câu hỏi mở từ E10), hay dừng hẳn dòng
method-derivation từ E10 tại đây và coi E10 là đóng góp diagnostic thuần túy.

**[SUPERSEDED bởi §31]** Đoạn "Việc CHƯA làm" ở trên là snapshot ngay sau khi DBTA đóng — user sau đó
quyết định thử thêm 1 candidate cuối (BTFA, §31) thay vì tăng steps/n_points hay điều tra
`mask_rcnn_swin_t` riêng lẻ; BTFA đã đóng luôn cả dòng method-derivation từ E10.

## 31. BTFA — Boundary Transition Field Attack: candidate cuối, NO-GO — đóng hẳn dòng method-derivation từ E10

Đề xuất ngay sau DBTA (§30), đúng logic tiến hóa: TGA nén O/E/Bn về 3 scalar/object (quá thưa, fail
white-box) → DBTA giữ cấu trúc điểm rời rạc dọc perimeter (~24 điểm/object, pass white-box nhưng vẫn
thưa về không gian, chỉ phủ đúng đường viền) → **BTFA thay hẳn bằng một trường liên tục**: dựng
**signed-distance field** `S(x,y)` cho mỗi GT box (công thức SDF hình chữ nhật chuẩn — âm bên trong,
0 đúng tại cạnh box, dương bên ngoài), giữ một **band** `-shrink_frac·repr_dim ≤ S ≤ expand_frac·
repr_dim` (`repr_dim = √(box_w·box_h)`, gộp 2 margin theo trục của TGA/DBTA thành 1 scale đẳng hướng
vì SDF là khoảng cách đẳng hướng), tính **normal direction** `n(p) = ∇S(p)/‖∇S(p)‖` (finite-difference
Sobel, không cần gradient vì S là hình học cố định), rồi ascend MSE của **directional derivative**
`D_nF(p) = ∇F(p)·n(p)` (Sobel áp lên chính feature map, có gradient) giữa adv và clean, trung bình
qua mọi pixel trong band × object × stage. Đây chính là **giới hạn liên tục của DBTA's finite
difference** `F_E-F_O`/`F_B-F_E` — không cần chọn discrete O/E/B triplet cho từng điểm nữa. Code:
`transfer_attack/btfa.py`, tích hợp `attack_type="btfa"` vào `transfer_attack/attack.py` (RRB chưa hỗ
trợ — raise `NotImplementedError` tường minh nếu gọi với `use_rrb=True`, tránh lặp lại lỗi hình học
đã gặp ở DBTA round 2 nếu ai đó bật RRB mà không port field-based point-transform trước). Pilot:
`scripts/btfa_v0_pilot.py` (4 arm: `osfd_norrb`/`dbta_norrb`/`btfa_norrb`/`mi_fgsm`, `dbta_norrb`
craft lại mới trong cùng lần chạy để so sánh trực tiếp, không lấy số từ §30 do nhiễu GPU-nondeterminism
run-to-run đã quan sát được ở DBTA — xem §30's 2 lần chạy "osfd" RRB-on khác nhau vài điểm dù cùng
seed/code).

**Thiết kế pre-registered, 2 pha** (chốt trước khi chạy): Pha 1 — **white-box gate tuyệt đối**
(`ASR_btfa ≥ WB_MIN_ASR=80%` trên chính surrogate, không phải delta) vì TGA fail chính ở đây; nếu
fail thì đóng ngay không cần đọc hard target. Pha 2 (chỉ đọc nếu Pha 1 pass) — so hard-target delta
với CẢ `osfd_norrb` (chính, tiêu chí `≥+5` trên `≥2/3` hard target như TGA/DBTA) VÀ `dbta_norrb` (phụ,
đặc biệt quan tâm `mask_rcnn_swin_t` — điểm yếu nhất của DBTA).

**Kết quả (N=20, 100 step, `dev_50`, 6 model, `results/btfa_v0_pilot_summary.csv`)**:

| model | group | osfd_norrb | dbta_norrb | btfa_norrb | mi_fgsm | Δ(btfa−osfd) | Δ(btfa−dbta) |
|---|---|---:|---:|---:|---:|---:|---:|
| faster_rcnn_r50 (white-box) | — | 100.0 | 95.7 | 96.8 | 100.0 | −3.2 | +1.1 |
| yolox_l | B | 24.8 | 24.8 | 22.8 | 21.8 | −2.0 | −2.0 |
| dino_swin_l | C | 10.0 | 9.0 | 11.0 | 13.0 | +1.0 | +2.0 |
| mask_rcnn_swin_t | C | 22.7 | 12.4 | 15.5 | 22.7 | −7.2 | **+3.1** |
| dino_r50 | D | 100.0 | 92.6 | 96.8 | 93.6 | −3.2 | +4.3 |
| mask_rcnn_r50 | D | 100.0 | 94.5 | 93.4 | 100.0 | −6.6 | −1.1 |

**Pha 1: PASS mạnh** — `96.8%` white-box, tốt nhất trong cả 3 candidate E10-derived (TGA: collapse
sâu; DBTA: `90.4-95.7%`; BTFA: `96.8%`, gần sát ceiling `OSFD=100%` ở N=20 này). Xác nhận: dense field
không hề yếu hơn dense point-sampling về mặt tối ưu — vấn đề của TGA thực sự chỉ là độ thưa của
pooled-scalar, không phải "region-relational objective nói chung yếu".

**Pha 2: so `osfd_norrb`, NO-GO** (0/3 hard target đạt `≥+5`; delta dao động hẹp `−7.2` đến `+1.0` —
roughly tied, không phải thua đậm như DBTA+RRB). **So `dbta_norrb`: BTFA thắng trên 2/3 hard target**
(`dino_swin_l` +2.0, và đặc biệt **`mask_rcnn_swin_t` +3.1** — đúng điểm DBTA yếu nhất, ủng hộ giả
thuyết dense field khá hơn point-sampling rời rạc ở Swin hai-giai-đoạn) — nhưng cải thiện này KHÔNG
đủ để BTFA tự nó vượt qua chính OSFD trên bất kỳ hard target nào.

**Kết luận (đóng, theo đúng xác nhận của user — không sweep band width, không đổi derivative operator,
không thêm RRB, không thêm OSFD auxiliary loss, không tăng steps, không tune stage weight)**:

> BTFA closes the loop TGA opened: white-box ASR is no longer the bottleneck (96.8%, competitive with
> OSFD's near-ceiling 100% and clearly recovered from TGA's white-box collapse), yet cross-architecture
> transfer still does not beat OSFD (hard-target deltas -7.2 to +1.0, 0/3 at the pre-registered +5
> threshold) even though BTFA is a strict representational upgrade over DBTA (a continuous field vs
> discrete perimeter points) and beats DBTA on 2/3 hard targets, including DBTA's specific weak point
> (mask_rcnn_swin_t, +3.1). Because white-box optimization strength can no longer explain the transfer
> gap -- the objective converges just as well as OSFD's against the surrogate it was crafted on -- the
> bottleneck must be a TRANSFER-specific property of what each objective ascends, not an optimization
> or signal-density deficiency. This closes the "attack E10's relational invariant directly" line of
> method derivation (TGA -> DBTA -> BTFA) as a disciplined falsification chain, not a dead end from
> insufficient engineering effort: three qualitatively different operationalizations of the SAME
> confirmed cross-architecture invariant (pooled scalar rank/magnitude, discrete point transition,
> continuous field transition) were tried, and each ruled out a specific failure hypothesis for the
> next (TGA's sparsity -> DBTA's white-box competence -> BTFA's still-insufficient transfer despite
> full white-box strength).

**Bài học chính (đáng nhớ hơn cả kết quả NO-GO)**:

> Architecture-shared DIAGNOSTIC structure does not imply an architecture-shared ADVERSARIAL
> DIRECTION. E10 confirmed a real, reproducible, cross-architecture invariant (the O<->E/E<->nearBG
> relational signature) — that finding stands on its own and is not falsified by TGA/DBTA/BTFA's
> NO-GOs. What §29-31 falsify is the STRONGER, unstated assumption the method-derivation attempt made:
> that disrupting a structure multiple architectures are known to SHARE is the same as finding a
> direction in input space that multiple architectures are simultaneously VULNERABLE to moving along.
> A property can be common across representations (E10) without an attack on that property being
> common in its effect (TGA/DBTA/BTFA) — diagnostic universality and adversarial transferability are
> different claims, and this project's evidence now separates them cleanly for the first time.

**Việc KHÔNG làm (quyết định rõ theo user, tránh redo)**: không thêm RRB vào BTFA (chưa port
field-based point-transform, và DBTA's kinh nghiệm cho thấy RRB có thể làm RỘNG gap thay vì thu hẹp
ngay cả khi hình học đúng — §30); không sweep band width (`shrink_frac`/`expand_frac`); không đổi
derivative operator (Sobel → kernel khác); không thêm OSFD auxiliary loss (hướng "OSFD + L_rel
combine" mà user đã loại trừ từ đầu §29 để giữ novelty sạch — giờ càng không có lý do quay lại, vì
BTFA đã cho thấy ngay cả objective độc lập mạnh nhất cũng không đủ); không tăng `steps` (white-box đã
gần ceiling, vấn đề không phải hội tụ); không tune stage-weight; không mở diagnostic riêng cho
`mask_rcnn_swin_t` như một câu hỏi độc lập (đã có đủ evidence từ chính BTFA — cải thiện tương đối so
DBTA nhưng vẫn không đủ so OSFD, không cần thêm 1 experiment giải thích riêng).

**Trạng thái cuối**: **Dòng method-derivation từ E10 (TGA §29 → DBTA §30 → BTFA §31) ĐÃ ĐÓNG HẲN.**
E10 (§27-28) vẫn đứng vững như mechanism/diagnostic discovery — relational invariant cross-architecture
là có thật, confirmed ở N=296, KHÔNG bị đảo ngược bởi bất kỳ NO-GO nào ở trên. Nhưng con đường "biến
diagnostic invariant thành attack objective bằng cách attack trực tiếp chính invariant đó" đã bị loại
trừ có kỷ luật qua 3 công thức hóa khác nhau, không phải bỏ cuộc giữa chừng. Roadmap N6-B (§16-22,
method mạnh nhất đã xác nhận của project, không liên quan tới E10) không bị ảnh hưởng. **Bước tiếp
theo (chưa làm, cần bàn với user khi bắt đầu phiên sau)**: quay lại tìm attack principle mới dựa trên
câu hỏi khác — "cái gì tạo ra transferable adversarial gradient/direction" thay vì "cái gì được nhiều
architecture cùng biểu diễn/cùng bất biến" — hai câu hỏi đã được chứng minh là KHÔNG tương đương qua
chuỗi §29-31.

**[SUPERSEDED bởi §32]** Đoạn "Bước tiếp theo" ở trên là snapshot ngay sau khi BTFA đóng — user sau đó đề xuất
1 candidate mới không dựa trên E10 (CEFA, dựa trên transformation-equivariance thay vì relational geometry),
test bằng E11 trước khi build; E11 cũng NO-GO, xem §32 cho hướng đi thực tế tiếp theo.

## 32. E11 — Adversarial Equivariance Gap: existence/relevance test cho CEFA, NO-GO — đóng trước khi build attack

Sau khi TGA→DBTA→BTFA đóng hẳn dòng method-derivation từ E10 (§29-31, kết luận "architecture-shared
diagnostic structure ⇏ architecture-shared adversarial direction"), user đề xuất một attack principle hoàn
toàn khác, không dựa trên E10's relational invariant: **CEFA (Cross-view Equivariance Failure Attack)** —
thay vì tấn công giá trị/quan hệ feature, tấn công trực tiếp **transformation equivariance** của toàn bộ
feature field. Ý tưởng: một detector backbone tốt phải thỏa `F(τx) ≈ W_τF(x)` với τ là 1 phép biến đổi hình
học nhẹ (rotate/scale); adversarial perturbation thành công có thể là perturbation phá vỡ chính
"transformation law" này — property mà nhiều architecture (CNN/Transformer) đều buộc phải tuân theo để giữ
spatial correspondence cho detection, mạnh hơn requirement thuần semantic của classifier.

Motivation nối từ 2 finding cũ của project: RRB (driver transfer mạnh nhất, +13→+31 ASR, E3 §8) và
path-M3/N6-B (path-averaged gradient cải thiện DINO-Swin, 33.0%→39.1% ASR ở `dev_300`, §16-22) đều là can
thiệp KHÔNG tấn công trực tiếp 1 feature value cụ thể — cả hai đều "nhìn" adversarial image qua nhiều
góc/nhiều điểm trên trajectory. Hypothesis: cả hai có thể đang (vô tình) làm tăng equivariance failure.

**Thiết kế E11 (existence + relevance test, KHÔNG craft loss mới — dùng nguyên crafting code hiện có)**:
- `Q_m(x,δ) = E_τ [ ||F_m(τ(x+δ)) − W_τF_m(x+δ)|| / ||F_m(x+δ)|| ]`, τ ∈ {rotate±5°, scale0.9/1.1}
  (pre-registered, không sweep).
- `W_τ` định nghĩa qua `F.affine_grid`+`grid_sample` trong tọa độ chuẩn hóa [-1,1] — cùng MỘT hàm `warp()`
  (`transfer_attack/equivariance.py`) áp trực tiếp lên cả pixel-space image (định nghĩa τ(x)) và feature-map
  tensor (định nghĩa W_τF(x)), nên nhất quán hình học đúng theo construction (resolution-agnostic), không cần
  derive/verify point-transform riêng như DBTA's `rotate_points` đã phải làm cho RRB.
- 4 arm: `osfd_rrb` (=`osfd_local`), `osfd_norrb`, `path_m3` (M=3, N6-B, RNG-paired với `osfd_rrb` qua
  `craft_paired_local_path`, tái dùng nguyên từ `scripts/n6b_path_pilot.py`), `mi_fgsm` (context, không dùng
  trong verdict). Craft mới N=20/100 step trên `dev_50` (checkout này `results/` trống — script mới
  `scripts/e11_equivariance_gap.py`).
- GO criteria (pre-registered, chốt trước khi chạy):
  - RRB test: `sign(ΔQ_RRB) == sign(ΔASR_RRB)` trên **3/3** hard target (`yolox_l`, `mask_rcnn_swin_t`,
    `dino_swin_l`).
  - Path test: cùng dấu trên `dino_swin_l` (chính); `mask_rcnn_swin_t` cùng dấu là bonus, không bắt buộc.
  - Strong GO nếu cả 2 test pass → mới thiết kế CEFA. NO-GO nếu 1 trong 2 fail — không đổi transform
    set/reweight stage để cứu.

**Kết quả (N=20, 100 step, `dev_50`, 6 model, `results/e11_equivariance_gap_summary.csv`, run log
`runs/e11_equivariance_gap_dev_50_n20_20260910T154634Z.json`)**:

| model | group | ASR norrb→rrb (Δ) | ASR rrb→path (Δ) | Q norrb→rrb (ΔQ) | Q rrb→path (ΔQ) | RRB sign match | path sign match |
|---|---|---:|---:|---:|---:|---|---|
| faster_rcnn_r50 (WB) | – | 100.0→100.0 (0.0) | →98.9 (−1.1) | 0.603→0.512 (−0.092) | →0.521 (+0.009) | – | – |
| yolox_l | B | 32.7→80.2 (+47.5) | →84.2 (+4.0) | 0.639→0.575 (−0.065) | →0.581 (+0.006) | ❌ | ✅ |
| dino_swin_l | C | 9.0→34.0 (+25.0) | →43.0 (+9.0) | 0.560→0.584 (+0.025) | →0.585 (+0.0006) | ✅ | ✅ |
| mask_rcnn_swin_t | C | 21.6→82.5 (+60.8) | →82.5 (0.0) | 0.487→0.499 (+0.012) | →0.496 (−0.004) | ✅ | ❌ (ASR tie) |
| dino_r50 | D | 100→98.9 | →98.9 | 0.695→0.572 (−0.124) | →0.582 | – | – |
| mask_rcnn_r50 | D | 100→97.8 | →98.9 | 0.616→0.526 (−0.090) | →0.533 | – | – |

**RRB test: 2/3 (`yolox_l` FAIL, `mask_rcnn_swin_t`/`dino_swin_l` PASS) — cần 3/3 → FAIL.** **Path test: PASS
trên `dino_swin_l`** (cùng dấu) nhưng biên độ ΔQ chỉ +0.0006 — nhỏ hơn ~40-50 lần biên độ ΔQ_RRB (0.01–0.12),
nhiều khả năng nằm trong nhiễu N=20; `mask_rcnn_swin_t` không match nhưng do ASR tie tuyệt đối (82.47=82.47)
ở N nhỏ này — suy biến, không phải bằng chứng phản bác mạnh theo hướng ngược lại.

**Falsifier sạch nhất: `yolox_l`.** RRB tăng ASR +47.5 điểm (một trong những gain lớn nhất từng đo trong
project cho riêng 1 can thiệp) nhưng Q **giảm** (−0.065) — đúng kịch bản NO-GO đã pre-register ("RRB tăng ASR
rất lớn nhưng equivariance gap không đổi hoặc giảm"). Pattern rộng hơn: RRB chỉ tăng Q trên 2 target **Swin**
(khớp hướng ASR), nhưng giảm Q trên toàn bộ nhóm **R50** (kể cả white-box, đã gần ceiling — nhất quán với
hiện tượng saturation quen thuộc từ E10 §28's Table C) **và** trên `yolox_l` (CSP/Darknet, nhóm B) — tức
hypothesis chỉ đứng vững cho đúng backbone Swin, không generalize sang "hard target nói chung" như tiêu chí
RRB test đòi hỏi.

**Kết luận (theo đúng quyết định user, đóng ngay không rescue)**:

> E11 rejects feature-equivariance failure as a general transfer mechanism. RRB-induced transfer gains are
> not consistently accompanied by increased equivariance error across heterogeneous backbones; the strongest
> counterexample is YOLOX-L, where ASR rises sharply while equivariance error decreases. The effect appears
> architecture-dependent, with positive alignment mainly on Swin targets, so CEFA is closed and will not be
> pursued as a general transferable attack principle.

**Bài học (đáng nhớ hơn cả kết quả NO-GO, nối tiếp bài học của §31)**:

> RRB success is not explained by one universal representation property.

Sau cả E10 (§27-28, relational invariant) và E11 (equivariance-failure invariant), pattern lặp lại rõ: một
diagnostic property/invariant nhìn rất thuyết phục (tồn tại thật, đo được, đôi khi khớp dấu ASR trên VÀI
model) thường KHÔNG generalize thành một transferable objective/mechanism DUY NHẤT giải thích transfer trên
**toàn bộ** tập hard target heterogeneous (CNN non-ResNet + 2 họ Swin khác nhau). Cả E10's TGA/DBTA/BTFA
(attack trực tiếp 1 structural invariant) và E11's CEFA (attack trực tiếp 1 transformation-law invariant) đều
fail vì cùng lý do gốc: shared DIAGNOSTIC property ⇏ shared ADVERSARIAL mechanism — nhưng E11 thêm một lát
cắt mới: ngay cả khi không attack trực tiếp property đó (E11 chỉ ĐO xem 2 intervention đã biết có đi cùng
chiều với nó hay không, không craft loss mới), correlation giữa "known transfer driver" và "candidate
universal property" cũng đã gãy ngay ở bước đo tương quan, trước khi kịp thử craft attack.

**Việc KHÔNG làm (quyết định rõ, tránh redo)**: không đổi transform set (thêm góc/scale khác), không
reweight theo stage, không tăng N để cố gắng "cứu" `yolox_l`'s falsification (biên độ mismatch của nó, −0.065
ngược chiều +47.5 ASR, đủ rõ để không phải vấn đề thiếu power) — theo đúng kỷ luật no-rescue đã áp dụng
xuyên suốt project (MVC §9, RCG §9, N2-B §12, DOB §15, E9 §26, TGA/DBTA/BTFA §29-31).

**Trạng thái cuối**: **CEFA đóng ngay ở bước existence/relevance test, không build attack.** E10 (§27-28) và
E11 (§32) cùng khẳng định: 2 "universal cross-architecture invariant" khác nhau (relational-geometry và
transformation-equivariance) đều tồn tại thật ở mức diagnostic nhưng đều KHÔNG giải thích được RRB's transfer
gain một cách nhất quán trên toàn bộ hard-target set. **Bước tiếp theo (chưa làm, cần bàn với user khi bắt
đầu phiên sau)**: đổi hẳn góc tìm kiếm — thay vì tiếp tục tìm 1 scalar/property của REPRESENTATION giải
thích transfer, chuyển sang tìm cấu trúc của chính PERTURBATION/UPDATE RULE mà nhiều hard target cùng "chấp
nhận" (structure of the perturbation itself, not a property of what it disrupts) — hướng này chưa được đặc
tả cụ thể, cần thiết kế cùng user trước khi code.

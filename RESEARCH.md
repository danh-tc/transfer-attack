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
- [ ] **(Ưu tiên chính)** Xác nhận N6-B trên `dev_300` (N lớn hơn) rồi held-out `val_100` theo
      đúng quy trình đã định. Cân nhắc thêm pilot trên nhóm A/B còn thiếu (fcos_r50,
      deformable_detr, yolov3_d53) để xác nhận không đánh đổi hiệu quả ở nhóm dễ transfer.
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

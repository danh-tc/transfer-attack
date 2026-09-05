# Session handoff

Mục đích: cho phiên Claude tiếp theo đọc nhanh để tiếp tục đúng ngữ cảnh sau khi máy restart,
không cần đọc lại toàn bộ `RESEARCH.md` từ đầu. `RESEARCH.md` vẫn là nguồn sự thật đầy đủ cho
mọi kết quả nghiên cứu (đặc biệt §21-§26 cho phiên này) — file này chỉ tóm tắt điều hướng +
engineering context không nằm trong journal.

**Cập nhật lần cuối**: 2026-09-05. Môi trường chạy là **checkout mới, không kế thừa state cũ**:
`results/` (gitignored) hoàn toàn trống — không noise, không prediction cache, không CSV nào từ
các phiên trước còn sống, chỉ `runs/*.json` (metrics tổng hợp) và các file `.md` là còn nguyên vì
có commit. Cũng thiếu 2 checkpoint (`dino_r50`, `mask_rcnn_r50`) — đã tải lại qua `mim download`
(xem "Engineering notes"). Phiên này **mở lại compute** (khác định hướng "paper closure" cuối phiên
trước) theo yêu cầu user, chạy 4 thực nghiệm liên tiếp: **E6 — Backbone Adversarial Response
Coupling** (`RESEARCH.md` §23, GO sau audit jackknife), **E7 — Downstream Amplification**
(`RESEARCH.md` §24, NO-GO), **E8 — Task-Relevant Response Alignment** (`RESEARCH.md` §25, NO-GO),
**E9 — Response-Coupling Intervention** (`RESEARCH.md` §26, NO-GO). E6 là mechanism candidate đầu
tiên sống sót qua cả replication ladder (N=49→120→296, 2 pool mode, 3 bootstrap_frac, delete-1
jackknife) mà không yếu đi — đây là finding quan trọng nhất phiên này. E7/E8 test 2 hướng giải
thích phần transfer mà `C_response` chưa giải thích hết (đều NO-GO); E9 thử chuyển sang can thiệp
trực tiếp (craft attack mới tăng `C_response`) — cũng NO-GO, cả ASR lẫn C_response đều không tăng
theo hướng cần. Đọc kỹ §23-§26 trước khi quyết định bước tiếp theo.

## Trạng thái hiện tại (quan trọng nhất, đọc trước)

**MỚI NHẤT (2026-09-05) — E6 Backbone Adversarial Response Coupling: ĐÃ ĐÓNG (GO, đọc thận trọng
sau audit)**, mechanism candidate cho H1 (`RESEARCH.md` §23, script `scripts/e6_response_coupling.py`).
Trả lời câu hỏi H1 để ngỏ ("tại sao backbone family là nguồn gốc gap, không chỉ rằng nó là") sau khi
`‖ΔF‖` (E1) và gradient cosine (§19) đều thất bại. Đo `C_response = CKA(Gram(ΔF_hat_surrogate),
Gram(ΔF_hat_target))` (linear CKA trong Gram space, per-image L2-normalized backbone-response-diff,
pool 7×7 mọi stage) so với `S_clean` (control, cùng CKA nhưng trên clean feature). Cả 2 matched-pair
(`dino_r50` vs `dino_swin_l`, `mask_rcnn_r50` vs `mask_rcnn_swin_t`) đúng hướng (R50>Swin) **không
đổi dấu một lần nào** qua N=49/120/296, mean/RMS pooling, bootstrap_frac 0.7/0.8/0.9.

**Audit CI bổ sung (delete-1 jackknife, sau khi user chỉ ra subsampling m<n cần rescale mà project
chưa làm)**: ở N=296, jackknife CI vẫn loại trừ 0 cho cả 4 trường hợp (2 pair × 2 pool mode) —
GO **giữ vững**. Nhưng cùng audit ở N=49 cho jackknife CI của DINO **cắt 0** (khác subsampling, vốn
báo excl-0 rất sát mép) — xác nhận CI subsampling lạc quan hơn thực tế ở N nhỏ, và jackknife chỉ
đồng thuận GO ở đúng N=296 (confirm), không phải N=49 (pilot). `|bias_jack|/SE_jack` khá lớn
(1.7×-5.6×, dấu hiệu CKA không hoàn toàn "smooth" cho delete-1) — ghi nhận như limitation của
inference, không đảo hướng kết luận (bias luôn âm, bias-correct sẽ đẩy estimate xa 0 hơn). Kết luận
cuối cùng viết ở mức không-causal (xem blockquote cuối §23). Gram matrix đã cache
(`results/e6_gram_dev300_{mean,rms}.npz`) — audit lại sau không cần re-extract.

**Đã đóng E6.** Chuyển sang **E7 — Downstream Amplification, ĐÃ CHẠY VÀ ĐÓNG, NO-GO** (`RESEARCH.md`
§24). Tái dùng nguyên vẹn `scripts/e2_pipeline_attenuation.py` (đã có sẵn `measure_mask_rcnn`/
`measure_dino` từ chuỗi diagnostic E1-E3, chỉ cần thêm `mask_rcnn_r50`/`dino_r50` vào `MEASURERS`),
chạy N=296 trên `dev_300` (noise tái dùng từ E6, không craft lại). Kết quả: amplification ratio
`A = rel_l2(final_stage)/rel_l2(backbone_stage)` cho chiều **không nhất quán** giữa 2 matched-pair
(Mask-RCNN: R50 giữ disturbance nhiều hơn Swin-T, ΔA=+0.162; DINO: Swin-L khuếch đại nhiều hơn R50,
ΔA=−0.410) — cả dùng `rel_l2` lẫn `cos_dist`. Đúng tiêu chí pre-registered → NO-GO, không đổi định
nghĩa `A` để cứu hypothesis.

**E8 — Task-Relevant Response Alignment, ĐÃ CHẠY VÀ ĐÓNG, NO-GO** (`RESEARCH.md` §25, script mới
`scripts/e8_task_relevant_alignment.py`). Hypothesis: không phải độ lớn `ΔF` mà là hướng — phần
`ΔF` align với gradient detection-loss của chính target (`g_t`) mới quyết định transfer. 4 bug thật
bắt được qua smoke-test trước khi chạy N=296 (đều là lần đầu `detector_task_loss` chạm 1 target
ngoài surrogate — xem "Engineering notes"). Kết quả: cả 2 matched-pair delta gần như 0, CI chồng
lấn hoàn toàn (DINO +0.0003, Mask-RCNN −0.0005) dù ASR chênh 31-66 điểm; `corr(P,ASR)` trên 8
target = **+0.054** (gần như không có gì). Giá trị `P` tuyệt đối (0.006-0.014) chỉ cao hơn 2-6 lần
so với "sàn nhiễu ngẫu nhiên" lý thuyết (~1/√D, D~70k-190k chiều đã pool) — gợi ý cosine-alignment
thô trên vector pool+concat toàn backbone (chiều quá cao) không phải cách operationalize tốt, dù
tín hiệu thật (nếu có) có thể nằm trong 1 subspace hạng thấp chưa thử. NO-GO, không đổi metric để
cứu.

**E9 — Response-Coupling Intervention, ĐÃ CHẠY VÀ ĐÓNG, NO-GO** (`RESEARCH.md` §26, script mới
`scripts/n9_response_coupling_intervention.py`). Chuyển từ correlation sang intervention: craft
`delta_coupling` = OSFD + chiếu low-pass radial-FFT (băng "low" của E4, `radius_frac≤1/3`, không
hyperparameter mới) sau mỗi step, lockstep với `delta_base` (OSFD chuẩn) để so sánh compute-matched.
Pilot N=20, 100 step, cả 8 target: **ΔASR âm ở 8/8 target** (lớn nhất `yolox_l` −47.5,
`yolov3_d53` −35.7 — ngược hẳn hypothesis vì đây là target CNN dễ transfer, không phải Swin khó),
**ΔC_response dao động ±0.001-0.009 quanh 0**, không có xu hướng tăng. Cả 2 vế hypothesis đều fail,
pattern đồng nhất chiều trên toàn bộ target → không cần scale N=296. Diễn giải: ép cứng `δ` về 1
băng tần suốt cả 100 step nhiều khả năng giới hạn nghiêm trọng không gian tìm kiếm optimization
(khác hẳn E4's post-hoc inject-1-lần lên noise đã crafted xong), giải thích ASR giảm đều mọi nơi
thay vì chọn lọc.

**Câu hỏi mở đầu E7 (tại sao ASR gap giữa 2 head cùng Swin backbone không tỷ lệ với C_response gap)
vẫn chưa có lời giải sau cả E7, E8, E9** — không theo đuổi thêm biến thể của cả 3 hướng trong phiên
này (kể cả proxy khác cho E9, vd soft regularizer thay vì hard projection, hay radius khác). Nếu
tiếp tục cần hướng đo/can thiệp khác hẳn, nên coi là 1 thực nghiệm mới pre-register riêng, không
phải patch E7/E8/E9. **Không có proxy phía surrogate nào đã thử làm tăng được `C_response` đáng tin
cậy** — rào cản cụ thể cho bất kỳ ai muốn biến `C_response` thành method tấn công thật.

**H1 (nguồn gốc transfer difficulty = backbone/representation, không phải detector head) — giờ
support bởi 2 matched-pair độc lập, khá chắc**:

**H1 (nguồn gốc transfer difficulty = backbone/representation, không phải detector head) — giờ
support bởi 2 matched-pair độc lập, khá chắc**:

| pair | head/decoder | backbone | ASR(local) | ASR(path) | Δ | 95% CI |
|---|---|---|---|---|---|---|
| `dino_swin_l` | DINO | Swin-L | 33.0% | 39.1% | +6.2 | [+4.6,+7.8] |
| `dino_r50` | DINO (giống hệt) | R50 | 99.2% | 98.8% | −0.3 | [−0.9,+0.2] cắt 0 |
| `mask_rcnn_swin_t` | Mask R-CNN | Swin-T | 67.8% | 71.1% | +3.2 | [+1.9,+4.6] |
| `mask_rcnn_r50` | Mask R-CNN (giống hệt) | R50 | 99.3% | 98.6% | −0.7 | [−1.22,−0.22] không cắt 0 (âm) |

2 detector-head family khác hẳn nhau (DETR-style decoder vs two-stage RPN+RoI), cùng đổi
Swin→R50 giữ nguyên head, cùng nhảy ASR_local lên ~99%. Chi tiết + cách đọc `mask_rcnn_r50`'s CI
âm (không mâu thuẫn H1, khớp pattern "gần-ceiling → path hơi âm" đã thấy ở mọi CNN target khác):
`RESEARCH.md` §21, mục "Matched-pair #2".

**H2 (path-averaging đặc biệt hữu ích khi backbone Swin) — ĐÃ ĐÓNG, kết luận: chỉ đúng cho DINO,
không generalize sang Mask-Swin-T**: DINO-only N=49 novelty-control cho `interaction=+6.0,
CI=[+0.9,+10.5]` không cắt 0 (xem `RESEARCH.md` §20). `mask_rcnn_swin_t` được test cùng chuẩn ở
N=50 (2026-08-29): `interaction=+2.48, CI=[−2.96,+8.0]` **cắt 0** — và quan trọng hơn, tăng N từ
20 (pilot, interaction +5.2, CI cắt 0) lên 50 làm signal **yếu đi** (point estimate giảm, same_side
0.915→0.820) thay vì siết chặt về phía dương như đã xảy ra với DINO — loại trừ khả năng đây chỉ là
thiếu power. Kết luận cuối: **OSFD-path interaction là hiện tượng đặc thù DINO/decoder DINO, không
phải property chung của backbone Swin nói chung**. H1 (nguồn gốc transfer difficulty = backbone)
vẫn support mạnh và KHÔNG bị ảnh hưởng bởi kết quả này — H1 và H2 là 2 câu hỏi tách biệt. Không cần
chạy thêm Mask novelty-control ở N lớn hơn — câu hỏi đã đủ dữ liệu để đóng (xem `RESEARCH.md` §20,
mục "Mask-Swin-T N=50").

**Compute-matched control (objection "N6-B thắng chỉ vì 3x gradient evaluation, không phải path
structure") — ĐÃ ĐÓNG, GO cho DINO**: 3-way lockstep mới (`osfd_local`/`rrb_avg_k3`/`path_m3`,
script `scripts/n6cm_compute_matched_pilot.py`) — pilot N=20 cho DINO point estimate +8.0 nhưng CI
chạm 0 (chưa decisive); confirmation N=49 (2026-08-29): `path−avg` trên DINO = **+6.77, CI=
[+2.67,+11.15] KHÔNG cắt 0** → đạt GO theo tiêu chí pre-registered (≥+3 và CI>0). Kết luận: gain
của N6-B trên DINO không giải thích được chỉ bằng "nhiều gradient evaluation hơn" — cấu trúc path
đóng góp thật, vượt trên naive K=3 RRB averaging cùng compute. `yolox_l` đảo chiều từ tín hiệu sạch
ở N=20 (+6.93, CI không cắt 0) sang cắt 0 ở N=49 (+3.19) — không bền, không dùng làm evidence.
`mask_rcnn_swin_t` inconclusive cả 2 N. `rcg_avg` (Phase M cũ) bị loại khỏi evidence chính do
không khớp `rrb_avg_k3` (N6-A) trên cùng ảnh — nghi ngờ là implementation/RNG artifact riêng của
script đó. Theo stopping rule đã chốt trước khi chạy, **không scale N=300** — xem `RESEARCH.md` §22.

## Đang ở đâu trong roadmap

Roadmap "prove" đã thống nhất với user (N6-B = path-integrated OSFD gradient, candidate mạnh nhất
project, không đổi method nữa — chỉ đi chứng minh):

`Held-out val_100 → Prior-art verification → Mechanism proof → Novelty control → Breadth test`

1. **[DONE]** Held-out `val_100` — DINO **CONFIRMED mạnh** (+6.5 ASR, CI=[+2.9,+10.3] không cắt
   0, robust qua 3 lần đo độc lập). Mask-Swin-T KHÔNG đạt tiêu chí held-out (CI cắt 0 ở N=98, dù
   `dev_300` N=300 đạt). → `RESEARCH.md` §16-B.
2. **[DONE]** Prior-art — IJCNN 2024 CLOSED (zero overlap, full-text đã đọc). HIFA vẫn OPEN
   (chỉ abstract/TLDR, risk đã hạ sau fresh scan vì IG dùng như attribution chứ không phải thay
   ascent gradient). → `RESEARCH.md` §18, §18-C.
3. **[DONE]** Mechanism proof (gradient alignment) — NO-GO theo pre-registered threshold (biên độ
   Δcos chỉ ~11-20% ngưỡng dù CI không cắt 0). Dấu khớp ΔASR trên 4/4 model nhưng không đủ mạnh là
   cơ chế chính. User quyết định KHÔNG đo alignment dọc trajectory (tránh post-hoc rescue). →
   `RESEARCH.md` §19.
4. **[DONE — cả 2 hard target, kết luận cuối]** Novelty control (`G_osfd` vs `G_det`) — pilot N=20
   (3 target) cho pattern đúng hướng nhưng CI chưa cắt 0. **DINO-only N=49: interaction +6.0,
   CI=[+0.9,+10.5] KHÔNG cắt 0** — confirmed cho DINO. **Mask-Swin-T N=50 (2026-08-29): interaction
   +2.48, CI=[−2.96,+8.0] CẮT 0**, signal yếu đi so với pilot N=20 (không phải thiếu power) —
   KHÔNG generalize. Kết luận H2: đặc thù cho DINO, không phải property chung của backbone Swin. →
   `RESEARCH.md` §20.
5. **[DONE — 2 matched-pair]** Breadth test — `dino_r50` VÀ `mask_rcnn_r50` đều đã thêm vào
   `MODEL_REGISTRY`, eval xong (reuse noise `dev_300`, N=296, gần như free về compute). **H1 giờ
   support bởi 2 pair độc lập** (xem bảng ở trên). Không cần thêm matched-pair thứ 3 trừ khi có lý
   do cụ thể mới. → `RESEARCH.md` §21.
6. **[DONE — thêm sau roadmap 5-bước, GO cho DINO]** Compute-matched control — pilot N=20 rồi
   confirmation N=49, 3-way lockstep `osfd_local`/`rrb_avg_k3`/`path_m3`. DINO
   `path−avg=+6.77, CI=[+2.67,+11.15]` không cắt 0 → loại trừ "compute nhiều hơn" như lời giải
   thích thay thế cho gain của N6-B. → `RESEARCH.md` §22.

Roadmap 6 bước trên là cho N6-B cụ thể (candidate method), đã đóng hoàn toàn. Phiên 2026-09-05 mở
một **nhánh song song mới**, không phải bước tiếp theo của roadmap N6-B: quay lại câu hỏi H1 còn để
ngỏ ("tại sao backbone family là nguồn gốc gap") bằng E6 (xem "Trạng thái hiện tại" + `RESEARCH.md`
§23). E6 là mechanism candidate cho **H1**, không phải method mới cạnh tranh với N6-B — hai việc
độc lập, không thay thế nhau. Bước tiếp theo cho E6 (chưa làm): tìm proxy chỉ-dùng-surrogate cho
`C_response` (hiện cần forward target thật, chỉ dùng được cho diagnostic).

## Engineering notes quan trọng (không nằm trong RESEARCH.md, dễ quên/redo nhầm)

- **Checkpoint `dino_r50`/`mask_rcnn_r50` không tự động có sẵn trên checkout mới** — dù đã có trong
  `MODEL_REGISTRY` từ phiên trước, file `.pth` không commit (gitignored). Tải lại bằng:
  `mim download mmdet --config dino-4scale_r50_8xb2-12e_coco --dest checkpoints/` và
  `mim download mmdet --config mask-rcnn_r50_fpn_1x_coco --dest checkpoints/` (network access hoạt
  động bình thường trong sandbox, ~250MB + ~170MB). Chạy `scripts/check_env.py` sau khi tải để xác
  nhận cả 9 model resolve được config+checkpoint trước khi làm gì khác.
- **`results/` trống hoàn toàn trên checkout mới** (gitignored, không mang theo khi clone) — không
  có noise, prediction cache, hay CSV nào từ phiên trước. Bất kỳ diagnostic nào định "tái dùng noise
  đã crafted" đều phải **recraft trước** bằng `scripts/run_attack.py` (config đọc từ `runs/*.json`
  cũ để khớp hyperparameter, nhưng RNG draw sẽ khác — verify lại bằng cách so ASR/mAP_drop của noise
  mới với số cũ trong `runs/`, nên khớp trong biên ~1 điểm nếu config đúng).
- **Bug bootstrap-with-replacement cho thống kê Gram/CKA** (bắt được khi viết `e6_response_coupling.py`,
  xem `RESEARCH.md` §23): resample-with-replacement chuẩn (đúng quy ước `paired_bootstrap_asr_delta`
  dùng cho ASR-delta ở khắp project) **thiên lệch có hệ thống** khi áp cho CKA/Gram-based statistic —
  ảnh trùng lặp có similarity=1 tuyệt đối với chính nó, thổi phồng Gram. Nếu viết diagnostic mới nào
  khác cũng dùng Gram/kernel-matrix similarity (không phải ASR/mAP đơn giản), **không dùng
  with-replacement bootstrap** — dùng subsampling không hoàn lại (m<n, vd 80%) thay thế. Convention
  ASR-delta bootstrap hiện có (N4, N6-B, N6-CM) không bị ảnh hưởng — bug này chỉ xảy ra với thống kê
  dạng Gram/kernel, không phải tỷ lệ đếm như ASR.
- **`center_gram` trong `e6_response_coupling.py` dùng công thức O(n²)** (row/col/grand-mean), không
  phải `H@K@H` matmul tường minh O(n³) — quan trọng nếu code chạm vào bootstrap ở N lớn (N=300+),
  vì O(n³) từng làm phần CI-computation mất ~15 phút ở N=120 trước khi tối ưu, có thể trở thành
  cost chính của cả diagnostic nếu không sửa.
- **`e6_response_coupling.py` giờ có `jackknife_matched_pair_delta()` và `--save-gram-npz`** — delete-1
  jackknife SE tính trực tiếp trên statistic difference (không phải trên từng CKA riêng lẻ, để giữ
  đúng dependence giữa 2 term dùng chung ảnh/surrogate); Gram matrix cache ra `.npz`
  (`results/e6_gram_dev300_{mean,rms}.npz`) để audit CI sau này không cần re-extract feature.
- **`scripts/e2_pipeline_attenuation.py`'s `MEASURERS` giờ có 4 model** (thêm `mask_rcnn_r50` →
  `measure_mask_rcnn`, `dino_r50` → `measure_dino`, dùng lại đúng function cũ không đổi logic — cả
  2 function đã architecture-generic, không hardcode gì riêng cho Swin). Nếu cần thêm matched-pair
  thứ 3 trong tương lai cho pipeline-attenuation, chỉ cần model đó cùng detector family (Mask-RCNN
  hoặc DINO/Deformable-DETR) là dùng lại được ngay, không cần viết measurer mới.

- **`transfer_attack/losses.py::detector_task_loss` trước phiên này CHỈ từng chạy trên surrogate
  (`faster_rcnn_r50`, cho mi_fgsm)** — E8 là lần đầu chạy trên target bất kỳ, và bắt được 4 bug thật
  (chi tiết đầy đủ ở `RESEARCH.md` §25, mục "Engineering"), tóm tắt để không lặp lại nếu viết script
  mới cần gradient qua `.loss()` của target:
  1. DINO's `pre_decoder` chỉ tạo 3 arg bắt buộc của `DINOHead.loss()`
     (`enc_outputs_class`/`enc_outputs_coord`/`dn_meta`) khi `self.training=True` — cần bật
     `model.train()` NHƯNG re-freeze mọi `BatchNorm`/`Dropout` submodule về `.eval()` ngay sau đó
     (pattern `_set_training_keep_norm_frozen()` trong `e8_task_relevant_alignment.py`) vì
     `dino_r50`/`mask_rcnn_r50` dùng BatchNorm thật (không phải GN/LN như 2 biến thể Swin) — bật
     train() trần sẽ dùng batch-statistics của 1 ảnh, sai lệch hoàn toàn so với eval-mode forward
     dùng ở mọi nơi khác trong project.
  2. `gt_boxes` cần `.to(device)` tường minh trước `build_gt_data_sample` (khác `craft_one_image`
     vốn đã làm sẵn việc này) — thiếu sẽ crash sâu bên trong DINO's `CdnQueryGenerator`.
  3. Model có `out_indices` bao gồm stage bị `frozen_stages` đóng băng (vd `mask_rcnn_r50`'s stage
     0) sẽ có `requires_grad=False` — guard `retain_grad()` bằng `if t.requires_grad` trước khi gọi.
  4. Mask R-CNN's `roi_head.mask_head.loss()` cần `gt_instances.masks` (segmentation GT) mà project
     không bao giờ tạo (chỉ đánh giá box mAP/ASR) — tạm gỡ `roi_head.mask_head=None` trước khi gọi
     `detector_task_loss`, khôi phục lại sau, đúng phạm vi "chỉ box task-loss".

- **Bug đã fix trong `scripts/evaluate.py`** (`evaluate_one_model`, 2 chỗ build coco-results cho
  "clean" row và cho `adv_preds`): cache dự đoán dùng chung thư mục giữa các manifest khác nhau,
  code cũ build từ toàn bộ dict cache thay vì lọc theo đúng `image_ids`/`common_ids` hiện tại →
  `KeyError` khi 2 manifest disjoint dùng chung cache. Đã sửa (filter dict trước khi gọi
  `to_identity_coco_results`) — fix đã áp dụng, ảnh hưởng mọi script gọi
  `evaluate_mod.evaluate_one_model`, không cần sửa lại.
- **DINO (`dino_swin_l` và `dino_r50`) dùng gradient checkpointing nội bộ**, không tương thích
  `torch.autograd.grad(loss, inputs=...)`. Phải dùng `loss.backward()` + đọc `.grad` trực tiếp.
  Nếu viết script mới cần gradient qua model DINO bất kỳ, nhớ pattern này.
- **`transfer_attack/models.py` giờ có 9 model** (7 gốc + `dino_r50` + `mask_rcnn_r50`, cả 2 group
  "D" — group mới, không phải A/B/C cũ vì đây là trục backbone khác. `group` chỉ là metadata hiển
  thị, không dùng trong logic filter ở đâu — an toàn để thêm group mới). Checkpoint mới:
  `checkpoints/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth` (~263MB) và
  `checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth` (~178MB), cả 2 từ
  `download.openmmlab.com` (network access trong sandbox hoạt động bình thường). Config resolve
  tự động qua `.mim/configs` cài sẵn trong mmdet package, không cần tải config riêng.
- **3 script mới đã viết trong phiên này** (không có trong repo trước đó):
  - `scripts/n6b_alignment_diagnostic.py` — mechanism-proof: `cos(g_local^sur, g_target)` vs
    `cos(g_path^sur, g_target)`, tái dùng noise đã crafted, target's own gradient không RRB.
  - `scripts/n6b_novelty_control.py` — novelty control: 4 variant `det_local`/`det_path`/
    `osfd_local`/`osfd_path`, `osfd_*` tái dùng `craft_paired_local_path` từ `n6b_path_pilot.py`
    (import), `det_*` viết mới (task loss, không RRB, đúng baseline `mi_fgsm` sẵn có). Dùng
    `--targets <model>` để giới hạn phạm vi (đã dùng cho DINO-only N=49).
  - `scripts/n6b_breadth_eval.py` — breadth test: eval model MỚI trên noise ĐÃ CRAFTED sẵn (không
    craft lại) — đã dùng cho cả `dino_r50` và `mask_rcnn_r50`, có thể tái dùng cho matched-pair
    thứ 3 trong tương lai nếu cần (thêm ModelSpec + checkpoint rồi gọi script với `--models
    <tên>`).
- **1 script mới thêm ở phiên compute-matched control (2026-08-29)**:
  - `scripts/n6cm_compute_matched_pilot.py` — 3-way lockstep `osfd_local`/`rrb_avg_k3`/`path_m3`
    trong CÙNG 1 hàm craft (không compose lại 2 hàm cũ `craft_paired_local_path`/
    `craft_paired_avg_cr` vì mỗi hàm tự quản lý RNG snapshot riêng theo step, ghép lại sẽ phá vỡ
    lockstep) — 1 RNG snapshot/step, restore riêng trước draw của local, trước draw ĐẦU của avg
    (draw 2-3 của avg advance tự nhiên, đúng bản chất "K view độc lập"), và trước MỖI draw của
    path (path chia sẻ đúng 1 augmentation instance qua cả 3 λ, theo thiết kế gốc
    `n6b_path_pilot.py`). Default models cho pilot: `{faster_rcnn_r50, yolox_l,
    mask_rcnn_swin_t, dino_swin_l}` (giống N6-A/RCG, không phải full 7-model registry).
    `--out-csv` cần chỉ định riêng nếu chạy nhiều N (mặc định ghi đè
    `results/n6cm_compute_matched_pilot_summary.csv`) — N=49 đã dùng
    `--out-csv results/n6cm_compute_matched_pilot_n50.csv` để giữ cả 2 file.

## File kết quả liên quan (phiên này)

**Lưu ý (2026-09-05)**: toàn bộ danh sách CSV dưới đây là từ phiên 2026-08-29, **không còn tồn tại
trên checkout hiện tại** (`results/` gitignored, trống khi bắt đầu phiên này) — giữ lại danh sách vì
số liệu đã trích dẫn vào `RESEARCH.md` (đứng vững độc lập với việc file gốc còn hay mất), không phải
vì file còn đọc được. Danh sách mới của phiên 2026-09-05 (E6) nằm ngay dưới đây, ĐANG tồn tại:

- `results/e6_response_coupling_dev300.csv` / `_summary.csv` — **bảng chính**, N=296, mean-pool
  (bảng dùng trong `RESEARCH.md` §23), kèm `results/e6_gram_dev300_mean.npz` (Gram cache, audit lại
  không cần re-extract)
- `results/e6_response_coupling_dev300_rms.csv` / `_summary.csv` + `e6_gram_dev300_rms.npz` — N=296,
  RMS-pool (sensitivity)
- `results/e6_response_coupling_n120*.csv`, `results/e6_response_coupling.csv`/`_rms.csv` (N=49) —
  các bước sensitivity-check trước khi scale (N=49→120→296), giữ lại để trace lại tiến trình siết
  chặt CI theo N nếu cần
- `results/e7_pipeline_attenuation_matched.csv` — E7, N=296, per-stage cos_dist/rel_l2 cho cả 4
  matched-pair model (`mask_rcnn_r50`/`mask_rcnn_swin_t`/`dino_r50`/`dino_swin_l`), NO-GO (§24)
- `results/e8_task_relevant_alignment.csv` — E8, N=296, `mean_P`/CI cho cả 8 target, NO-GO (§25)
- `results/n9_response_coupling_pilot_n20.csv` — E9, pilot N=20, ΔASR/ΔC_response cho cả 8 target,
  NO-GO (§26). Noise crafted mới nằm ở `results/noise/dev_50/n9_base/` và `n9_coupling/`.
- `runs/run_attack_osfd_dev_50_20260905T085244Z.json`, `runs/run_attack_osfd_dev_300_20260905T095420Z.json`
  (N=120, dùng cho sensitivity), `runs/run_attack_osfd_dev_300_20260905T110900Z.json` (N=296, dùng
  cho bảng chính) — noise recraft mới, KHÔNG phải cùng file tensor với noise gốc §16-22 (xem
  "Engineering notes")

Danh sách CSV cũ (2026-08-29, không còn trên đĩa):

- `results/n6b_path_pilot_summary_val_100.csv` — held-out confirm (bước 1)
- `results/n6b_alignment_diagnostic.csv` — mechanism-proof N=100 (bước 3)
- `results/n6b_novelty_control_summary.csv` — novelty control pilot N=20, 3 target (bước 4)
- `results/n6b_novelty_control_dino_n50.csv` — DINO-only N=49, interaction CI=[+0.9,+10.5] không
  cắt 0 (bước 4, DONE)
- `results/n6b_novelty_control_mask_n50.csv` — Mask-Swin-T N=50 (2026-08-29), interaction
  CI=[−2.96,+8.0] cắt 0, KHÔNG generalize (bước 4, ĐÃ ĐÓNG — xem `RESEARCH.md` §20)
- `results/n6b_breadth_dino_r50.csv` — matched-pair #1 (bước 5, DONE)
- `results/n6b_breadth_mask_rcnn_r50.csv` — matched-pair #2 (bước 5, DONE)
- `results/n6cm_compute_matched_pilot_summary.csv` — compute-matched control pilot N=20 (bước 6,
  promising nhưng chưa decisive trên DINO)
- `results/n6cm_compute_matched_pilot_n50.csv` — compute-matched control confirmation N=49
  (bước 6, DONE — GO cho DINO, `path−avg=+6.77, CI=[+2.67,+11.15]` không cắt 0)
- `runs/run_attack_osfd_n6b_*_val_100_*.json`, `runs/run_attack_n6bctl_*.json`,
  `runs/run_attack_osfd_n6cm_*.json` — run logs (`EXPERIMENTS.md` đã regenerate, tự sinh, không
  sửa tay)

## Việc KHÔNG làm (đã quyết định rõ, tránh redo/tranh luận lại)

- Không sweep `M` (=3 pre-registered từ đầu N6-B, giữ nguyên xuyên suốt kể cả sau NO-GO)
- Không đo alignment dọc theo trajectory sau khi mechanism-proof NO-GO (tránh post-hoc rescue)
- Không chạy N6-C (cross-layer relational feature distortion) trừ khi N6-B thất bại hoàn toàn
- Không tune hyperparameter dựa trên kết quả `val_100` (chỉ dùng đúng 1 lần để confirm, đã dùng)
- Không theo route ngoài luồng (vd sci-hub) để lấy full-text HIFA
- **Không viết "H2 đã confirmed cho mọi Swin target"** — đã test cả 2 hard target nhóm C ở N đủ
  lớn: chỉ DINO confirmed (interaction CI không cắt 0), Mask-Swin-T đã test và KHÔNG generalize
  (CI cắt 0, signal yếu đi khi tăng N) — đây là kết luận cuối, không phải "chưa đủ dữ liệu"
- Không chạy thêm Mask novelty-control ở N lớn hơn (N=100+) — signal đã đi ngược hướng cần thiết
  khi tăng N 20→50 (yếu đi, không siết chặt), nên đây không phải vấn đề thiếu power; thêm compute
  khó có khả năng đổi kết luận
- Không launch novelty-control N=100 full 4-target hay compute lớn khác mà không hỏi user trước
  — user đã nhiều lần nhấn mạnh ưu tiên information-gain/compute-cost, không phải "chạy cho chắc"
- Không chạy MI-FGSM ở `dev_300`/`val_100` để "làm đẹp" bảng final baseline — user quyết định
  MI-FGSM ở `dev_50` N=49 đã đủ vai trò (chứng minh gap task-loss≪feature-attack), thêm N không
  đổi kết luận, information gain thấp so với chi phí (~50 phút)
- Không scale compute-matched control (`n6cm_compute_matched_pilot.py`) lên N=300 — đã đạt GO
  sạch ở N=49 theo đúng tiêu chí pre-registered (point estimate ≥+3 và CI>0 trên DINO), stopping
  rule đã chốt trước khi chạy nói rõ không cần N=300 trừ khi final table cần precision cao hơn
- Không dùng `rcg_avg` (Phase M, `results/m_rcg_pilot_summary.csv`) làm evidence cho câu hỏi
  compute-matched nữa — không khớp `rrb_avg_k3` (N6-A) trên cùng ảnh dù cùng config danh nghĩa,
  nghi ngờ là RNG/implementation artifact riêng của script đó (xem `RESEARCH.md` §22)

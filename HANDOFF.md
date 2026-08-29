# Session handoff

Mục đích: cho phiên Claude tiếp theo đọc nhanh để tiếp tục đúng ngữ cảnh sau khi máy restart,
không cần đọc lại toàn bộ `RESEARCH.md` từ đầu. `RESEARCH.md` vẫn là nguồn sự thật đầy đủ cho
mọi kết quả nghiên cứu (đặc biệt §16-§21 cho phiên này) — file này chỉ tóm tắt điều hướng +
engineering context không nằm trong journal.

**Cập nhật lần cuối**: 2026-08-29. Roadmap "prove" 5-bước cho N6-B đã đóng hoàn toàn (xem mục dưới).
Sau đó, project chuyển sang giai đoạn "paper closure" theo đề nghị của user: (1) final baseline
table (MI-FGSM/OSFD/N6-B path-M3, lắp từ dữ liệu sẵn có, 0 compute mới — chưa ghi vào
`RESEARCH.md`, chỉ trình bày trong hội thoại) rồi (2) **compute-matched control — ĐÃ CHẠY, GO cho
DINO** (xem "Trạng thái hiện tại" và `RESEARCH.md` §22). Đây là compute-experiment cuối cùng user
chủ định chạy trước khi chuyển hẳn sang final paper comparison/writing — **không launch thêm
compute lớn** trừ khi final baseline review phát hiện lỗ hổng bắt buộc phải lấp (vd MI-FGSM ở
`dev_300` để đồng nhất N, đã được user cân nhắc và quyết định KHÔNG làm — xem "Việc KHÔNG làm").

## Trạng thái hiện tại (quan trọng nhất, đọc trước)

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

## Engineering notes quan trọng (không nằm trong RESEARCH.md, dễ quên/redo nhầm)

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

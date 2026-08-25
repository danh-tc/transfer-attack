# Session handoff

Mục đích: cho phiên Claude tiếp theo đọc nhanh để tiếp tục đúng ngữ cảnh sau khi máy restart,
không cần đọc lại toàn bộ `RESEARCH.md` từ đầu. `RESEARCH.md` vẫn là nguồn sự thật đầy đủ cho
mọi kết quả nghiên cứu (đặc biệt §16-§21 cho phiên này) — file này chỉ tóm tắt điều hướng +
engineering context không nằm trong journal.

**Cập nhật lần cuối**: 2026-08-25. Cả 2 việc ưu tiên của lần dừng trước (DINO-only N=49
novelty-control + `mask_rcnn_r50` matched-pair breadth) **đã xong**. Chưa có việc mới được user
chốt cho phiên tiếp theo — đọc phần "Trạng thái hiện tại" bên dưới rồi hỏi user muốn đi hướng nào
tiếp (đừng tự launch thêm compute lớn mà không hỏi, đúng tinh thần thận trọng đã giữ xuyên suốt).

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

**H2 (path-averaging đặc biệt hữu ích khi backbone Swin) — vẫn CHỈ confirmed cho DINO, chưa cho
Mask**: DINO-only N=49 novelty-control cho `interaction=+6.0, CI=[+0.9,+10.5]` không cắt 0 (xem
`RESEARCH.md` §20). `mask_rcnn_swin_t` mới chỉ có ASR-gain riêng confirmed (§16-B, +3.2 dev_300)
nhưng chưa có bài test interaction (`G_osfd` vs `G_det`) ở N đủ lớn — pilot N=20 cho +5.2 nhưng CI
cắt 0.

**Việc CÓ THỂ làm tiếp nếu muốn đóng nốt H2 cho Mask** (chưa quyết định, hỏi user trước khi làm):
chạy `scripts/n6b_novelty_control.py --manifest data/manifests/dev_50.json --n-images 50
--targets mask_rcnn_swin_t --out-csv results/n6b_novelty_control_mask_n50.csv` — y hệt cách đã làm
cho DINO, tốn ~50 phút craft. Đây sẽ là việc cuối cùng để có thể phát biểu H2 cho cả 2 hard target
thay vì chỉ DINO.

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
4. **[DONE cho DINO, chưa làm cho Mask]** Novelty control (`G_osfd` vs `G_det`) — pilot N=20 (3
   target) cho pattern đúng hướng nhưng CI chưa cắt 0. **DINO-only N=49 (đã xong): interaction
   +6.0, CI=[+0.9,+10.5] KHÔNG cắt 0** — confirmed cho DINO cụ thể. Mask-Swin-T chưa được re-run
   ở N lớn tương tự (vẫn dừng ở pilot N=20, CI cắt 0) — xem "Việc CÓ THỂ làm tiếp" ở trên. →
   `RESEARCH.md` §20.
5. **[DONE — 2 matched-pair]** Breadth test — `dino_r50` VÀ `mask_rcnn_r50` đều đã thêm vào
   `MODEL_REGISTRY`, eval xong (reuse noise `dev_300`, N=296, gần như free về compute). **H1 giờ
   support bởi 2 pair độc lập** (xem bảng ở trên). Không cần thêm matched-pair thứ 3 trừ khi có lý
   do cụ thể mới. → `RESEARCH.md` §21.

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

## File kết quả liên quan (phiên này)

- `results/n6b_path_pilot_summary_val_100.csv` — held-out confirm (bước 1)
- `results/n6b_alignment_diagnostic.csv` — mechanism-proof N=100 (bước 3)
- `results/n6b_novelty_control_summary.csv` — novelty control pilot N=20, 3 target (bước 4)
- `results/n6b_novelty_control_dino_n50.csv` — DINO-only N=49, interaction CI=[+0.9,+10.5] không
  cắt 0 (bước 4, DONE)
- `results/n6b_breadth_dino_r50.csv` — matched-pair #1 (bước 5, DONE)
- `results/n6b_breadth_mask_rcnn_r50.csv` — matched-pair #2 (bước 5, DONE)
- `runs/run_attack_osfd_n6b_*_val_100_*.json`, `runs/run_attack_n6bctl_*.json` — run logs (chưa
  chạy `scripts/gen_experiment_log.py` để cập nhật `EXPERIMENTS.md` — tự sinh, không sửa tay)

## Việc KHÔNG làm (đã quyết định rõ, tránh redo/tranh luận lại)

- Không sweep `M` (=3 pre-registered từ đầu N6-B, giữ nguyên xuyên suốt kể cả sau NO-GO)
- Không đo alignment dọc theo trajectory sau khi mechanism-proof NO-GO (tránh post-hoc rescue)
- Không chạy N6-C (cross-layer relational feature distortion) trừ khi N6-B thất bại hoàn toàn
- Không tune hyperparameter dựa trên kết quả `val_100` (chỉ dùng đúng 1 lần để confirm, đã dùng)
- Không theo route ngoài luồng (vd sci-hub) để lấy full-text HIFA
- **Không viết "H2 đã confirmed cho mọi Swin target"** — chỉ DINO có interaction-test confirmed;
  Mask-Swin-T mới có ASR-gain riêng, chưa có interaction-test ở N đủ lớn (xem "Việc CÓ THỂ làm
  tiếp" ở trên nếu muốn đóng nốt)
- Không launch novelty-control N=100 full 4-target hay compute lớn khác mà không hỏi user trước
  — user đã nhiều lần nhấn mạnh ưu tiên information-gain/compute-cost, không phải "chạy cho chắc"

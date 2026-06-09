# Mini LLaMA 4 MoE — code ghi nhớ bài học

Repo bổ sung file `llama4_moe_lesson_vi.py` để bạn có một bản code ngắn, dễ đọc và có thể đưa lên GitHub nhằm ôn lại bài học **Implementing LLaMA 4 from Scratch**.

## Notebook/bài học nói về gì?

Bài học giải thích cách xây dựng một **decoder-only Transformer kiểu LLaMA 4 thu nhỏ** với trọng tâm là **Mixture-of-Experts (MoE)**. Thay vì mọi token đi qua cùng một feed-forward network, mỗi token được một router chọn `top_k` expert phù hợp để xử lý.

## Mục đích của code

Code không nhằm tái tạo LLaMA 4 thật. Mục đích là giúp ghi nhớ các ý chính:

- `CharTokenizer`: biến từng ký tự thành token ID để đơn giản hóa tokenizer.
- `RMSNorm`: normalization kiểu LLaMA, nhẹ hơn LayerNorm vì không trừ mean.
- `apply_rope`: Rotary Positional Embedding để đưa thông tin vị trí vào query/key.
- `CausalSelfAttention`: attention có causal mask để token hiện tại không nhìn tương lai.
- `MoEFeedForward`: router chọn `top_k` expert cho từng token và cộng thêm shared expert.
- `MiniLlama4MoE`: ghép embedding, Transformer blocks, final norm và language-model head.

## Cách chạy

Cài PyTorch trước, ví dụ:

```bash
python -m pip install torch
```

Train nhanh vài bước trên CPU:

```bash
python llama4_moe_lesson_vi.py --steps 20 --batch-size 8 --generate 80 --device cpu
```

Nếu có GPU/CUDA, có thể bỏ `--device cpu`:

```bash
python llama4_moe_lesson_vi.py --steps 200 --batch-size 32 --generate 120
```

## Output kỳ vọng

Model được train trên một đoạn văn bản rất nhỏ về Facebook, nên kết quả sinh text chủ yếu là minh họa/memorization. Nếu prompt gần với dữ liệu train, output sẽ dễ nhìn hơn; nếu prompt lạ, output có thể sai chính tả hoặc thiếu logic. Đây là điều bình thường với model đồ chơi.

## Gợi ý học tiếp

- Tăng `hidden_dim`, `num_layers`, `num_experts` trong `ModelConfig`.
- Thử đổi `top_k` từ `2` sang `1` hoặc `3` để quan sát routing.
- Thay `TRAIN_TEXT` bằng dữ liệu dài hơn.
- Thêm temperature/top-k sampling nâng cao trong hàm `generate`.
- In `top_indices` trong `MoEFeedForward` để xem token được route tới expert nào.

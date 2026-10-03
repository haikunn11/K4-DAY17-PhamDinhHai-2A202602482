# Kết quả triển khai Memory Agent

## Phạm vi đã hoàn thành

- Offline mode có kết quả xác định và không cần API key.
- Live OpenAI mode đọc khóa từ `.env`, khởi tạo `ChatOpenAI` và đã được smoke-test thành công với `gpt-4o-mini`.
- Live end-to-end test đã ghi tên, nơi ở và nghề nghiệp trong một thread, khởi tạo lại agent rồi recall đúng cả ba fact ở thread mới.
- Baseline chỉ giữ memory trong từng thread, không có persistent profile.
- Advanced kết hợp `User.md`, rolling summary có giới hạn và các message gần nhất.
- Correction mới thay thế location và profession cũ.
- Câu hỏi, câu đùa và thông tin chuyến đi không ghi đè profile.
- Benchmark sử dụng state tạm biệt lập, nên có thể chạy lại mà không bị dữ liệu cũ làm sai kết quả.

## Kiểm thử

```text
9 passed
```

Bộ test bao phủ:

- Chuẩn hóa provider.
- Đọc, ghi, sửa và đo kích thước `User.md`.
- Fact extraction, correction và thông tin nhiễu.
- Compact trigger và giới hạn recent messages.
- Cross-session recall.
- So sánh prompt load trên hội thoại dài.
- Recall toàn bộ fact được yêu cầu trong cả hai dataset.

## Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1577 | 14848 | 0.000 | 0.230 | 0 | 0 |
| Advanced | 2894 | 24421 | 1.000 | 1.000 | 310 | 0 |

Trong hội thoại ngắn, Advanced tốn nhiều prompt-token hơn vì mỗi lượt phải tải persistent profile và quản lý memory. Đây là overhead có chủ ý để đổi lấy khả năng nhớ qua nhiều session.

## Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 350 | 22631 | 0.000 | 0.225 | 0 | 0 |
| Advanced | 521 | 15122 | 1.000 | 1.000 | 237 | 6 |

Trong hội thoại dài, compact memory giúp Advanced xử lý ít hơn khoảng 33% prompt-token so với Baseline, đồng thời vẫn giữ recall đầy đủ. Đây là lợi ích chính của compact: giảm lượng context phải gửi lặp lại, không nhất thiết giảm số token mà agent sinh ra.

## Bonus kỹ thuật

Triển khai hiện có hai mở rộng thuộc nhóm 90–100 điểm:

- **Entity extraction có cấu trúc**: profile được lưu theo các field như name, location, profession, interests, response style và sở thích.
- **Conflict handling**: correction cập nhật giá trị hiện hành thay vì lưu đồng thời fact cũ và mới; các mẫu phủ định, câu hỏi và thông tin gây nhiễu được lọc trước khi ghi.

## Rủi ro và giới hạn

- `User.md` vẫn tăng theo số loại fact được lưu, dù mỗi field không bị nhân bản vô hạn.
- Heuristic extraction có thể cần thêm pattern khi gặp cách diễn đạt ngoài dataset.
- Summary quá ngắn có thể làm mất chi tiết tạm thời; summary quá dài lại giảm lợi ích token.
- Live API phụ thuộc quota, quyền truy cập model và kết nối mạng.
- Automated test mặc định luôn dùng offline mode để không phát sinh chi phí API và giữ kết quả tái lập.

## Lệnh chạy

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m pytest -v
py -3 src\benchmark.py
```

Khóa API chỉ được lưu trong `.env`, file này đã nằm trong `.gitignore` và không được đưa vào commit.

# Bước 8 — Phân tích kết quả Memory System

## Kết quả tổng quan

| Bộ dữ liệu | Agent | Prompt tokens processed | Cross-session recall | Memory growth | Compactions |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard | Baseline | 14.848 | 0.000 | 0 bytes | 0 |
| Standard | Advanced | 24.421 | 1.000 | 310 bytes | 0 |
| Long-context | Baseline | 22.631 | 0.000 | 0 bytes | 0 |
| Long-context | Advanced | 15.122 | 1.000 | 237 bytes | 6 |

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ lưu message theo `thread_id`. Khi câu hỏi recall được gửi trong một thread mới, lịch sử của thread cũ không còn nằm trong context nên Baseline không biết tên, nghề nghiệp, nơi ở hoặc preference của người dùng. Vì vậy Baseline đạt cross-session recall bằng `0.000`.

Advanced tách thông tin ổn định khỏi lịch sử hội thoại và lưu vào `state/profiles/<user>/User.md`. Profile này được đọc lại ở mọi thread của cùng người dùng. Khi nơi ở thay đổi từ Đà Nẵng sang Huế hoặc nghề nghiệp thay đổi từ backend engineer sang MLOps engineer, cơ chế upsert thay giá trị cũ bằng correction mới. Nhờ đó Advanced đạt recall `1.000` trên cả hai bộ dữ liệu.

## 2. Vì sao Advanced có thể tốn hơn trong hội thoại ngắn?

Ở Standard Benchmark, Advanced xử lý `24.421` prompt-token, cao hơn `14.848` của Baseline. Nguyên nhân là mỗi lượt của Advanced phải mang thêm:

- Nội dung persistent profile trong `User.md`.
- Summary của phần lịch sử đã compact, nếu có.
- Các message gần nhất trong short-term memory.
- Chi phí trích xuất và cập nhật fact ổn định.

Trong hội thoại ngắn, lịch sử của Baseline chưa đủ lớn để trở thành gánh nặng. Phần profile bổ sung của Advanced vì thế là overhead lớn hơn lợi ích tiết kiệm context. Đổi lại, Advanced có khả năng nhớ qua session còn Baseline thì không.

## 3. Vì sao compact có lợi thế trong hội thoại dài?

Baseline luôn đưa toàn bộ lịch sử thread vào context. Khi số lượt và độ dài message tăng, cùng một nội dung cũ bị xử lý lại nhiều lần, làm tổng prompt-token tăng nhanh.

Advanced chỉ giữ nguyên một số message gần nhất. Khi context vượt ngưỡng, các message cũ được chuyển thành rolling summary có kích thước giới hạn. Trong Long-Context Stress Benchmark, compact xảy ra `6` lần và Advanced chỉ xử lý `15.122` prompt-token, thấp hơn khoảng `33%` so với `22.631` của Baseline, trong khi recall vẫn đạt `1.000`.

Compact chủ yếu tối ưu **prompt tokens processed**, không nhất thiết giảm **agent tokens only**, vì số token câu trả lời còn phụ thuộc nội dung và cách trình bày của agent.

## 4. Memory file tăng trưởng ra sao và có rủi ro gì?

Baseline không có persistent memory nên memory growth bằng `0`. Advanced tạo `User.md`, tăng `310` bytes trong benchmark chuẩn và `237` bytes trong stress benchmark.

Structured upsert giúp mỗi field chỉ giữ giá trị hiện hành, tránh lưu lặp vô hạn cùng một fact. Tuy vậy, hệ thống vẫn có các rủi ro:

- **Lưu sai fact:** heuristic có thể hiểu nhầm câu hỏi, câu đùa hoặc thông tin tạm thời.
- **Xung đột thông tin:** correction mới có thể không được nhận diện và fact cũ tiếp tục được sử dụng.
- **Memory tăng theo thời gian:** số loại sở thích, thực thể và preference mới vẫn có thể làm profile phình ra.
- **Mất thông tin khi compact:** summary quá ngắn có thể bỏ mất chi tiết cần cho follow-up.
- **Tốn token do profile:** profile quá dài sẽ được đưa lại vào nhiều prompt và làm giảm lợi ích compact.

Triển khai hiện tại giảm các rủi ro này bằng structured fields, conflict handling, lọc câu hỏi/thông tin nhiễu, giới hạn rolling summary và test correction trên cả hai dataset. Trong hệ thống production, có thể bổ sung confidence threshold, timestamp và memory decay để kiểm soát memory tốt hơn.

## Kết luận

Kết quả benchmark thể hiện đúng trade-off của bài lab: Baseline đơn giản và rẻ hơn khi hội thoại ngắn nhưng không nhớ qua session; Advanced phải trả thêm chi phí quản lý persistent memory, đổi lại recall tốt hơn và có lợi thế rõ rệt về prompt cost khi hội thoại đủ dài để compact phát huy tác dụng.

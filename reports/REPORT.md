# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** `Python: 3.13.15 PyTorch: 2.11.0+cu128 Ultralytics: 8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
	`"rank": 1, 
	`"class_id": 468,
	`"class_name": "cab", 
	`"score": 0.510915,
	`"taxonomy_name": "ImageNet-1K",
- Record này mô tả toàn ảnh như thế nào?
	-   Bức ảnh dưới miêu tả tác vụ phân loại ảnh gán một nhãn duy nhất ở cấp độ toàn bức ảnh. Qua ảnh report, model có thể thấy số lượng xe con có độ tin cậy cao nhất, với điểm đánh giá tin cậy lên tới 51,09% , tiếp sau là xe bus con với độ tin cậy chỉ ~20%.![[Pasted image 20260911093103.png]]
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
	-  Người
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
	-  Để quản lý loại hình data đã thu thập (mã 468 dành cho xe con), hiệu chỉnh và thống kê loại data mà máy đã dự đoán cùng phép phân loại (taxonomy) đã được sử dụng, giúp tinh gọn quá trình đánh giá độ tin cậy.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
	- 1. Số lượng chủ thể có trên ảnh: cần có ID, tên gọi cho riêng từng chủ thể (vd xe con = cab, xe bus = minibus). Không được gom chung tất cả chủ thể thành một nhãn chung (vd trong trường hợp cần phân biệt xe nhưng nhãn tất cả là car)
	- 2. Nếu chủ thể thành 1 nhúm khó phân biệt từng cá thể, phải label là group hoặc ignore để tránh clustered model
- Vì sao model score không phải ground truth?
	- Model score chỉ là số liệu đánh giá model tự tin mình đoán đúng chủ thể trong bức ảnh hay không, chứ không phải chỉ số ground truth để so sánh/training vì mô hình có thể tự tin rất cao vào một dự đoán sai do thiên kiến dữ liệu, và ngược lại có thể tự tin thấp vào một đối tượng hiển nhiên đúng do nhiễu. Ground truth chỉ được thiết lập sau khi con người kiểm duyệt và xác nhận theo guideline chuẩn.


## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
	`"class_id": 5, 
	`"class_name": "bus", 
	`"score": 0.912558, 
	`"bbox_xyxy": [ 93.17, 187.95, 223.01, 320.91 ],
	`"bbox_width": 129.84,
	`"bbox_height": 132.96,
- Diễn giải vị trí box bằng lời:
	- Box đang detect một chiếc xe bus, chiều rộng box là 129.84 pixel, chiều cao box là 132.96 pixel. tọa độ của box góc ($x_1, y_1$)=(93.17, 187.95); ($x_{2},y_{2}$)=(223.01, 320.91) pixel
- So sánh số prediction ở hai threshold:
- threshold=0.20: 17 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle'] threshold=0.35: 11 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl'] 
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
	- Threshold nhỏ hơn, bao phủ rộng hơn, nhiều vật thể được detect hơn, khối lượng vật thể cần review nhiều hơn và ngược lại.
- Đề xuất một quy tắc box chặt:
	- **Mục tiêu:** Bounding box phải là hình chữ nhật nhỏ nhất song song với các trục tọa độ ($x, y$) chứa toàn bộ các điểm ảnh (pixels) thuộc về chủ thể, giảm thiểu tối đa diện tích background lọt vào bên trong.
	- Bộ phận nhô ra cũng phải bao trùm cả tóc bay, ngón tay vươn ra, vạt áo/dây giày, ăng-ten, gương xe,... nếu chúng thuộc về cá thể đó và còn rõ nét. Nếu mép chủ thể có vùng mờ (blend màu với nền), cạnh box phải chạm vào ranh giới ngoài cùng nơi màu của chủ thể bắt đầu hòa vào nền, không co cụm vào core của chủ thể.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
	- 1. Xử lý che khuất/cắt mép: quy định tỉ lệ hiển thị để được đưa vào model train (vd >50% hiển thị), bbox bao trọn vùng cơ thể thực tế nhìn thấy, hay dự đoán cả phần bị khuất? - Nếu chủ thể nằm sát rìa khung hình và bị cắt ngang, cần quy định tỷ lệ tối thiểu còn lại trong ảnh để gán nhãn. Hộp nhãn phải dừng chính xác tại biên ảnh (`x=0, y=0, x=W, y=H`), không được ước lượng kéo dài box ra ngoài viền ảnh số.
	- 2. Quy định rõ bài toán cần bắt tất cả mọi cá thể trong ảnh hay chỉ tập trung vào chủ thể chính để không detect cả một nhúm. Trường hợp trong ảnh xuất hiện hình ảnh ảo như phản chiếu qua gương, ảnh in trên áo,... cần ghi rõ nhãn riêng.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
	`"instance_id": "traffic-002",
	`"class_id": 2, 
	`"class_name": "car", 
	`"score": 0.891092, 
	`"bbox_xyxy": [ 418.82, 264.54, 498.47, 339.52 ],  
	`"polygon_xy": [ [ 497.0, 287.0 ], [ 497.0, 287.0 ], [ 492.0, 287.0 ], [ 491.0, 287.0 ],
- Polygon bổ sung chi tiết gì so với box?
	- polygon bổ sung được vị trí chi tiết của từng pixel định dạng vật thể (mask segmentation).
- `instance_id` dùng để làm gì và không phải loại ID nào?
	- Phân biệt các cá thể cùng Class: Nếu trong một ảnh có 3 chiếc xe con, nhãn lớp không thể giúp mô hình biết pixel nào thuộc về xe A, xe B hay xe C. `instance_id` gán một mã định danh riêng cho từng cá thể độc lập (ví dụ: traffic 001, traffic 002)
	-  không phải `category_id` / `class_id`/`image_id`/`annotation_id`
- Đề xuất một quy tắc biên mask:
	- **Mục tiêu:** Đường biên mask của Polygon phải đi chính xác qua ranh giới chuyển tiếp giữa chủ thể và nền, phản ánh đúng hình dạng thực tế của vật thể.
	- Nguyên tắc "50% Transition": Đường viền mask phải đi qua dải pixel có tỷ lệ pha trộn màu xấp xỉ 50% giữa foreground và background.
    - Tránh over-segmentation: Tuyệt đối không vẽ lấn ra các pixel thuần thuộc về nền chỉ để đảm bảo không bị cắt hụt vật thể.
    - Tránh under-segmentation : Không co cụm mask vào bên trong vùng core của vật thể làm mất đi độ dày tự nhiên của biên.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
	- Cấm chồng lấn: Biên giữa hai mask phải tiếp giáp khít nhau (zero overlap, zero gap). Không để hở khoảng trống nền giữa 2 mask chạm nhau.
    - Modal Mask (Required): Chủ thể nằm trước đè lên chủ thể nằm sau. Biên của vật nằm trước cắt ngang vật nằm sau một cách dứt khoát theo đúng đường viền nhìn thấy.
    - Amodal Mask (Optional): Phần bị che khuất của vật nằm sau được vẽ ước lượng theo suy luận hình học, mang chung `instance_id`.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`


| **Tác vụ**              | **Đơn vị / Định dạng Ground Truth**                                                                                                                                                            | **Lỗi hoặc điểm mơ hồ thường gặp**                                                                                                                                                                                                                                       | **Annotator làm gì?**                                                                                                                                                                                           | **Reviewer xem gì?**                                                                                                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phân loại ảnh**       | - **Đơn vị:** Toàn bộ bức ảnh.<br>- **Định dạng:** Chuỗi nhãn / Class ID (Single-label hoặc Multi-label vector `[0, 1, 0]`).                                                                   | - Ảnh có nhiều đối tượng thuộc các lớp khác nhau (không rõ đối tượng nào là chủ đạo).<br>- Ảnh chất lượng thấp, chủ thể mơ hồ khó nhận diện.                                                                                                                             | - Xác định chủ thể chính hoặc liệt kê toàn bộ các lớp có mặt theo quy ước.<br>- Đánh dấu cờ `ambiguous` / `unclear` nếu không thể phân định.                                                                    | - Kiểm tra tính nhất quán với taxonomy.<br>- Soi các ca biên xem có bị nhầm giữa các lớp gần giống nhau  không.                                                                                           |
| **Phát hiện vật thể** _ | - **Đơn vị:** Từng cá thể (Instance).<br>- **Định dạng:** Hộp bao tọa độ `[x_min, y_min, x_max, y_max]` (hoặc `[x, y, w, h]`) kèm `class_id`.                                                  | - **Box lỏng:** Dư nhiều pixel nền.<br>- **Cắt xén :** Cắt phạm vào chi tiết vật thể.<br>- Bỏ sót vật thể bị che khuất hoặc kích thước nhỏ sát ngưỡng.<br>- Nhầm lẫn giữa 1 box gộp hay nhiều box khi bị vật khác chắn ngang.                                            | - Áp dụng quy tắc _Tight Bounding Box_.<br>- Gắn nhãn riêng cho từng cá thể.<br>- Xử lý che khuất và cắt mép theo đúng guideline.                                                                               | - Kiểm tra 4 mép tiếp xúc của box (đảm bảo độ khít).<br>- Quét vùng rìa ảnh và vùng che khuất xem có bị bỏ sót đối tượng (false negative) không.<br>- Kiểm tra phân loại nhãn của từng box.               |
| **Phân vùng cá thể**    | - **Đơn vị:** Từng cá thể ở cấp độ pixel.<br>- **Định dạng:** Tọa độ đa giác `Polygon [[x1, y1], [x2, y2], ...]`, chuỗi mã hóa `RLE`, hoặc `Binary Mask (PNG)` kèm `instance_id` + `class_id`. | - **Lem viền :** Lấn sang background hoặc vật thể kế bên.<br>- **Mất chi tiết:** Bỏ qua các khoảng rỗng (lỗ thủng/donut) hoặc sợi mỏng.<br>- Mật độ điểm neo quá thưa (gây gãy khúc) hoặc quá dày không cần thiết.<br>- Trùng lặp `instance_id` giữa 2 cá thể khác nhau. | - Phóng to để vẽ bám sát đường biên thực tế (ngưỡng chuyển màu 50%).<br>- Đục lỗ cho các khoảng hở bên trong.<br>- Gán đúng và duy nhất `instance_id` cho từng cá thể (kể cả khi bị chia cắt thành nhiều mảng). | - Bật/tắt lớp mask để kiểm tra độ khít của đường viền.<br>- Soi các vùng rỗng (lỗ hổng giữa tay/chân, quai xách) xem đã đục mask chưa.<br>- Kiểm tra tính toàn vẹn của `instance_id` trên toàn ảnh/video. |
## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
	- **Mục tiêu:** Đảm bảo toàn bộ hình ảnh và nhãn dữ liệu khi lưu trữ, huấn luyện hoặc phân phối cho bên thứ ba đều tuân thủ các quy định pháp lý (như GDPR, PDPA, Nghị định 13/2023/NĐ-CP), triệt tiêu hoàn toàn khả năng nhận diện danh tính cá thể ngoài đời thực mà không làm suy giảm đặc trưng bài toán.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Người quản lý tổng của kho dữ liệu (Project Manager/QA Lead, Mentor) qua kênh liên lạc chính thức, cung cấp image ID và mô tả vắn tắt vấn đề mà không tự ý sao chép hay lan truyền ảnh đó.


## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

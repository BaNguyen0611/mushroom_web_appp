
<img width="611" height="585" alt="Screen Shot 2025-08-25 at 10 15 05" src="https://github.com/user-attachments/assets/2108ab3b-9a53-4f93-bd11-43e44c2b6846" />

# 🍄 Mushroom Classifier Web App (ID3)

Dự án web app dự đoán nấm ăn được hay nấm độc dựa trên **thuật toán ID3 tự cài đặt** và **dataset mushrooms.csv**.

## 📌 Tính năng
- Dựa hoàn toàn vào dataset `mushrooms.csv`.
- Thuật toán **ID3** viết từ đầu, không dùng scikit-learn.
- Giao diện web dự đoán nấm.
- Truy cập các API `/api/schema`, `/api/tree` và `/api/predict`.

## 🛠️ Cài đặt & chạy
```bash
pip install flask pandas numpy
python app.py
```

## Truy cập web
```
http://127.0.0.1:5000
```

## 📂 Cấu trúc
```
mushroom_web_app_fixed/
├── app.py
├── mushrooms.csv
└── README.md
```

## 📊 Công nghệ
- Python 3.x
- Flask
- Pandas
- Numpy

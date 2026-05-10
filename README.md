# TradingCode-VN
Nơi chứa các mã nguồn định lượng giao dịch thị trường chứng khoán Việt Nam.
Nghiên cứu tập trung vào mã nguồn toán học định lượng để xây dựng các chỉ báo, đặc biệt cho giao dịch hợp đồng tương lai phái sinh mã VN30F1.

---

## Hermes Research Workspace

**Hermes** là Quant Analyst & Strategy Advisor AI agent cho dự án TradingCode-VN.  
Workspace tập trung vào nghiên cứu định lượng, phân tích dữ liệu lịch sử, và prototype chiến lược — **không phải production execution**.

### Cấu trúc thư mục

```
research/
├── data/               # Tiện ích load dữ liệu OHLCV (VN30F)
├── features/           # Feature engineering: kỹ thuật & microstructure
├── regime/             # Phát hiện chế độ thị trường (HMM, Volatility-based)
├── signals/            # Prototype tín hiệu alpha (Mean Reversion, Momentum)
├── backtest/           # Engine backtest vector hoá, walk-forward, metrics
└── notebooks/          # Jupyter notebooks nghiên cứu
    ├── 01_feature_engineering_eda.ipynb
    └── 02_regime_detection_backtest.ipynb
```

### Cài đặt môi trường nghiên cứu

```bash
pip install -r requirements.txt
```

### Khởi chạy notebooks

```bash
cd research/notebooks
jupyter lab
```

### Phạm vi của Hermes

| Trong phạm vi ✅ | Ngoài phạm vi 🚫 |
|---|---|
| Alpha research & feature engineering | Tự động đặt lệnh production |
| Regime detection | Ra quyết định giao dịch thật |
| Backtest / walk-forward validation | Hệ thống execution production |
| Statistical arbitrage research | Tư vấn không có kiểm chứng dữ liệu |
| Prototype Python cho kiểm chứng ý tưởng | |

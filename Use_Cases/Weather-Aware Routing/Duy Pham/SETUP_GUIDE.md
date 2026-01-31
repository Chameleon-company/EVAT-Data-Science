# 伺服器設置指南 (Server Setup Guide)

## 問題診斷 (Issue Diagnosis)

伺服器無法啟動，因為缺少必要的 Python 套件。需要確保所有依賴都已正確安裝。

## 解決步驟 (Solution Steps)

### 1. 檢查 Python 環境
```bash
cd "/Users/wingyip/Desktop/EVAT-Data-Science/Use_Cases/Weather-Aware Routing/Duy Pham"
python3 --version
which python3
```

### 2. 安裝所有必要的套件
```bash
# 確保使用正確的 Python 環境
pip3 install pandas scikit-learn shapely geopandas flask flask-cors joblib

# 或者如果你使用虛擬環境
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn shapely geopandas flask flask-cors joblib
```

### 3. 驗證安裝
```bash
python3 -c "import pandas, sklearn, shapely, geopandas, flask, joblib; print('所有套件已安裝！')"
```

### 4. 啟動伺服器
```bash
python3 api.py
```

你應該會看到：
```
==================================================
Starting EV Traffic & Weather Model API...
Server will be available at http://localhost:5000
==================================================
 * Running on http://0.0.0.0:5000
```

### 5. 測試 API

在另一個終端視窗中：

**測試根端點：**
```bash
curl http://localhost:5000/
```

**測試預測端點：**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2023, "start_lat": -37.8136, "start_lon": 144.9631}'
```

## 常見問題 (Common Issues)

### 問題：ModuleNotFoundError
**解決方法：** 確保在正確的 Python 環境中安裝了所有套件

### 問題：Port 5000 已被使用
**解決方法：** 修改 `api.py` 最後一行的端口號，例如改為 5001

### 問題：CORS 錯誤（前端無法連接）
**解決方法：** 安裝 flask-cors：`pip3 install flask-cors`

## 下一步 (Next Steps)

1. ✅ 修復後端代碼（已完成）
2. ✅ 創建前端 TypeScript 客戶端（已完成）
3. ⏳ 安裝依賴套件（需要你手動執行）
4. ⏳ 啟動伺服器（需要你手動執行）
5. ⏳ 測試 API 連接（需要你手動執行）

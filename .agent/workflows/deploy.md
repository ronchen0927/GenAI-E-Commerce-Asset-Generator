---
description: How to activate CI/CD and deploy to Cloud Run
---

# 啟用 CI/CD 並部署到 Cloud Run

## Part 1: GitHub Actions CI（自動啟用）

### 前置條件
- 程式碼已經推到 GitHub（✅ 已完成：`ronchen0927/ecommerce-ai-helper`）

### 啟用步驟

CI pipeline 已經寫好在 `.github/workflows/ci.yml`，**只要 push 到 GitHub 就自動啟用**。

```bash
# 1. 確認所有變更已 commit
git add -A
git commit -m "feat: add CI/CD pipeline, IaC configs, and monitoring docs"

# 2. Push 到 GitHub — CI 會自動跑
git push origin main
```

### 驗證

1. 到 https://github.com/ronchen0927/ecommerce-ai-helper/actions 確認 CI 跑起來了
2. 三個 stage 都應該通過：Lint → Test → Docker Build

### 常見問題

- **CI 紅燈？** 點進去看 log，通常是 lint 或 test 錯誤
- **Docker build 失敗？** 確認 `Dockerfile` 和 `pyproject.toml` 一致

---

## Part 2: 部署到 Cloud Run

### 前置條件

1. **安裝 Google Cloud CLI**
   - 到 https://cloud.google.com/sdk/docs/install 下載安裝
   - 安裝後重開 terminal

2. **登入 GCP**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

3. **建立或選擇 GCP Project**
   ```bash
   # 建立新 project（或用現有的）
   gcloud projects create ai-ecommerce-media-studio --name="AI E-Commerce Media Studio"
   
   # 設定為預設 project
   gcloud config set project ai-ecommerce-media-studio
   ```

4. **啟用必要的 API**
   ```bash
   gcloud services enable \
     cloudbuild.googleapis.com \
     run.googleapis.com \
     artifactregistry.googleapis.com \
     secretmanager.googleapis.com
   ```

5. **建立 Artifact Registry Repository**（存 Docker image）
   ```bash
   gcloud artifacts repositories create cloud-run-source-deploy \
     --repository-format=docker \
     --location=asia-east1 \
     --description="Docker images for Cloud Run deployment"
   ```

6. **設定 IAM 權限**（讓 Cloud Build 可以部署 Cloud Run）
   ```bash
   # 取得 project number
   PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
   
   # 授權 Cloud Build service account 部署 Cloud Run
   gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
     --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
     --role="roles/run.admin"
   
   gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
     --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
     --role="roles/iam.serviceAccountUser"
   ```

### 部署方式 A：使用 Cloud Build（推薦）

```bash
# 從專案根目錄執行
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions=_REGION=asia-east1,_SERVICE_NAME=ai-ecommerce-media-studio
```

這會自動執行：Build Docker image → Push to Artifact Registry → Deploy to Cloud Run

### 部署方式 B：使用 gcloud run deploy（快速）

```bash
# 直接從 source 部署（Cloud Run 會自動 build）
gcloud run deploy ai-ecommerce-media-studio \
  --source . \
  --region asia-east1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300
```

### 設定 Secrets（如果要用 Replicate API）

```bash
# 1. 建立 secret
echo -n "your-replicate-api-token" | \
  gcloud secrets create replicate-api-token --data-file=-

# 2. 授權 Cloud Run 讀取 secret
gcloud secrets add-iam-policy-binding replicate-api-token \
  --member="serviceAccount:$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# 3. 更新 Cloud Run service 掛載 secret
gcloud run services update ai-ecommerce-media-studio \
  --region asia-east1 \
  --set-secrets=REPLICATE_API_TOKEN=replicate-api-token:latest
```

### 驗證部署

```bash
# 取得 service URL
gcloud run services describe ai-ecommerce-media-studio \
  --region asia-east1 \
  --format='value(status.url)'

# 測試 health check
curl https://YOUR_SERVICE_URL/health
# 應該回傳 {"status": "healthy"}
```

---

## Part 3: 完整流程（日常開發循環）

```
1. 寫 code → commit → push
2. GitHub Actions 自動跑 CI（lint + test + docker build）
3. CI 通過後，手動觸發部署：
   gcloud builds submit --config deploy/cloudbuild.yaml \
     --substitutions=_REGION=asia-east1,_SERVICE_NAME=ai-ecommerce-media-studio
4. 到 Cloud Run console 確認部署成功
```

---

## 注意事項

### 成本控制
- Cloud Run 設定了 `maxScale: 3`，限制最多 3 個 instance
- CPU throttling 啟用，沒有請求時不收 CPU 費用
- `minScale: 0`，完全沒流量時 scale to zero（$0）
- **Replicate API 才是主要成本來源**，每次 inference 約 $0.01-0.05

### 架構限制（Cloud Run 版本）
- Cloud Run 沒有 Redis，所以 **Celery async task queue 不會運作**
- Cloud Run 版本只能用 sync mode（直接在 request 內處理）
- 如果要完整的 async 架構，需要使用 GKE 或 Cloud Compute + docker-compose

### 如果只是面試 Demo
- 不需要真的部署
- 用 `gcloud run deploy --source .` 跑一次證明可以就夠了
- 然後 `gcloud run services delete ai-ecommerce-media-studio` 刪掉省錢

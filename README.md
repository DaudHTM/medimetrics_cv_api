# medimetrics_cv_api — Cloud Run deployment

This repository contains a FastAPI application (`main_cloudrun.py`) which performs hand and face measurements using MediaPipe and OpenCV. The included `Dockerfile`, `requirements.txt`, and `.dockerignore` make it straightforward to deploy to Google Cloud Run.

Important notes before deploy
- Use `opencv-python-headless` in the container (already in `requirements.txt`).
- `mediapipe` requires a compatible Python version and OS packages. The provided Dockerfile installs common system libs used by OpenCV and MediaPipe. If you hit Mediapipe import errors, try changing the Python minor version (3.10/3.11) or consult the MediaPipe installation docs.

Build & deploy (PowerShell)

1. Build an image and push to Container Registry (example):

```powershell
# set your GCP project
$env:PROJECT = "your-gcp-project-id"
gcloud config set project $env:PROJECT

# build and push (Container Registry)
gcloud builds submit --tag gcr.io/$env:PROJECT/medimetrics-api .
```

2. Deploy to Cloud Run:

```powershell
$env:SERVICE = "medimetrics-api"
gcloud run deploy $env:SERVICE --image gcr.io/$env:PROJECT/medimetrics-api --platform managed --region us-central1 --allow-unauthenticated
```

3. After deployment, Cloud Run will provide a HTTPS URL. Use that to POST images to `/measure-hand` and `/measure-face`.

Local testing (optional)

```powershell
# run locally (ensure PORT is same as Dockerfile default if you want parity)
python -m uvicorn main_cloudrun:app --host 0.0.0.0 --port 8080

# then POST an image with curl (example)
curl -X POST -F "file=@test_image.png" http://localhost:8080/measure-hand
```

Security
- By default the Cloud Run service is deployed with `--allow-unauthenticated`. For production lock it down and use authentication or an API key.

If you'd like, I can also:
- Add a small `cloudbuild.yaml` for automated builds
- Add a minimal test script that posts `test_image.png` to the deployed endpoint and saves JSON

# SatQuery AI — Backend MVP

## 1. Create environment

### Windows PowerShell

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Start API

```powershell
uvicorn app.main:app --reload
```

API:
- http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs

## 3. Test

Open `/docs`, select `POST /api/analyze`, enter a question and upload images.

Examples:

- `Is there water in this image?` -> VQA
- `Where is the water?` -> Grounding
- `What changed between these dates?` + 2 images -> Change Detection
- `Identify built-up areas using optical and SAR` + 2 images -> Optical + SAR Fusion

This first milestone intentionally uses placeholder specialist functions.
The routing layer is real and will be replaced/enhanced with LangGraph + LLM after the API contract is stable.

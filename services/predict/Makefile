PORT=8000
run-api:
	uv run python run.py
run-new-api:
	uv run uvicorn run:app --host 0.0.0.0 --port 8000

run-app:
	uv run python -m streamlit run app.py

upload-models:
	uv run python upload_models.py

install:
	uv add streamlit plotly requests pillow torch torchvision fastapi uvicorn python-multipart
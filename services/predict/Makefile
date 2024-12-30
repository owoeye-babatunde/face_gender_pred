run-api:
	uv run python run.py

run-app:
	uv run python -m streamlit run app.py

install:
	uv add streamlit plotly requests pillow torch torchvision fastapi uvicorn python-multipart

build-image:
	docker-compose build

start-services:
	docker-compose up
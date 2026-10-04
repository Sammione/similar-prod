FROM python:3.11-slim
WORKDIR /srv
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Bake the pretrained models into the image so startup doesn't download them
RUN python -c "from transformers import CLIPModel, CLIPProcessor; from sentence_transformers import SentenceTransformer; \
CLIPModel.from_pretrained('patrickjohncyh/fashion-clip'); CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip'); \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

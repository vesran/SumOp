FROM python:3.8.2
LABEL version="0.1"

WORKDIR /sumop

# Envs
ENV PYTHONPATH "${PYTHONPATH}:/code"

# Install Python's requirements
COPY requirements.txt /sumop
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy projects files
COPY models/ /sumop/models/
COPY sumop/ /sumop/sumop/
RUN mkdir /sumop/data

# Process data
RUN python /sumop/sumop/main.py

EXPOSE 8501

CMD ["streamlit", "run", "/sumop/sumop/dashboard.py"]

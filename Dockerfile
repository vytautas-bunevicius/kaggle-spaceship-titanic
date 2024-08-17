# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies, including those required for h5py
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    default-jdk \
    git \
    pkg-config \
    libhdf5-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install custom package if needed
RUN pip install --no-cache-dir git+https://github.com/vytautas-bunevicius/kaggle-spaceship-titanic.git@d69f41e03e056f2ada1bf35de5d609edec713901

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies, including those required for h5py
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jdk \
    libhdf5-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the model and application code
COPY models/StackedEnsemble_Best1000_1_AutoML_1_20240811_214618 /app/models/StackedEnsemble_Best1000_1_AutoML_1_20240811_214618
COPY app.py /app/
COPY templates /app/templates

EXPOSE 8080

CMD ["python", "app.py"]

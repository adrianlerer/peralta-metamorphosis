# Paper 11: Political Actor Network Analysis - Docker Container
# Multi-stage build for optimized production image

# Stage 1: Base Python environment with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    gcc \
    g++ \
    gfortran \
    # Graphics and plotting dependencies
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Network tools
    curl \
    wget \
    # Git for version control
    git \
    # Additional utilities
    unzip \
    vim \
    nano \
    htop \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Development environment with Jupyter and full dependencies
FROM base as development

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install additional Jupyter extensions and tools
RUN pip install --no-cache-dir \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all] \
    jupyter-dash \
    voila

# Create necessary directories
RUN mkdir -p /app/data /app/notebooks /app/code /app/results /app/docs /app/visualizations

# Copy project files
COPY . .

# Set up Jupyter configuration
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_lab_config.py

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting Paper 11 Analysis Environment..."\n\
echo "Jupyter Lab: http://localhost:8888"\n\
echo "Interactive Dashboard: http://localhost:8050"\n\
echo ""\n\
echo "Available notebooks:"\n\
ls -la /app/notebooks/*.ipynb\n\
echo ""\n\
echo "To run analysis:"\n\
echo "  python /app/code/analysis.py"\n\
echo ""\n\
exec "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Expose ports
EXPOSE 8888 8050 5000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["jupyter", "lab", "--allow-root", "--no-browser", "--ip=0.0.0.0", "--port=8888"]

# Stage 3: Production environment (minimal)
FROM base as production

# Install only essential production dependencies
WORKDIR /app
COPY requirements.txt .

# Install minimal production requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy pandas scipy scikit-learn \
    networkx matplotlib plotly \
    flask gunicorn

# Copy only necessary files
COPY code/ ./code/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash paper11user && \
    chown -R paper11user:paper11user /app

USER paper11user

# Expose port for web interface
EXPOSE 8000

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "code.app:app"]

# Labels for metadata
LABEL maintainer="Paper 11 Research Team" \
      description="Political Actor Network Analysis - Academic Replication Package" \
      version="1.0.0" \
      license="MIT" \
      repository="https://github.com/your-username/paper11-analysis"
# 1. Use the Playwright Python image based on Ubuntu 24.04 (Noble)
FROM mcr.microsoft.com/playwright/python:v1.52.0-noble

# 2. Set working directory
WORKDIR /optimization-agent

# 3. Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# 4. Copy & install your Python dependencies (including playwright)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 5. Copy your application code
COPY . .

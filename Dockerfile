# 1. Use the Playwright Python image based on Ubuntu 24.04 (Noble)
FROM mcr.microsoft.com/playwright/python:v1.52.0-noble

# 2. Set working directory
WORKDIR /optimization-agent

# 3. Install tini for proper init and signal handling
RUN apt-get update && apt-get install -y tini && rm -rf /var/lib/apt/lists/*

# 4. Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# 5. Copy & install your Python dependencies (including playwright)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 6. Copy your application code
COPY . .

# 7. Use tini as init system to properly reap zombie processes
ENTRYPOINT ["tini", "--"]
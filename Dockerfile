FROM python:3.12.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV OMP_NUM_THREADS=1
ENV OPENCV_OPENCL_RUNTIME=disabled

WORKDIR /app

RUN apt-get update
RUN apt-get install -y --no-install-recommends libgl1
RUN apt-get install -y --no-install-recommends libglib2.0-0
RUN apt-get install -y --no-install-recommends libsm6
RUN apt-get install -y --no-install-recommends libxext6
RUN apt-get install -y --no-install-recommends libxrender1
RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY src ./src
COPY start.sh ./start.sh

RUN chmod +x /app/start.sh

EXPOSE 11001

CMD ["/app/start.sh"]

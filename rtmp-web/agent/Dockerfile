# Run the on-site agent as a container on any always-on machine on the camera LAN.
#   docker build -t nvr-agent .
#   docker run -d --name nvr-agent --restart unless-stopped \
#     --network host -v "$PWD/agent.json:/app/agent.json:ro" nvr-agent
# --network host lets ONVIF discovery (LAN multicast) and RTSP pull reach the cameras.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY nvr_agent.py .
ENTRYPOINT ["python3", "nvr_agent.py", "--config", "/app/agent.json"]

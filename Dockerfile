# 单阶段构建
FROM ubuntu:24.04

# 设置工作目录
WORKDIR /app

# 使用阿里云镜像源
RUN cat > /etc/apt/sources.list << 'EOF'
deb http://mirrors.aliyun.com/ubuntu/ noble main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ noble-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ noble-backports main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ noble-security main restricted universe multiverse
EOF

# 安装 Python 3.12 和 pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 设置 pip 使用清华大学镜像
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 设置 uv 使用清华大学镜像
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件
COPY . .

# 安装 uv 并同步依赖
RUN pip3 install --no-cache-dir uv --break-system-packages && uv sync

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 创建日志和配置目录
RUN mkdir -p /app/logs /app/config

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["uv", "run", "main.py"]

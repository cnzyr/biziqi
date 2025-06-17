FROM qwen2_5-vl:base
LABEL maintainer=
# 设置工作目录
RUN mkdir -p /usr/workspace
WORKDIR /usr/workspace
ADD . /usr/workspace
RUN pip install uwsgi opencv-python-headless flask qwen_vl_utils -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN ["chmod", "+x", "/usr/workspace/uwsgi_start_docker.sh"]

# 容器启动命令
ENTRYPOINT ["/bin/bash", "/usr/workspace/uwsgi_start_docker.sh"]
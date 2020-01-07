ARG PYTHON_VERSION=3.6

ARG NAME_SPACE

FROM registry.cn-shanghai.aliyuncs.com/${NAME_SPACE}/cuda_docker_base:${PYTHON_VERSION} as builder

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install --no-cache-dir pyarmor

WORKDIR /build

COPY . /build

RUN bash tools/compress.sh

FROM registry.cn-shanghai.aliyuncs.com/${NAME_SPACE}/cuda_docker_base:${PYTHON_VERSION}

WORKDIR /workspace

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

COPY --from=builder /build /workspace

RUN pip install torch==1.3.1 torchvision==0.4.2

RUN pip install cython

RUN pip install -e fvcore && pip install -e cocoapi/PythonAPI && pip install -e detectron2 && pip install -r requirements.txt

ENTRYPOINT [ "/usr/bin/dumb-init", "--" ]

CMD [ "bash" ]

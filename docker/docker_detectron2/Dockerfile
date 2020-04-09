ARG PYTHON_VERSION=3.7

ARG NAME_SPACE

FROM registry.cn-shanghai.aliyuncs.com/${NAME_SPACE}/suanpan-python-sdk:${PYTHON_VERSION} as builder

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install --no-cache-dir pyarmor

WORKDIR /build

COPY . /build

RUN bash tools/compress.sh

FROM registry.cn-shanghai.aliyuncs.com/${NAME_SPACE}/suanpan-python-sdk:${PYTHON_VERSION}

WORKDIR /workspace

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN apt-get update -y && apt-get install git -y

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

COPY --from=builder /build /workspace

RUN pip install torch==1.4.0 torchvision==0.5.0

RUN pip install cython pyyaml==5.1

RUN pip install -e fvcore && pip install -e cocoapi/PythonAPI && pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html && pip install -r requirements.txt

ENTRYPOINT [ "/sbin/my_init", "--" ]

CMD [ "bash" ]

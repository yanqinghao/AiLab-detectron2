ARG PYTHON_VERSION=3.7

FROM registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/stream_base:${PYTHON_VERSION} as builder

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install --no-cache-dir pyarmor

WORKDIR /build

COPY . /build

RUN bash tools/compress.sh

FROM registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/stream_base:${PYTHON_VERSION}

WORKDIR /workspace

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

COPY --from=builder /build /workspace

RUN pip install ./whl/cpu/torch-1.3.1+cpu-cp37-cp37m-linux_x86_64.whl

RUN pip install ./whl/cpu/torchvision-0.4.2+cpu-cp37-cp37m-linux_x86_64.whl

RUN rm -rf whl

RUN pip install cython && pip install -r requirements.txt

RUN pip install -e detectron2

ENTRYPOINT [ "/sbin/my_init", "--" ]

CMD [ "bash" ]

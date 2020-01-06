NAMESPACE=("shuzhi-amd64")
for i in ${NAMESPACE[*]}
do
    docker build --build-arg NAME_SPACE=${i} -t registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker-gpu:$1 -f docker/docker_detectron2_gpu/Dockerfile .
    docker build --build-arg NAME_SPACE=${i} -t registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream-gpu:$1 -f docker/stream_detectron2_gpu/Dockerfile .
    sed -i "s/cpu/cu100/g" .dockerignore
    docker build --build-arg NAME_SPACE=${i} -t registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker:$1 -f docker/docker_detectron2/Dockerfile .
    docker build --build-arg NAME_SPACE=${i} -t registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream:$1 -f docker/stream_detectron2/Dockerfile .
    sed -i "s/cu100/cpu/g" .dockerignore

    docker push registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker:$1
    docker push registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream:$1
    docker push registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker-gpu:$1
    docker push registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream-gpu:$1
done
NAMESPACE=("shuzhi-amd64")
for i in ${NAMESPACE[*]}
do
    docker pull registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker-gpu:$1
    docker tag registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker-gpu:$1 registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker-gpu:$2
    docker push registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker-gpu:$2
    docker pull registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream-gpu:$1
    docker tag registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream-gpu:$1 registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream-gpu:$2
    docker push registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream-gpu:$2
    sed -i "s/cpu/cu100/g" .dockerignore
    docker pull registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker:$1
    docker tag registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker:$1 registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker:$2
    docker push registry.cn-shanghai.aliyuncs.com/${i}/detectron2-docker:$2
    docker pull registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream:$1
    docker tag registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream:$1 registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream:$2
    docker push registry.cn-shanghai.aliyuncs.com/${i}/detectron2-stream:$2
    sed -i "s/cu100/cpu/g" .dockerignore
done
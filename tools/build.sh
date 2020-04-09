NAMESPACE=("shuzhi-amd64")
for i in ${NAMESPACE[*]}
do
    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/detectron2-docker-gpu:$1 -f docker/docker_detectron2_gpu/Dockerfile .
    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/detectron2-docker-gpu:$1

    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/detectron2-docker:$1 -f docker/docker_detectron2/Dockerfile .
    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/detectron2-docker:$1
done
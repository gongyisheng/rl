# rl
rl experiments

infra: [miles](https://github.com/radixark/miles)

standard configurations are in [tasks](tasks/readme.md)

experiment variants are in [experiments](experiments/)

## Environment

```bash
docker create --gpus all --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng/Documents/checkpoints:/data --name miles-dev radixark/miles sleep infinity
docker start miles-dev
docker exec -it miles-dev bash
```

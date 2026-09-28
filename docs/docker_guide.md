# Docker 使用与保存指南

本项目新增的 Dockerfile 用于主程序、格式转换和离线实验。构建环境为 Python 3.10，
没有打包 EnzGFM、EpHod、UniStab 权重或 CUDA 环境。真实模型需另行配置容器内兼容环境，
不能仅凭 `--gpus all` 就认为模型依赖已经安装。本次工作环境没有 docker 命令，未执行镜像构建。

## 构建与运行

在项目根目录运行：

```bash
docker build -t enzyme-agent:verifier-reflection .
mkdir -p results
docker run --name enzyme-agent-smoke \
  -v "$PWD/results:/results" enzyme-agent:verifier-reflection
```

默认命令在 /results/smoke 运行离线 smoke test。再次运行应选择新结果目录：

```bash
docker run --name enzyme-agent-bench \
  -v "$PWD/results:/results" enzyme-agent:verifier-reflection \
  python experiments/run_verifier_reflection_ablation.py --mode offline \
  --max-tasks 20 --seed 42 --failure-probability 0 0.1 0.2 0.3 \
  --output-dir /results/matrix_1
```

如果容器已经具备模型所需的 Linux Python 环境、权重和驱动支持，可以使用 GPU、配置与目录挂载：

```bash
docker run --gpus all --name enzyme-agent-real --env-file .env.server \
  -v "$PWD/results:/results" \
  -v "$PWD/config:/work/config:ro" \
  -v "$PWD/data:/work/data:ro" \
  -v /srv/enzyme-models:/models:ro \
  -v /srv/enzyme-cache:/cache \
  enzyme-agent:verifier-reflection \
  fixed-enzyme-agent route /work/config/test.json /work/data/input.fasta \
  --request '按配置评估活性、最适 pH 和稳定性' \
  --output-dir /results/real_1 --reflection --execute
```

这条真实运行命令是部署模板，基础镜像自身不能提供外部模型环境。模型路径须指向容器中路径，
不能沿用宿主机 Windows 路径；宿主机虚拟环境也不保证可直接挂载后执行。
.env.server 只包含 DEEPSEEK_API_KEY 等运行时环境变量，必须排除在 Git 和镜像构建上下文之外。
不要使用 Dockerfile ENV 或 ARG 将 secret 烘焙到 image。

```bash
docker ps -a
docker logs -f enzyme-agent-bench
docker exec -it enzyme-agent-real /bin/bash
docker stop enzyme-agent-real
docker rm enzyme-agent-real
```

docker exec 只适用于仍在运行的容器；离线实验完成后容器会退出。

## 保存和跨服务器传输

```bash
docker save -o enzyme_agent.tar enzyme-agent:verifier-reflection
gzip enzyme_agent.tar
scp enzyme_agent.tar.gz user@server:/path/to/images/
```

在目标服务器：

```bash
gunzip enzyme_agent.tar.gz
docker load -i enzyme_agent.tar
docker images
```

docker image 是程序和依赖的只读模板；docker container 是 image 的一个运行实例，
包含进程和可写层；mounted result directory 是挂载进容器的宿主机结果目录。
删除 container 不会删除 bind mount 的宿主机结果，但要避免在容器内主动删除挂载文件。

docker save 保存的是 image，不自动保存运行产生的数据。实验结果优先保存在
`-v "$PWD/results:/results"` 指定的 bind mount，或明确管理的 volume，并单独备份。
tar.gz 镜像、模型权重和实验原始大文件不要上传到要求小于 50 MB 的源代码仓库。

如确实需要保存手动修改过的容器，可以：

```bash
docker commit enzyme-agent-real enzyme-agent:manual-snapshot
docker save -o enzyme_agent_snapshot.tar enzyme-agent:manual-snapshot
```

docker commit 不保存挂载卷内的数据，并且可能把容器内残留敏感文件带入新镜像。
它不是推荐的正式可复现流程。推荐保留 Dockerfile、经验证的 dependency lock、源代码版本
和挂载结果目录。本轮基础 Dockerfile 使用现有依赖范围，完整目标平台 lock 仍待服务器验证后生成。

## 集群不允许 Docker 时

遵守集群政策，可考虑 Apptainer/Singularity 导入已构建的镜像，并按集群文档配置 GPU 与 bind mount。
本轮没有改为 Apptainer 项目，也没有声称已验证该集群运行方式。

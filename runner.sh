docker build --tag task3 . && \
  docker run --rm -u $(id -u):$(id -g) -v "$( cd "$( dirname "$0" )" && pwd )":/results task3
# -v "$( cd "$( dirname "$0" )" && pwd )":/results
#-v /Users/lukas/ETH/PAI/PAI_Project3/plots:/code/plots task3
```powershell
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

pip config set global.index-url https://pypi.org/simple

```

## 在vscode中，tensorflow.keras模块的下属模块无法自动补全，我在github官方respository的issue中找到了解决问题的方法，即进入tensorflow/__init__.py文件，将下列代码粘贴进去即可。

位置:
```shell
\\wsl.localhost\<wsl distro name>\home\<linux user name>\anaconda3\envs\<conda env name(current)>\lib\python3.10\site-packages\tensorflow\__init__.py
```
```shell
\\wsl.localhost\Ubuntu\home\ubuntu\anaconda3\envs\py310\lib\python3.10\site-packages\tensorflow\__init__.py
```


1. ``` NOT WORKED ``` 在tensorflow/__init__.py文件中，将下列代码粘贴进去即可。
```python
# Explicitly import lazy-loaded modules to support autocompletion.
# pylint: disable=g-import-not-at-top
if _typing.TYPE_CHECKING:
  from tensorflow_estimator.python.estimator.api._v2 import estimator as estimator
  from keras.api._v2 import keras
  from keras.api._v2.keras import losses
  from keras.api._v2.keras import metrics
  from keras.api._v2.keras import optimizers
  from keras.api._v2.keras import initializers
# pylint: enable=g-import-not-at-top
from tensorflow_estimator.python.estimator.api._v2 import estimator
```

2. ``` NOT WORKED ``` GPT 建议添加代码到此文件
```python
# Explicitly import lazy-loaded modules to support autocompletion.
# pylint: disable=g-import-not-at-top
_keras_module = "keras.api._v2.keras"
_keras = _LazyLoader("keras", globals(), _keras_module)
_module_dir = _module_util.get_parent_dir_for_name(_keras_module)
if _module_dir:
    _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "keras", _keras)
```

## 安装GPU支持的tensorflow

1. 安装 CUDA Toolkit

```bash
https://developer.nvidia.com/cuda-downloads

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network

```


### 3.9.4. Common Installation Instructions for WSL[](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#common-installation-instructions-for-wsl#common-installation-instructions-for-wsl "Permalink to this headline")

These instructions apply to both local and network installation for WSL.

1.  **Update the Apt repository cache:**

```bash
sudo apt-get update
```

2.  **Install CUDA SDK:**

```bash
sudo apt-get install cuda-toolkit=12.3.* # 指定tf都已经的版本，去tf release中查询
```

3.  **【Must do】** Perform the [post-installation actions.](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)


4. other ways to install CUDA on WSL-Ubuntu

```bash
cd $HOME
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run
```
5. To perform a basic install of all CUDA Toolkit components using Conda, run the following command:

```bash
conda install cuda -c nvidia
```


## instal tensorflow[and-cuda]
安装TF

```python
pip install --upgrade pip

# For GPU users
pip install tensorflow[and-cuda]
# For CPU users
#pip install tensorflow
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```

# 最后测试,直接安装2.15.* 版本，可以支持gpu

```pwsh
 pip install tensorflow[and-cuda]==2.15.*
 ```

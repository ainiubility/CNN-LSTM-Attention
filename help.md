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

GPT 建议添加代码到此文件
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

# from distutils.core import setup
# # from setuptools import setup
#
# # python setup.py sdist
# # python setup.py install
# # python setup.py install --record installed.txt
# setup(
#     name='myai',#需要打包的名字
#     version='v1.0',#版本
#     packages=['tool','base'],
#     py_modules=['tool/utils','base/baseai2','base/baseai5'], # 需要打包的模块
# )

from setuptools import setup, find_packages

setup(
    name="myfunc",  # 需要打包的名字
    version="0.1",  # 版本
    # packages = find_packages(),
    packages=['tool'],  # 需要打包的模块
    py_modules=['tool/utils', 'base/baseai', 'tool/sockutils'],  # 需要打包的单个文件
)
# python setup.py bdist_egg
# python setup.py install

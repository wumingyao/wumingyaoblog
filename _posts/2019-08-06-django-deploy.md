---
layout:     post
title:      "Django项目部署"
subtitle:   "Django项目从python环境安装到部署"
date:       2019-08-06
author:     "wumingyao"
header-img: "img/in-post/2019.08/06/bg.jpg"
tags: [教程,Django,Python]
categories: [教程]
---

### 1、安装python环境

(1)下载python3.tar.xz: wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz.xz

(2)解压成.tar: xz -d Python-3.5.2.tar.xz

(3)解压tar包：tar -xvzf Python-3.5.2.tar

(4)进入Python-3.5.2目录

(5)编译：./configure --prefix=/usr/local/python3 --enable-shared

(6)安装：make && make install

(7)Python3创建软链接：ln -s /usr/local/python3/bin/python3.5 /usr/bin/python3

(8)Pip3创建软链接：ln -s /usr/local/python3/bin/pip3.5 /usr/bin/pip3

注：python编译错误：Ignoring ensurepip failure: pip 8.1.1 requires SSL/TLS

原因是系统中少了openssl-devel包，执行以下命令：

sudo apt-get install libssl-dev

sudo apt-get install openssl

### 2、安装项目依赖包

(1)导出项目所需的包到requirement.txt中：pip3 freeze > requirements.txt

(2)安装：pip3 install --ignore-installed  -r requirements.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

注：安装的时候报错：subprocess.CalledProcessError: Command '('lsb_release', '-a')' returned non-zero exit status 1.

解决：sudo rm /usr/bin/lsb_release

注：pip install pycurl "Error: Could not run curl-config: [Errno 2] No such file or directory"

解决：sudo apt-get install libcurl4-openssl-dev

### 3、打包django项目：

(1)安装pyinstaller：pip3 install pyinstaller

(2) pyinstaller -F manage.py --hidden-import rest_framework.authentication --hidden-import rest_framework.permissions  --hidden-import rest_framework.parsers --hidden-import rest_framework.negotiation --hidden-import rest_framework.metadata  --hidden-import  rest_framework.apps  --hidden-import  corsheaders --hidden-import  corsheaders.middleware

(3)执行：nohup ./manage runserver 0.0.0.0:8000 >RunInfo.log >&1 &

注: [Errno 2] No such file or directory: '/tmp/_MEImHg8uh/django/contrib/auth/password_validation.pyc'

解决：django 版本固定到 2.0

注：权限不足

解决：chmod +x manage

注：Traceback (most recent call last):

File "/usr/lib/command-not-found", line 27, in <module>

from CommandNotFound.util import crash_guard

ImportError: No module named 'CommandNotFound'

解决：将python3加入环境变量，并且执行source /etc/profile.

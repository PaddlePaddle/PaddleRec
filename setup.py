# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as file:
    REQUIRED_PACKAGES = file.read()

setup(
    name='paddlerec',
    description=(
        'A quick start tool of search & recommendation algorithm based on PaddlePaddle.'
    ),
    long_description='',
    url='https://github.com/PaddlePaddle/PaddleRec',
    download_url='https://github.com/PaddlePaddle/PaddleRec.git',
    author='PaddlePaddle Author',
    author_email='',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    # PyPI package information.
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=('paddlerec paddlepaddle recommendation-algorithm'))

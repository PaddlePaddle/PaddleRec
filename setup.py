"""
setup for fleet-rec.
"""
import os
import sys
from setuptools import setup, find_packages
import tempfile
import shutil

if sys.version_info.major == 2:
    requires = [
        "paddlepaddle == 1.7.2",
        "netron >= 0.0.0",
        "pyyaml >= 5.1.1"
    ]
else:
    requires = [
        "paddlepaddle >= 0.0.0",
        "netron >= 0.0.0",
        "pyyaml >= 5.1.1"
    ]

about = {}
about["__title__"] = "fleet-rec"
about["__version__"] = "0.0.2"
about["__description__"] = "fleet-rec"
about["__author__"] = "seiriosPlus"
about["__author_email__"] = "tangwei12@baidu.com"
about["__url__"] = "https://github.com/seiriosPlus/FleetRec"

readme = "..."


def run_cmd(command):
    assert command is not None and isinstance(command, str)
    return os.popen(command).read().strip()


def build(dirname):
    print(dirname)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    run_cmd("cp -r {}/* {}".format(package_dir, dirname))
    run_cmd("mv {} {}".format(os.path.join(dirname, "demo"), os.path.join(dirname, "fleetrec")))
    run_cmd("mv {} {}".format(os.path.join(dirname, "doc"), os.path.join(dirname, "fleetrec")))
    run_cmd("mv {} {}".format(os.path.join(dirname, "models"), os.path.join(dirname, "fleetrec")))
    run_cmd("mv {} {}".format(os.path.join(dirname, "tools"), os.path.join(dirname, "fleetrec")))

    packages = find_packages(dirname)
    package_dir = {'': dirname}

    setup(
        name=about["__title__"],
        version=about["__version__"],
        description=about["__description__"],
        long_description=readme,
        author=about["__author__"],
        author_email=about["__author_email__"],
        url=about["__url__"],
        packages=packages,
        package_dir=package_dir,
        python_requires=">=2.7",
        install_requires=requires,
        zip_safe=False
    )


dirname = tempfile.mkdtemp()
build(dirname)
shutil.rmtree(dirname)

print('''
\033[32m
  _   _   _   _   _   _   _   _   _  
 / \ / \ / \ / \ / \ / \ / \ / \ / \ 
( F | L | E | E | T | - | R | E | C )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ 
\033[0m
\033[34m
Installation Complete. Congratulations!
How to use it ? Please visit our webside: https://github.com/seiriosPlus/FleetRec
\033[0m
''')

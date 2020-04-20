"""
setup for fleet-rec.
"""

from setuptools import setup

packages = ["fleetrec", "fleetrec.models", "fleetrec.models.ctr_dnn",
            "fleetrec.examples", "fleetrec.core", "fleetrec.core.engine",
            "fleetrec.core.metrics", "fleetrec.core.models", "fleetrec.core.trainers", "fleetrec.core.utils"]

requires = [
    "paddlepaddle >= 0.0.0"
]

about = {}
about["__title__"] = "fleet-rec"
about["__version__"] = "0.0.2"
about["__description__"] = "fleet-rec"
about["__author__"] = "seiriosPlus"
about["__author_email__"] = "tangwei12@baidu.com"
about["__url__"] = "https://github.com/seiriosPlus/FleetRec"

readme = "..."

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    long_description=readme,
    author=about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    packages=packages,
    python_requires=">=2.7",
    install_requires=requires,
    zip_safe=False
)

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

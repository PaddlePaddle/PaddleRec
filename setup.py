"""
setup for fleet-rec.
"""

from setuptools import setup

packages = ["fleetrec", "fleetrec.examples", "fleetrec.metrics", "fleetrec.models", "fleetrec.reader",
            "fleetrec.trainer", "fleetrec.utils"]

requires = [
    "paddlepaddle>=1.6.2"
]

about = {}
about["__title__"] = "fleet-rec"
about["__version__"] = "0.0.2"
about["__description__"] = "fleet-rec"

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
    python_requires=">=3.0, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3*",
    install_requires=requires,
    zip_safe=False
)

print('''
\033[32m
███████╗██╗     ███████╗███████╗████████╗
██╔════╝██║     ██╔════╝██╔════╝╚══██╔══╝
█████╗  ██║     █████╗  █████╗     ██║   
██╔══╝  ██║     ██╔══╝  ██╔══╝     ██║   
██║     ███████╗███████╗███████╗   ██║   
╚═╝     ╚══════╝╚══════╝╚══════╝   ╚═╝   
\033[0m
\033[34m
Installation Complete. Congratulations!
How to use it ? Please visit our webside: https://github.com/seiriosPlus/FleetRec
\033[0m
''')

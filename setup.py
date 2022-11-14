
import setuptools
import os

def get_requirements(path):
    ret = []
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as freq:
        for line in freq.readlines():
            ret.append( line.strip() )
    return ret


path = os.path.dirname(os.path.abspath(__file__))
requires =  get_requirements(path)
print(requires)
setuptools.setup(
        name = 'opendelta',
        version = '0.0.1',
        python_requires=">=3.8.0",
        install_requires=requires,
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ]
)

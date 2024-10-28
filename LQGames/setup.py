from setuptools import setup, find_packages

setup(
    name="LQGames",
    version="0.1.0",
    author="Joaquim Antônio Costa Bermudes",
    author_email="joaquimacbermudes@gmail.com",
    description="Package for synthesis and simulation of linear quadratic differential games",
    url="https://github.com/joaquimbermudes/LQGames",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "warning",
        "copy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

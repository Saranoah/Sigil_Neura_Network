from setuptools import setup, find_packages

setup(
    name="sigil-network",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0"
    ],
    author="Israa Ali",
    description="Kintsugi-inspired neural network with Value-Weighted Pathway Reinforcement",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Saranoah/Sigil_Neura_Network",
    python_requires=">=3.8",
)


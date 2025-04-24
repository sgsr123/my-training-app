from setuptools import find_packages, setup

setup(
    name="trainer",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    description="Vertex AI model training application",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=[
        "tensorflow>=2.8.0",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "cloudml-hypertune",
    ],
    python_requires=">=3.7",
)
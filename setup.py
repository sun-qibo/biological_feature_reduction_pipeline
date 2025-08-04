from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="biological-feature-reduction-pipeline",
    version="0.1.0",
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="A feature reduction pipeline for biological datasets with interpretability preservation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/biological_feature_reduction_pipeline",  # Replace with your repo URL
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/biological_feature_reduction_pipeline/issues",
        "Documentation": "https://github.com/yourusername/biological_feature_reduction_pipeline#readme",
        "Source Code": "https://github.com/yourusername/biological_feature_reduction_pipeline",
    },
    packages=find_packages(),
    py_modules=["biological_feature_reducer"],  # Since you have a single module
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    keywords=[
        "bioinformatics",
        "feature-reduction",
        "machine-learning",
        "genomics", 
        "microbiome",
        "data-preprocessing",
        "correlation-analysis",
        "interpretable-ml"
    ],
    entry_points={
        "console_scripts": [
            # Uncomment if you want command-line interface
            # "bio-feature-reduce=biological_feature_reducer:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

"""
Morgan RAG Setup Configuration.

Installs Morgan AI Assistant with CLI commands:
- morgan: Main user CLI
- morgan-admin: Admin CLI
"""

from pathlib import Path
from setuptools import find_packages, setup

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="morgan-rag",
    version="2.0.0",
    description="Morgan AI Assistant - Intelligent, emotionally-aware assistant with RAG and learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Morgan Team",
    author_email="team@morgan.ai",
    url="https://github.com/yourusername/morgan",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "morgan=morgan.cli.app:main",
            "morgan-admin=morgan.cli.distributed_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai assistant rag emotion-detection learning nlp",
    project_urls={
        "Documentation": "https://morgan.readthedocs.io/",
        "Source": "https://github.com/yourusername/morgan",
        "Tracker": "https://github.com/yourusername/morgan/issues",
    },
)

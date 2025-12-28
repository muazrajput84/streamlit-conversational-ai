from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="ai-chatbot-streamlit",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered chatbot with Streamlit UI and Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-chatbot-streamlit",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.29.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chatbot-train=src.train_model:main",
            "chatbot-app=src.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["data/*.json", "assets/**/*"],
    },
    keywords="chatbot ai machine-learning streamlit nlp",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ai-chatbot/issues",
        "Source": "https://github.com/yourusername/ai-chatbot",
        "Documentation": "https://github.com/yourusername/ai-chatbot/blob/main/README.md",
    },
)

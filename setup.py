from setuptools import setup, find_packages

setup(
    name="financial-advisor-llm",
    version="0.1.0",
    description="Privacy-focused LLM financial advisor for local document analysis",
    author="Financial Advisor Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "langchain>=0.0.350",
        "langchain-community>=0.0.10",
        "faiss-cpu>=1.7.4",
        "PyMuPDF>=1.23.0",
        "sentence-transformers>=2.2.2",
        "ollama>=0.1.7",
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tiktoken>=0.5.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
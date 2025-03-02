from setuptools import setup, find_packages

# Read version from package __init__.py
with open('skype_analyzer/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"\'')
            break

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description from README.md
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="skype_analyzer",
    version=version,
    description="A tool for analyzing Skype chat history using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sten Tamkivi",
    author_email="sten@tamkivi.com",
    url="https://github.com/stentamkivi/skype-chatdb-to-llm",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "skype-analyzer=skype_analyzer.cli:main",
        ],
    },
) 
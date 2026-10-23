from setuptools import setup, find_packages

setup(
    name="poolforge-agent",
    version="0.1.0",
    description="PoolForge distributed worker agent CLI",
    packages=find_packages(),
    py_modules=["cli", "worker"],
    entry_points={
        "console_scripts": [
            "poolforge=cli:main",
        ],
    },
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "psutil>=5.9.0",
        "websockets>=11.0",
        "requests>=2.28.0",
        "pydantic>=2.0.0",
        "cryptography>=41.0.0",
    ],
)

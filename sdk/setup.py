from setuptools import setup, find_packages

setup(
    name='fedn',
    version='0.0.1',
    description="""Scaleout Federated Learning""",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Morgan Ekmefjord',
    author_email='morgan@scaleout.se',
    url='https://www.scaleoutsystems.com',
    include_package_data=True,
    py_modules=['fedn'],
    python_requires='>=3.5,<4',
    install_requires=[
        "attrdict>=2.0.1",
        "certifi>=2018.11.29",
        "chardet>=3.0.4",
        "Click>6.6",
        "PyYAML>=4.2b1",
        "requests==2.21.0",
        "urllib3==1.24.2",
        "minio==5.0.6",
        "six>=1.14.0",
        "python-slugify",
        "prettytable",
        "grpcio-tools",
        "grpcio>=1.30",
        "protobuf",
        "pypandoc",
        "wheel",
        "pymongo",
    ],
    license="Copyright Scaleout Systems AB. See license for details",
    zip_safe=False,
    entry_points={
        'console_scripts': ["fedn=cli:main"]
    },
    keywords='Federated learning',
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)

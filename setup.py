from setuptools import setup, find_packages

setup(
    name='toxic',
    version='0.0.1',
    packages=['toxic'],
    entry_points={
        'console_scripts': [
            'toxic-server=toxic.entrypoints:server',
            'toxic-client=toxic.entrypoints:client',
            'toxic-train=toxic.entrypoints:train',
            'toxic-optim=toxic.entrypoints:optimization'
        ],
    },
    install_requires=[
        'pip>=20.1.1',
        'wheel>=0.34.2',
        'setuptools>=41.0.0',
        'pandas==1.0.4',
        'tensorflow==2.2.0',
        'optuna==1.4.0',
        'sacremoses==0.0.43',
        'numpy==1.18.4',
        'transformers==2.10.0',
        'neptune-client==0.4.114',
        'tensorflow-datasets==3.1.0',
        'wild-nlp==1.0.2',
        'nlpaug==0.0.14',
        'requests==2.23.0',
        'nltk==3.5',
        'psutil==5.7.0'
    ],
    include_package_data=True,
)

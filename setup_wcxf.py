from setuptools import setup, find_packages

setup(name='wcxf',
      version='1.2.3',
      author='David M. Straub, Jason Aebischer',
      author_email='david.straub@tum.de, jason.aebischer@tum.de',
      license='MIT',
      packages=find_packages(),
      package_data={
        'wcxf': ['data/*.yml',
                 'data/*.yaml',
                 'data/*.json',
                 'bases/*.json',
                 'bases/child/*.json',
                ]
      },
      install_requires=['pyyaml', 'ckmutil>=0.3', 'rundec>=0.5'],
      extras_require={
            'testing': ['nose', 'smeftrunner'],
      },
      entry_points={
        'console_scripts': [
            'wcxf = wcxf.cli:wcxf_cli',
            'wcxf2eos = wcxf.cli:eos',
            'wcxf2smeftsim = wcxf.cli:smeftsim',
        ]
      },
    )

from setuptools import setup, find_packages


with open('wilson/_version.py') as f:
    exec(f.read())


setup(name='wilson',
      version=__version__,
      author='Jason Aebischer, David M. Straub',
      author_email='jason.aebischer@tum.de, david.straub@tum.de',
      url='https://github.com/wilsoneft/wilson',
      description='Toolkit for effective field theories beyond the Standard Model',
      license='MIT',
      packages=find_packages(),
      package_data={
      'wilson': ['smeftrunner/tests/data/*',
                 'data/*.yml',
                 'data/*.yaml',
                 'data/*.json',
                 'wetrunner/tests/data/*',
                 ],
      },
      install_requires=['scipy>=1.0', 'numpy', 'pylha>=0.2', 'pyyaml',
                        'ckmutil>=0.3', 'wcxf>=1.2', 'rundec>=0.5'],
      extras_require={
            'testing': ['nose', 'smeftrunner'],
      },
    )

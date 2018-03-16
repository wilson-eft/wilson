from setuptools import setup, find_packages

setup(name='wetrunner',
      version='0.2',
      author='Jason Aebischer, Xuanyou Pan, David M. Straub',
      author_email='jason.aebischer@tum.de, xuanyou.pan@tum.de, david.straub@tum.de',
      url='https://github.com/DsixTools/python-wetrunner',
      description='A Python package for the renormalization group evolution in the Weak Effective Theory (WET).',
      license='MIT',
      packages=find_packages(),
      package_data={'wetrunner': ['tests/data/*']},
      extras_require={'testing': ['nose']},
      install_requires=['numpy', 'rundec>=0.5', 'wcxf>=1.2'],
      )

from setuptools import setup, find_packages


with open('wilson/_version.py') as f:
    exec(f.read())

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(name='wilson',
      version=__version__,
      author='Jason Aebischer, David M. Straub',
      author_email='jason.aebischer@tum.de, david.straub@tum.de',
      url='https://github.com/wilson-eft/wilson',
      description='Toolkit for effective field theories beyond the Standard Model',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      license='MIT',
      packages=find_packages(),
      package_data={
      'wilson': ['run/smeft/tests/data/*.*',
                 'data/tests/*.*',
                 'data/*.*',
                 ],
      },
      install_requires=['scipy>=1.0', 'numpy', 'pylha>=0.2', 'pyyaml',
                        'ckmutil>=0.3', 'wcxf==1.4.7', 'rundec>=0.5',
                        'voluptuous'],
      extras_require={
            'testing': ['nose',],
      },
    )

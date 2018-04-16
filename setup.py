from setuptools import setup, find_packages


with open('wilson/_version.py') as f:
    exec(f.read())

#
setup(name='wilson',
      version=__version__,
      author='Jason Aebischer, David M. Straub',
      author_email='jason.aebischer@tum.de, david.straub@tum.de',
      url='https://github.com/wilson-eft/wilson',
      description='Toolkit for effective field theories beyond the Standard Model',
      long_description="""``wilson`` is a Python library for matching and running Wilson coefficients of
higher-dimensional operators beyond the Standard Model. Provided with the numer-
ical values of the Wilson coefficients at a high new physics scale, it automatically per-
forms the renormalization group evolution within the Standard Model effective field
theory (SMEFT), matching onto the weak effective theory (WET) at the electroweak
scale, and QCD/QED renormalization group evolution below the electroweak scale
down to hadronic scales relevant for low-energy precision tests. The matching and
running encompasses the complete set of dimension-six operators in both SMEFT
and WET. The program builds on the Wilson coefficient exchange format (WCxf)
and can thus be easily combined with a number of existing public codes.""",
      license='MIT',
      packages=find_packages(),
      package_data={
      'wilson': ['run/wet/tests/data/*',
                 'run/wet/tests/data/*',
                 'data/tests/*',
                 'data/*',
                 ],
      },
      install_requires=['scipy>=1.0', 'numpy', 'pylha>=0.2', 'pyyaml',
                        'ckmutil>=0.3', 'wcxf>=1.3', 'rundec>=0.5'],
      extras_require={
            'testing': ['nose', 'smeftrunner'],
      },
    )

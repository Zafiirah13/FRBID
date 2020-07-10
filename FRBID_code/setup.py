from setuptools import setup, find_packages

setup(name = 'FRBID',
      version = '1.0.0',
      description = 'Fast Radio Burst Intelligent Distinguisher using Deep Learning',
      author = 'Zafiirah Hosenie',
      author_email = 'zafiirah.hosenie@gmail.com',
      license = 'MIT',
      url = 'https://github.com/Zafiirah13/FRBID',
      packages = find_packages(),
      install_requires=['pandas==0.24.1',
                        'numpy==1.14.6',
                        'tensorflow-gpu==1.9.0',
                        'imbalanced_learn==0.4.3',
                        'matplotlib==3.0.3',
                        'scipy==1.2.1',
                        'Keras==2.0.9',
                        'imblearn==0.0',
                        'Pillow==7.1.2',
                        'scikit_learn==0.23.1'
                        ],
      classifiers=[
                  'Programming Language :: Python :: 3.6',
                  ],
      keywords=['Convolutional Neural Network', 'Deep Learning', 'FRB and RFI', 'MeerKAT'],
      include_package_data = True)


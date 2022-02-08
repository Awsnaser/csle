from setuptools import setup

setup(name='csle_common',
      version='0.0.1',
      install_requires=['gym', 'pyglet', 'numpy', 'torch', 'docker', 'paramiko', 'stable_baselines3', 'scp',
                        'random_username', 'jsonpickle', 'Sphinx', 'sphinxcontrib-napoleon',
                        'sphinx-rtd-theme', 'psycopg', 'click', 'flask', 'waitress'],
      author='Kim Hammar',
      author_email='hammar.kim@gmail.com',
      description='csle is a platform for evaluating and developing reinforcement learning agents for '
                  'control problems in cyber security; csle-common contains the common functionality of csle modules',
      license='Creative Commons Attribution-ShareAlike 4.0 International',
      keywords='Reinforcement-Learning Cyber-Security Markov-Games Markov-Decision-Processes',
      url='https://github.com/Limmen/csle',
      download_url='https://github.com/Limmen/csle/archive/0.0.1.tar.gz',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Programming Language :: Python :: 3.8'
      ]
)
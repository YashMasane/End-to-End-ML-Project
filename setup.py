from setuptools import find_packages, setup
from typing import List

def get_requirments(file_path)->List[str]:

    '''
    This function will return all requirements for this project
    '''
    HYPEN_E_DOT = '-e .'
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [requirement.replace('\n' ,'') for requirement in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements
        

setup(
    name='End to End ML Project',
    version='0.0.1',
    author='Yash Masane',
    author_email='masaneyash6@gmail.com',
    packages=find_packages(),
    install_requires=get_requirments('requirements.txt')

)
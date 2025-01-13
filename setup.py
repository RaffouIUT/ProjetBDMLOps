import setuptools

with open('README.md','r',encoding="utf-8") as f:
    long_description=f.read()




__version__= "0.0.0"

REPO_NAME='mlops_template_project'
AUTHOR_USER_NAME="asini"
SRC_REPO = "gridPower"
AUTHOR_EMAIL="aghilas.sini@univ-lemans.fr"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='a small  python package for  mlops ',
    long_description=long_description,
    long_destription_content='text/markdown',
    url=f"https://git.univ-lemans.fr/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={

        "Bug Tracker":f"https://git.univ-lemans.fr/{AUTHOR_USER_NAME}/{REPO_NAME}/",

    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where='src'),


)

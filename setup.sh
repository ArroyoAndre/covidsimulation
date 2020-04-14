#/bin/bash

pip3 install -r requirements.txt
python3 setup.py build_ext --inplace
python3 setup.py install

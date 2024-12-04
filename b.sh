bash ./scripts/install_deps.sh

pip install -r ./requirements/requirements.txt

pip install mmcv
pip install loguru
pip install chardet
pip install glumpy

sh /scripts/compile_all.sh


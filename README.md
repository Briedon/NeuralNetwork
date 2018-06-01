# NeuralNetwork

nainstalovanie python packagov je potrebne mat pip a python aspon 3.5

pip by mal byt normalne nainstalovany ak nie je, podla verzie linuxu nainstaluj https://packaging.python.org/guides/installing-using-linux-tools/

pak pre kazdy balicek napises
sudo pip3 install jmeno balicku

dynet:
http://dynet.readthedocs.io/en/latest/python.html
pre cpu verzi
pip3 install dynet
pre gpu a najnovejsi verzi
BACKEND=cuda pip install git+https://github.com/clab/dynet#egg=dynet

networkx:
to je package na spracovanie grafov
https://networkx.github.io/documentation/stable/install.html
pip install neetworkx

pytorch:
pytorch ma problem s poliami/listami premenych ma sa na to pouzivat specialny modul ale ten nie je rychly spis pomaly, co je velky problem ak chcem nacitat nahodne grafy s roznym poctom vrstiev, ale ak sa to napise staticky tak je to fakt rychle
https://pytorch.org/
uz v zaklade ma gpu
pip3 install torch torchvision

tensorflow:
https://www.tensorflow.org/install/install_linux
sudo apt-get install cuda-command-line-tools
pak pridas vygenerovane cuda kniznice do cesty pre kniznice
xport LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64
pak pre gpu
pip3 install tensorflow-gpu
pak pre cpu
pip3 install tensorflow

pre pouzivanie c++ potrebujes eigen
na to aby to fungoval musis stiahnut a extractovat kody pomocou hg clone https://bitbucket.org/eigen/eigen/ nebo 
git clone https://github.com/eigenteam/eigen-git-mirror
a nalinkovat do /usr/include, pak by to malo fungovat
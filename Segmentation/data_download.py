import os
import tarfile
import urllib.request as request

url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
data_dir = "./data"
target_dir = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(target_dir):
    request.urlretrieve(url=url, filename=target_dir)

    tar = tarfile.TarFile(target_dir)
    tar.extractall(data_dir)

    tar.close()

os.remove(target_dir)


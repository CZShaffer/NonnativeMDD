import urllib
import urllib.request
import os
from tqdm import tqdm
import tarfile
import zipfile
import gdown

def url_retrieve(url, dir=None, filename=None, deleteExisting=False, compression=None, gdrive=False):
    '''
    Downloads data from a given url
    :param url: url of file to download
    :param dir: directory where file will be written to, defaults to current directory
    :param filename: filename of file being saved, defaults to name of retrieved file
    :param deleteExisting: whether or not to delete file if it already exists, defaults to skip download if file exists
    :param compression: compression type of the file to be extracted. Deletes compressed file after extracting.
    Setting to none leaves the file compressed and will not delete the compressed file
    :param gdrive: Whether or not the url links to google drive, defaults to false
    :return:
    '''
    file_name = url.split('/')[-1] if filename is None else filename
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_name = dir + file_name
    if os.path.exists(file_name) or os.path.exists(file_name.split('.')[0]):
        if not deleteExisting:
            print(f"{file_name} already exists. Skipping Download.")
            return
        print(f"{file_name} already exists. Deleting existing file.")

    if gdrive:
        gdown.download(url, "L2_ARCTIC", quiet=False)
    else:
        # based on https://stackoverflow.com/questions/22676/how-to-download-a-file-over-http
        f = open(file_name, 'wb')
        with urllib.request.urlopen(url) as u:
            file_size = int(u.getheader("Content-Length"))
            print(f"Downloading: {file_name} Bytes: {file_size}")
            progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024)
            file_size_dl = 0
            block_sz = 8192*4
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                f.write(buffer)
                progress_bar.update(block_sz)
            progress_bar.close()
            print(file_size_dl)
        f.close()
        # end cite
    print(f"downloaded {file_name}")
    extract(file_name, compression)
    print("Done")

def extract(file_name, compression):
    '''
    extracts files from a given archive to a directory of the same name
    :param file_name: file path to the compressed file
    :param compression: compression type of the file to be extracted. Deletes compressed file after extracting.
    Setting to none leaves the file compressed and will not delete the compressed file
    :return:
    '''
    if compression == "tar":
        tar = tarfile.open(file_name, "r:")
        print(f"Extracting to {file_name[:-4]}")
        if not os.path.exists(file_name[:-4]):
            os.makedirs(file_name[:-4])
        tar.extractall(file_name[:-4])
        tar.close()
    elif compression == "tar.gz":
        tar = tarfile.open(file_name, "r:gz")
        print(f"Extracting to {file_name[:-7]}")
        if not os.path.exists(file_name[:-7]):
            os.makedirs(file_name[:-7])
        tar.extractall(file_name[:-7])
        tar.close()
        os.remove(file_name)
    elif compression == "zip":
        zip = zipfile.ZipFile(file_name, "r:")
        print(f"Extracting to {file_name[:-4]}")
        if not os.path.exists(file_name[:-4]):
            os.makedirs(file_name[:-4])
        zip.extractall(file_name[:-4])
        zip.close()



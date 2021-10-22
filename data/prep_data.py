import os
from url_retrieve import url_retrieve, extract

def prep_librispeech(dataset="mini", deleteExisting=False, compression="tar.gz"):
    '''
    Downloads and extracts the Librispeech dataset to the data folder
    :param dataset: which Librispeech dataset to download, defaults to Librispeech mini. Must be one of "mini", "100", or "360"
    :param deleteExisting: whether or not to delete file if it already exists, defaults to skip download if file exists
    :param compression: compression type of the file to be extracted. Deletes compressed file after extracting.
    Setting to none leaves the file compressed and will not delete the compressed file
    :return:
    '''
    mini_train = "http://www.openslr.org/resources/31/train-clean-5.tar.gz"
    libri_100_train = "http://www.openslr.org/resources/31/train-clean-100.tar.gz"
    libri_360_train = "http://www.openslr.org/resources/31/train-clean-360.tar.gz"
    if dataset == "mini":
        url_retrieve(mini_train, dir="Librispeech/",deleteExisting=deleteExisting, compression=compression)
    if dataset == "100":
        url_retrieve(libri_100_train, dir="Librispeech/",deleteExisting=deleteExisting, compression=compression)
    if dataset == "360":
        url_retrieve(libri_360_train, dir="Librispeech/",deleteExisting=deleteExisting, compression=compression)

def prep_l2_arctic(deleteExisting=False, compression="zip", extract_subdirectory=False):
    '''
    Downloads and extracts the L2 ARCTIC dataset to the data folder
    :param deleteExisting: whether or not to delete file if it already exists, defaults to skip download if file exists
    :param compression: compression type of the file to be extracted. Deletes compressed file after extracting.
    Setting to none leaves the file compressed and will not delete the compressed file
    :param extract_subdirectory: whether or not to extrat all subdirecotires within the L2 Arctic dataset. Due to the
    size of the dataset this is false by default.
    :return:
    '''
    # Before accessing this data, please request permission from https://psi.engr.tamu.edu/l2-arctic-corpus/
    # The l2_arctic variable should contain the URL of the dataset you recieved from site above. An example url is shown below for formatting purposes
    # l2_arctic = "https://drive.google.com/uc?id=1JvI3ktRJdC-CSHEieQMpS8Wca2FTPxjr"
    l2_arctic = ""
    dir = "L2_ARCTIC/"
    filename = "L2_ARCTIC.zip"
    url_retrieve(l2_arctic, dir=dir, filename=filename, deleteExisting=deleteExisting, compression=compression, gdrive=True)
    if extract_subdirectory:
        dirs = []
        print("Directories to be extracted:")
        for root, dirs, files in os.walk(dir+filename[:-4]):
            for file in files:
                if file.endswith(".zip"):
                    zipf = os.path.join(root, file).replace("\\","/")
                    print(zipf)
                    dirs.append(zipf)
        for dir in dirs:
            extract(dir, "zip")

# prep_librispeech(dataset="mini", deleteExisting=False)
# prep_l2_arctic(deleteExisting=False, extract_subdirectory=False)
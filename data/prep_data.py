import os
from url_retrieve import url_retrieve

def prep_librispeech(dataset="mini", deleteExisting=True, compression="tar.gz"):
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

def prep_l2_arctic(deleteExisting=True, compression="zip"):
    '''
    Downloads and extracts the L2 ARCTIC dataset to the data folder
    :param deleteExisting: whether or not to delete file if it already exists, defaults to skip download if file exists
    :param compression: compression type of the file to be extracted. Deletes compressed file after extracting.
    Setting to none leaves the file compressed and will not delete the compressed file
    :return:
    '''
    # Before accessing this data, please request permission from https://psi.engr.tamu.edu/l2-arctic-corpus/
    l2_arctic = "https://drive.google.com/uc?id=1JvI3ktRJdC-CSHEieQMpS8Wca2FTPxjr"
    url_retrieve(l2_arctic, dir="L2_ARCTIC/",filename="L2_ARCTIC" ,deleteExisting=deleteExisting, compression=compression, gdrive=True)
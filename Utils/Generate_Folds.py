# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:32:56 2021
Modified from [repo]
@author: jpeeples
"""

"""
Splits the following dataset into k-folds:
1. BreakHis.
2. BACH (part A) 2018.
3. GlaS.
4. CAMELYON16.
"""

__author__ = "Soufiane Belharbi, https://sbelharbi.github.io/"
__copyright__ = "Copyright 2018, ÉTS-Montréal"
__license__ = "GPL"
__version__ = "3"
__maintainer__ = "Soufiane Belharbi"
__email__ = "soufiane.belharbi.1@etsmtl.net"


import glob
from os.path import join
import os
import traceback
import random
import sys
import math
import csv
import copy
import pdb


import reproducibility

def chunk_it(l, n):
    """
    Create chunks with the same size (n) from the iterable l.
    :param l: iterable.
    :param n: int, size of the chunk.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Dict2Obj(object):
    """
    Convert a dictionary into a class where its attributes are the keys of the dictionary, and
    the values of the attributes are the values of the keys.
    """
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs

def split_valid_SFBHI(args):
    """
    Create a validation/train sets in SFBHI dataset.
    csv file format: relative path to the image, relative path to the mask, class (str: benign, malignant).
    :param args:
    :return:
    """
    
    all_samples = []
    # Read the file Labeled Image Reference Length.csv
    baseurl = args.baseurl
    with open(join(baseurl, "Labeled Image Reference Length.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # get rid of the header.
        for row in reader:
            # not sure why they thought it is a good idea to put a space before the class. Now, I have to get rid of
            # it and possibly other hidden spaces ...
            row = [r.replace(" ", "") for r in row]
            # assert row[2] in classes, "The class `{}` is not within the predefined classes `{}`".format(row[2], classes)
            all_samples.append(row[0])

    assert len(all_samples) == 105, "The number of samples {} do not match what they said (105) .... [NOT " \
                                    "OK]".format(len(all_samples))

    # Take test samples aside. They are fix.
    pdb.set_trace()
    # test_samples = [s for s in all_samples if s[0].startswith("test")]
    # assert len(test_samples) == 80, "The number of test samples {} is not 80 as they said .... [NOT OK]".format(len(
    #     test_samples))

    # all_train_samples = [s for s in all_samples if s[0].startswith("train")]
    # assert len(all_train_samples) == 85, "The number of train samples {} is not 85 as they said .... [NOT OK]".format(
    #     len(all_train_samples))

    # benign = [s for s in all_train_samples if s[1] == "benign"]
    # malignant = [s for s in all_train_samples if s[1] == "malignant"]

    # Split
    splits = []
    # for i in range(args.nbr_splits):
    #     for _ in range(1000):
    #         random.shuffle(benign)
    #         random.shuffle(malignant)
    #     splits.append({"benign": copy.deepcopy(benign),
    #                    "malignant": copy.deepcopy(malignant)}
    #                   )

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold contains a train, and valid set with a
        predefined size.
        Note: Samples are expected to be shuffled beforehand.
        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set, ts_set): where each element is the list (str paths)
                 of the samples of each set: train, valid, and test, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the provided sizes."

        # chunk the data into chunks of size ts (the size of the test set), so we can rotate the test set.
        list_chunks = list(chunk_it(lsamps, s_vl))
        list_folds = []

        for i in range(len(list_chunks)):
            vl_set = list_chunks[i]

            right, left = [], []
            if i < len(list_chunks) - 1:
                right = list_chunks[i + 1:]
            if i > 0:
                left = list_chunks[:i]

            leftoverchunks = right + left

            leftoversamples = []
            for e in leftoverchunks:
                leftoversamples += e

            tr_set = leftoversamples
            list_folds.append((tr_set, vl_set))

        return list_folds

    # Save the folds.
    # Save the folds into *.csv files.
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.
        csv file format: relative path to the image, relative path to the mask, class (str: benign, malignant).
        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for name, clas in lsamples:
                filewriter.writerow([name + ".png", name + ".png", clas])

    def create_one_split(split_i, test_samples, benign, malignant, nbr_folds):
        """
        Create one split of k-folds.
        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param benign: list, list of train benign samples.
        :param malignant: list, list of train maligant samples.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        vl_size_benign = math.ceil(len(benign) * args.folding["vl"] / 100.)
        vl_size_malignant = math.ceil(len(malignant) * args.folding["vl"] / 100.)

        list_folds_benign = create_folds_of_one_class(benign, len(benign) - vl_size_benign, vl_size_benign)
        list_folds_malignant = create_folds_of_one_class(malignant, len(malignant) - vl_size_malignant,
                                                         vl_size_malignant)

        assert len(list_folds_benign) == len(list_folds_malignant), "We didn't obtain the same number of fold" \
                                                                    " .... [NOT OK]"
        assert len(list_folds_benign) == 5, "We did not get exactly 5 folds, but `{}` .... [ NOT OK]".format(
            len(list_folds_benign))
        print("We found {} folds .... [OK]".format(len(list_folds_malignant)))

        outd = args.fold_folder
        for i in range(nbr_folds):
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv(test_samples, join(out_fold, "test_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the train set
            train = list_folds_malignant[i][0] + list_folds_benign[i][0]
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv(train, join(out_fold, "train_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the valid set
            valid = list_folds_malignant[i][1] + list_folds_benign[i][1]
            dump_fold_into_csv(valid, join(out_fold, "valid_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])

        with open(join(outd, "readme.md"), 'w') as fx:
            fx.write("csv format:\nrelative path to the image, relative path to the mask, class "
                     "(str: benign, malignant).")

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)


    # Creates the splits
    for i in range(args.nbr_splits):
        create_one_split(i, test_samples, splits[i]["benign"], splits[i]["malignant"], args.nbr_folds)

    print("All GlaS splitting (`{}`) ended with success .... [OK]".format(args.nbr_splits))


def do_SFBHI():
    """
    SFBHI
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    baseurl = os.getcwd()

    args = {"baseurl": baseurl,
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "SFBHI",
            "fold_folder": "folds/SFBHI",
            "img_extension": "png",
            "nbr_folds": 5,
            "nbr_splits": 1  # how many times to perform the k-folds over the available train samples.
            }
    split_valid_SFBHI(Dict2Obj(args))




if __name__ == "__main__":

    # ============== CREATE FOLDS OF GlaS DATASET
    do_SFBHI()
    pdb.set_trace()

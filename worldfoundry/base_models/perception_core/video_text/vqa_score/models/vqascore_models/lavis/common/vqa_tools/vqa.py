"""
Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

__author__ = "aagrawal"
__version__ = "0.9"

# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import copy
import datetime
import json


class VQA:
    def __init__(self, annotation_file=None, question_file=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if not annotation_file == None and not question_file == None:  # noqa: SIM201, E711 - Keeps `not ==` for operands that may compare elementwise.; Keeps `==` for operands that may compare elementwise.
            print("loading VQA annotations and questions into memory...")
            _time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file))  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
            questions = json.load(open(question_file))  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
            self.dataset = dataset
            self.questions = questions
            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        imgToQA = {ann["image_id"]: [] for ann in self.dataset["annotations"]}
        qa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        qqa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        for ann in self.dataset["annotations"]:
            imgToQA[ann["image_id"]] += [ann]
            qa[ann["question_id"]] = ann
        for ques in self.questions["questions"]:
            qqa[ques["question_id"]] = ques
        print("index created!")

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.datset["info"].items():
            print("%s: %s" % (key, value))  # noqa: UP031 - Keeps printf conversions that depend on argument types.

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):  # noqa: B006 - Keeps the upstream default object and signature.
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param 	imgIds    (int array)   : get question ids for given imgs
                        quesTypes (str array)   : get question ids for given question types
                        ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) is list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) is list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) is list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:  # noqa: SIM201 - Keeps `not ==` for operands that may compare elementwise.
                anns = sum(  # noqa: RUF017 - Keeps the upstream sum() list concatenation.
                    [self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA],
                    [],
                )
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann["question_type"] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann["answer_type"] in ansTypes]
        ids = [ann["question_id"] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):  # noqa: B006 - Keeps the upstream default object and signature.
        """
         Get image ids that satisfy given filter conditions. default skips that filter
         :param quesIds   (int array)   : get image ids for given question ids
        quesTypes (str array)   : get image ids for given question types
        ansTypes  (str array)   : get image ids for given answer types
         :return: ids     (int array)   : integer array of image ids
        """
        quesIds = quesIds if type(quesIds) is list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) is list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) is list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(quesIds) == 0:  # noqa: SIM201 - Keeps `not ==` for operands that may compare elementwise.
                anns = sum([self.qa[quesId] for quesId in quesIds if quesId in self.qa], [])  # noqa: RUF017 - Keeps the upstream sum() list concatenation.
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann["question_type"] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann["answer_type"] in ansTypes]
        ids = [ann["image_id"] for ann in anns]
        return ids

    def loadQA(self, ids=[]):  # noqa: B006 - Keeps the upstream default object and signature.
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) is list:
            return [self.qa[id] for id in ids]
        elif type(ids) is int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann["question_id"]
            print("Question: %s" % (self.qqa[quesId]["question"]))  # noqa: UP031 - Keeps printf conversions that depend on argument types.
            for ans in ann["answers"]:
                print("Answer %d: %s" % (ans["answer_id"], ans["answer"]))  # noqa: UP031 - Keeps printf conversions that depend on argument types.

    def loadRes(self, resFile, quesFile):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = VQA()
        res.questions = json.load(open(quesFile))  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
        res.dataset["info"] = copy.deepcopy(self.questions["info"])
        res.dataset["task_type"] = copy.deepcopy(self.questions["task_type"])
        res.dataset["data_type"] = copy.deepcopy(self.questions["data_type"])
        res.dataset["data_subtype"] = copy.deepcopy(self.questions["data_subtype"])
        res.dataset["license"] = copy.deepcopy(self.questions["license"])

        print("Loading and preparing results...     ")
        time_t = datetime.datetime.utcnow()
        anns = json.load(open(resFile))  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
        assert type(anns) is list, "results is not an array of objects"
        annsQuesIds = [ann["question_id"] for ann in anns]
        assert set(annsQuesIds) == set(self.getQuesIds()), (
            "Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file."
        )
        for ann in anns:
            quesId = ann["question_id"]
            if res.dataset["task_type"] == "Multiple Choice":
                assert ann["answer"] in self.qqa[quesId]["multiple_choices"], (
                    "predicted answer is not one of the multiple choices"
                )
            qaAnn = self.qa[quesId]
            ann["image_id"] = qaAnn["image_id"]
            ann["question_type"] = qaAnn["question_type"]
            ann["answer_type"] = qaAnn["answer_type"]
        print("DONE (t=%0.2fs)" % ((datetime.datetime.utcnow() - time_t).total_seconds()))  # noqa: UP031 - Keeps printf conversions that depend on argument types.

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

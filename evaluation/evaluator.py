#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import inspect
import json
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import SimpleITK as sitk
from evaluation.metrics import ConfusionMatrix, ALL_METRICS


class Evaluator:
    """Object that holds test and reference segmentations with label information
    and computes a number of metrics on the two. 'labels' must either be an
    iterable of numeric values (or tuples thereof) or a dictionary with string
    names and numeric values.
    """

    default_metrics = [
        "False Positive Rate",
        "Dice",
        "Jaccard",
        "Precision",
        "Recall",
        "Accuracy",
        "False Omission Rate",
        "Negative Predictive Value",
        "False Negative Rate",
        "True Negative Rate",
        "False Discovery Rate",
        "Total Positives Test",
        "Total Positives Reference"
    ]

    default_advanced_metrics = [
        "Hausdorff Distance",
        "Hausdorff Distance 95",
        "Avg. Surface Distance",
        "Avg. Symmetric Surface Distance"
    ]

    # metrics in mm: computed with the advanced ones only when evaluate() gets the voxel spacing
    # (spacing_mm=...). The voxel-unit distances above stay unchanged, comparable with earlier runs
    default_spacing_metrics = [
        "NSD 3mm",
        "Hausdorff Distance 95 mm",
        "Avg. Symmetric Surface Distance mm"
    ]

    def __init__(self,
                 test=None,
                 reference=None,
                 labels=None,
                 metrics=None,
                 advanced_metrics=None,
                 nan_for_nonexisting=True,
                 spacing_metrics=None):

        self.test = None
        self.reference = None
        self.confusion_matrix = ConfusionMatrix()
        self.labels = None
        self.nan_for_nonexisting = nan_for_nonexisting
        self.result = None

        self.metrics = []
        if metrics is None:
            for m in self.default_metrics:
                self.metrics.append(m)
        else:
            for m in metrics:
                self.metrics.append(m)

        self.advanced_metrics = []
        if advanced_metrics is None:
            for m in self.default_advanced_metrics:
                self.advanced_metrics.append(m)
        else:
            for m in advanced_metrics:
                self.advanced_metrics.append(m)

        self.spacing_metrics = list(self.default_spacing_metrics if spacing_metrics is None
                                    else spacing_metrics)

        self.set_reference(reference)
        self.set_test(test)
        if labels is not None:
            self.set_labels(labels)
        else:
            if test is not None and reference is not None:
                self.construct_labels()

    def set_test(self, test):
        """Set the test segmentation."""

        self.test = test

    def set_reference(self, reference):
        """Set the reference segmentation."""

        self.reference = reference

    def set_labels(self, labels):
        """Set the labels.
        :param labels= may be a dictionary (int->str), a set (of ints), a tuple (of ints) or a list (of ints). Labels
        will only have names if you pass a dictionary"""

        if not isinstance(labels, (dict, set, list, tuple)):
            raise ValueError("Labels must be either list, tuple, set or dict")
        elif isinstance(labels, dict):
            self.labels = collections.OrderedDict(labels)
        elif isinstance(labels, set):
            self.labels = list(labels)
        elif isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            raise TypeError("Can only handle dict, list, tuple, set & numpy array, but input is of type {}".format(type(labels)))

    def construct_labels(self):
        """Construct label set from unique entries in segmentations."""

        if self.test is None and self.reference is None:
            raise ValueError("No test or reference segmentations.")
        elif self.test is None:
            labels = np.unique(self.reference)
        else:
            labels = np.union1d(np.unique(self.test),
                                np.unique(self.reference))
        self.labels = list(map(lambda x: int(x), labels))

    def set_metrics(self, metrics):
        """Set evaluation metrics"""

        if isinstance(metrics, set):
            self.metrics = list(metrics)
        elif isinstance(metrics, (list, tuple, np.ndarray)):
            self.metrics = metrics
        else:
            raise TypeError("Can only handle list, tuple, set & numpy array, but input is of type {}".format(type(metrics)))

    def add_metric(self, metric):

        if metric not in self.metrics:
            self.metrics.append(metric)

    def evaluate(self, test=None, reference=None, advanced=False, **metric_kwargs):
        """Compute metrics for segmentations."""
        if test is not None:
            self.set_test(test)

        if reference is not None:
            self.set_reference(reference)

        if self.test is None or self.reference is None:
            raise ValueError("Need both test and reference segmentations.")

        if self.labels is None:
            self.construct_labels()

        self.metrics.sort()

        # get functions for evaluation
        # somewhat convoluted, but allows users to define additonal metrics
        # on the fly, e.g. inside an IPython console
        _funcs = {m: ALL_METRICS[m] for m in self.metrics + self.advanced_metrics + self.spacing_metrics}
        frames = inspect.getouterframes(inspect.currentframe())
        for metric in self.metrics:
            for f in frames:
                if metric in f[0].f_locals:
                    _funcs[metric] = f[0].f_locals[metric]
                    break
            else:
                if metric in _funcs:
                    continue
                else:
                    raise NotImplementedError(
                        "Metric {} not implemented.".format(metric))

        # get results
        self.result = {}

        eval_metrics = list(self.metrics)
        if advanced:
            eval_metrics += self.advanced_metrics
            if metric_kwargs.get("spacing_mm") is not None:
                eval_metrics += self.spacing_metrics

        if isinstance(self.labels, dict):

            for label, name in self.labels.items():
                self.result[name] = {}
                if not hasattr(label, "__iter__"):
                    self.confusion_matrix.set_test(self.test == label)
                    self.confusion_matrix.set_reference(self.reference == label)
                else:
                    current_test = 0
                    current_reference = 0
                    for l in label:
                        current_test += (self.test == l)
                        current_reference += (self.reference == l)
                    self.confusion_matrix.set_test(current_test)
                    self.confusion_matrix.set_reference(current_reference)
                for metric in eval_metrics:
                    self.result[name][metric] = _funcs[metric](confusion_matrix=self.confusion_matrix,
                                                               nan_for_nonexisting=self.nan_for_nonexisting,
                                                               **metric_kwargs)

        else:

            for i, l in enumerate(self.labels):
                self.result[l] = {}
                self.confusion_matrix.set_test(self.test == l)
                self.confusion_matrix.set_reference(self.reference == l)
                for metric in eval_metrics:
                    self.result[l][metric] = _funcs[metric](confusion_matrix=self.confusion_matrix,
                                                            nan_for_nonexisting=self.nan_for_nonexisting,
                                                            **metric_kwargs)

        return self.result

    def to_dict(self):

        if self.result is None:
            self.evaluate()
        return self.result

    def to_array(self):
        """Return result as numpy array (labels x metrics)."""

        if self.result is None:
            self.evaluate

        result_metrics = sorted(self.result[list(self.result.keys())[0]].keys())

        a = np.zeros((len(self.labels), len(result_metrics)), dtype=np.float32)

        if isinstance(self.labels, dict):
            for i, label in enumerate(self.labels.keys()):
                for j, metric in enumerate(result_metrics):
                    a[i][j] = self.result[self.labels[label]][metric]
        else:
            for i, label in enumerate(self.labels):
                for j, metric in enumerate(result_metrics):
                    a[i][j] = self.result[label][metric]

        return a

    def to_pandas(self):
        """Return result as pandas DataFrame."""

        a = self.to_array()

        if isinstance(self.labels, dict):
            labels = list(self.labels.values())
        else:
            labels = self.labels

        result_metrics = sorted(self.result[list(self.result.keys())[0]].keys())

        return pd.DataFrame(a, index=labels, columns=result_metrics)


class NiftiEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):

        self.test_nifti = None
        self.reference_nifti = None
        super(NiftiEvaluator, self).__init__(*args, **kwargs)

    def set_test(self, test):
        """Set the test segmentation."""

        if test is not None:
            self.test_nifti = sitk.ReadImage(test)
            super(NiftiEvaluator, self).set_test(sitk.GetArrayFromImage(self.test_nifti))
        else:
            self.test_nifti = None
            super(NiftiEvaluator, self).set_test(test)

    def set_reference(self, reference):
        """Set the reference segmentation."""

        if reference is not None:
            self.reference_nifti = sitk.ReadImage(reference)
            super(NiftiEvaluator, self).set_reference(sitk.GetArrayFromImage(self.reference_nifti))
        else:
            self.reference_nifti = None
            super(NiftiEvaluator, self).set_reference(reference)

    def evaluate(self, test=None, reference=None, voxel_spacing=None, **metric_kwargs):

        if voxel_spacing is None:
            voxel_spacing = np.array(self.test_nifti.GetSpacing())[::-1]
            metric_kwargs["voxel_spacing"] = voxel_spacing

        return super(NiftiEvaluator, self).evaluate(test, reference, **metric_kwargs)


def _eval_one(args):
    """Evaluate a single (test, ref) pair \u2014 module-level for Pool pickling."""
    test, ref, evaluator_cls, labels, metric_kwargs = args
    ev = evaluator_cls()
    if labels is not None:
        ev.set_labels(labels)
    ev.set_test(test)
    ev.set_reference(ref)
    if ev.labels is None:
        ev.construct_labels()
    scores = ev.evaluate(**metric_kwargs)
    if isinstance(test, str):
        scores["test"] = test
    if isinstance(ref, str):
        scores["reference"] = ref
    return scores


def aggregate_scores(test_ref_pairs,
                     evaluator=NiftiEvaluator,
                     labels=None,
                     nanmean=True,
                     json_output_file=None,
                     json_name="",
                     json_description="",
                     json_author="Fabian",
                     json_task="",
                     num_workers=1,
                     case_names=None,
                     case_kwargs=None,
                     **metric_kwargs):
    """
    test = predicted image
    :param test_ref_pairs:
    :param evaluator:
    :param labels: must be a dict of int-> str or a list of int
    :param nanmean:
    :param json_output_file:
    :param json_name:
    :param json_description:
    :param json_author:
    :param json_task:
    :param case_names: optional, one name per pair, stored as "case" in its entry of results["all"]
    :param case_kwargs: optional, one dict per pair, added to metric_kwargs for that pair only, e.g.
                        {"spacing_mm": (s0, s1, s2)} for the metrics in mm (stored in the entry too)
    :param metric_kwargs:
    :return:
    """

    evaluator_cls = evaluator if isinstance(evaluator, type) else type(evaluator)

    all_scores = {}
    all_scores["all"] = []
    all_scores["mean"] = {}

    test_ref_pairs = list(test_ref_pairs)
    for name, per_case in (("case_names", case_names), ("case_kwargs", case_kwargs)):
        if per_case is not None and len(per_case) != len(test_ref_pairs):
            raise ValueError(f"{name} has {len(per_case)} entries for {len(test_ref_pairs)} cases")

    # evaluate all cases (parallel when num_workers > 1)
    tasks = [(test, ref, evaluator_cls, labels,
              {**metric_kwargs, **(case_kwargs[i] if case_kwargs is not None else {})})
             for i, (test, ref) in enumerate(test_ref_pairs)]
    if num_workers > 1 and len(tasks) > 1:
        from multiprocessing import Pool
        with Pool(min(num_workers, len(tasks))) as pool:
            case_scores = list(pool.imap(_eval_one, tasks))
    else:
        case_scores = [_eval_one(t) for t in tasks]

    for i, current_scores in enumerate(case_scores):
        meta = {}
        if case_names is not None:
            meta["case"] = case_names[i]
        if case_kwargs is not None and case_kwargs[i].get("spacing_mm") is not None:
            meta["spacing_mm"] = [float(s) for s in case_kwargs[i]["spacing_mm"]]
        current_scores = {**meta, **current_scores}
        all_scores["all"].append(current_scores)

        # append score list for mean (labels only: skip "case", "spacing_mm", "test", "reference")
        for label, score_dict in current_scores.items():
            if not isinstance(score_dict, dict):
                continue
            if label not in all_scores["mean"]:
                all_scores["mean"][label] = {}
            for score, value in score_dict.items():
                if score not in all_scores["mean"][label]:
                    all_scores["mean"][label][score] = []
                all_scores["mean"][label][score].append(value)

    for label in all_scores["mean"]:
        for score in all_scores["mean"][label]:
            if nanmean:
                all_scores["mean"][label][score] = float(np.nanmean(all_scores["mean"][label][score]))
            else:
                all_scores["mean"][label][score] = float(np.mean(all_scores["mean"][label][score]))

    # save to file if desired
    # we create a hopefully unique id by hashing the entire output dictionary
    if json_output_file is not None:
        if type(json_output_file) == str:
            json_output_file = open(json_output_file, "w")
        json_dict = {}
        json_dict["name"] = json_name
        json_dict["description"] = json_description
        timestamp = datetime.today()
        json_dict["timestamp"] = str(timestamp)
        json_dict["task"] = json_task
        json_dict["author"] = json_author
        json_dict["results"] = all_scores
        json_dict["id"] = hashlib.md5(json.dumps(json_dict).encode("utf-8")).hexdigest()[:12]
        json.dump(json_dict, json_output_file, indent=4, separators=(",", ": "))
        json_output_file.close()

    return all_scores


def aggregate_scores_for_experiment(score_file,
                                    labels=None,
                                    metrics=Evaluator.default_metrics,
                                    nanmean=True,
                                    json_output_file=None,
                                    json_name="",
                                    json_description="",
                                    json_author="Fabian",
                                    json_task=""):

    scores = np.load(score_file)
    scores_mean = scores.mean(0)
    if labels is None:
        labels = list(map(str, range(scores.shape[1])))

    results = []
    results_mean = {}
    for i in range(scores.shape[0]):
        results.append({})
        for l, label in enumerate(labels):
            results[-1][label] = {}
            results_mean[label] = {}
            for m, metric in enumerate(metrics):
                results[-1][label][metric] = float(scores[i][l][m])
                results_mean[label][metric] = float(scores_mean[l][m])

    json_dict = {}
    json_dict["name"] = json_name
    json_dict["description"] = json_description
    timestamp = datetime.today()
    json_dict["timestamp"] = str(timestamp)
    json_dict["task"] = json_task
    json_dict["author"] = json_author
    json_dict["results"] = {"all": results, "mean": results_mean}
    json_dict["id"] = hashlib.md5(json.dumps(json_dict).encode("utf-8")).hexdigest()[:12]
    if json_output_file is not None:
        json_output_file = open(json_output_file, "w")
        json.dump(json_dict, json_output_file, indent=4, separators=(",", ": "))
        json_output_file.close()

    return json_dict

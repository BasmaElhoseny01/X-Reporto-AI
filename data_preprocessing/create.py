
import csv
import json
import logging
import os
import re

import imagesize
import spacy
import torch
from tqdm import tqdm

from data_preprocessing.constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE, SUBSTRINGS_TO_REMOVE
import data_preprocessing.section_parser as sp
from paths import path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg, path_full_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# to log certain statistics during dataset creation
txt_file_for_logging = "log_file_dataset_creation.txt"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)


NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES = 10


class DataPreprocessing:
    def __init__(self,train_only = False):
        self.path_chest_imagenome = path_chest_imagenome
        self.path_mimic_cxr = path_mimic_cxr
        self.path_mimic_cxr_jpg = path_mimic_cxr_jpg
        self.path_full_dataset = path_full_dataset

        self.path_to_images_to_avoid = os.path.join(self.path_chest_imagenome, "silver_dataset", "splits", "images_to_avoid.csv")

        self.image_ids_to_avoid = self.get_images_to_avoid()

        if train_only:
            self.csv_files_dict = self.get_train_files()
        else:
            self.csv_files_dict = self.get_train_val_test_csv_files()
    def get_train_files():
        """Return a dict with datasets as keys and paths to the corresponding csv files in the chest-imagenome dataset as values"""
        path_to_splits_folder = os.path.join(path_chest_imagenome, "silver_dataset", "splits")
        return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train"]}
    def get_train_val_files():
        """Return a dict with datasets as keys and paths to the corresponding csv files in the chest-imagenome dataset as values"""
        path_to_splits_folder = os.path.join(path_chest_imagenome, "silver_dataset", "splits")
        return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid"]}
    def get_train_val_test_csv_files(self):
        """Return a dict with datasets as keys and paths to the corresponding csv files in the chest-imagenome dataset as values"""
        path_to_splits_folder = os.path.join(self.path_chest_imagenome, "silver_dataset", "splits")
        return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}
    def get_images_to_avoid(self):
        image_ids_to_avoid = set()

        with open(self.path_to_images_to_avoid) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")

            # print header line
            print(next(csv_reader))

            for row in csv_reader:
                image_id = row[2]
                image_ids_to_avoid.add(image_id)

        return image_ids_to_avoid
    def create_new_csv_files(self):
        if os.path.exists(self.path_full_dataset):
            log.error(f"Full dataset folder already exists at {self.path_full_dataset}.")
            log.error("Delete dataset folder or rename variable path_full_dataset in src/path_datasets_and_weights.py before running script to create new folder!")
            return None

        os.mkdir(self.path_full_dataset)
        for dataset, path_csv_file in self.csv_files_dict.items():
            self.create_new_csv_file(dataset, path_csv_file)

    def create_new_csv_file(self, dataset, path_csv_file):
        log.info(f"Creating new {dataset}.csv file...")

        csv_rows = self.get_rows(dataset, path_csv_file, self.image_ids_to_avoid)

        self.write_rows_in_new_csv_file(dataset, csv_rows)

    def check_coordinate(self,coordinate: int, dim: int) -> int:
        if coordinate < 0:
            coordinate = 0
        elif coordinate > dim:
            coordinate = dim
        return coordinate
    def coordinates_faulty(self,height, width, x1, y1, x2, y2) -> bool:
        area_of_bbox_is_zero = x1 == x2 or y1 == y2
        smaller_than_zero = x2 <= 0 or y2 <= 0
        exceeds_limits = x1 >= width or y1 >= height

        return area_of_bbox_is_zero or smaller_than_zero or exceeds_limits
    def determine_if_abnormal(self,attributes_list: list[list]) -> bool:
        for attributes in attributes_list:
            for attribute in attributes:
                if attribute == "nlp|yes|abnormal":
                    return True
        return False

    def convert_phrases_to_single_string(self,phrases: list[str], sentence_tokenizer) -> str:
        def remove_substrings(phrases):
            def remove_wet_read(phrases):
                index_slices_to_remove = []
                for index in range(len(phrases)):
                    if phrases[index:index + 8] == "WET READ":

                        for curr_index in range(index + 8, len(phrases)):
                            if phrases[curr_index:curr_index + 2] in ["AM", "PM"] or phrases[curr_index:curr_index + 8] == "WET READ":
                                break

                        if phrases[curr_index:curr_index + 2] in ["AM", "PM"]:
                            index_slices_to_remove.append((index, curr_index + 2))

                for indices_tuple in reversed(index_slices_to_remove):
                    start_index, end_index = indices_tuple
                    phrases = phrases[:start_index] + phrases[end_index:]

                return phrases

            phrases = remove_wet_read(phrases)
            phrases = re.sub(SUBSTRINGS_TO_REMOVE, "", phrases, flags=re.DOTALL)

            return phrases

        def remove_whitespace(phrases):
            phrases = " ".join(phrases.split())
            return phrases

        def capitalize_first_word_in_sentence(phrases, sentence_tokenizer):
            sentences = sentence_tokenizer(phrases).sents

            phrases = " ".join(sent.text[0].upper() + sent.text[1:] for sent in sentences)

            return phrases

        def remove_duplicate_sentences(phrases):
            if phrases[-1] == ".":
                phrases = phrases[:-1]

            phrases_dict = {phrase: None for phrase in phrases.split(". ")}

            phrases = ". ".join(phrase for phrase in phrases_dict)

            return phrases + "."

        phrases = " ".join(phrases)

        phrases = remove_substrings(phrases)

        phrases = remove_whitespace(phrases)

        phrases = capitalize_first_word_in_sentence(phrases, sentence_tokenizer)

        phrases = remove_duplicate_sentences(phrases)

        return phrases
    
    def get_attributes_dict(self,image_scene_graph: dict, sentence_tokenizer) -> dict[tuple]:
        attributes_dict = {}
        for attribute in image_scene_graph["attributes"]:
            region_name = attribute["bbox_name"]

            if region_name not in ANATOMICAL_REGIONS:
                continue

            phrases = self.convert_phrases_to_single_string(attribute["phrases"], sentence_tokenizer)
            is_abnormal = self.determine_if_abnormal(attribute["attributes"])

            attributes_dict[region_name] = (phrases, is_abnormal)

        return attributes_dict
    
    def get_reference_report(self,subject_id: str, study_id: str, missing_reports: list[str]):
        custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

        if f"s{study_id}" in custom_section_names or f"s{study_id}" in custom_indices:
            return -1

        path_to_report = os.path.join(path_mimic_cxr, "files", f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")

        if not os.path.exists(path_to_report):
            shortened_path_to_report = os.path.join(f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")
            missing_reports.append(shortened_path_to_report)
            return -1

        with open(path_to_report) as f:
            report = "".join(f.readlines())

        sections, section_names, _ = sp.section_text(report)

        if "findings" in section_names:
            findings_index = len(section_names) - section_names[-1::-1].index("findings") - 1
            report = sections[findings_index]
        else:
            return -1

        report = " ".join(report.split())

        return report
    
    def get_total_num_rows(self,path_csv_file: str) -> int:
        with open(path_csv_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")

            next(csv_reader)

            return sum(1 for row in csv_reader)
    
    def get_rows(self, dataset: str, path_csv_file: str, image_ids_to_avoid: set) -> list[list]:
        
        csv_rows = []
        num_rows_created = 0

        # we split the test set into 1 that contains all images that have bbox coordinates for all 29 regions
        # (which will be around 31271 images in total, or around 95% of all test set images),
        # and 1 that contains the rest of the images (around 1440 images) that do not have bbox coordinates for all 29 regions
        # this is done such that we can efficiently evaluate the first test set (since vectorized code can be written for it),
        # and evaluate the second test set a bit more inefficiently (using for loops) afterwards
        if dataset == "test":
            csv_rows_less_than_29_regions = []

        total_num_rows = self.get_total_num_rows(path_csv_file)

        # used in function convert_phrases_to_single_string
        sentence_tokenizer = spacy.load("en_core_web_trf")

        # stats will be logged in path_to_log_file
        num_images_ignored_or_avoided = 0
        num_faulty_bboxes = 0
        num_images_without_29_regions = 0
        missing_images = []
        missing_reports = []

        with open(path_csv_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")

            # skip the first line (i.e. the header line)
            next(csv_reader)

            # iterate over all rows of the given csv file (i.e. over all images), if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is not set to a specific value
            for row in tqdm(csv_reader, total=total_num_rows):
                subject_id = row[1]
                study_id = row[2]
                image_id = row[3]

                # all images in set IMAGE_IDS_TO_IGNORE seem to be failed x-rays and thus have to be discarded
                # (they also don't have corresponding scene graph json files anyway)
                # all images in set image_ids_to_avoid are image IDs for images in the gold standard dataset,
                # which should all be excluded from model training and validation
                if image_id in IMAGE_IDS_TO_IGNORE or image_id in image_ids_to_avoid:
                    num_images_ignored_or_avoided += 1
                    continue

                # image_file_path is of the form "files/p10/p10000980/s50985099/6ad03ed1-97ee17ee-9cf8b320-f7011003-cd93b42d.dcm"
                # i.e. f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{image_id}.dcm"
                # since we have the MIMIC-CXR-JPG dataset, we need to replace .dcm by .jpg
                image_file_path = row[4].replace(".dcm", ".jpg")
                mimic_image_file_path = os.path.join(self.path_mimic_cxr_jpg, image_file_path)

                if not os.path.exists(mimic_image_file_path):
                    missing_images.append(mimic_image_file_path)
                    continue

                # for the validation and test sets, we only want to include images that have corresponding reference reports with "findings" sections
                if dataset in ["valid", "test"]:
                    reference_report = self.get_reference_report(subject_id, study_id, missing_reports)

                    # skip images that don't have a reference report with "findings" section
                    if reference_report == -1:
                        continue

                    # the reference_report will be appended to new_image_row (declared further below, which contains all information about a single image)
                    # just before new_image_row itself is appended to csv_rows (because the image could still be rejected from the validation set,
                    # if it doesn't have 29 bbox coordinates)

                chest_imagenome_scene_graph_file_path = os.path.join(self.path_chest_imagenome, "silver_dataset", "scene_graph", image_id) + "_SceneGraph.json"

                with open(chest_imagenome_scene_graph_file_path) as fp:
                    image_scene_graph = json.load(fp)

                anatomical_region_attributes = self.get_attributes_dict(image_scene_graph, sentence_tokenizer)

                # new_image_row will store all information about 1 image as a row in the csv file
                new_image_row = [subject_id, study_id, image_id, mimic_image_file_path]
                bbox_coordinates = []
                bbox_labels = []
                bbox_phrases = []
                bbox_phrase_exist_vars = []
                bbox_is_abnormal_vars = []

                width, height = imagesize.get(mimic_image_file_path)

                # counter to see if given image contains bbox coordinates for all 29 regions
                # if image does not bbox coordinates for 29 regions, it's still added to the train and test dataset,
                # but not the val dataset (see reasoning in the module docstring on top of this file)
                num_regions = 0

                region_to_bbox_coordinates_dict = {}
                # objects is a list of obj_dicts where each dict contains the bbox coordinates for a single region
                for obj_dict in image_scene_graph["objects"]:
                    region_name = obj_dict["bbox_name"]
                    x1 = obj_dict["original_x1"]
                    y1 = obj_dict["original_y1"]
                    x2 = obj_dict["original_x2"]
                    y2 = obj_dict["original_y2"]

                    region_to_bbox_coordinates_dict[region_name] = [x1, y1, x2, y2]

                for anatomical_region in ANATOMICAL_REGIONS:
                    bbox_coords = region_to_bbox_coordinates_dict.get(anatomical_region, None)

                    # if there are no bbox coordinates or they are faulty, then don't add them to image information
                    if bbox_coords is None or self.coordinates_faulty(height, width, *bbox_coords):
                        num_faulty_bboxes += 1
                    else:
                        x1, y1, x2, y2 = bbox_coords

                        # it is possible that the bbox is only partially inside the image height and width (if e.g. x1 < 0, whereas x2 > 0)
                        # to prevent these cases from raising an exception, we set the coordinates to 0 if coordinate < 0, set to width if x-coordinate > width
                        # and set to height if y-coordinate > height
                        x1 = self.check_coordinate(x1, width)
                        y1 = self.check_coordinate(y1, height)
                        x2 = self.check_coordinate(x2, width)
                        y2 = self.check_coordinate(y2, height)

                        bbox_coords = [x1, y1, x2, y2]

                        # since background has class label 0 for object detection, shift the remaining class labels by 1
                        class_label = ANATOMICAL_REGIONS[anatomical_region] + 1

                        bbox_coordinates.append(bbox_coords)
                        bbox_labels.append(class_label)

                        num_regions += 1

                    # get bbox_phrase (describing the region inside bbox) and bbox_is_abnormal boolean variable (indicating if region inside bbox is abnormal)
                    # if there is no phrase, then the region inside bbox is normal and thus has "" for bbox_phrase (empty phrase) and False for bbox_is_abnormal
                    bbox_phrase, bbox_is_abnormal = anatomical_region_attributes.get(anatomical_region, ("", False))
                    bbox_phrase_exist = True if bbox_phrase != "" else False

                    bbox_phrases.append(bbox_phrase)
                    bbox_phrase_exist_vars.append(bbox_phrase_exist)
                    bbox_is_abnormal_vars.append(bbox_is_abnormal)

                new_image_row.extend([bbox_coordinates, bbox_labels, bbox_phrases, bbox_phrase_exist_vars, bbox_is_abnormal_vars])

                # for train set, add all images (even those that don't have bbox information for all 29 regions)
                # for val set, only add images that have bbox information for all 29 regions
                # for test set, distinguish between test set 1 that contains test set images that have bbox information for all 29 regions
                # (around 95% of all test set images)
                if dataset == "train" or (dataset in ["valid", "test"] and num_regions == 29):
                    if dataset in ["valid", "test"]:
                        new_image_row.append(reference_report)

                    csv_rows.append(new_image_row)

                    num_rows_created += 1
                # test set 2 will contain the remaining 5% of test set images, which do not have bbox information for all 29 regions
                elif dataset == "test" and num_regions != 29:
                    new_image_row.append(reference_report)
                    csv_rows_less_than_29_regions.append(new_image_row)

                if num_regions != 29:
                    num_images_without_29_regions += 1

                # break out of loop if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is specified
                if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES and num_rows_created >= NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES:
                    break

        self.write_stats_to_log_file(dataset, num_images_ignored_or_avoided, missing_images, missing_reports, num_faulty_bboxes, num_images_without_29_regions)

        if dataset == "test":
            return csv_rows, csv_rows_less_than_29_regions
        else:
            return csv_rows
        
    def write_stats_to_log_file(self,dataset: str, num_images_ignored_or_avoided: int, missing_images: list[str], missing_reports: list[str], num_faulty_bboxes: int, num_images_without_29_regions: int):
        with open(txt_file_for_logging, "a") as f:
            f.write(f"{dataset}:\n")
            f.write(f"\tnum_images_ignored_or_avoided: {num_images_ignored_or_avoided}\n")

            f.write(f"\tnum_missing_images: {len(missing_images)}\n")
            for missing_img in missing_images:
                f.write(f"\t\tmissing_img: {missing_img}\n")

            f.write(f"\tnum_missing_reports: {len(missing_reports)}\n")
            for missing_rep in missing_reports:
                f.write(f"\t\tmissing_rep: {missing_rep}\n")

            f.write(f"\tnum_faulty_bboxes: {num_faulty_bboxes}\n")
            f.write(f"\tnum_images_without_29_regions: {num_images_without_29_regions}\n\n")

    def write_rows_in_new_csv_file(self,dataset: str, csv_rows: list[list]) -> None:
        log.info(f"Writing rows into new {dataset}.csv file...")

        if dataset == "test":
            csv_rows, csv_rows_less_than_29_regions = csv_rows

        new_csv_file_path = os.path.join(path_full_dataset, dataset)
        new_csv_file_path += ".csv" if not NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES else f"-{NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES}.csv"

        header = ["subject_id", "study_id", "image_id", "mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_phrases", "bbox_phrase_exists", "bbox_is_abnormal"]
        if dataset in ["valid", "test"]:
            header.append("reference_report")

        with open(new_csv_file_path, "w") as fp:
            csv_writer = csv.writer(fp)

            csv_writer.writerow(header)
            csv_writer.writerows(csv_rows)

        if dataset == "test":
            new_csv_file_path = new_csv_file_path.replace(".csv", "-2.csv")

            with open(new_csv_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows(csv_rows_less_than_29_regions)


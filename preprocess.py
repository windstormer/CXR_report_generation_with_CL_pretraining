import numpy as np
import cv2
import xml.etree.ElementTree as ET
import argparse
import os
# import pydicom
# from pydicom.pixel_data_handlers.util import apply_voi_lut
import glob
from tqdm import tqdm
import SimpleITK as sitk

def read_xray(path, patch_size, voi_lut = True, fix_monochrome = True):
    # dicom = pydicom.read_file(path)
    # print(dicom)
    # if voi_lut:
    #     data = apply_voi_lut(dicom.pixel_array, dicom)
    # else:
    #     data = dicom.pixel_array
               
    # if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
    #     data = np.amax(data) - data
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    sitkdata = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(sitkdata)[0]
    
    # print(data.shape, data.max(), data.min())

    data = cv2.resize(data, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def load_kaggle_image(path, patch_size):
    if not os.path.exists(os.path.join(path, "data.npy")):
        image_path_list = glob.glob(os.path.join(path, "*.dicom"))
        image_list = []
        for img_path in tqdm(image_path_list):
            img = read_xray(img_path, patch_size)
            image_list.append(img)
        image_list = np.asarray(image_list)
        np.save(os.path.join(path, "data.npy"), image_list)
    else:
        image_list = np.load(os.path.join(path, "data.npy"))
    return image_list


def load_image(path, image_name_list, img_size, patch_size):
    image_list = []
    for file in image_name_list:
        # print(file)
        image = cv2.imread(os.path.join(path, "{}.png".format(file)), cv2.IMREAD_GRAYSCALE)
        h_pad = img_size[0] - image.shape[0]
        w_pad = img_size[1] - image.shape[1]
        if h_pad < 0:
            h_pad = 0
        if w_pad < 0:
            w_pad = 0
        image = cv2.copyMakeBorder(image, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image = image[(image.shape[0]-img_size[0])//2:(image.shape[0]+img_size[0])//2, (image.shape[1]-img_size[1])//2:(image.shape[1]+img_size[1])//2]
        image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        image_list.append(image)
    image_list = np.asarray(image_list)
    return image_list
    
def load_image_seg(path, image_name_list, img_size, patch_size):
    image_list = []
    seg_list = []
    for file in image_name_list:
        # print(file)
        image = cv2.imread(os.path.join(path, "{}.png".format(file)), cv2.IMREAD_GRAYSCALE)
        h_pad = img_size[0] - image.shape[0]
        w_pad = img_size[1] - image.shape[1]
        if h_pad < 0:
            h_pad = 0
        if w_pad < 0:
            w_pad = 0
        image = cv2.copyMakeBorder(image, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image = image[(image.shape[0]-img_size[0])//2:(image.shape[0]+img_size[0])//2, (image.shape[1]-img_size[1])//2:(image.shape[1]+img_size[1])//2]
        image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        image_list.append(image)
        seg = cv2.imread(os.path.join("/hdd/vincent18/IUXR_Lung_Seg", "{}.png".format(file)), cv2.IMREAD_GRAYSCALE)
        seg_list.append(seg)
    image_list = np.asarray(image_list)
    seg_list = np.asarray(seg_list)
    return image_list, seg_list

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def load_report(path):
    file_list = os.listdir(path)
    file_list.sort()
    image_name_list = []
    report_list = []
    for file in file_list:
        tree = ET.parse(os.path.join(path, file))
        root = tree.getroot()
        # print(root.find('IUXRId').get('id'))
        # print(root.find('MedlineCitation').find('Article').find('Abstract')[2].get('Label'))
        # print(root.find('MedlineCitation').find('Article').find('Abstract')[2].text)
        # print(root.findall('parentImage')[0].get('id'))
        if len(root.findall('parentImage')) != 0:
            if root.find('MedlineCitation').find('Article').find('Abstract')[2].text == None and root.find('MedlineCitation').find('Article').find('Abstract')[3].text == None:
                continue
            for image in root.findall('parentImage'):
                image_name_list.append(image.get('id'))
                if root.find('MedlineCitation').find('Article').find('Abstract')[2].text == None:
                    paragraph = root.find('MedlineCitation').find('Article').find('Abstract')[3].text
                elif root.find('MedlineCitation').find('Article').find('Abstract')[3].text == None:
                    paragraph = root.find('MedlineCitation').find('Article').find('Abstract')[2].text
                else:
                    paragraph = root.find('MedlineCitation').find('Article').find('Abstract')[2].text + " " + root.find('MedlineCitation').find('Article').find('Abstract')[3].text
                report_list.append(paragraph)
    return report_list, image_name_list

def report_check(report_list, image_name_list):
    new_report_list = []
    new_image_name_list = []
    for idx, report in enumerate(report_list):
        new_report = report.lower().replace("..", ".")
        new_report = new_report.replace("'", "")
        new_sentences = []
        # print(nltk.tokenize.sent_tokenize(new_report))
        for sentence in nltk.tokenize.sent_tokenize(new_report):
            new_sentence = sentence.replace("/", " / ")
            if "xxxx" not in sentence and not hasNumbers(sentence):
                new_sentences.append(sentence)
        new_report = " ".join(new_sentences)
        if len(new_report) > 0:
            new_report_list.append(new_report)
            new_image_name_list.append(image_name_list[idx])
    
    return new_report_list, new_image_name_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path",
                        type=str, 
                        default="../IUXR_png/",
                        help="path of dataset")

    parser.add_argument("--report_path",
                        type=str, 
                        default="../IUXR_report/",
                        help="path of report")
    parser.add_argument("--img_size",
                        type=int,
                        nargs=2,
                        default=[512, 512],
                        help="image size [x, y]")
    args = parser.parse_args()
    
    report, image_name_list = load_report(args.report_path)
    image = load_image(args.dataset_path, image_name_list, args.img_size)
    n_case = len(report)

    for i in range(n_case):
        print(image_name_list[i], image[i].shape, report[i])
    


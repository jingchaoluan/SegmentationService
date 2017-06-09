# -*- coding: utf-8 -*-
from django.conf import settings
import sys, os, os.path, subprocess, shutil

# Get the directory of ocropy script
ocropyDir = settings.BASE_DIR + "/ocropy"

# Get the directory which stores all input and output files
dataDir = settings.MEDIA_ROOT

# Execute ocr binarization script: binarize the original image
# Parameter: the original image name
# Return: a list conaining two files: {iamgename}.bin.png, {iamgename}.nrm.png
def segmentation_exec(imagename):

    # Prepare path for OCR service
    inputPath = dataDir+"/"+imagename
    image_base = imagename.split(".")[0]
    outputDir = dataDir+"/"+image_base
    
    # Call segmentation script
    segmentation_cmd = ocropyDir + "/ocropus-gpageseg -n --minscale 1.0 " + inputPath
    r_segmentation = subprocess.call([segmentation_cmd], shell=True)
    if r_segmentation != 0:
        sys.exit("Error: Page segmentation process failed")

    if os.path.exists(outputDir):
	return outputDir
    else:
	sys.exit("Error: the output folder does not exist.")

# Delete all files related to this service time, including inputs and outputs
def del_service_files(dataDir):

    for the_file in os.listdir(dataDir):
	file_path = os.path.join(dataDir, the_file)
	try:
	    if os.path.isfile(file_path):
		os.unlink(file_path)
	    elif os.path.isdir(file_path):
		shutil.rmtree(file_path)
	except Exception as e:
	    print(e)


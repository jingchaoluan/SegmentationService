# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.conf import settings
from django.shortcuts import render
from wsgiref.util import FileWrapper
from .serializers import ParameterSerializer
from .segmentation import segmentation_exec
from .extrafunc import del_service_files
import sys, os, os.path, zipfile, StringIO


# Get the directory which stores all input and output files
dataDir = settings.MEDIA_ROOT

def index(request):
    return render(request, 'index.html')

@csrf_exempt
@api_view(['GET', 'POST'])
def segmentationView(request, format=None):
    if request.data.get('image') is None:
        return HttpResponse("Please upload at least one binarized image.")

    # Receive specified parameters values
    # Receive parameters with model and serializer
    data_dict = request.data.dict()
    del data_dict['image']   # Image will be processed seperately for receiving multiple images
    # Serialize the specified parameters, only containing the specified parameters
    # If we want to generate the parameters object with all of the default paremeters, call parameters.save()
    paras_serializer = ParameterSerializer(data=data_dict)
    if paras_serializer.is_valid():
        pass # needn't parameters.save(), since we needn't to store these parameters in DB

    # Receive and store uploaded image(s)
    # One or multiple images/values in one field
    imagenames = []
    imagepaths = []
    images = request.data.getlist('image')
    for image in images:
        image_str = str(image)
        imagenames.append(image_str)
        imagepaths.append(dataDir+"/"+image_str)
        default_storage.save(dataDir+"/"+image_str, image)
    
    # Call segmentation function
    output_dirs = segmentation_exec(imagepaths, paras_serializer.data)

    # Put all output files path in a list
    outputfiles_path = []
    for output_dir in output_dirs:
        for output_file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, output_file)
            outputfiles_path.append(file_path)

    # return the multiple files in zip type
    # Folder name in ZIP archive which contains the above files
    zip_dir = "output_segmentation"
    zip_filename = "%s.zip" % zip_dir
    # Open StringIO to grab in-memory ZIP contents
    strio = StringIO.StringIO()
    # The zip compressor
    zf = zipfile.ZipFile(strio, "w")

    for fpath in outputfiles_path:
        # Caculate path for file in zip
        fdir, fname = os.path.split(fpath)
        subdir = os.path.basename(os.path.normpath(fdir))
        zip_path = os.path.join(zip_dir+"/"+subdir, fname)
        # Add file, at correct path
        zf.write(fpath, zip_path)

    zf.close()
    # Grab ZIP file from in-memory, make response with correct MIME-type
    response = HttpResponse(strio.getvalue(), content_type="application/x-zip-compressed")
    # And correct content-disposition
    response["Content-Disposition"] = 'attachment; filename=%s' % zip_filename
    
    # Delete all files related to this service time
    del_service_files(dataDir)

    return response

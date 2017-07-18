#!/usr/bin/env python

# TODO:
# ! add option for padding
# - fix occasionally missing page numbers
# - treat large h-whitespace as separator
# - handle overlapping candidates
# - use cc distance statistics instead of character scale
# - page frame detection
# - read and use text image segmentation mask
# - pick up stragglers
# ? laplacian as well

from __future__ import print_function

from pylab import *
import glob,os,os.path
import traceback
from scipy.ndimage import measurements
from scipy.misc import imsave
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter
from multiprocessing import Pool
import ocrolib
from ocrolib import psegutils,morph,sl
from ocrolib.exceptions import OcropusException
from ocrolib.toplevel import *


# The default parameters values
# Users can custom the first 11 parameters
args_default = {
    # limits
    'maxlines':300,  # maximum # lines permitted

    # scale parameters
    'scale':0.0,     # the basic scale of the document (roughly, xheight) 0=automatic
    'hscale':1.0,    # non-standard scaling of horizontal parameters
    'vscale':1.0,    # non-standard scaling of vertical parameters

    # line parameters
    'threshold':0.2, # baseline threshold
    'noise':8,       # noise threshold for removing small components from lines
    'usegause':False,# use gaussian instead of uniform

    # column separator parameters
    'maxseps':0,     # maximum # black column separators
    'sepwiden':10,   # widen black separators (to account for warping)
    'maxcolseps':3,  # maximum # whitespace column separators
    'csminheight':10.0,# minimum column height (units=scale)

    'parallel':0,    # number of parallel CPUs to use

    ### The following parameters needn't be overwritten by users
    # limits
    'minscale':1.0,  # minimum scale permitted
    # output parameters
    'pad':3,         # adding for extracted lines
    'expand':3,      # expand mask for grayscale extraction
    # other parameters
    'nocheck':True,  # disable error checking on inputs
    'quiet':False,   # be less verbose, usally use with parallel together
    'debug':False
}

args = {}

# The entry of segmentation service
# Return the directories, each directory related to a input image and stored the segmented line images  
def segmentation_exec(images, parameters):
    # Update parameters values customed by user
    # Each time update the args with the default args dictionary, avoid the effect of the previous update
    global args
    args = args_default.copy()
    args.update(parameters)
    print("==========")
    print(args)

    if len(images)<1:
        sys.exit(0)

    # Unicode to str
    for i, image in enumerate(images):
        images[i] = str(image)

    if args['parallel']>1:
        args['quiet'] = True

    output_dirs = []
    if args['parallel']<2:
        for i,imagepath in enumerate(images):
            if args['parallel']==1: print_info(imagepath)
            output_dir = safe_process((imagepath,i+1))
            output_dirs.append(output_dir)
    else:
        pool = Pool(processes=args['parallel'])
        jobs = []
        for i,imagepath in enumerate(images): jobs += [(imagepath,i+1)]
        result = pool.map(process,jobs)
    
    return output_dirs


def norm_max(v):
    return v/amax(v)


def check_page(image):
    if len(image.shape)==3: return "input image is color image %s"%(image.shape,)
    if mean(image)<median(image): return "image may be inverted"
    h,w = image.shape
    if h<600: return "image not tall enough for a page image %s"%(image.shape,)
    if h>10000: return "image too tall for a page image %s"%(image.shape,)
    if w<600: return "image too narrow for a page image %s"%(image.shape,)
    if w>10000: return "line too wide for a page image %s"%(image.shape,)
    slots = int(w*h*1.0/(30*30))
    _,ncomps = measurements.label(image>mean(image))
    if ncomps<10: return "too few connected components for a page image (got %d)"%(ncomps,)
    if ncomps>slots: return "too many connnected components for a page image (%d > %d)"%(ncomps,slots)
    return None


def print_info(*objs):
    print("INFO: ", *objs, file=sys.stdout)

def print_error(*objs):
    print("ERROR: ", *objs, file=sys.stderr)

def B(a):
    if a.dtype==dtype('B'): return a
    return array(a,'B')

def DSAVE(title,image):
    if not args['debug']: return
    if type(image)==list:
        assert len(image)==3
        image = transpose(array(image),[1,2,0])
    fname = "_"+title+".png"
    print_info("debug " + fname)
    imsave(fname,image)



################################################################
### Column finding.
###
### This attempts to find column separators, either as extended
### vertical black lines or extended vertical whitespace.
### It will work fairly well in simple cases, but for unusual
### documents, you need to tune the parameters.
################################################################

def compute_separators_morph(binary,scale):
    """Finds vertical black lines corresponding to column separators."""
    d0 = int(max(5,scale/4))
    d1 = int(max(5,scale))+args['sepwiden']
    thick = morph.r_dilation(binary,(d0,d1))
    vert = morph.rb_opening(thick,(10*scale,1))
    vert = morph.r_erosion(vert,(d0//2,args['sepwiden']))
    vert = morph.select_regions(vert,sl.dim1,min=3,nbest=2*args['maxseps'])
    vert = morph.select_regions(vert,sl.dim0,min=20*scale,nbest=args['maxseps'])
    return vert

def compute_colseps_mconv(binary,scale=1.0):
    """Find column separators using a combination of morphological
    operations and convolution."""
    h,w = binary.shape
    smoothed = gaussian_filter(1.0*binary,(scale,scale*0.5))
    smoothed = uniform_filter(smoothed,(5.0*scale,1))
    thresh = (smoothed<amax(smoothed)*0.1)
    DSAVE("1thresh",thresh)
    blocks = morph.rb_closing(binary,(int(4*scale),int(4*scale)))
    DSAVE("2blocks",blocks)
    seps = minimum(blocks,thresh)
    seps = morph.select_regions(seps,sl.dim0,min=args['csminheight']*scale,nbest=args['maxcolseps'])
    DSAVE("3seps",seps)
    blocks = morph.r_dilation(blocks,(5,5))
    DSAVE("4blocks",blocks)
    seps = maximum(seps,1-blocks)
    DSAVE("5combo",seps)
    return seps

def compute_colseps_conv(binary,scale=1.0):
    """Find column separators by convoluation and
    thresholding."""
    h,w = binary.shape
    # find vertical whitespace by thresholding
    smoothed = gaussian_filter(1.0*binary,(scale,scale*0.5))
    smoothed = uniform_filter(smoothed,(5.0*scale,1))
    thresh = (smoothed<amax(smoothed)*0.1)
    DSAVE("1thresh",thresh)
    # find column edges by filtering
    grad = gaussian_filter(1.0*binary,(scale,scale*0.5),order=(0,1))
    grad = uniform_filter(grad,(10.0*scale,1))
    # grad = abs(grad) # use this for finding both edges
    grad = (grad>0.5*amax(grad))
    DSAVE("2grad",grad)
    # combine edges and whitespace
    seps = minimum(thresh,maximum_filter(grad,(int(scale),int(5*scale))))
    seps = maximum_filter(seps,(int(2*scale),1))
    DSAVE("3seps",seps)
    # select only the biggest column separators
    seps = morph.select_regions(seps,sl.dim0,min=args['csminheight']*scale,nbest=args['maxcolseps'])
    DSAVE("4seps",seps)
    return seps

def compute_colseps(binary,scale):
    """Computes column separators either from vertical black lines or whitespace."""
    print_info("considering at most %g whitespace column separators" % args['maxcolseps'])
    colseps = compute_colseps_conv(binary,scale)
    DSAVE("colwsseps",0.7*colseps+0.3*binary)
    
    print_info("considering at most %g black column separators" % args['maxseps'])
    seps = compute_separators_morph(binary,scale)
    DSAVE("colseps",0.7*seps+0.3*binary)
    colseps = maximum(colseps,seps)
    binary = minimum(binary,1-seps)
    return colseps,binary



################################################################
### Text Line Finding.
###
### This identifies the tops and bottoms of text lines by
### computing gradients and performing some adaptive thresholding.
### Those components are then used as seeds for the text lines.
################################################################

def compute_gradmaps(binary,scale):
    # use gradient filtering to find baselines
    boxmap = psegutils.compute_boxmap(binary,scale)
    cleaned = boxmap*binary
    DSAVE("cleaned",cleaned)
    if args['usegause']:
        # this uses Gaussians
        grad = gaussian_filter(1.0*cleaned,(args['vscale']*0.3*scale,
                                            args['hscale']*6*scale),order=(1,0))
    else:
        # this uses non-Gaussian oriented filters
        grad = gaussian_filter(1.0*cleaned,(max(4,args['vscale']*0.3*scale),
                                            args['hscale']*scale),order=(1,0))
        grad = uniform_filter(grad,(args['vscale'],args['hscale']*6*scale))
    bottom = ocrolib.norm_max((grad<0)*(-grad))
    top = ocrolib.norm_max((grad>0)*grad)
    return bottom,top,boxmap

def compute_line_seeds(binary,bottom,top,colseps,scale):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = args['threshold']
    vrange = int(args['vscale']*scale)
    bmarked = maximum_filter(bottom==maximum_filter(bottom,(vrange,0)),(2,2))
    bmarked = bmarked*(bottom>t*amax(bottom)*t)*(1-colseps)
    tmarked = maximum_filter(top==maximum_filter(top,(vrange,0)),(2,2))
    tmarked = tmarked*(top>t*amax(top)*t/2)*(1-colseps)
    tmarked = maximum_filter(tmarked,(1,20))
    seeds = zeros(binary.shape,'i')
    delta = max(3,int(scale/2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y,1) for y in find(bmarked[:,x])]+[(y,0) for y in find(tmarked[:,x])])[::-1]
        transitions += [(0,0)]
        for l in range(len(transitions)-1):
            y0,s0 = transitions[l]
            if s0==0: continue
            seeds[y0-delta:y0,x] = 1
            y1,s1 = transitions[l+1]
            if s1==0 and (y0-y1)<5*scale: seeds[y1:y0,x] = 1
    seeds = maximum_filter(seeds,(1,int(1+scale)))
    seeds = seeds*(1-colseps)
    DSAVE("lineseeds",[seeds,0.3*tmarked+0.7*bmarked,binary])
    seeds,_ = morph.label(seeds)
    return seeds



################################################################
### The complete line segmentation process.
################################################################

def remove_hlines(binary,scale,maxsize=10):
    labels,_ = morph.label(binary)
    objects = morph.find_objects(labels)
    for i,b in enumerate(objects):
        if sl.width(b)>maxsize*scale:
            labels[b][labels[b]==i+1] = 0
    return array(labels!=0,'B')

def compute_segmentation(binary,scale):
    """Given a binary image, compute a complete segmentation into
    lines, computing both columns and text lines."""
    binary = array(binary,'B')

    # start by removing horizontal black lines, which only
    # interfere with the rest of the page segmentation
    binary = remove_hlines(binary,scale)

    # do the column finding
    if not args['quiet']: print_info("computing column separators")
    colseps,binary = compute_colseps(binary,scale)

    # now compute the text line seeds
    if not args['quiet']: print_info("computing lines")
    bottom,top,boxmap = compute_gradmaps(binary,scale)
    seeds = compute_line_seeds(binary,bottom,top,colseps,scale)
    DSAVE("seeds",[bottom,top,boxmap])

    # spread the text line seeds to all the remaining
    # components
    if not args['quiet']: print_info("propagating labels")
    llabels = morph.propagate_labels(boxmap,seeds,conflict=0)
    if not args['quiet']: print_info("spreading labels")
    spread = morph.spread_labels(seeds,maxdist=scale)
    llabels = where(llabels>0,llabels,spread*binary)
    segmentation = llabels*binary
    return segmentation



################################################################
### Processing each file.
################################################################

def process(job):
    imagepath,i = job
    global base
    base,_ = ocrolib.allsplitext(imagepath)
    outputdir = base
    imagename_base = os.path.basename(os.path.normpath(base))

    try:
        binary = ocrolib.read_image_binary(imagepath)
    except IOError:
        if ocrolib.trace: traceback.print_exc()
        print_error("cannot open either %s.bin.png or %s" % (base, imagepath))
        return

    checktype(binary,ABINARY2)

    if not args['nocheck']:
        check = check_page(amax(binary)-binary)
        if check is not None:
            print_error("%s SKIPPED %s (use -n to disable this check)" % (imagepath, check))
            return

    binary = 1-binary # invert

    if args['scale']==0:
        scale = psegutils.estimate_scale(binary)
    else:
        scale = args['scale']
    print_info("scale %f" % (scale))
    if isnan(scale) or scale>1000.0:
        print_error("%s: bad scale (%g); skipping\n" % (imagepath, scale))
        return
    if scale<args['minscale']:
        print_error("%s: scale (%g) less than --minscale; skipping\n" % (imagepath, scale))
        return

    # find columns and text lines

    if not args['quiet']: print_info("computing segmentation")
    segmentation = compute_segmentation(binary,scale)
    if amax(segmentation)>args['maxlines']:
        print_error("%s: too many lines %g" % (imagepath, amax(segmentation)))
        return
    if not args['quiet']: print_info("number of lines %g" % amax(segmentation))

    # compute the reading order

    if not args['quiet']: print_info("finding reading order")
    lines = psegutils.compute_lines(segmentation,scale)
    order = psegutils.reading_order([l.bounds for l in lines])
    lsort = psegutils.topsort(order)

    # renumber the labels so that they conform to the specs

    nlabels = amax(segmentation)+1
    renumber = zeros(nlabels,'i')
    for i,v in enumerate(lsort): renumber[lines[v].label] = 0x010000+(i+1)
    segmentation = renumber[segmentation]

    # finally, output everything
    if not args['quiet']: print_info("writing lines")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    lines = [lines[i] for i in lsort]
    ocrolib.write_page_segmentation("%s.pseg.png"%outputdir,segmentation)
    cleaned = ocrolib.remove_noise(binary,args['noise'])
    for i,l in enumerate(lines):
        binline = psegutils.extract_masked(1-cleaned,l,pad=args['pad'],expand=args['expand'])
        ocrolib.write_image_binary("%s/%s_01%04x.bin.png"%(outputdir,imagename_base,i+1),binline)
    print_info("%6d  %s %4.1f %d" % (i, imagepath,  scale,  len(lines)))
    return outputdir


def safe_process(job):
    imagepath,i = job
    outputdir = None
    try:
        outputdir = process(job)
    except OcropusException as e:
        if e.trace:
            traceback.print_exc()
        else:
            print_info(imagepath+":"+e)
    except Exception as e:
        traceback.print_exc()
    return outputdir

__copyright__  = "Copyright (c) 2022-2025, Intelligent Imaging Innovations, Inc. All rights reserved.  All rights reserved."
__license__  = "This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree."

from SBReadFile import *
from matplotlib import pyplot as plt
import numpy as np
import sys, getopt

def usage():
    print('usage: python test_SBReadFile.py -i <inputfile> [options]')
    print('options:')
    print('  -R                Show RGB composite (uses channels 0,1,2 by default)')
    print('  -m r,g,b          Channel indices for RGB (e.g., -m 2,0,1)')
    print('  -n capture        Capture index (default 0)')
    print('  -t timepoint      Timepoint index (default 0)')
    print('  -p plane          Z-plane index (default middle)')
    print('  -P position       Position index (default 0)')


def main(argv):
    theSBFileReader = SBReadFile()

    if len(sys.argv) < 3:
        usage()
        sys.exit(2)

    theFileName = ''
    rgb_mode = False
    rgb_map = (0, 1, 2)
    theCapture = 0
    theTimepoint = 0
    thePlane = None
    thePosition = 0
    try:
        opts, args = getopt.getopt(argv, "hRi:m:n:t:p:P:", ["ifile="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '-R':
            rgb_mode = True
        elif opt == '-m':
            parts = arg.split(',')
            if len(parts) != 3:
                print('Invalid -m mapping, expected r,g,b')
                sys.exit(2)
            rgb_map = tuple(int(x) for x in parts)
        elif opt == '-n':
            theCapture = int(arg)
        elif opt == '-t':
            theTimepoint = int(arg)
        elif opt == '-p':
            thePlane = int(arg)
        elif opt == '-P':
            thePosition = int(arg)
        elif opt in ("-i", "--ifile"):
            theFileName = arg
    print('Input file is ', theFileName)

    theRes = theSBFileReader.Open(theFileName)
    if not theRes:
        sys.exit()

    theNumTimepoints = theSBFileReader.GetNumTimepoints(theCapture)
    theNumChannels = theSBFileReader.GetNumChannels(theCapture)
    theNumPlanes = theSBFileReader.GetNumZPlanes(theCapture)
    if thePlane is None:
        thePlane = int(theNumPlanes / 2)

    if rgb_mode:
        if theNumChannels < 3:
            print('RGB mode requested but less than 3 channels present')
            sys.exit(2)
        r, g, b = rgb_map
        for c in (r, g, b):
            if c < 0 or c >= theNumChannels:
                print('RGB mapping channel out of range: ', rgb_map)
                sys.exit(2)
        # Read planes for each channel
        r_im = theSBFileReader.ReadImagePlaneBuf(theCapture, thePosition, theTimepoint, thePlane, r, True)
        g_im = theSBFileReader.ReadImagePlaneBuf(theCapture, thePosition, theTimepoint, thePlane, g, True)
        b_im = theSBFileReader.ReadImagePlaneBuf(theCapture, thePosition, theTimepoint, thePlane, b, True)
        rgb = np.dstack([r_im, g_im, b_im])
        plt.figure(1)
        plt.title(f'RGB C{r},{g},{b} T{theTimepoint} Z{thePlane}')
        plt.imshow(rgb)
        plt.axis('off')
        plt.show(block=False)
        data = input("Press Enter to exit or Ctrl+C to quit...\n")
        return

    # Default behavior: print metadata and show channel 0 for each timepoint
    theChannel = 0
    theThumbnail = theSBFileReader.GetThumbnail(theCapture)

    theImageName = theSBFileReader.GetImageName(theCapture)
    print("*** the image name: ", theImageName)

    theImageComments = theSBFileReader.GetImageComments(theCapture)
    print("*** the image comments: ", theImageComments)

    theNumPositions = theSBFileReader.GetNumPositions(theCapture)
    print("*** the image num positions: ", theNumPositions)

    print("*** the image num timepoints: ", theNumTimepoints)
    print("*** the image num channels: ", theNumChannels)

    theNumAnnotations = theSBFileReader.GetNumROIAnnotations(theCapture)
    print("*** the image num ROI annotations: ", theNumAnnotations)
    for theAnnoIndex in range(0, theNumAnnotations):
        theShape, theVertexes = theSBFileReader.GetROIAnnotation(theCapture, theAnnoIndex)
        print("theShape: ", theShape)
        for theVertex in theVertexes:
            print(" x: ", theVertex.mX, " y: ", theVertex.mY)

    for theTimepoint in range(0, theNumTimepoints):
        theNumRegions = theSBFileReader.GetNumFRAPRegions(theCapture, theTimepoint)
        if (theNumRegions == 0):
            continue
        print("*** the image num FRAP Regions ", theNumRegions, " for timepoint: ", theTimepoint)
        theShape, theVertexes = theSBFileReader.GetFRAPAnnotation(theCapture, theTimepoint)
        print("the FRAPP Annotation shape : ", theShape)
        for theVertex in theVertexes:
            print(" x: ", theVertex.mX, " y: ", theVertex.mY)
        for theRegionIndex in range(0, theNumRegions):
            theShape, theVertexes = theSBFileReader.GetFRAPRegion(theCapture, theTimepoint, theRegionIndex)
            print("the Frap Region shape: ", theShape)
            for theVertex in theVertexes:
                print(" x: ", theVertex.mX, " y: ", theVertex.mY)

    theX, theY, theZ = theSBFileReader.GetVoxelSize(theCapture)
    print("*** the the voxel x,y,z size is: ", theX, theY, theZ)

    theY, theM, theD, theH, theMn, theS = theSBFileReader.GetCaptureDate(theCapture)
    print("*** the the date yr/mn/day/hr/min/sec is: ", theY, theM, theD, theH, theMn, theS)

    theXmlDescriptor = theSBFileReader.GetAuxDataXMLDescriptor(theCapture, theChannel)
    print("*** theXmlDescriptor is ", theXmlDescriptor)

    theLen, theType = theSBFileReader.GetAuxDataNumElements(theCapture, theChannel)
    print("*** theLen,theType ", theLen, theType)

    theXmlData = theSBFileReader.GetAuxSerializedData(theCapture, theChannel, 0)
    print("*** theXmlData is ", theXmlData)

    theNumRows = theSBFileReader.GetNumYRows(theCapture)
    theNumColumns = theSBFileReader.GetNumXColumns(theCapture)
    theZplane = int(theNumPlanes / 2)
    for theTimepoint in range(0, theNumTimepoints):
        image = theSBFileReader.ReadImagePlaneBuf(theCapture, 0, theTimepoint, theZplane, 0, True)  # captureid,position,timepoint,zplane,channel,as 2d
        print("*** The read buffer len is: ", len(image))

        plt.figure(theTimepoint + 1)
        plt.imshow(image)

    plt.pause(100)
    data = input("Please hit Enter to exit:\n")
    print("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
    


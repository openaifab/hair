import cv2 
import numpy as np

def trimap(image, size, erosion=False):
    
    row    = image.shape[0];
    col    = image.shape[1];

    pixels = 2*size + 1;                                     ## Double and plus 1 to have an odd-sized kernel
    kernel = np.ones((pixels,pixels),np.uint8)               ## How many pixel of extension do I get

    if erosion is not False:
        erosion = int(erosion)
        erosion_kernel = np.ones((3,3), np.uint8)                     ## Design an odd-sized erosion kernel
        image = cv2.erode(image, erosion_kernel, iterations=erosion)  ## How many erosion do you expect
        image = np.where(image > 0, 255, image)                       ## Any gray-clored pixel becomes white (smoothing)
        # Error-handler to prevent entire foreground annihilation
        if cv2.countNonZero(image) == 0:
            print("ERROR: foreground has been entirely eroded");
            sys.exit();

    dilation  = cv2.dilate(image, kernel, iterations = 1)

    dilation  = np.where(dilation == 255, 128, dilation) 	## WHITE to GRAY
    remake    = np.where(dilation != 128, 0, dilation)		## Smoothing
    remake    = np.where(image > 128, 200, dilation)		## mark the tumor inside GRAY

    remake    = np.where(remake < 128, 0, remake)		## Embelishment
    remake    = np.where(remake > 200, 0, remake)		## Embelishment
    remake    = np.where(remake == 200, 255, remake)		## GRAY to WHITE

    #############################################
    # Ensures only three pixel values available #
    # TODO: Optimization with Cython            #
    #############################################    
    for i in range(0,row):
        for j in range (0,col):
            if (remake[i,j] != 0 and remake[i,j] != 255):
                remake[i,j] = 128;
    print("generate trimap(size: " + str(size) + ", erosion: " + str(erosion) + ")")
    return remake
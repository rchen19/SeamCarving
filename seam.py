"""
@rchen
email: ranchen@gatech.edu
"""
import numpy as np
import cv2
import os
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# I/O directories
input_dir = "images"
output_dir = "output"

#define energy functions
#only sobel gradient is used here
#but can add more if desired

def sobel_gradient_map(image):
    """
    image: 3D array, dtype = uint8
    """
    #image = np.float64(image)
    gradient_x = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3, scale=1.0/1.0)
    gradient_x = np.absolute(gradient_x)
    gradient_x = np.mean(gradient_x, axis=2)

    gradient_y = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3, scale=1.0/1.0)
    gradient_y = np.absolute(gradient_y)
    gradient_y = np.mean(gradient_y, axis=2)

    gradient_map = 0.5 * gradient_x + 0.5 * gradient_y
    gradient_map = cv2.normalize(src=gradient_map, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return gradient_map

def find_energy_map(image, energy_type="sobel"):
    if energy_type == "sobel":
        return sobel_gradient_map(image)
    else:
        raise ValueError("energy type undefined")

@jit
def find_vertical_seam(energy_map, k=1):
    """
    energy_map: 2D array, dtype = float64
    k: max possible shift from one line to the next, default is 1
    return: 2d array, h x 2, w is the height of image
    """
    h, w = energy_map.shape
    #min_energy = np.full((h,w), np.inf, dtype=np.float64) #matrix for cumulative minimun energy
    path = np.zeros((h,w), dtype=np.int64)
    min_energy = energy_map.copy()
    #create a row with element being col numbers, length w
    row_n = np.arange(0,w,1)
    #print "row_n:", row_n.shape
    for i in range(1, h): #iterate over rows, starting from the 2nd row
        previous_row = min_energy[i-1]
        energy_array = np.full((2*k+1, w), np.inf, dtype=np.float64)
        for m in range(0, 2*k+1):
            n = m - k #number of col shifted, - means shift left, + means shift right
            
            if n<=0:
                energy_array[m, 0:w+n] = previous_row[-n:w]
            else: #n>0
                energy_array[m, n:w] = previous_row[0:w-n]

            #alternatively:
            #energy_array[m, max(0,n):w+min(0,n)] = previous_row[-min(0,n):w-max(0,n)]

        min_energy[i] = np.min(energy_array, axis=0) + min_energy[i]
        path[i] = row_n - (np.argmin(energy_array, axis=0) -k)
    #print np.max(row_n)
    #print "path"
    #print path.shape
    #print np.max(path)
    #print np.min(path)      

    min_energy_end = np.min(min_energy[h-1,:])
    min_path = [(h-1, np.argmin(min_energy[h-1,:]))]
    for i in range(h-1, 0, -1): #starting from the last row
        #print "i is:", i
        #print "the other is:", min_path[-1][1]
        ind = path[i, min_path[-1][1]]
        min_path.append((i-1,ind))
    min_path = np.array(min_path)
    min_path = min_path[min_path[:,0].argsort()]

    return min_path, min_energy_end

@jit
def find_horizontal_seam(energy_map, k=1):
    """
    energy_map: 2D array, dtype = float64
    k: max possible shift from one line to the next, default is 1
    return: 2d array, w x 2, w is the width of image
    """
    h, w = energy_map.shape
    #min_energy = np.full((h,w), np.inf, dtype=np.float64) #matrix for cumulative minimun energy
    path = np.zeros((h,w), dtype=np.int64)
    min_energy = energy_map.copy()
    #create a col with element being row numbers, length h
    col_n = np.arange(0,h,1)
    for j in range(1, w): #iterate over cols, starting from the 2nd col
        previous_col = min_energy[:,j-1]
        energy_array = np.full((h, 2*k+1), np.inf, dtype=np.float64)
        for m in range(0, 2*k+1):
            n = m - k #number of col shifted, - means shift up, + means shift down
            
            if n<=0:
                energy_array[0:h+n, m] = previous_col[-n:h]
            else: #n>0
                energy_array[n:h, m] = previous_col[0:h-n]

            #alternatively:
            #energy_array[max(0,n):h+min(0,n), m] = previous_row[-min(0,n):h-max(0,n)]

        min_energy[:,j] = np.min(energy_array, axis=1) + min_energy[:,j]
        path[:,j] = col_n - (np.argmin(energy_array, axis=1) - k)

    #print path

    min_energy_end = np.min(min_energy[:,w-1])
    min_path = [(np.argmin(min_energy[:,w-1]), w-1)]
    for j in range(w-1, 0, -1): #starting from the last col
        #print "j is:", j
        #print min_path[-1]
        ind = path[min_path[-1][0],j]
        min_path.append((ind, j-1))
    min_path = np.array(min_path)
    min_path = min_path[min_path[:,1].argsort()]

    return min_path, min_energy_end

def visualize_seam(image, seam):
    """
    image: 3D array
    seam: 2D array, 2 cols
    """
    img_out = image.copy()
    img_out[seam[:,0], seam[:,1], :] = np.array([0,0,255]) 

    cv2.imshow('dst_seam', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_out

@jit
def remove_vertical_seam_2D(image, seam):
    """
    image: 2D array
    seam: 2D array, 2cols, [[r,c],...]
    """
    h,w = image.shape
    img_out = image.copy()
    img_out = img_out.flatten()
    ind_to_remove = np.ravel_multi_index(seam.T, (h,w))
    img_out = np.delete(img_out, ind_to_remove)
    img_out = img_out.reshape((h, w-1))
    return img_out

@jit
def remove_vertical_seam(image, seam):
    """
    image: 3D array
    seam: 2D array, 2cols, [[r,c],...]
    """
    if len(image.shape) == 2:
        img_out = remove_vertical_seam_2D(image, seam)
    else:
        b,g,r = cv2.split(image)
        b_out = remove_vertical_seam_2D(b, seam)
        g_out = remove_vertical_seam_2D(g, seam)
        r_out = remove_vertical_seam_2D(r, seam)
        img_out = cv2.merge((b_out, g_out, r_out))

    return img_out

@jit
def remove_horizontal_seam_2D(image, seam):
    """
    image: 2D array
    seam: 2D array, 2cols, [[r,c],...]
    """
    h,w = image.shape
    img_out = image.copy()
    img_out = img_out.flatten(order='F')
    #print "seam:", seam.T
    ind_to_remove = np.ravel_multi_index(seam.T, (h,w), order='F')
    #print "raveled:", ind_to_remove
    img_out = np.delete(img_out, ind_to_remove)
    img_out = img_out.reshape((h-1, w), order='F')
    return img_out

@jit
def remove_horizontal_seam(image, seam):
    """
    image: 3D array
    seam: 2D array, 2cols, [[r,c],...]
    """
    if len(image.shape) == 2:
        img_out = remove_horizontal_seam_2D(image, seam)
    else:
        b,g,r = cv2.split(image)

        b_out = remove_horizontal_seam_2D(b, seam)
        g_out = remove_horizontal_seam_2D(g, seam)
        r_out = remove_horizontal_seam_2D(r, seam)
        img_out = cv2.merge((b_out, g_out, r_out))
        #print "ok"

    return img_out



@jit
def shrink_horizontal(image, target_width): #remove vertical seam repeatedly
    #check target_width
    if target_width >= image.shape[1]:
        raise ValueError("error in target shape: target width too large")
    else:
        c = image.shape[1] - target_width
        img_out = image.copy()
        while c > 0:
            #print "c is:", c
            seam, _ = find_vertical_seam(find_energy_map(img_out))
            img_out = remove_vertical_seam(img_out, seam)
            c -= 1
        return img_out

@jit
def shrink_vertical(image, target_height): #remove horizontal seam repeatedly
    #check target_height
    if target_height >= image.shape[0]:
        raise ValueError("error in target shape: target height too large")

    else:
        r = image.shape[0] - target_height
        img_out = image.copy()
        while r > 0:
            seam, _ = find_horizontal_seam(find_energy_map(img_out))
            img_out = remove_horizontal_seam(img_out, seam)
            r -= 1
        return img_out

@jit
def find_k_vertical_seams(energy_map, k):
    h,w = energy_map.shape
    seams = [np.arange(0,h,1)] #place the row indices as the first 1D array
    m = k
    while m > 0:
        seam, _ = find_vertical_seam(energy_map)
        seams.append(seam[:,1]) #append an 1D array of col numbers in the seam
        energy_map = remove_vertical_seam_2D(energy_map, seam)
        m -= 1
    #create an 2D array of seams, shape h x (k+1)
    #1st col is the row indices, the rest k cols are the col indicies of the k seams
    seams = (np.array(seams)).T
    for n in range(k, 1, -1): #should exclude m=1
        #n is is the col number in array seams 
        #where all seams on the rightside to col(n) (including col(n)) need to be converted
        #sub-array to be converted is
        seams_sub = seams[:, n:k+1]
        previous_seam = seams[:, n-1].reshape(-1,1) #here n>=2
        #if the value in seams_sub (the col numbers of the seams) < the value from previous_seam
        #then no need for change, meaning previous_seam pixel is inserted on the right of the current seam pixel
        #if the value in seams_sub (the col numbers of the seams) >= the value from previous_seam
        #then the values of seams_sub need to increment by 1
        #meaning the previous_seam pixel is inserted on the left of the current seam pixel
        diff = np.int64(seams_sub >= previous_seam) #convert boolean to 1s and 0s
        seams[:, n:k+1] = seams_sub + diff

    #when enlarging an original image, insert the rightmost seam from seams array first
    #all seams coordinates are converted to original image now
    #but when insert one seam, the rest un-inserted seams' coordinates need to be changed
    for m in range(k, 1, -1): #should also exclude m=1
        seams_sub = seams[:, 1:m]
        current_seam = seams[:,m].reshape(-1,1)

        diff = np.int64(seams_sub >= current_seam)
        seams[:, 1:m] = seams_sub + diff #here m >=2


    return seams


def add_vertical_seam_2D(image, seam, pixel_val=None):
    """
    image: 2D array
    seam: 2D array, 2cols, [[r,c],...]
    """
    h,w = image.shape
    img_out = np.zeros((h, w+1), dtype=np.uint8)
    for i in range(0,h,1): #iterate over rows, ith row:
        seam_loc = seam[i,1] #col number of ith row in the image
        if pixel_val is None:
            pixel = [image[i, seam_loc]]
            if seam_loc != 0: pixel.append(image[i, seam_loc-1])
            if seam_loc != w-1: pixel.append(image[i, seam_loc+1])
            #print pixel
            pixel = np.uint8(np.mean(pixel))
            #print "pixel:", pixel
        else: pixel = pixel_val
        #print i
        #print "seam loc:",seam_loc
        #print "width:", w
        #print pixel
        row_temp = np.insert(image[i], seam_loc+1, pixel) #"+1" for insertion on the right
        img_out[i] = np.uint8(row_temp)
    return img_out

def add_vertical_seam(image, seam):
    """
    image: 3D array
    seam: 2D array, 2cols, [[r,c],...]
    """
    if len(image.shape) == 2:
        img_out = add_vertical_seam_2D(image, seam)
    else:
        b,g,r = cv2.split(image)
        b_out = add_vertical_seam_2D(b, seam)
        g_out = add_vertical_seam_2D(g, seam)
        r_out = add_vertical_seam_2D(r, seam)
        img_out = cv2.merge((b_out, g_out, r_out))
    return img_out

def visualize_added_vertical_seam(image, seam):
    """
    image: 3D array
    seam: 2D array, 2cols, [[r,c],...]
    """
    if len(image.shape) == 2:
        img_out = add_vertical_seam_2D(image, seam, pixel_val=255)
    else:
        b,g,r = cv2.split(image)
        b_out = add_vertical_seam_2D(b, seam, pixel_val=0)
        g_out = add_vertical_seam_2D(g, seam, pixel_val=0)
        r_out = add_vertical_seam_2D(r, seam, pixel_val=255)
        img_out = cv2.merge((b_out, g_out, r_out))
    return img_out

def add_horizontal_seam_2D(image, seam, pixel_val=None):
    """
    image: 2D array
    seam: 2D array, 2cols, [[r,c],...]
    """
    h,w = image.shape
    img_out = np.zeros((h+1, w), dtype=np.uint8)
    for j in range(0, w, 1): #iterate over cols, jth col
        seam_loc = seam[j,0] #row number for jth col in the image
        if pixel_val is None:
            pixel = [image[seam_loc, j]]
            if seam_loc != 0: pixel.append(image[seam_loc-1, j])
            if seam_loc != h-1: pixel.append(image[seam_loc+1, j])
            pixel_val = np.uint8(np.mean(pixel))
        else: pixel = pixel_val
        col_temp = np.insert(image[:,j], seam_loc+1, pixel) #"+1" for insertion on the below
        img_out[:,j] = np.uint8(col_temp)

    return img_out

def add_horizontal_seam(image, seam):
    """
    image: 3D array
    seam: 2D array, 2cols, [[r,c],...]
    """
    if len(image.shape) == 2:
        img_out = add_horizontal_seam_2D(image, seam)
    else:
        b,g,r = cv2.split(image)
        b_out = add_horizontal_seam_2D(b, seam)
        g_out = add_horizontal_seam_2D(g, seam)
        r_out = add_horizontal_seam_2D(r, seam)
        img_out = cv2.merge((b_out, g_out, r_out))
    return img_out

def visualize_added_horizontal_seam(image, seam):
    """
    image: 3D array
    seam: 2D array, 2cols, [[r,c],...]
    """
    if len(image.shape) == 2:
        img_out = add_horizontal_seam_2D(image, seam, pixel_val=255)
    else:
        b,g,r = cv2.split(image)
        b_out = add_horizontal_seam_2D(b, seam, pixel_val=0)
        g_out = add_horizontal_seam_2D(g, seam, pixel_val=0)
        r_out = add_horizontal_seam_2D(r, seam, pixel_val=255)
        img_out = cv2.merge((b_out, g_out, r_out))
    return img_out


def expand_horizontal(image, target_width): #insert vertical seam repeatedly
    #check target_width
    if target_width <= image.shape[1]:
        raise ValueError("error in target shape: target width too small")

    else:
        c = target_width - image.shape[1] 
        print "c:", c
        img_out = image.copy()
        img_out_viz = image.copy()
        energy_map = find_energy_map(img_out)
        seams = find_k_vertical_seams(energy_map, k=c)
        while c > 0:
            img_out = add_vertical_seam(img_out, seams[:, [0, c]])
            img_out_viz = visualize_added_vertical_seam(img_out_viz, seams[:, [0, c]])
            c -= 1
        return img_out, img_out_viz

def expand_vertical(image, target_height): #insert horizontal seam repeatedly
    #check target_width
    if target_height <= image.shape[0]:
        raise ValueError("error in target shape: target height too small")

    else:
        r = arget_height - image.shape[0]
        img_out = image.copy()
        while r > 0:
            seam, _ = find_horizontal_seam(find_energy_map(img_out))
            img_out = add_horizontal_seam(img_out, seam)
            r -= 1
        return img_out

@jit
def transport_map(image):
    """
    image: 3D array
    """
    h,w,_ = image.shape
    img = image.copy()
    #transport map, keeps track of the lowest energy
    T = np.full((h,w), np.inf, dtype=np.float64) 
    #step map, keeps track of the steps taken
    #0 means previous step from left, vertical seam removal
    #1 means previous step from above, horizontal seam removal
    S = np.zeros((h,w), dtype=np.int64) 

    #keep track of two rows of reduced images
    #current_row = []
    previous_row = []

    for i in range(0, h):
        current_row = []
        for j in range(0, w):
            if i==0 and j==0:
                T[i,j] = 0
                current_row.append(img)
            elif i==0:
                img_left = current_row[j-1]
                seam_left, energy_left = find_vertical_seam(find_energy_map(img_left))
                T[i,j] = T[i, j-1] + energy_left
                S[i,j] = 0
                current_row.append(remove_vertical_seam(img_left, seam_left))
            elif j==0:
                img_top = previous_row[j]
                seam_top, energy_top = find_horizontal_seam(find_energy_map(img_top))
                T[i,j] = T[i-1, j] + energy_top
                S[i,j] = 1
                current_row.append(remove_horizontal_seam(img_top, seam_top))
            else:
                img_left = current_row[j-1]
                img_top = previous_row[j]
                seam_left, energy_left = find_vertical_seam(find_energy_map(img_left))
                seam_top, energy_top = find_horizontal_seam(find_energy_map(img_top))
                energy_left += T[i, j-1]
                energy_top += T[i-1, j]
                              
                if energy_left<=energy_top:
                    current_row.append(remove_vertical_seam(img_left, seam_left))
                    T[i,j] = energy_left
                    S[i,j] = 0
                else:
                    current_row.append(remove_horizontal_seam(img_top, seam_top))
                    T[i,j] = energy_top
                    S[i,j] = 1

        previous_row = current_row[:]

    return T, S

def get_sequence(r, c, S):
    """
    S: step map
    """
    sequence = [S[r,c]]
    trace = [(r,c)]
    while r + c > 1:
        if sequence[-1] == 0:
            c = c - 1
        else:
            r = r - 1
        sequence.append(S[r,c])
        trace.append((r,c))

    sequence.reverse()
    trace.reverse()
    return sequence, trace

def retarget(image, r, c, step_map=None):
    """
    image: 3D array
    r: int, number of rows to reduce
    c: int, numver of cols to reduce
    return: an optimal sequence of vertical and horizontal seam removal
    """
    if step_map is None:
        T, S = transport_map(image)
    else:
        S = step_map
    sequence, trace = get_sequence(r, c, S)
    #print "r+c:", r+c
    #print "sequence:", len(sequence)
    return sequence
    #0:move from left, remove vertical seam
    #1:move from top, remove horizontal seam


def reduce_2d(image, n, m, step_map=None):
    """
    image: 3D array
    n: int, new height
    m: int, new width
    return: 3D array of reduced image
    """
    img_out = image.copy()
    sequence = retarget(image, image.shape[0]-n, image.shape[1]-m, step_map=step_map)
    #print image.shape[0]-n + image.shape[1]-m
    #print sequence
    #sequence = [0]*54
    #print sequence
    for step in sequence:
        #0:move from left, remove vertical seam
        #1:move from top, remove horizontal seam
        if step == 0:
            seam, _ = find_vertical_seam(find_energy_map(img_out))
            img_out = remove_vertical_seam(img_out, seam)
        else:
            seam, _ = find_horizontal_seam(find_energy_map(img_out))
            img_out = remove_horizontal_seam(img_out, seam)
    return img_out

    




if __name__ == "__main__":
    image_file = os.path.join(input_dir, "fig7.png") 
    image = cv2.imread(image_file)
    energy_map = find_energy_map(image)
    #print energy_map.shape
    #cv2.imshow('dst_rt2', energy_map)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #v_seam, _ = find_vertical_seam(energy_map)
    #print v_seam
    #visualize_seam(image, v_seam)

    #h_seam, _ = find_horizontal_seam(energy_map)
    #print h_seam.T
    #visualize_seam(image, h_seam)

    #img_out = shrink_horizontal(image, 250)
    #img_out = reduce_2d(image, 500, 400)

    #cv2.imshow('dst_rt2', img_out)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    if os.path.isfile("T.csv") and os.path.isfile("S.csv"):
        T = pd.read_csv("T.csv", header=None, index_col=False).values
        S = pd.read_csv("S.csv", header=None, index_col=False).values
    else:
        T, S = transport_map(image)
        T_df = pd.DataFrame(T)
        S_df = pd.DataFrame(S)
        T_df.to_csv("T.csv", header=False, index=False, index_label=False)
        S_df.to_csv("S.csv", header=False, index=False, index_label=False)

    print T.shape
    print image.shape
    sequence, trace = get_sequence(90, 125, S)
    #print np.array(trace)[0]
    #print np.array(trace)[0]
    mask = np.zeros(T.shape, dtype=int)
    print mask

    mask[np.array(trace)[:,0], np.array(trace)[:,1]] = 1
    print mask
    T_masked = np.ma.array(T, mask=mask)
    cmap = cm.jet
    cmap.set_bad(color='white', alpha=1.)
    plt.imshow(T_masked, \
           interpolation=None, cmap=cmap)
    plt.colorbar()
    plt.show()






import numpy as np
import cv2
import os
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from seam import *

"""
fig5 700 x 466
fig7 254 x 350
fig8 200 x 239
"""
# I/O directories
input_dir = "images"
output_dir = "output"

def part1(filename="fig5.png"):
    image_file = os.path.join(input_dir, filename) 
    image = cv2.imread(image_file)
    energy_map = find_energy_map(image)
    #print energy_map.shape
    cv2.imshow('dst_rt2', energy_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(output_dir, "fig5_energy_map.png"), \
        np.uint8(cv2.normalize(src=energy_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)))

    v_seam, _ = find_vertical_seam(energy_map)
    #print v_seam
    vis_v_seam = visualize_seam(image, v_seam)
    cv2.imwrite(os.path.join(output_dir, "fig5_v_seam.png"), vis_v_seam)

    h_seam, _ = find_horizontal_seam(energy_map)
    #print h_seam.T
    vis_h_seam = visualize_seam(image, h_seam)
    cv2.imwrite(os.path.join(output_dir, "fig5_h_seam.png"), vis_h_seam)

    fig5_h_shrunken = shrink_horizontal(image, target_width=350)
    cv2.imshow('dst_rt2', fig5_h_shrunken)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, "fig5_h_shrunken.png"), fig5_h_shrunken)





def part2(filename="fig8.png"):
    image_file = os.path.join(input_dir, filename) 
    image = cv2.imread(image_file)
    energy_map = find_energy_map(image)
    cv2.imshow('dst_rt1', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    seams = find_k_vertical_seams(energy_map, 100)
    print energy_map.shape
    print seams.shape
    for i in range(100, 0, -1):
        n = 100 - i
        if np.max(seams[:, i]) > energy_map.shape[1] + n -1:
            print  np.max(seams[:, i]) - (energy_map.shape[1] + n -1)
            print "wrong:", i
            print energy_map.shape[1] + n -1
            print seams[:, i]

    """
    fig8_expanded_50, fig8_visualize_50 = expand_horizontal(image, target_width=358)

    cv2.imshow('dst_rt2', fig8_expanded_50)
    cv2.imshow('dst_rt3', fig8_visualize_50)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, "fig8_expanded_50.png"), fig8_expanded_50)
    cv2.imwrite(os.path.join(output_dir, "fig8_visualize_50.png"), fig8_visualize_50)

    fig8_expanded_50_50, fig8_visualize_50_50 = expand_horizontal(fig8_expanded_50, target_width=478)
    cv2.imwrite(os.path.join(output_dir, "fig8_expanded_50_50.png"), fig8_expanded_50_50)
    cv2.imwrite(os.path.join(output_dir, "fig8_visualize_50_50.png"), fig8_visualize_50_50)



def part3(filename="fig7.png"):
    image_file = os.path.join(input_dir, filename) 
    image = cv2.imread(image_file)
    energy_map = find_energy_map(image)
    if os.path.isfile("T.csv") and os.path.isfile("S.csv"):
        T = pd.read_csv("T.csv", header=None, index_col=False).values
        S = pd.read_csv("S.csv", header=None, index_col=False).values
    else:
        T, S = transport_map(image)
        T_df = pd.DataFrame(T)
        S_df = pd.DataFrame(S)
        T_df.to_csv("T.csv", header=False, index=False, index_label=False)
        S_df.to_csv("S.csv", header=False, index=False, index_label=False)

    #print T.shape
    #print image.shape
    sequence, trace = get_sequence(254/2, 350/2, S)
    #print np.array(trace)[0]
    #print np.array(trace)[0]
    mask = np.zeros(T.shape, dtype=int)
    #print mask

    mask[np.array(trace)[:,0], np.array(trace)[:,1]] = 1
    #print mask
    T_masked = np.ma.array(T, mask=mask)
    cmap = cm.jet
    cmap.set_bad(color='white', alpha=1.)
    plt.imshow(T_masked, \
           interpolation=None, cmap=cmap)
    plt.colorbar()
    plt.show()
    #plt.imsave("transport.png", T_masked, cmap=cmap)
    #plt.colorbar()

    plt.imshow(S, \
           interpolation=None, cmap=colors.ListedColormap(['blue', 'red']))
    plt.colorbar(boundaries=[0,0.5,1], ticks=[0, 1])
    plt.show()
    #plt.imsave("step.png", T_masked, cmap=cmap)

    fig = plt.figure(figsize = (9.,6.)) # Your image (W)idth and (H)eight in inches
    # Stretch image to full figure, removing "grey region"
    plt.subplots_adjust(left = 0.1, right = 0.8, top = 1, bottom = 0)
    im = plt.imshow(T_masked, \
           interpolation=None, cmap=cmap) # Show the image
    pos = fig.add_axes([0.83,0.15,0.02,0.7]) # Set colorbar position in fig
    fig.colorbar(im, cax=pos) # Create the colorbar
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'T_out.png'))

    fig = plt.figure(figsize = (9.,6.)) # Your image (W)idth and (H)eight in inches
    # Stretch image to full figure, removing "grey region"
    plt.subplots_adjust(left = 0.1, right = 0.8, top = 1, bottom = 0)
    im = plt.imshow(S, \
           interpolation=None, cmap=colors.ListedColormap(['blue', 'red'])) # Show the image
    pos = fig.add_axes([0.83,0.15,0.02,0.7]) # Set colorbar position in fig
    fig.colorbar(im, cax=pos, boundaries=[0,0.5,1], ticks=[0, 1]) # Create the colorbar
    plt.savefig(os.path.join(output_dir, 'S_out.png'))

    fig7_2D_reduce = reduce_2d(image, 254/2, 350/2, step_map=S) #fig7 254 x 350
    cv2.imwrite(os.path.join(output_dir, "fig7_2D_reduce.png"), fig7_2D_reduce)

def part4(): #random tests, please ignore
    image_file = os.path.join(input_dir, "fig7.png") 
    image = cv2.imread(image_file)
    energy_map = find_energy_map(image)
    h_seam, _ = find_horizontal_seam(energy_map)
    #print h_seam.T
    vis_h_seam = visualize_seam(image, h_seam)
    out_h = remove_horizontal_seam(image, h_seam)

    #v_seam, _ = find_vertical_seam(energy_map)
    #vis_v_seam = visualize_seam(image, v_seam)
    #out_v = remove_vertical_seam(image, v_seam)
    #out = shrink_vertical(image, 225)
    #out = shrink_horizontal(image, 225)
    cv2.imshow('dst_rt3', out_h)
    #cv2.imshow('dst_rt4', out_v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #part1()
    #part2(filename="fig8.png")
    part3(filename="fig7.png")
    #part4()

# python Scripts/visualize_2d.py --data_root /home/max/projects/data_ifl_real/data_ifl_pool  --viewpt 3 --scene _2023_6_15_17_56_3 --real_data --real_data_grasps --real_data_amodal


#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
import cv2
import copy
from skimage import io
from skimage import color
from skimage import segmentation
from skimage import measure
import random
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import visualize_layout

COLORS = ['black', 'dimgray', 'dimgrey', 'gray', 'grey', 'darkgray', 'darkgrey', 'silver', 'lightgray', 'lightgrey',
            'gainsboro', 'whitesmoke', 'white', 'snow', 'rosybrown', 'lightcoral', 'indianred', 'brown', 'firebrick',
            'maroon', 'darkred', 'red', 'mistyrose', 'salmon', 'tomato', 'darksalmon', 'coral', 'orangered', 'lightsalmon',
            'sienna', 'seashell', 'chocolate', 'saddlebrown', 'sandybrown', 'peachpuff', 'peru', 'linen', 'bisque',
            'darkorange', 'burlywood', 'antiquewhite', 'tan', 'navajowhite', 'blanchedalmond', 'papayawhip', 'moccasin',
            'orange', 'wheat', 'oldlace', 'floralwhite', 'darkgoldenrod', 'goldenrod', 'cornsilk', 'gold', 'lemonchiffon',
            'khaki', 'palegoldenrod', 'darkkhaki', 'ivory', 'beige', 'lightyellow', 'lightgoldenrodyellow', 'olive',
            'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'greenyellow', 'chartreuse', 'lawngreen', 'honeydew',
            'darkseagreen', 'palegreen', 'lightgreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen',
            'mediumseagreen', 'springgreen', 'mintcream', 'mediumspringgreen', 'mediumaquamarine', 'aquamarine', 'turquoise',
            'lightseagreen', 'mediumturquoise', 'azure', 'lightcyan', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan',
            'aqua', 'cyan', 'darkturquoise', 'cadetblue', 'powderblue', 'lightblue', 'deepskyblue', 'skyblue', 'lightskyblue',
            'steelblue', 'aliceblue', 'dodgerblue', 'lightslategray', 'lightslategrey', 'slategray', 'slategrey',
            'lightsteelblue', 'cornflowerblue', 'royalblue', 'ghostwhite', 'lavender', 'midnightblue', 'navy', 'darkblue',
            'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'blueviolet', 'indigo',
            'darkorchid', 'darkviolet', 'mediumorchid', 'thistle', 'plum', 'violet', 'purple', 'darkmagenta', 'fuchsia',
            'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink', 'lavenderblush', 'palevioletred', 'crimson',
            'pink', 'lightpink']

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset Obejct detection viewer.")
    parser.add_argument(
        "--data_root",
        type=str, 
        default="./dataset_real",
        help="Path to data.")
    parser.add_argument(
        "--scene",
        type=str, default="0",
        help="Specify which scene to load.")
    parser.add_argument(
        "--viewpt",
        type=int, default=0,
        help="Specify which viwpt to load.")
    parser.add_argument(
        "--visualize_layout",
        default=False, action='store_true',
        help="Set flag to vis scene labels.")
    parser.add_argument(
        "--real_data",
        default=False, action='store_true',
        help="Set flag for real world data.")
    parser.add_argument(
        "--amodal",
        default=False, action='store_true',
        help="Set flag for visualizing amodal mask.")
    parser.add_argument(
        "--real_data_grasps",
        default=False, action='store_true',
        help="Set flag for real world data grasps.")
    parser.add_argument(
        "--real_data_mats",
        default=False, action='store_true',
        help="Set flag for real world data materials.")
    parser.add_argument(
        "--real_data_amodal",
        default=False, action='store_true',
        help="Set flag for real world data materials.")
    parser.add_argument(
        "--save_figures",
        default=False, action='store_true',
        help="Set flag for real world data materials.")

    args = parser.parse_args()

    PATH_TO_DATA = pathlib.Path(args.data_root)
    SCENE = args.scene
    VIEWPT = args.viewpt
    PATH_TO_SCENE = PATH_TO_DATA / f"scene{SCENE}"
    PATH_TO_RGB = PATH_TO_SCENE / f"{VIEWPT}_rgb.png"
    PATH_TO_DEPTH = PATH_TO_SCENE / f"{VIEWPT}.npz"
    PATH_TO_GRASPS = PATH_TO_SCENE / f"{VIEWPT}_grasps.npz"
    PATH_TO_MATERIALS = PATH_TO_SCENE / f"{VIEWPT}_mats.npz"
    PATH_TO_AMODAL = PATH_TO_SCENE / f"{3}_amodal.npz"

    color_bgr = cv2.imread(str(PATH_TO_RGB))
    color_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    with np.load(str(PATH_TO_DEPTH)) as data:
        depth = data['depth']
        try:
            instances_objects = data['instances_objects']
        except:
            print(f"WARNING : instances_objects not in {PATH_TO_DEPTH}.")
            instances_objects = np.zeros_like(depth)

        try:
            instances_semantic = data['instances_semantic']
        except:
            print(f"WARNING : instances_semantic not in {PATH_TO_DEPTH}.")
            instances_semantic = np.zeros_like(depth)

        if args.real_data is False:
            occlusion = data['occlusion']
            try:
                occlusion_masks = data['occlusion_objects']
            except:
                print(f"WARNING : occlusion_masks not in {PATH_TO_DEPTH}.")
                pass
            try:
                occlusion_masks = data['seg_masks_single']
            except:
                print(f"WARNING : seg_masks_single not in {PATH_TO_DEPTH}.")
                pass

    if args.real_data and args.real_data_amodal:
        with np.load(str(PATH_TO_AMODAL)) as data:
            # amodal_bitmaps_full_size = data['amodal_bitmaps_full_size']
            amodal_bitmaps_full_size_instances = data['amodal_bitmaps_full_size_instances']
            occlusion_bitmap_full_size=data['occlusion_bitmap_full_size']

    if args.real_data_grasps and args.real_data:
        with np.load(str(PATH_TO_GRASPS)) as data:
            try:
                suction_bitmap = data['suction_bitmap']
            except:
                print(f"WARNING : suction_bitmap not in {PATH_TO_GRASPS}.")
                suction_bitmap = np.zeros_like(depth)

            #suction_bitmap = data['suction_bitmap']
            try:
                parallel_jaw_2d = data['parallel_jaw_2d']
            except:
                print(f"WARNING : parallel_jaw_2d not in {PATH_TO_GRASPS}.")
                parallel_jaw_2d = np.array([])
            
            #parallel_jaw_2d = data['parallel_jaw_2d']

        suction_mask = np.zeros_like(color_img)
        for i in range(suction_mask.shape[0]):
            for j in range(suction_mask.shape[1]):
                if suction_bitmap[i,j] != 0:
                    suction_mask[i,j] = (255,0,0)
                else:
                    suction_mask[i,j] = color_img[i,j]

        parallel_jaw_mask = copy.deepcopy(color_img)
        for pt_pair in parallel_jaw_2d:
            parallel_jaw_mask = cv2.line(parallel_jaw_mask, pt_pair[0], pt_pair[1], (0,0,255), thickness=10)
    else:
        suction_mask = np.zeros_like(color_img)
        parallel_jaw_mask = np.zeros_like(color_img)

    if args.real_data_mats and args.real_data:
        with np.load(str(PATH_TO_MATERIALS)) as data:
            try:
                material_bitmap = data['material_bitmap']
            except:
                print(f"WARNING : material_bitmap not in {PATH_TO_MATERIALS}.")
                material_bitmap = np.zeros_like(depth)
            #material_bitmap = data['material_bitmap']
    else:
        material_bitmap = np.zeros_like(depth)

    if args.real_data is False:
        # Segment image with SLIC - Simple Linear Iterative Clustering
        seg = segmentation.slic(color_img, n_segments=30, compactness=40.0, enforce_connectivity=True, sigma=3)
        ## generate seg from occlusion masks
        seg = np.zeros(color_img.shape[:2], dtype=np.uint8)
        mask = instances_objects
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                if mask[i,j] != 0:
                    #print(mask[i,j])
                    seg[i,j] = int(mask[i,j])
                else:
                    #print(mask[i,j])
                    pass


        random.shuffle(COLORS)

        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(color.label2rgb(seg,color_img, saturation=0.4, colors=COLORS, kind='overlay'))

        #print(occlusion_masks.shape[0])
        for idx in range(occlusion_masks.shape[0]):
            mask = occlusion_masks[int(idx)]
            contours_ = measure.find_contours(mask, 0.8)
            for contour in contours_:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color=COLORS[idx])


        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('amodal segmentation')
        plt.show()

    if args.real_data is False:
        height, width = depth.shape
        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.title('RGB')
        plt.imshow(color_img)
        plt.subplot(2, 3, 2)
        plt.axis('off')
        plt.title('Depth')
        plt.imshow(depth)
        plt.subplot(2, 3, 3)
        plt.axis('off')
        plt.title('Instances (Object Ids)')
        plt.imshow(instances_objects)
        plt.subplot(2, 3, 4)
        plt.axis('off')
        plt.title('Instances (Categories)')
        plt.imshow(instances_semantic)
        plt.subplot(2, 3, 5)
        plt.title('Occlusion')
        plt.imshow(occlusion*100, cmap=plt.get_cmap('YlGnBu'))
        plt.axis('off')
        plt.colorbar()
        plt.show()
    elif args.real_data and args.real_data_amodal:
        random.shuffle(COLORS)

        #color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0), (255, 192, 203), (0, 255, 255), (255, 0, 255),]
        def hex_to_rgb(value):
            value = value.lstrip('#')
            lv = len(value)
            return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

        color_list_hex = ['#be254a', '#dc484c', '#ef6645', '#f88c51', '#fdb365', '#fed27f', '#feeb9d', '#fffebe',
                          '#f0f9a7', '#d8ef9b', '#b3e0a2', '#89d0a4', '#60bba8', '#3f97b7', '#4273b3', '#be254a', '#dc484c', '#ef6645', '#f88c51', '#fdb365', '#fed27f', '#feeb9d', '#fffebe',
                          '#f0f9a7', '#d8ef9b', '#b3e0a2', '#89d0a4', '#60bba8', '#3f97b7', '#4273b3']

        color_list = [hex_to_rgb(color) for color in color_list_hex]

        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        labels = list(range(8))
        cats = ['airfilter', 'airfilter', 'airfilter', 'airfilter', 'tennis_ball', 'tennis_ball', 'moustard_bottle', 'moustard_bottle']
        captions = ["{}".format(val)for val, cat in zip(labels, cats)]
        # ax.imshow(imgviz.instances2rgb(color_img, labels=labels, captions=captions, masks=amodal_bitmaps_full_size_instances[:, 0].astype(bool), colormap=np.array(color_list)))

        color_img_seg = np.zeros_like(color_img)
        for idx in range(amodal_bitmaps_full_size_instances.shape[0]):
            mask_vis = amodal_bitmaps_full_size_instances[int(idx), 0]
            for i in range(color_img.shape[0]):
                for j in range(color_img.shape[1]):
                    if mask_vis[i,j]:
                        color_img_seg[i,j] = color_list[idx * 2]
        
        color_img_seg = cv2.addWeighted(color_img, 0.5, color_img_seg, 0.5, 0.0)
        # ax.imshow(color_img_seg)

        # ax.imshow(imgviz.instances2rgb(color_img, labels=labels, captions=captions, masks=amodal_bitmaps_full_size_instances[:, 0].astype(bool), colormap=np.array(color_list)))

        color_img_seg = np.zeros_like(color_img)
        for idx in range(amodal_bitmaps_full_size_instances.shape[0]):
            mask_vis = amodal_bitmaps_full_size_instances[int(idx), 0]
            for i in range(color_img.shape[0]):
                for j in range(color_img.shape[1]):
                    if mask_vis[i,j]:
                        color_img_seg[i,j] = color_list[idx * 2]
        
        color_img_seg = cv2.addWeighted(color_img, 0.5, color_img_seg, 0.5, 0.0)
        # ax.imshow(color_img_seg)

        for idx in range(amodal_bitmaps_full_size_instances.shape[0]):
            mask_invis = amodal_bitmaps_full_size_instances[int(idx), 1]
            contours_ = measure.find_contours(mask_invis, 0.8)
            for contour in contours_:
                color_ = (color_list[idx * 2][0]/255, color_list[idx * 2][1]/255, color_list[idx * 2][2]/255)
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2.5, color=color_)

        for idx in range(amodal_bitmaps_full_size_instances.shape[0]):
            mask_vis = amodal_bitmaps_full_size_instances[int(idx), 0]

            contours_ = measure.find_contours(mask_vis, 0.8)
            for contour in contours_:
                for i in range(color_img.shape[0]):
                    for j in range(color_img.shape[1]):
                        if mask_vis[i,j]:
                            color_img[i,j] = color_list[idx * 2]
                color_ = (color_list[idx * 2][0]/255, color_list[idx * 2][1]/255, color_list[idx * 2][2]/255)
                # ax.plot(contour[:, 1], contour[:, 0] , linewidth=2.5, color=color_)
        
        ax.imshow(color_img_seg)
        if args.save_figures:
            plt.savefig("plot.png", dpi=800, bbox_inches="tight")
        plt.show()
    
        plt.title('Occlusion')
        plt.imshow(occlusion_bitmap_full_size*100, cmap=plt.get_cmap('bwr'))
        plt.axis('off')
        plt.colorbar()
        if args.save_figures:
            plt.savefig("occlusion_plot_bwr.png", dpi=2000, bbox_inches="tight")
        plt.show()
    
    if args.real_data:
        height, width = depth.shape
        plt.subplot(2, 4, 1)
        plt.axis('off')
        plt.title('RGB')
        plt.imshow(color_img)
        plt.subplot(2, 4, 2)
        plt.axis('off')
        plt.title('Depth')
        plt.imshow(depth, cmap='Greys')
        plt.colorbar()
        plt.subplot(2, 4, 3)
        plt.axis('off')
        plt.title('Instances (Object Ids)')
        plt.imshow(instances_objects)
        plt.subplot(2, 4, 4)
        plt.axis('off')
        plt.title('Instances (Categories)')
        plt.imshow(instances_semantic)
        plt.subplot(2, 4, 5)
        plt.title('Vacuum')
        plt.imshow(suction_mask)
        plt.axis('off')
        plt.subplot(2, 4, 6)
        plt.title('Parallel-Jaw')
        plt.imshow(parallel_jaw_mask)
        plt.axis('off')
        plt.subplot(2, 4, 7)
        plt.axis('off')
        plt.title('Materials')
        plt.imshow(material_bitmap)
        plt.show()

    if args.visualize_layout:
        PATH_TO_RGB = PATH_TO_SCENE / f"0" if not args.real_data else PATH_TO_SCENE / f"3"
        visualize_layout.visualize(str(PATH_TO_RGB))

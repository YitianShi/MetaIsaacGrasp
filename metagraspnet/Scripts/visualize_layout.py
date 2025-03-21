from skimage.io import imread
from matplotlib import pyplot as plt
import numpy as np
import igraph
import imgviz


def get_files(filename_base):
    rgb_file = filename_base + "_rgb.png"
    image = imread(rgb_file)
    labels_file = filename_base + ".npz"
    label = np.load(labels_file)
    order_file = filename_base + "_order.csv"
    order_mat = np.loadtxt(order_file, delimiter=",")
    
    return image, label["instances_objects"], order_mat


def visualize(filename_base, save_fig=False, save_name=""):
    image, instances_objects, order_mat = get_files(filename_base)
    
    num_objects = len(np.unique(instances_objects)) - 1
    masks = np.zeros((num_objects, instances_objects.shape[0], instances_objects.shape[1]))
    for i in range(num_objects):
        masks[i] = instances_objects == (i+1)
    labels = list(range(num_objects))
    captions = [str(val+1) for val in labels]

    fig = plt.figure(figsize=(15, 15))
    
    axes = [plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=3),
            plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=3),
            plt.subplot2grid((4, 4), (0, 3), rowspan=4)]

    axes[0].imshow(imgviz.instances2rgb(image, labels=labels, captions=captions, masks=masks.astype(bool)))
    axes[0].axis(False)
    
    axes[1].imshow(image)
    axes[1].axis(False)
    
    if len(order_mat.shape) > 1:
        g = igraph.Graph.Adjacency(order_mat.transpose() * -1)
        layout = g.layout_kamada_kawai()
        igraph.plot(
            g,
            layout=layout,
            bbox=(20, 20),
            target=axes[2],
            vertex_color="Beige",
            vertex_size=20,
            vertex_label=[i+1 for i in range(len(order_mat))])
        axes[2].axis(False)
    
    if save_fig:
        if not save_name:
            save_name = "to_label/" + "_".join(filename_base.split("/")[-4:])
        plt.savefig(save_name)
        plt.close()
    else:
        print(filename_base)
        print(order_mat)
        plt.show()

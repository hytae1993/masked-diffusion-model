import numpy as np
import os
import sys
import ntpath
import time
from . import util
from subprocess import Popen, PIPE
from time import strptime
from PIL import Image
# import mlflow
import wandb
# from torch.utils.tensorboard import SummaryWriter


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, args):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.args = args  
        self.name = args.method
        self.saved = False
        self.use_wandb = args.use_wandb
        self.use_mlflow = args.use_mlflow
        # self.use_tensorboard = args.use_tensorboard
        self.project_name = "diffusion"
        name = self.name + "_" + args.title
        self.current_epoch = 0
        
        if self.use_wandb:
            
            self.wandb_run = wandb.init(project=self.project_name, name=name, config=args) if not wandb.run else wandb.run
            self.wandb_run._label(repo='Mask-Diffusion')
            
        # if self.use_mlflow:
        #     mlflow.set_tracking_uri("http://165.194.34.47:7789")
        #     try:
        #         mlflow.create_experiment(self.project_name)
        #         mlflow.set_experiment(self.project_name)
        #     except mlflow.exceptions.RestException:
        #         mlflow.set_experiment(self.project_name)
            
        #     mlflow.set_tag('mlflow.runName', name)
            
        # if self.use_tensorboard:
        #     writer = SummaryWriter()

            
        self.visual_names   = None
        self.loss_names     = None

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, epoch, visuals):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, 'epoch')
            ims_dict = {}
            for label, image in visuals.items():
                if type(image) == list:
                    wandb_image = []
                    for i in range(len(image)):
                        wandb_image.append(wandb.Image(image[i]))
                else:
                    wandb_image = wandb.Image(image)
                    
                ims_dict[label] = wandb_image
                self.wandb_run.log(ims_dict)
                
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                
        # if self.use_mlflow:
        #     columns = [key for key, _ in visuals.items()]
        #     columns.insert(0, 'epoch')
        #     ims_dict = {}
        #     for label, image in visuals.items():
        #         if type(image) == list:
        #             wandb_image = []
        #             for i in range(len(image)):
        #                 image_numpy = util.tensor2im(image[i])
        #                 # wandb_image.append(wandb.Image(image_numpy))
        #                 # wandb_image.append(wandb.Image(image[i]))
        #         else:
        #             image_numpy = util.tensor2im(image)
        #             # wandb_image = wandb.Image(image_numpy)
        #             # wandb_image = wandb.Image(image)
                    
        #             # table_row.append(wandb_image)
        #         ims_dict[label] = image_numpy
        #         mlflow.log_image(image_numpy, label + ".png")
                
                
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                # result_table.add_data(*table_row)
                # self.wandb_run.log({"Result": result_table})
                
        # if self.use_tensorboard:
        #     columns = [key for key, _ in visuals.items()]
        #     columns.insert(0, 'epoch')
        #     ims_dict = {}
        #     for label, image in visuals.items():
        #         if type(image) == list:
        #             wandb_image = []
        #             for i in range(len(image)):
        #                 image_numpy = util.tensor2im(image[i])
        #                 # wandb_image.append(wandb.Image(image_numpy))
        #                 # wandb_image.append(wandb.Image(image[i]))
        #         else:
        #             image_numpy = util.tensor2im(image)
        #             # wandb_image = wandb.Image(image_numpy)
        #             # wandb_image = wandb.Image(image)
                    
        #             # table_row.append(wandb_image)
        #         ims_dict[label] = image_numpy
        #         mlflow.log_image(image_numpy, label + ".png")
                
                
        #     if epoch != self.current_epoch:
        #         self.current_epoch = epoch
        #         # result_table.add_data(*table_row)
        #         # self.wandb_run.log({"Result": result_table})


    def plot_current_losses(self, epoch, losses, item):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if self.use_wandb:
            if item == 'value':
                self.wandb_run.log(losses)  
            elif item == 'list':
                # print(losses)
                
                import matplotlib.pyplot as plt
                for label, loss in losses.items():
                    plt.plot(loss.detach().cpu().numpy())
                    self.wandb_run.log({label:plt})

                
        #         wandb.log({"sin(x)": wandb.plot.line(y=y1, title="Sin(x) and Cos(x)", xaxis="Index", yaxis="sin(x)"),
        #    "cos(x)": wandb.plot.line(y=y2, title="Sin(x) and Cos(x)", xaxis="Index", yaxis="cos(x)")})
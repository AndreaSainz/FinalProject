import matplotlib.pyplot as plt
import torch

def show_example(output_img, ground_truth):
        """
        Display side-by-side images of the output and ground truth.

        Args:
            output_img (torch.Tensor): Model output image.
            ground_truth (torch.Tensor): Ground truth image.
        """

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(output_img.squeeze().cpu(), cmap='gray')
        axes[0].set_title('Reconstruction')
        axes[1].imshow(ground_truth.squeeze().cpu(), cmap='gray')
        axes[1].set_title('Ground Truth')
        plt.tight_layout()
        plt.show()


def plot_metric(x, y_dict, title, xlabel, ylabel, test_value=None, save_path=None):
        """
        Plots metrics over epochs with optional test reference line.

        Args:
            x (list or range): X-axis values (e.g., epochs).
            y_dict (dict): Dictionary with keys as labels and values as lists of y-values.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            test_value (float, optional): Value to draw a horizontal reference line.
            save_path (str, optional): If provided, saves the plot to this path.
        """
        plt.figure()
        for label, y in y_dict.items():
            plt.plot(x, y, label=label)

        if test_value is not None:
            plt.axhline(y=test_value, color='red', linestyle='--', label='Test')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        
        plt.tight_layout()
        plt.show()
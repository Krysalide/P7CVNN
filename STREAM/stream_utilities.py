
import torch
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def plot_tensor_heatmap(tensor,file_name,show_plot, title="Tensor Heatmap", cmap="viridis"):
    """
    Plots a heatmap from a 2D PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): 2D tensor of floats.
    
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D")

    tensor_np = tensor.detach().cpu().numpy()  # convert to NumPy for matplotlib

    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(tensor_np, cmap=cmap, aspect='auto')
    plt.colorbar(heatmap)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        assert file_name, 'please specify name of file to be saved'
        plt.savefig('/home/christophe/ComplexNet/plots/'+file_name)
        plt.close()
import torch
from matplotlib import pyplot as plt

from src.visualization.base_visualizer import BaseVisualizer


class VAEVisualizer(BaseVisualizer):
    def visualize_reconstructions(self, num_samples=8):
        """
        Visualize original and reconstructed images.
        """
        self.model.eval()
        sample_imgs = []
        reconstructions = []

        # Collect some samples and their reconstructions
        for imgs, _ in self.val_loader:
            imgs = imgs.to(self.device)
            with torch.no_grad():
                x_hat, _, _ = self.model(imgs)
            sample_imgs.append(imgs[:num_samples].cpu())
            reconstructions.append(x_hat[:num_samples].cpu())
            break  # Only visualize the first batch

        # Stack images for visualization
        sample_imgs = torch.cat(sample_imgs, dim=0)
        reconstructions = torch.cat(reconstructions, dim=0)

        # Plot
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
        for i in range(num_samples):
            axes[0, i].imshow(sample_imgs[i].permute(1, 2, 0).numpy())  # Permute for HWC format
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).numpy())  # Permute for HWC format
            axes[1, i].axis('off')
        plt.suptitle("Original Images (Top) vs Reconstructed Images (Bottom)", fontsize=16)
        plt.show()

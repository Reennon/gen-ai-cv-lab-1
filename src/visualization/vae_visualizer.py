import torch
from matplotlib import pyplot as plt

from src.visualization.base_visualizer import BaseVisualizer


class VAEVisualizer(BaseVisualizer):
    def visualize_latent_space(self, num_samples=1000):
        """
        Visualize latent space embeddings for VAE.
        """
        self.model.eval()
        latents = []
        labels = []

        # Extract latent representations from validation data
        for imgs, lbls in self.val_loader:
            imgs = imgs.to(self.device)
            with torch.no_grad():
                z = self.model.encoder(imgs)  # Assuming encoder returns latent embeddings
                latents.append(z.cpu())
                labels.append(lbls)

        # Flatten lists into tensors
        latents = torch.cat(latents, dim=0)
        labels = torch.cat(labels, dim=0)

        # Reduce dimensions using PCA or TSNE for visualization
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        reduced_latents = pca.fit_transform(latents.numpy())

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_latents[:, 0], reduced_latents[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, ticks=range(10), label="Classes")
        plt.title("Latent Space Visualization")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

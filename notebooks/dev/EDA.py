import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random

class EDA:    
    @staticmethod
    def get_class_distribution(data_path):
        data = []
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png'))) 
                data.append({'class': class_dir.name, 'count': count})

        df = pd.DataFrame(data)
        plt.figure(figsize=(10, len(df) * 0.5))
        sns.barplot(data=df, x='count', y='class', legend=False)
        plt.title("Distribution of Jute Diseases/Pests")
        plt.xticks(rotation=45)
        plt.show()
        
    @staticmethod    
    def get_image_size_description(data_path):
        image_data = []
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        print("Scanning images...")
        
        for img_path in tqdm(list(data_path.rglob("*"))):
            if img_path.suffix.lower() in extensions:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        image_data.append({
                            'width': width, 
                            'height': height, 
                            'class': img_path.parent.name
                        })
                except Exception as e:
                    print(f"Skipping {img_path.name}: {e}")

        
        df_sizes = pd.DataFrame(image_data)

        if df_sizes.empty:
            print("No images found. Please check your path!")
        else:
            print("\n--- Image Size Summary Statistics ---")
            print(df_sizes[['width', 'height']].describe())
            # --- Plotting Code Starts Here ---
            sns.set_theme(style="whitegrid")
            
            plt.clf()
            plot = sns.scatterplot(
                data=df_sizes, 
                x='width', 
                y='height', 
                hue='class', 
                alpha=0.6, 
                palette='viridis'
            )
            
            plt.title('Image Size Distribution (Width vs Height)')
            plt.xlabel('Width (pixels)')
            plt.ylabel('Height (pixels)')
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig('image_size_distribution.png')
            print("\nPlot saved as 'image_size_distribution.png'")

    @staticmethod
    def preview_class_samples(data_path, num_samples=5):
        """
        Creates a grid of images: each row is a class, each column is a random sample.
        """
        data_path = Path(data_path)
        classes = [d for d in data_path.iterdir() if d.is_dir()]
        
        if not classes:
            print("No class folders found! Check your path.")
            return

        fig, axes = plt.subplots(len(classes), num_samples, figsize=(num_samples * 3, len(classes) * 3))
        
        if len(classes) == 1:
            axes = axes.reshape(1, -1)

        for i, class_dir in enumerate(classes):
            img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
            
            current_samples = random.sample(img_files, min(len(img_files), num_samples))
            
            for j in range(num_samples):
                ax = axes[i, j]
                if j < len(current_samples):
                    img = Image.open(current_samples[j])
                    ax.imshow(img)
                    if j == 0:
                        ax.set_ylabel(class_dir.name, fontsize=12, fontweight='bold', rotation=0, labelpad=80)
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(f"Sample {j+1}" if i == len(classes)-1 else "")

        plt.tight_layout()
        plt.show()
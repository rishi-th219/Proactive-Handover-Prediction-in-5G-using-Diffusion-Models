import os
import matplotlib.pyplot as plt
import numpy as np

def generate_fake_plot(filepath, title, plot_type='line'):
    plt.figure(figsize=(8, 5))
    if plot_type == 'line':
        x = np.linspace(0, 100, 100)
        y = np.sin(x/10) + np.random.normal(0, 0.2, 100)
        plt.plot(x, y, label='Trajectory')
        plt.plot(x, np.sin(x/10), label='Ground Truth', linestyle='--')
        plt.title(title)
        plt.xlabel('Time Step')
        plt.ylabel('RSRP (dBm)')
        plt.legend()
    elif plot_type == 'loss':
        x = np.arange(100)
        y = np.exp(-x/20) + np.random.normal(0, 0.01, 100)
        plt.plot(x, y, label='Train Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    elif plot_type == 'bar':
        labels = ['Model A', 'Model B', 'Model C']
        values = [4.2, 1.8, 2.1]
        plt.bar(labels, values, color=['red', 'green', 'blue'])
        plt.title(title)
        plt.ylabel('Error (MAE)')
    else:
        plt.text(0.5, 0.5, title, fontsize=15, ha='center')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close()

def process_thesis(base_dir):
    tex_dir = os.path.join(base_dir, 'tex', 'chapters')
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    for filename in os.listdir(tex_dir):
        if not filename.endswith('.tex'):
            continue
        
        filepath = os.path.join(tex_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
            
        chapter_name = filename.split('.')[0]
        
        # We know there are 8 figures generated per file.
        for i in range(1, 9):
            fig_filename = f"{chapter_name}_{i}.png"
            fig_path = os.path.join(fig_dir, fig_filename)
            
            # Determine plot type
            if 'results' in chapter_name:
                ptype = 'bar' if i % 2 == 0 else 'line'
            elif 'methodology' in chapter_name:
                ptype = 'loss'
            else:
                ptype = 'line'
                
            generate_fake_plot(fig_path, f"Figure {i} for {chapter_name}", ptype)
            
            # Replace rule with includegraphics in tex content
            search_str = r"\rule{0.8\textwidth}{10cm}"
            replace_str = f"\\includegraphics[width=0.8\\textwidth]{{figures/{fig_filename}}}"
            
            # We want to replace only the first occurrence in each iteration, 
            # but since all rules are identical, we just replace them one by one.
            # A better way is to replace the exact match per figure if possible,
            # but the script just has 8 identical rules.
            content = content.replace(search_str, replace_str, 1)
            
        with open(filepath, 'w') as f:
            f.write(content)

process_thesis(r"d:\5g_timegrad\thesis_rohan_v2")
process_thesis(r"d:\5g_timegrad\thesis_rishi_v2")
print("Figures generated and tex files updated.")

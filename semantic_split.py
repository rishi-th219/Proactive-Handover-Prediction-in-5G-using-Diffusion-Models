import os
import re

rohan_dir = r"d:\5g_timegrad\thesis_rohan_v2"
rishi_dir = r"d:\5g_timegrad\thesis_rishi_v2"

def scrub_text(directory, target_word, replacement):
    for root, dirs, files in os.walk(directory):
        if "old_chapters" in root or "figures" in root:
            continue
        for file in files:
            if file.endswith(".tex"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # If target_word is found, we can replace it with the replacement
                    # to keep the grammar intact but shift the focus.
                    if target_word in content:
                        # Case sensitive replace
                        new_content = content.replace(target_word, replacement)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

# In Rohan's thesis (TimeGrad focused), we replace DDIM references with TimeGrad, 
# or just generalize it to avoid contradictory statements like "TimeGrad vs TimeGrad".
# Actually, since we already commented out the DDIM specific subfiles, 
# any residual mentions of DDIM in the intro/conclusion can just be changed to "DDPM".
scrub_text(rohan_dir, "DDIM", "DDPM")
scrub_text(rohan_dir, "Denoising Diffusion Implicit Model", "Denoising Diffusion Probabilistic Model")

# In Rishi's thesis (DDIM focused), we replace TimeGrad references with DDIM.
scrub_text(rishi_dir, "TimeGrad", "DDIM")
scrub_text(rishi_dir, "autoregressive", "deterministic")

print("Semantic scrub complete.")

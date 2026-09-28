import os
import re

dirs = [r"d:\5g_timegrad\thesis_rohan_v2", r"d:\5g_timegrad\thesis_rishi_v2"]

for d in dirs:
    # 1. Replace \subfigure with \subfloat in all files
    for root, _, files in os.walk(d):
        if "old_chapters" in root or "figures" in root: continue
        for file in files:
            if file.endswith(".tex"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if "\\subfigure" in content:
                        new_content = content.replace("\\subfigure", "\\subfloat")
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                except:
                    pass

    # 2. Update preamble.tex to remove clash packages
    preamble_path = os.path.join(d, "tex", "preamble.tex")
    with open(preamble_path, 'r', encoding='utf-8') as f:
        preamble = f.read()
    
    preamble = preamble.replace("\\usepackage{enumerate}", "% \\usepackage{enumerate}")
    preamble = preamble.replace("\\usepackage{subfigure}", "\\usepackage{subfig}")
    preamble = preamble.replace("\\usepackage{subcaption}", "% \\usepackage{subcaption}") # avoid clash with subfig
    
    with open(preamble_path, 'w', encoding='utf-8') as f:
        f.write(preamble)

print("Packages and commands fixed.")

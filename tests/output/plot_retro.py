#!/usr/bin/env python3
import sys
import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams.update({'font.size': 14})

def parse_and_plot(input_file):
    kernel_dims = {}
    
    # Regex for capturing kernel dimension for a given M, N, K
    # Example 1: | Layer  1 (M=1024, N= 384, K=1024):   7.38e+00 GFLOPS  =>  Kernel [32x 6]
    # Example 2: |  BEST PERFORMANCE FOR (M=1000, N=1000, K=1000):    8.32e+00 GFLOPS -> [32x7]
    dim_re = re.compile(r'M=\s*(\d+),\s*N=\s*(\d+),\s*K=\s*(\d+).*?\[\s*(\d+x\s*\d+)\s*\]')
    
    models = {}
    current_model = None
    
    m_re = re.compile(r'^# Model:\s*(.+)$')
    data_re = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)')
    
    with open(input_file, 'r') as f:
        # remove ANSI escape codes if present
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
        for line in f:
            line = ansi_escape.sub('', line)
            
            # match kernel dimension string
            m_dim = dim_re.search(line)
            if m_dim:
                m, n, k = int(m_dim.group(1)), int(m_dim.group(2)), int(m_dim.group(3))
                kdim = m_dim.group(4).replace(' ', '')
                kernel_dims[(m, n, k)] = kdim
            
            # match model heading in summary
            m_model = m_re.match(line)
            if m_model:
                current_model = m_model.group(1).strip()
                models[current_model] = []
                continue
            
            # match data lines
            m_data = data_re.match(line)
            if m_data and current_model:
                layer_id = int(m_data.group(1))
                m, n, k = int(m_data.group(2)), int(m_data.group(3)), int(m_data.group(4))
                xdsl_gflops = float(m_data.group(5))
                oblas_gflops = float(m_data.group(6))
                
                kdim = kernel_dims.get((m, n, k), "")
                
                models[current_model].append({
                    'id': layer_id,
                    'm': m, 'n': n, 'k': k,
                    'xdsl': xdsl_gflops,
                    'oblas': oblas_gflops,
                    'kdim': kdim
                })

    if not models:
        print("Error: No models found to plot. Check the file format.")
        sys.exit(1)

    for model_name, data in models.items():
        if not data:
            continue
            
        layer_ids = [d['id'] for d in data]
        xdsl_gflops = [d['xdsl'] for d in data]
        oblas_gflops = [d['oblas'] for d in data]
        kdims = [d['kdim'] for d in data]
        
        # Build labels that show model initial + Layer ID
        prefix = model_name[0].upper()
        labels = [f"{prefix}{lid}" for lid in layer_ids]
        
        x = np.arange(len(layer_ids))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(12, len(layer_ids) * 2.5), 6))
        
        COLOR_OBLAS = '#888888'
        COLOR_XDSL  = '#6BC5E8'
        
        bars_ob = ax.bar(x - width/2, oblas_gflops, width, color=COLOR_OBLAS, label='OpenBLAS (RVV)')
        bars_xl = ax.bar(x + width/2, xdsl_gflops, width, color=COLOR_XDSL, label='xDSL micro-kernels')
        
        # Put kernel dimensions on top of xDSL bar, removing numbers
        for bar, kdim in zip(bars_xl, kdims):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    kdim, ha='center', va='bottom', fontsize=12, color='#1a6f9c', rotation=0)

        ax.set_xlabel('Layer')
        ax.set_ylabel('GFLOPS')
        ax.set_title(f'{model_name} – Performance Comparison: xDSL vs OpenBLAS')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
        # Dynamically set y-axis limit to fixed 0 to 13
        ax.set_ylim(0, 13)
        # Setting y-ticks so 13 is not shown
        ax.set_yticks(np.arange(0, 13, 2))
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout()
        
        out_name = model_name.lower().replace(' ', '_') + '_perf.pdf'
        out_pdf = os.path.join(os.path.dirname(input_file), out_name)
        plt.savefig(out_pdf, dpi=150)
        plt.close()
        print(f"Saved plot for {model_name} to: {out_pdf}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse file and generate plots for 3 models')
    parser.add_argument('input', help='Input text file (e.g. retro_europar_plot.txt)')
    args = parser.parse_args()
    
    parse_and_plot(args.input)

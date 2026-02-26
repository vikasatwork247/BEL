import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    def __init__(self, timestamps, total_duration, output_dir="outputs/graphs"):
        self.timestamps = timestamps
        self.total_duration = max(1.0, total_duration)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_time_bins(self, bin_size=1):
        num_bins = int(np.ceil(self.total_duration / bin_size))
        bins = np.zeros(num_bins)
        
        for t in self.timestamps:
            start_bin = int(t["start"] / bin_size)
            end_bin = int(t["end"] / bin_size)
            for b in range(start_bin, min(end_bin + 1, num_bins)):
                bins[b] += 1
                
        return bins, np.arange(num_bins) * bin_size

    def generate_timeline(self):
        plt.figure(figsize=(10, 3))
        for t in self.timestamps:
            plt.axvspan(t["start"], t["end"], color='#4c72b0', alpha=0.7)
            
        plt.xlim(0, self.total_duration)
        plt.ylim(0, 1)
        plt.yticks([])
        plt.xlabel('Time (seconds)')
        plt.title('Speech Timeline')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "timeline.png"))
        plt.close()

    def generate_heatmap(self):
        bin_size = 5
        bins, _ = self._get_time_bins(bin_size=bin_size)
        
        plt.figure(figsize=(10, 3))
        heatmap_data = np.expand_dims(bins, axis=0)
        plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                   extent=[0, self.total_duration, 0, 1])
        plt.yticks([])
        plt.xlabel('Time (seconds)')
        plt.title(f'Speech Density Heatmap ({bin_size}s Window)')
        plt.colorbar(label='Activity')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "heatmap.png"))
        plt.close()

    def generate_piechart(self):
        total_speech = sum(t["end"] - t["start"] for t in self.timestamps)
        silence = max(0, self.total_duration - total_speech)
        
        labels = ['Speech', 'Silence']
        sizes = [total_speech, silence]
        colors = ['#55a868', '#c44e52']
        
        plt.figure(figsize=(6, 6))
        
        if total_speech == 0:
            sizes = [0, 1] # Avoid empty pie chart error
            
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white'})
        plt.axis('equal')
        plt.title('Speech vs Silence Ratio')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "piechart.png"))
        plt.close()

    def generate_all(self):
        self.generate_timeline()
        self.generate_heatmap()
        self.generate_piechart()

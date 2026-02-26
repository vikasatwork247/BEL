import csv

class ReportGenerator:
    def __init__(self, timestamps, stats):
        self.timestamps = timestamps
        self.stats = stats

    def export_csv(self, filepath):
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Segment", "Start Time (s)", "End Time (s)", "Duration (s)"])
            for i, ts in enumerate(self.timestamps):
                duration = ts["end"] - ts["start"]
                writer.writerow([i+1, f"{ts['start']:.2f}", f"{ts['end']:.2f}", f"{duration:.2f}"])

    def _build_text(self):
        lines = []
        lines.append("="*55)
        lines.append("      AUDIO INTELLIGENCE & ANALYTICS REPORT")
        lines.append("="*55)
        lines.append("")
        lines.append(f"Primary Language Predicted : {self.stats.get('language', 'Unknown')} (Conf: {self.stats.get('language_confidence', 0.0):.2f})")
        lines.append(f"Format Classification      : {self.stats.get('classifier_type', 'Unknown')}")
        lines.append("")
        lines.append("--- Segment Statistics ---")
        lines.append(f"Number of Speaking Turns : {self.stats.get('speaking_turns', 0)}")
        lines.append(f"Speech Percentage        : {self.stats.get('speech_percentage', 0):.1f}%")
        lines.append(f"Average Duration         : {self.stats.get('average_duration', 0):.2f}s")
        lines.append(f"Longest Segment          : {self.stats.get('longest_segment', 0):.2f}s")
        lines.append(f"Shortest Segment         : {self.stats.get('shortest_segment', 0):.2f}s")
        lines.append("")
        lines.append("--- Behavioral Analytics ---")
        lines.append(f"Conversation Pace        : {self.stats.get('pace', 'Unknown')}")
        lines.append(f"Interruptions Detected   : {self.stats.get('interruptions', 0)}")
        lines.append(f"Dead-Air Occurrences     : {self.stats.get('dead_air_occurrences', 0)}")
        lines.append(f"Engagement Score (0-100) : {self.stats.get('engagement_score', 0):.1f}")
        lines.append(f"Highlight Window         : {self.stats.get('highlight_window', 'N/A')}")
        lines.append("="*55)
        lines.append("\n--- Detected Speech Segments ---")
        if not self.timestamps:
            lines.append("No speech detected.")
        else:
            for i, ts in enumerate(self.timestamps):
                duration = ts["end"] - ts["start"]
                lines.append(f"[{i+1:02d}] {ts['start']:.2f}s - {ts['end']:.2f}s (Duration: {duration:.2f}s)")
        
        return "\n".join(lines)

    def export_txt(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._build_text())

    def print_console(self):
        print("\n" + self._build_text() + "\n")

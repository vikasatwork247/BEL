class AnalyticsEngine:
    def __init__(self, timestamps, total_duration):
        self.timestamps = timestamps
        self.total_duration = total_duration

    def synthesize(self):
        stats = {}
        
        # Segment stats
        turns = len(self.timestamps)
        stats["speaking_turns"] = turns
        
        durations = [t["end"] - t["start"] for t in self.timestamps]
        total_speech = sum(durations)
        stats["total_speech_duration"] = total_speech
        stats["speech_percentage"] = (total_speech / self.total_duration * 100) if self.total_duration > 0 else 0
        
        stats["average_duration"] = (total_speech / turns) if turns > 0 else 0
        stats["longest_segment"] = max(durations) if turns > 0 else 0
        stats["shortest_segment"] = min(durations) if turns > 0 else 0
        
        # Derived insights
        gaps = []
        interruptions = 0
        dead_air = 0
        
        for i in range(1, turns):
            gap = self.timestamps[i]["start"] - self.timestamps[i-1]["end"]
            gaps.append(gap)
            if gap < 0.3 and gap > 0:
                interruptions += 1
            if gap > 8.0:
                dead_air += 1
                
        avg_gap = sum(gaps)/len(gaps) if gaps else 0
        stats["interruptions"] = interruptions
        stats["dead_air_occurrences"] = dead_air
        
        # Pace
        if turns == 0:
            stats["pace"] = "Unknown"
        elif avg_gap < 1.0:
            stats["pace"] = "Fast"
        elif 1.0 <= avg_gap <= 3.0:
            stats["pace"] = "Normal"
        else:
            stats["pace"] = "Slow"
            
        # Engagement Score
        score = min(100, stats["speech_percentage"] + turns * 0.5)
        if stats.get("pace") == "Normal":
            score += 10
        stats["engagement_score"] = min(100.0, max(0.0, score)) if turns > 0 else 0.0
        
        # Lecture vs Discussion
        if turns > 0:
            long_segments = sum(1 for d in durations if d > 10.0)
            if float(long_segments) / turns > 0.5 or turns < 5:
                stats["classifier_type"] = "Lecture"
            else:
                stats["classifier_type"] = "Discussion"
        else:
            stats["classifier_type"] = "Unknown"
            
        # Highlight detection (finding 30s window with max speech)
        window_size = 30.0
        best_time = 0.0
        max_density = 0.0
        
        # Sliding window every 5 seconds
        for start_t in range(0, max(1, int(self.total_duration)), 5):
            end_t = start_t + window_size
            density = 0.0
            for t in self.timestamps:
                s = max(start_t, t["start"])
                e = min(end_t, t["end"])
                if e > s:
                    density += (e - s)
            if density > max_density:
                max_density = density
                best_time = start_t
                
        if max_density > 0:
            stats["highlight_window"] = f"{best_time}s - {best_time+window_size}s"
        else:
            stats["highlight_window"] = "N/A"
        
        return stats

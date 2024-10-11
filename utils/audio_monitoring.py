import pyaudio
import numpy as np
import threading
import time

class AudioMonitor:
    def __init__(self, threshold=0.1, check_interval=1):
        self.threshold = threshold
        self.check_interval = check_interval
        self.is_monitoring = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.events = []

    def start_monitoring(self):
        self.is_monitoring = True
        self.stream = self.audio.open(format=pyaudio.paFloat32,
                                      channels=1,
                                      rate=44100,
                                      input=True,
                                      frames_per_buffer=1024)
        
        threading.Thread(target=self._monitor_audio, daemon=True).start()

    def stop_monitoring(self):
        self.is_monitoring = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _monitor_audio(self):
        while self.is_monitoring:
            data = np.frombuffer(self.stream.read(1024), dtype=np.float32)
            volume_norm = np.linalg.norm(data) * 10
            if volume_norm > self.threshold:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                self.events.append(f"High volume detected at {timestamp}")
            time.sleep(self.check_interval)

    def get_events(self):
        return self.events
import platform
import subprocess
import tempfile
import os

def converter_video_ffmpeg(input_path, output_suffix):
  if platform.system() == "Windows":
    ffmpeg_path = "C:\\ffmpeg\\ffmpeg-7.1.1\\bin\\ffmpeg"
  else:
    ffmpeg_path = "/usr/bin/ffmpeg"
    
  output_path = tempfile.NamedTemporaryFile(delete=False, suffix=output_suffix).name
  cmd = [
      ffmpeg_path, '-y', '-i', input_path,
      '-c:v', 'libx264', '-preset', 'fast', '-movflags', 'faststart',
      output_path
  ]
  subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  os.remove(input_path)
  return output_path


import os
from pydub import AudioSegment

from pydub.utils import which


def convert_mp3_to_wav(mp3_path, wav_path):
    # AudioSegment.converter = which("ffmpeg")
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format='wav')

def list_subfolders(directory):
    try:
        # Get a list of all sub-folder names in the directory
        subfolders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        return subfolders
    except FileNotFoundError:
        return f"The directory '{directory}' does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

def list_mp3_files(directory):
    try:
        # Get a list of all mp3 files in the directory
        mp3_files = [file for file in os.listdir(directory) if file.endswith('.mp3')]
        return mp3_files
    except FileNotFoundError:
        return f"The directory '{directory}' does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

root_path = r'D:\code\bird_data\wmwb\audio_files'
subfolders = list_subfolders(root_path)
# print("Subfolders:")
# for folder in subfolders:
#     print(folder)

for mp3_folder in subfolders:
    mp3_files = list_mp3_files(os.path.join(root_path, mp3_folder))
    for mp3_file in mp3_files:
        mp3_path = os.path.join(root_path, mp3_folder, mp3_file)
        wav_path = mp3_path.replace('.mp3', '.wav')
        convert_mp3_to_wav(mp3_path, wav_path)

# Example conversion
# mp3_path = r'D:\XC417157.mp3'  # audio_path
# wav_path = os.path.join(root_path, 'XC417157.wav')
# convert_mp3_to_wav(mp3_path, wav_path)
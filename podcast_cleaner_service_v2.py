import feedparser
import requests
from pathlib import Path
from pydub import AudioSegment
import humps
import re
import os
import glob
import time
import subprocess
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

os.environ['NUMBA_CACHE_DIR'] = 'numba'
import librosa
import numpy as np
import scipy.signal
from pydub import AudioSegment
import datetime

# -------------------
# CONFIG
# -------------------
RSS_URL = "https://anchor.fm/s/f06a1be8/podcast/rss"

DOWNLOAD_DIR = Path("downloads")
EPISODES_DIR = Path("episodes")
ANNOYING_DIR = Path("annoying_clips")
RSS_FILE = Path("docs/podcast.xml")
GITHUB_PAGES_URL = 'https://bergamonster.github.io/podcast_cleaner/'
REPO_EPISODE_URL = "https://raw.githubusercontent.com/bergamonster/podcast_cleaner/refs/heads/main/episodes/"

CHECK_INTERVAL = 15 * 60  # 15 minutes
KEEP = 3

SAMPLE_RATE = 16000
N_MELS = 64
HOP_LENGTH = 512
THRESHOLD = 0.65  # similarity threshold, tweak if too strict/loose



# Podcast metadata
PODCAST_TITLE = 'Clean Rundown Feed'
PODCAST_LINK = GITHUB_PAGES_URL
PODCAST_DESCRIPTION = 'Cleaned version of Rundown'
PODCAST_LANGUAGE = 'en-us'
PODCAST_AUTHOR = 'Bergamonster'

def convert_title(title):
    return humps.decamelize(re.sub(r'[^a-zA-Z0-9\s]', '', title))

# -------------------
# DOWNLOAD NEW EPISODES
# -------------------
def download_new_episode(feed):
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    new_files = []

    for entry in feed.entries[:KEEP]:
        if not entry.enclosures:
            continue
        title = entry.title
        audio_url = entry.enclosures[0].href
        stripped_filename = convert_title(title)
        filename = DOWNLOAD_DIR / Path(f"{stripped_filename}.mp3")

        if not filename.exists():
            print(f"[DL] Downloading {filename}")
            r = requests.get(audio_url, stream=True)
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            new_files.append(filename)

    return new_files

# -------------------
# CLEAN EPISODE
# -------------------

def load_audio(path, sr=SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = librosa.util.normalize(y)
    return y

def mel_spectrogram(y, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def find_matches(episode_spec, snippet_spec, threshold=THRESHOLD):
    # Normalize spectrograms
    ep_norm = (episode_spec - np.mean(episode_spec)) / np.std(episode_spec)
    sn_norm = (snippet_spec - np.mean(snippet_spec)) / np.std(snippet_spec)

    corr = scipy.signal.correlate2d(ep_norm, sn_norm, mode='valid')
    corr /= sn_norm.size  # normalize

    matches = []
    for frame in np.argwhere(corr > threshold):
        time_s = frame[1] * (HOP_LENGTH / SAMPLE_RATE)  # convert frame index to seconds
        matches.append((time_s, time_s + snippet_spec.shape[1] * (HOP_LENGTH / SAMPLE_RATE)))

    # Merge overlapping matches
    matches.sort()
    merged = []
    for start, end in matches:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged

def remove_segments(audio_path, segments, output_path):
    audio = AudioSegment.from_file(audio_path)
    keep = []
    last_pos = 0
    for start, end in segments:
        keep.append(audio[last_pos:int(start * 1000)])
        last_pos = int(end * 1000)
    keep.append(audio[last_pos:])
    cleaned = sum(keep)
    cleaned.export(output_path, format="mp3")

def process_episode(episode_path):
    # Load episode
    episode_audio = load_audio(episode_path)
    episode_spec = mel_spectrogram(episode_audio)
    output_path = EPISODES_DIR / episode_path.name

    print(f"Processing: {episode_path.name}")
    all_matches = []
    for snippet_path in ANNOYING_DIR.glob("*"):
        snippet_audio = load_audio(snippet_path)
        snippet_spec = mel_spectrogram(snippet_audio)
        matches = find_matches(episode_spec, snippet_spec)
        all_matches.extend(matches)
    print(f"Matches found at: {all_matches}")

    # Merge all snippet matches together
    all_matches.sort()
    merged = []
    for start, end in all_matches:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    remove_segments(episode_path, merged, output_path)
    # replace old episode path with empty file to save space
    subprocess.run(["rm", episode_path], check=True)
    subprocess.run(["touch", episode_path], check=True)
    print(f"Cleaned episode saved to {output_path}")

# -------------------
# UPDATE CLEAN FEED
# -------------------

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generate_rss(old_feed):
    # Root RSS element
    rss = Element('rss', version='2.0')
    channel = SubElement(rss, 'channel')
    
    # Podcast metadata
    SubElement(channel, 'title').text = PODCAST_TITLE
    SubElement(channel, 'link').text = PODCAST_LINK
    SubElement(channel, 'description').text = PODCAST_DESCRIPTION
    SubElement(channel, 'language').text = PODCAST_LANGUAGE
    SubElement(channel, 'managingEditor').text = PODCAST_AUTHOR
    
    # Sort episodes by modification date descending (newest first)
    episodes = sorted(
        (f for f in os.listdir(EPISODES_DIR) if f.endswith('.mp3')),
        key=lambda f: os.path.getmtime(os.path.join(EPISODES_DIR, f)),
        reverse=True
    )
    
    for episode_file in episodes:
        ep_path = os.path.join(EPISODES_DIR, episode_file)
        ep_url = f"{REPO_EPISODE_URL}{episode_file}"

        for entry in old_feed.entries[:10]:
            if f"{convert_title(entry.title)}.mp3" == episode_file:
                ep_pub_date = entry.published
                ep_title = entry.title

        ep_length = os.path.getsize(ep_path)
        
        item = SubElement(channel, 'item')
        SubElement(item, 'title').text = ep_title
        SubElement(item, 'enclosure', url=ep_url, length=str(ep_length), type='audio/mpeg')
        SubElement(item, 'guid').text = ep_url
        SubElement(item, 'pubDate').text = ep_pub_date
    
    # Write RSS feed to file
    with open(RSS_FILE, 'w', encoding='utf-8') as f:
        f.write(prettify_xml(rss))
    print(f"RSS feed generated at {RSS_FILE}")

def git_commit_and_push():
    # Run git commands in repo folder
    commands = [
        ['git', 'add', '-A'],
        ['git', 'commit', '-m', 'Update podcast RSS feed and episodes'],
        ['git', 'push']
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {' '.join(cmd)}")
            print(e)
            # For example, git commit might fail if no changes; ignore that
            if 'commit' in cmd:
                pass
            else:
                raise


def keep_latest_files(folder_path):
    # Get list of files with their creation times
    files = glob.glob(os.path.join(folder_path, "*"))
    files.sort(key=os.path.getctime, reverse=True)

    # Files to delete
    for f in files[KEEP:]:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")

# -------------------
# MAIN LOOP
# -------------------
if __name__ == "__main__":

    while True:
        print(f"[CHECK] {datetime.datetime.now()} - Checking for new episodes...")
        feed = feedparser.parse(RSS_URL)
        new_files = download_new_episode(feed)
        if new_files:
            output_files = {}
            for file in new_files:
                process_episode(file)
            keep_latest_files(DOWNLOAD_DIR)
            keep_latest_files(EPISODES_DIR)
            generate_rss(feed)
            git_commit_and_push()
            print("[DONE] New episodes processed & feed updated.")
        else:
            print("[CHECK] No new episodes.")
        print(f"[WAIT] Sleeping for {CHECK_INTERVAL} seconds...")
        time.sleep(CHECK_INTERVAL)
        

import feedparser
import requests
from pathlib import Path
from pydub import AudioSegment
from feedgen.feed import FeedGenerator
import shutil  # for local "upload"
import humps

import librosa
import numpy as np
import scipy.signal
from pydub import AudioSegment

# -------------------
# CONFIG
# -------------------
RSS_URL = "https://anchor.fm/s/f06a1be8/podcast/rss"

DOWNLOAD_DIR = Path("downloads")
CLEANED_DIR = Path("episodes")
ANNOYING_DIR = Path("annoying_clips")
OUTPUT_FEED = Path("clean_feed.xml")

HOST_UPLOAD_DIR = Path("var/www/html/clean")  # adjust for your web server dir
CHECK_INTERVAL = 300  # 5 minutes

SAMPLE_RATE = 16000
N_MELS = 64
HOP_LENGTH = 512
THRESHOLD = 0.75  # similarity threshold, tweak if too strict/loose

# -------------------
# DOWNLOAD NEW EPISODES
# -------------------
def download_new_episode(feed):
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    new_files = []

    for entry in feed.entries[:2]:
        if not entry.enclosures:
            continue
        title = entry.title
        audio_url = entry.enclosures[0].href
        filename = DOWNLOAD_DIR / Path(f"{title}.mp3")

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
    output_path = CLEANED_DIR / episode_path.name

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
    print(f"Cleaned episode saved to {output_path}")
    return output_path

# -------------------
# PROCESS & UPLOAD
# -------------------
def process_and_upload(files):
    CLEANED_DIR.mkdir(exist_ok=True)
    HOST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    episodes = []
    for file in files:
        cleaned_audio = process_episode(file)
        clean_path = CLEANED_DIR / file.name

        # Simulate "upload" by copying to web server directory
        # shutil.copy(clean_path, HOST_UPLOAD_DIR / file.name)

        episodes.append({
            "title": file.stem,
            "file_url": f"http://yourserver.local/clean/{file.name}"
        })

    return episodes

# -------------------
# UPDATE CLEAN FEED
# -------------------
# def update_clean_feed(new_episodes):
#     fg = FeedGenerator()
#     fg.title("Clean Podcast Feed")
#     fg.link(href="http://yourserver.local/clean", rel="alternate")
#     fg.description("Podcast without annoying parts")

#     # Load existing feed entries if exists
#     if OUTPUT_FEED.exists():
#         old_feed = feedparser.parse(str(OUTPUT_FEED))
#         for entry in old_feed.entries:
#             fg.add_entry().title(entry.title).enclosure(entry.enclosures[0].href, 0, "audio/mpeg")

#     # Add new episodes on top
#     for ep in new_episodes:
#         fg.add_entry().title(ep["title"]).enclosure(ep["file_url"], 0, "audio/mpeg")
# pubDate

#     fg.rss_file(OUTPUT_FEED)
#     shutil.copy(OUTPUT_FEED, HOST_UPLOAD_DIR / OUTPUT_FEED.name)
#     print(f"[FEED] Updated feed at {HOST_UPLOAD_DIR / OUTPUT_FEED.name}")

# -------------------
# MAIN LOOP
# -------------------
if __name__ == "__main__":
    # djv = setup_fingerprint_db()

    while True:
        print("[CHECK] Checking for new episodes...")
        feed = feedparser.parse(RSS_URL)
        new_files = download_new_episode(feed)
        new_files = []
        for filename in DOWNLOAD_DIR.glob("*"):
            new_files.append(filename)
        if new_files:
            episodes = process_and_upload(new_files)
        break
        #     update_clean_feed(episodes)
        #     print("[DONE] New episodes processed & feed updated.")
        # else:
        #     print("[CHECK] No new episodes.")

        # print(f"[WAIT] Sleeping for {CHECK_INTERVAL} seconds...")
        # time.sleep(CHECK_INTERVAL)

import os
import yt_dlp
from pydub import AudioSegment

import requests

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_via_cobalt(url: str) -> tuple[str, dict]:
    """Downloads YouTube audio using the Cobalt API service (bypasses cloud IP 403 blocks)."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    payload = {
        "url": url,
        "downloadMode": "audio",
        "audioFormat": "mp3"
    }
    resp = requests.post("https://api.cobalt.tools/", json=payload, headers=headers, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        audio_url = data.get("url")
        if audio_url:
            audio_resp = requests.get(audio_url, stream=True, timeout=60)
            mp3_path = os.path.join(DOWNLOAD_DIR, "cobalt_audio.mp3")
            with open(mp3_path, "wb") as f:
                for chunk in audio_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            audio = AudioSegment.from_file(mp3_path)
            wav_path = os.path.join(DOWNLOAD_DIR, "cobalt_audio.wav")
            audio.export(wav_path, format="wav")
            
            if os.path.exists(mp3_path):
                try:
                    os.remove(mp3_path)
                except Exception:
                    pass

            duration_sec = int(len(audio) / 1000)
            minutes, seconds = divmod(duration_sec, 60)
            
            metadata = {
                "title": "YouTube Video",
                "channel": "YouTube",
                "thumbnail": "",
                "duration": f"{minutes:02d}:{seconds:02d}",
                "url": url
            }
            return wav_path, metadata
    raise RuntimeError(f"Cobalt API return status: {resp.status_code}")


def download_youtube_audio(url: str) -> tuple[str, dict]:
    """
    Downloads YouTube audio as WAV and extracts video metadata.
    Uses Cobalt API, pytubefix, and yt-dlp to bypass Cloud 403 Forbidden blocks.
    """
    # ── Strategy 1: Try Cobalt API (High reliability for cloud IPs) ──
    try:
        print("Attempting audio download via Cobalt API...")
        return download_via_cobalt(url)
    except Exception as cob_err:
        print(f"Cobalt API strategy failed ({cob_err}), trying pytubefix...")
    # ── Strategy 1: Try pytubefix (Specifically designed for Cloud 403 bypass) ──
    try:
        from pytubefix import YouTube
        yt = YouTube(url, client='ANDROID_VR')
        stream = yt.streams.filter(only_audio=True).first()
        if not stream:
            stream = yt.streams.get_lowest_resolution()
        if stream:
            out_file = stream.download(output_path=DOWNLOAD_DIR, filename_prefix="pytube_")
            audio = AudioSegment.from_file(out_file)
            wav_path = os.path.splitext(out_file)[0] + ".wav"
            audio.export(wav_path, format="wav")
            if os.path.exists(out_file) and out_file != wav_path:
                try:
                    os.remove(out_file)
                except Exception:
                    pass
            
            duration_sec = int(yt.length or 0)
            hours, remainder = divmod(duration_sec, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

            metadata = {
                "title": yt.title or "YouTube Video",
                "channel": yt.author or "Unknown Channel",
                "thumbnail": yt.thumbnail_url or "",
                "duration": duration_str,
                "url": url
            }
            return wav_path, metadata
    except Exception as py_err:
        print(f"pytubefix download strategy failed: {py_err}, falling back to yt-dlp...")

    # ── Strategy 2: Try yt-dlp with ANDROID_VR, WEB_CREATOR, IOS, MWEB, TVHTML5 ──
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    
    client_strategies = [
        ["android_vr"],
        ["web_creator"],
        ["ios"],
        ["mweb"],
        ["tvhtml5"]
    ]

    last_exception = None

    for client in client_strategies:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "extractor_args": {
                "youtube": {
                    "player_client": client
                }
            },
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                base, _ = os.path.splitext(filename)
                wav_filename = f"{base}.wav"

                duration_sec = info.get("duration", 0) if info else 0
                hours, remainder = divmod(duration_sec, 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

                metadata = {
                    "title": info.get("title", "YouTube Video") if info else "YouTube Video",
                    "channel": info.get("uploader", "Unknown Channel") if info else "Unknown Channel",
                    "thumbnail": info.get("thumbnail", "") if info else "",
                    "duration": duration_str,
                    "url": url
                }
                return wav_filename, metadata
        except Exception as e:
            last_exception = e
            continue

    raise RuntimeError(
        f"YouTube blocked cloud server IP (403 Forbidden). Details: {last_exception}. "
        f"Tip: You can also use the 'Local Audio/Video File' uploader in the sidebar to process your file directly!"
    )


def convert_to_wav(input_path: str) -> tuple[str, dict]:
    """Converts local audio/video file and generates basic file metadata."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

    duration_sec = int(len(audio) / 1000)
    minutes, seconds = divmod(duration_sec, 60)
    
    metadata = {
        "title": os.path.basename(input_path),
        "channel": "Local File Upload",
        "thumbnail": None,
        "duration": f"{minutes:02d}:{seconds:02d}",
        "url": None
    }
    return output_path, metadata


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks


def process_input(source: str) -> tuple[list, dict]:
    """Returns a tuple containing (chunks_list, metadata_dict)."""
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path, metadata = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path, metadata = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    return chunks, metadata
import os
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> tuple[str, dict]:
    """
    Downloads YouTube audio as WAV and extracts video metadata.
    Handles YouTube 403 Forbidden errors with client fallbacks.
    """
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    
    client_strategies = [
        ["mweb", "ios"],
        ["android", "ios"],
        ["web", "mweb"],
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
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
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

    raise RuntimeError(f"Failed to download YouTube audio (403 Forbidden). Error details: {last_exception}")


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
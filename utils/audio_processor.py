import os
import re
import requests
import yt_dlp
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def extract_youtube_video_id(url: str) -> str:
    """Extracts 11-character video ID from various YouTube URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'youtube\.com\/shorts\/([0-9A-Za-z_-]{11})',
        r'youtube\.com\/embed\/([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def format_seconds(seconds: float) -> str:
    """Formats float seconds into HH:MM:SS or MM:SS."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_youtube_oembed_metadata(url: str, video_id: str) -> dict:
    """Fetches video title, channel, and thumbnail via YouTube oEmbed API."""
    oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        resp = requests.get(oembed_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title": data.get("title", "YouTube Video"),
                "channel": data.get("author_name", "YouTube Channel"),
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
                "duration": "YouTube Video",
                "url": url
            }
    except Exception:
        pass
    return {
        "title": f"YouTube Video ({video_id})",
        "channel": "YouTube Channel",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        "duration": "YouTube Video",
        "url": url
    }


def get_youtube_transcript_fast(url: str) -> tuple[dict, dict]:
    """Retrieves timestamped transcript directly via YouTube Subtitles API."""
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL. Could not extract Video ID.")

    metadata = get_youtube_oembed_metadata(url, video_id)
    
    fetched = None
    errors = []

    # Strategy A: Direct get_transcript with common languages
    try:
        fetched = YouTubeTranscriptApi.get_transcript(
            video_id, 
            languages=['en', 'en-US', 'en-GB', 'en-IN', 'hi', 'es', 'fr', 'de']
        )
    except Exception as e1:
        errors.append(f"get_transcript: {e1}")

    # Strategy B: List transcripts and find manual or generated transcript
    if not fetched:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                transcript_obj = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'en-IN', 'hi'])
            except Exception:
                transcript_obj = next(iter(transcript_list))
            fetched = transcript_obj.fetch()
        except Exception as e2:
            errors.append(f"list_transcripts: {e2}")

    # Strategy C: Explicitly find generated transcript
    if not fetched:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript_obj = transcript_list.find_generated_transcript(['en', 'hi'])
            fetched = transcript_obj.fetch()
        except Exception as e3:
            errors.append(f"generated_transcript: {e3}")

    if not fetched:
        raise RuntimeError(f"No YouTube transcript/captions found for this video ({'; '.join(errors)})")
        
    full_text_parts = []
    segments = []
    
    for item in fetched:
        start_sec = item['start']
        duration = item.get('duration', 0.0)
        end_sec = start_sec + duration
        text = item['text'].replace('\n', ' ').strip()
        
        if not text:
            continue
            
        full_text_parts.append(text)
        segments.append({
            "start": format_seconds(start_sec),
            "end": format_seconds(end_sec),
            "start_raw": start_sec,
            "end_raw": end_sec,
            "text": text
        })
        
    full_text = " ".join(full_text_parts)
    transcript_data = {
        "full_text": full_text,
        "segments": segments
    }
    return transcript_data, metadata


def process_text_input(raw_text: str, title: str = "Custom Text Input") -> tuple[dict, dict]:
    """Processes raw text transcript directly into structured segments."""
    words = raw_text.strip().split()
    segments = []
    words_per_segment = 50
    for i in range(0, len(words), words_per_segment):
        seg_text = " ".join(words[i:i + words_per_segment])
        start_sec = (i // words_per_segment) * 30
        end_sec = start_sec + 30
        segments.append({
            "start": format_seconds(start_sec),
            "end": format_seconds(end_sec),
            "start_raw": start_sec,
            "end_raw": end_sec,
            "text": seg_text
        })
    metadata = {
        "title": title,
        "channel": "User Text Input",
        "thumbnail": None,
        "duration": format_seconds(len(segments) * 30),
        "url": None
    }
    return {"full_text": raw_text.strip(), "segments": segments}, metadata


def download_via_cobalt(url: str) -> tuple[str, dict]:
    """Downloads YouTube audio using Cobalt API microservice."""
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
    """Fallback audio download handler for YouTube videos without captions."""
    try:
        return download_via_cobalt(url)
    except Exception as cob_err:
        print(f"Cobalt API failed: {cob_err}, trying pytubefix...")

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
        print(f"pytubefix strategy failed: {py_err}, trying yt-dlp...")

    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    client_strategies = [["android_vr"], ["web_creator"], ["ios"], ["mweb"], ["tvhtml5"]]

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
            "extractor_args": {"youtube": {"player_client": client}},
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
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
        f"YouTube audio download failed ({last_exception}). "
        f"Tip: Upload local audio/video file directly using sidebar!"
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
    """Splits long audio into chunks."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks


def process_input(source: str) -> tuple[str, any, dict]:
    """
    Unified input processor.
    Returns: (proc_type, proc_data, metadata_dict)
    proc_type: "FAST_TRANSCRIPT" or "CHUNKS"
    """
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Attempting instant transcript extraction...")
        try:
            transcript_data, metadata = get_youtube_transcript_fast(source)
            return "FAST_TRANSCRIPT", transcript_data, metadata
        except Exception as e:
            print(f"Fast transcript API failed ({e}), falling back to audio download...")
            wav_path, metadata = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path, metadata = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    return "CHUNKS", chunks, metadata
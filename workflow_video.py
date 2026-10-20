"""Explicit color policy for the supplied SDR-valued, HDR-tagged demo exports."""
import hashlib
import subprocess
from pathlib import Path

SDR_POLICY = 'reinterpret_sdr_bt709'
COLOR_OPTIONS = ['-color_primaries','bt709','-color_trc','bt709',
                 '-colorspace','bt709','-color_range','tv']
H264_COLOR_FILTER = ('h264_metadata=colour_primaries=1:transfer_characteristics=1:'
                    'matrix_coefficients=1:video_full_range_flag=0')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def decoded_signature(ffmpeg, path):
    """Hash every decoded YUV frame and copied audio packet, including timing."""
    result=subprocess.run([ffmpeg,'-nostdin','-hide_banner','-loglevel','error',
        '-i',str(path),'-map','0:v:0','-map','0:a:0?','-vsync','0',
        '-c:v','rawvideo','-c:a','copy','-f','framemd5','-'],
        capture_output=True,check=True)
    return hashlib.sha256(result.stdout).hexdigest()


def repair_sdr_tags(ffmpeg, path):
    """Remux H.264 tags only; refuse to replace the file if samples/timing change."""
    path=Path(path)
    before=sha256(path)
    signature=decoded_signature(ffmpeg,path)
    temporary=path.with_name(path.stem+'.color-fix.tmp.mp4')
    subprocess.run([ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-y',
        '-i',str(path),'-map','0:v:0','-map','0:a:0?','-c','copy',
        '-bsf:v',H264_COLOR_FILTER,*COLOR_OPTIONS,'-movflags','+faststart',
        str(temporary)],check=True)
    if decoded_signature(ffmpeg,temporary)!=signature:
        raise RuntimeError(f'Color repair changed decoded samples/timing; original retained: {path}')
    temporary.replace(path)
    return {'policy':SDR_POLICY,'previous_review_video_sha256':before,
            'decoded_frames_and_audio_timing_signature_sha256':signature,
            'decoded_frames_audio_and_timestamps_identical':True,
            'method':'H.264 VUI and MP4 color metadata only; no pixel re-encoding or tone mapping'}

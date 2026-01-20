import argparse
import subprocess
import json
import logging
import sys
import shutil
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 定数定義
DEFAULT_MIN_SSIM = 0.985
DEFAULT_MAX_SSIM = 1.0  # デフォルトは制限なし
DEFAULT_MAX_RETRIES = 3

CODEC_CONFIGS = {
    'av1_nvenc': {'quality': 38, 'preset': 'p7'},
    'hevc_nvenc': {'quality': 36, 'preset': 'p7'},
    'libx265': {'quality': 36, 'preset': 'medium'},
    'libx264': {'quality': 32, 'preset': 'slow'},
}

AVAILABLE_CODECS = list(CODEC_CONFIGS.keys())
DEFAULT_CODEC = 'av1_nvenc'

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg', '.ts', '.m2ts'}

def is_video_file(file_path: Path) -> bool:
    """
    ファイルが動画かどうかを判定する。
    1. 拡張子が動画形式であること
    2. ffprobeで1フレーム以上の映像ストリームが検出されること
    3. 継続時間が0より大きいこと
    """
    if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
        return False

    info = get_video_info(file_path)
    if not info:
        return False

    video_streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'video']
    if not video_streams:
        return False

    # 1フレームしかないものは画像（または動画ではない）とみなす
    # nb_frames が取得できない場合は duration で判断
    nb_frames = video_streams[0].get('nb_frames')
    if nb_frames and nb_frames.isdigit() and int(nb_frames) <= 1:
        return False

    # duration が 0 のものは画像（または静止画）とみなす
    duration = info.get('format', {}).get('duration')
    if duration and float(duration) <= 0:
        return False

    return True

def get_video_info(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    ffprobeを使用して動画ファイルの情報を取得する。
    動画ストリームが存在しない場合はNoneを返す。
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        str(file_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        if not data.get('streams'):
            return None
        return data
    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to probe file {file_path}: {e}")
        return None
    except json.JSONDecodeError:
        logger.debug(f"Failed to decode ffprobe output for {file_path}")
        return None

def calculate_ssim(original_path: Path, encoded_path: Path) -> float:
    """
    ffmpegを使用して2つの動画ファイル間のSSIMを計算する。
    動画全体を通して計算し、平均SSIM (All) を返す。
    """
    # フィルタコンプレックスでSSIMを計算し、標準エラー出力からパースする
    
    cmd = [
        'ffmpeg',
        '-i', str(encoded_path),
        '-i', str(original_path),
        '-filter_complex', '[0:v]settb=1/1000,setpts=PTS-STARTPTS[main];[1:v]settb=1/1000,setpts=PTS-STARTPTS[ref];[main][ref]ssim',
        '-f', 'null',
        '-'
    ]
    
    try:
        logger.info(f"Calculating SSIM for {encoded_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # ffmpegのエラーチェック
        if result.returncode != 0:
            logger.error(f"ffmpeg SSIM calculation failed for {encoded_path.name}")
            logger.error(f"ffmpeg stderr: {result.stderr}")
            return 0.0
        
        # ffmpegの出力はstderrに出る
        output = result.stderr
        
        # 出力例: [Parsed_ssim_0 @ 0000...] SSIM Y:0.99... U:0.99... V:0.99... All:0.99...
        match = re.search(r'SSIM.*?All:([0-9.]+)', output)
        if match:
            return float(match.group(1))
        else:
            logger.warning(f"Could not parse SSIM from output for {encoded_path.name}")
            logger.warning(f"ffmpeg output: {output}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating SSIM: {e}")
        return 0.0

def encode_video(
    input_path: Path,
    output_path: Path,
    codec: str,
    quality_value: float,
    preset: str,
    audio_codec: str = 'aac',
    audio_bitrate: Optional[str] = '192k',
    params: Optional[Dict[str, str]] = None
) -> bool:
    """
    動画をエンコードする。
    """
    if params is None:
        params = {}

    # 出力ディレクトリの作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg',
        '-y', # 上書き許可
        '-i', str(input_path),
        '-c:v', codec,
    ]

    # コーデックごとの画質オプション設定
    if codec in ['av1_nvenc', 'hevc_nvenc']:
        cmd.extend(['-cq:v', str(quality_value)])
        # NVENC向けの最適化設定
        cmd.extend(['-tune', 'hq', '-rc-lookahead', '60', '-pix_fmt', 'yuv420p'])
    elif codec in ['libx264', 'libx265']:
        cmd.extend(['-crf', str(quality_value)])
    
    # プリセット
    cmd.extend(['-preset', preset])

    # 音声設定
    cmd.extend(['-c:a', audio_codec])
    if audio_codec != 'copy' and audio_bitrate:
        cmd.extend(['-b:a', audio_bitrate])

    # メタデータ転送とMP4最適化
    cmd.extend(['-map_metadata', '0', '-movflags', '+faststart'])

    # その他のパラメータ
    for k, v in params.items():
        cmd.extend([k, v])

    # 出力パス
    cmd.append(str(output_path))

    try:
        logger.info(f"Encoding {input_path.name} (Codec: {codec}, Q: {quality_value}, Audio: {audio_codec}/{audio_bitrate})")
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Encoding failed for {input_path.name}: {e}")
        # エラー出力を表示
        if e.stderr:
            logger.error(e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else e.stderr)
        return False

def process_file(
    source_file: Path,
    source_root: Path,
    target_root: Path,
    args: argparse.Namespace
):
    """
    1つのファイルを処理する。
    """
    # 相対パスを計算してターゲットパスを決定
    rel_path = source_file.relative_to(source_root)
    # コンテナをmp4に変更
    target_file = target_root / rel_path.with_suffix('.mp4')

    # 動画情報の詳細を取得
    source_info = get_video_info(source_file)
    if not source_info:
        logger.warning(f"Failed to get video info for {source_file}. Skipping.")
        return

    # 音声設定の決定
    audio_codec = 'aac'
    audio_bitrate = '192k'
    
    audio_streams = [s for s in source_info.get('streams', []) if s.get('codec_type') == 'audio']
    if audio_streams:
        src_audio = audio_streams[0]
        if src_audio.get('codec_name') == 'aac':
            audio_codec = 'copy'
            audio_bitrate = None
        else:
            # ビットレート取得
            br = src_audio.get('bit_rate')
            if br:
                audio_bitrate = str(br)
            # ビットレートが取得できない場合はデフォルト設定を使用

    # エンコードパラメータの初期化
    current_quality = args.quality
    
    # 再試行ループ
    for attempt in range(args.max_retries + 1):
        success = encode_video(
            source_file,
            target_file,
            args.codec,
            current_quality,
            args.preset,
            audio_codec,
            audio_bitrate
        )

        if not success:
            logger.error(f"Failed to encode {source_file}. Skipping.")
            return

        # 解像度チェック
        target_info = get_video_info(target_file)
        if not target_info:
            logger.error(f"Failed to get info for encoded file {target_file}. Skipping.")
            return

        # 映像ストリームを探す
        src_video_streams = [s for s in source_info['streams'] if s.get('codec_type') == 'video']
        tgt_video_streams = [s for s in target_info['streams'] if s.get('codec_type') == 'video']
        
        if not src_video_streams or not tgt_video_streams:
            logger.error(f"Video stream not found for comparison. Skipping.")
            return

        src_stream = src_video_streams[0]
        tgt_stream = tgt_video_streams[0]
        
        if src_stream.get('width') != tgt_stream.get('width') or \
           src_stream.get('height') != tgt_stream.get('height'):
            logger.error(f"Resolution mismatch for {source_file.name}: "
                         f"{src_stream.get('width')}x{src_stream.get('height')} -> "
                         f"{tgt_stream.get('width')}x{tgt_stream.get('height')}")
            return

        # SSIMチェック
        ssim = calculate_ssim(source_file, target_file)
        logger.info(f"SSIM for {source_file.name}: {ssim}")

        if ssim < args.min_ssim:
            if attempt < args.max_retries:
                logger.info(f"Quality check (SSIM: {ssim} < {args.min_ssim}). Retrying with higher quality...")
                # 画質を上げる（値を下げる）
                current_quality -= 2
                if current_quality < 0:
                    current_quality = 0
            else:
                logger.warning(f"Quality check failed after {args.max_retries} retries. Keeping the best attempt.")
                log_file_size_comparison(source_file, target_file)
                break
        elif ssim > args.max_ssim:
            if attempt < args.max_retries:
                logger.info(f"Quality check (SSIM: {ssim} > {args.max_ssim}). Retrying with lower quality...")
                # 画質を下げる（値を上げる）
                current_quality += 2
            else:
                logger.warning(f"Quality check: SSIM still exceeds upper limit after {args.max_retries} retries. Keeping the best attempt.")
                log_file_size_comparison(source_file, target_file)
                break
        else:
            logger.info(f"Quality check passed (SSIM: {ssim} is within [{args.min_ssim}, {args.max_ssim}]).")
            log_file_size_comparison(source_file, target_file)
            break

def log_file_size_comparison(source_file: Path, target_file: Path):
    """
    ソースファイルとターゲットファイルのサイズを比較してログ出力する。
    """
    try:
        src_size = source_file.stat().st_size
        tgt_size = target_file.stat().st_size
        # 圧縮率（元のサイズに対する割合）を計算
        ratio_percent = (tgt_size / src_size) * 100 if src_size > 0 else 0
        
        src_mb = src_size / (1024 * 1024)
        tgt_mb = tgt_size / (1024 * 1024)
        
        logger.info(f"Size: {src_mb:.2f}MB -> {tgt_mb:.2f}MB (Compressed: {ratio_percent:.1f}%)")
    except Exception as e:
        logger.warning(f"Failed to calculate file size comparison: {e}")

def main():
    if len(sys.argv) == 1:
        # 対話形式での設定
        while True:
            source_dir_str = input("ソースディレクトリ: ").strip()
            if source_dir_str:
                break
            print("ソースディレクトリは必須です。")
            
        while True:
            target_dir_str = input("ターゲットディレクトリ: ").strip()
            if target_dir_str:
                break
            print("ターゲットディレクトリは必須です。")
            
        codec = input(f"コーデック (デフォルト: {DEFAULT_CODEC}): ").strip() or DEFAULT_CODEC
        if codec not in AVAILABLE_CODECS:
            logger.warning(f"'{codec}' は利用可能なコーデックではありません。デフォルトの '{DEFAULT_CODEC}' を使用します。")
            codec = DEFAULT_CODEC
        
        quality_str = input("画質 (デフォルト: 自動設定): ").strip()
        quality = float(quality_str) if quality_str else None
        
        preset = input("プリセット (デフォルト: 自動設定): ").strip() or None
        
        min_ssim_str = input(f"最小SSIM (デフォルト: {DEFAULT_MIN_SSIM}): ").strip()
        min_ssim = float(min_ssim_str) if min_ssim_str else DEFAULT_MIN_SSIM
        
        max_ssim_str = input(f"最大SSIM (デフォルト: {DEFAULT_MAX_SSIM}): ").strip()
        max_ssim = float(max_ssim_str) if max_ssim_str else DEFAULT_MAX_SSIM
        
        max_retries_str = input(f"最大再試行回数 (デフォルト: {DEFAULT_MAX_RETRIES}): ").strip()
        max_retries = int(max_retries_str) if max_retries_str else DEFAULT_MAX_RETRIES
        
        args = argparse.Namespace(
            source_dir=Path(source_dir_str),
            target_dir=Path(target_dir_str),
            codec=codec,
            quality=quality,
            preset=preset,
            min_ssim=min_ssim,
            max_ssim=max_ssim,
            max_retries=max_retries
        )
    else:
        # 通常の引数処理
        parser = argparse.ArgumentParser(description='Video Re-encoder with SSIM Quality Assurance')
        
        parser.add_argument('source_dir', type=Path, help='Source directory containing video files')
        parser.add_argument('target_dir', type=Path, help='Target directory for output files')
        
        parser.add_argument('--codec', type=str, default=DEFAULT_CODEC, 
                            choices=AVAILABLE_CODECS,
                            help=f'Video codec (default: {DEFAULT_CODEC})')
        
        parser.add_argument('--quality', type=float, help='Quality value (CQ for NVENC, CRF for libx264/265). Default depends on codec.')
        
        parser.add_argument('--preset', type=str, help='Encoding preset. Default depends on codec.')
        
        parser.add_argument('--min-ssim', type=float, default=DEFAULT_MIN_SSIM, help=f'Minimum SSIM threshold (default: {DEFAULT_MIN_SSIM})')
        parser.add_argument('--max-ssim', type=float, default=DEFAULT_MAX_SSIM, help=f'Maximum SSIM threshold (default: {DEFAULT_MAX_SSIM})')
        parser.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES, help=f'Maximum retries for quality adjustment (default: {DEFAULT_MAX_RETRIES})')

        args = parser.parse_args()

    # デフォルト値の設定（共通ロジック）
    # コーデックごとのデフォルト設定を取得
    config = CODEC_CONFIGS.get(args.codec)
    
    if args.quality is None:
        if config:
            args.quality = config['quality']
        else:
            # 未知のコーデックへの予備処理
            logger.warning(f"Unknown codec {args.codec}, cannot set default quality.")
            
    if args.preset is None:
        if config:
            args.preset = config['preset']
        else:
            logger.warning(f"Unknown codec {args.codec}, cannot set default preset.")

    if not args.source_dir.exists():
        logger.error(f"Source directory not found: {args.source_dir}")
        sys.exit(1)

    # 安全対策: ソースとターゲットが同じ場合はエラー
    if args.source_dir.resolve() == args.target_dir.resolve():
        logger.error("Error: Source and target directories must be different to prevent accidental overwriting.")
        sys.exit(1)

    # ファイル探索
    logger.debug(f"Scanning files in {args.source_dir}...")
    all_files = sorted([p for p in args.source_dir.rglob('*') if p.is_file()])
    
    # ファイルごとに動画判定を行いながら順次処理
    logger.info(f"Found {len(all_files)} files in {args.source_dir}")
    logger.info(f"Settings: Codec={args.codec}, Quality={args.quality}, Preset={args.preset}, MinSSIM={args.min_ssim}, MaxSSIM={args.max_ssim}")

    for file_path in all_files:
        # 隠しファイル除外
        if file_path.name.startswith('.'):
            continue

        if is_video_file(file_path):
            process_file(file_path, args.source_dir, args.target_dir, args)
        else:
            # 動画でない場合はそのままコピー
            rel_path = file_path.relative_to(args.source_dir)
            target_file = args.target_dir / rel_path
            
            # 出力先ディレクトリの準備
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Copying non-video file: {file_path.name} -> {target_file.name}")
            shutil.copy2(file_path, target_file)

if __name__ == '__main__':
    main()

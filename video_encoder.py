import argparse
import subprocess
import json
import logging
import sys
import shutil
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm

# ロギング設定
# プログレスバー（tqdm）と標準出力が混ざらないように、カスタムのTqdmLoggingHandlerを定義する
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

# フォーマットを [HH:MM:SS] [LEVEL] メッセージ という形にスッキリさせる
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
handler = TqdmLoggingHandler()
handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler]
)
logger = logging.getLogger(__name__)

# 定数定義
DEFAULT_MIN_SSIM = 0.985
DEFAULT_MAX_SSIM = 1.0  # デフォルトは制限なし
DEFAULT_MAX_RETRIES = 5

CODEC_CONFIGS = {
    'av1_nvenc': {'quality': 38, 'preset': 'p7'},
    'hevc_nvenc': {'quality': 36, 'preset': 'p7'},
    'libx265': {'quality': 36, 'preset': 'medium'},
    'libx264': {'quality': 32, 'preset': 'slow'},
}

AVAILABLE_CODECS = list(CODEC_CONFIGS.keys())
DEFAULT_CODEC = 'av1_nvenc'

# エンコーダー識別用メタデータ
ENCODER_TOOL_NAME = 'AdaptiveVideoEncoder'

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

def is_encoded_by_tool(file_path: Path) -> bool:
    """
    ファイルがこのツールでエンコード済みかどうかを判定する。
    メタデータの 'encoder_tool' フィールドを確認する。
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        str(file_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        tags = data.get('format', {}).get('tags', {})
        # MP4ではカスタムタグが 'comment' 等に格納されることが多いため広くチェック
        # または 'encoder_tool' そのものを直接チェック
        for key, value in tags.items():
            if key.lower() == 'encoder_tool' and value == ENCODER_TOOL_NAME:
                return True
            if key.lower() == 'comment' and f'tool:{ENCODER_TOOL_NAME}' in value:
                return True
        return False
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False

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
        # SSIM計算中は進行中を示すのみとするため、冗長なログは省くか短くする
        # logger.info(f"  Calculating SSIM...") # Encodingの横などに結果をくっつける方が綺麗なのでここでは出さない
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # ffmpegのエラーチェック
        if result.returncode != 0:
            logger.error(f"  ffmpeg SSIM calculation failed for {encoded_path.name}")
            logger.error(f"  ffmpeg stderr: {result.stderr}")
            return 0.0
        
        # ffmpegの出力はstderrに出る
        output = result.stderr
        
        # 出力例: [Parsed_ssim_0 @ 0000...] SSIM Y:0.99... U:0.99... V:0.99... All:0.99...
        match = re.search(r'SSIM.*?All:([0-9.]+)', output)
        if match:
            return float(match.group(1))
        else:
            logger.warning(f"  Could not parse SSIM from output")
            logger.warning(f"  ffmpeg output: {output}")
            return 0.0
            
    except Exception as e:
        logger.error(f"  Error calculating SSIM: {e}")
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

    # メタデータ転送とMP4最適化、エンコーダー識別タグの追加
    cmd.extend(['-map_metadata', '0', '-movflags', '+faststart'])
    # MP4コンテナの互換性を考慮し、comment タグにも情報を埋め込む
    cmd.extend(['-metadata', f'encoder_tool={ENCODER_TOOL_NAME}'])
    cmd.extend(['-metadata', f'comment=tool:{ENCODER_TOOL_NAME}'])

    # その他のパラメータ
    for k, v in params.items():
        cmd.extend([k, v])

    # 出力パス
    cmd.append(str(output_path))

    try:
        logger.info(f"  Encoding... (Codec: {codec}, Q: {quality_value}, Audio: {audio_codec}/{audio_bitrate})")
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  Encoding failed for {input_path.name}: {e}")
        # エラー出力を表示
        if e.stderr:
            logger.error(f"  {e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else e.stderr}")
        return False

def determine_audio_settings(source_info: dict) -> tuple[str, Optional[str]]:
    """
    動画情報から適切な音声コーデックとビットレートを決定する。
    Returns: (audio_codec, audio_bitrate)
    """
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
            # ビットレートが取得できない場合はデフォルト設定(192k)を使用

    return audio_codec, audio_bitrate

def run_encode_with_retry(
    source_file: Path,
    encode_target_file: Path,
    source_info: dict,
    audio_codec: str,
    audio_bitrate: Optional[str],
    args: argparse.Namespace,
    summary_data: list,
    in_place: bool
) -> tuple[bool, int]:
    """
    エンコードの実行とSSIM検証による再試行ループを管理する。
    Returns: (success_bool, final_attempt_count)
    """
    current_quality = args.quality
    
    for attempt in range(args.max_retries + 1):
        success = encode_video(
            source_file,
            encode_target_file,
            args.codec,
            current_quality,
            args.preset,
            audio_codec,
            audio_bitrate
        )

        if not success:
            logger.error(f"  Failed to encode. Skipping.")
            summary_data.append({
                'name': source_file.name,
                'original_size': source_file.stat().st_size,
                'encoded_size': None,
                'status': 'Failed (Encode)'
            })
            return False, attempt

        # 解像度チェック
        target_info = get_video_info(encode_target_file)
        if not target_info:
            logger.error(f"  Failed to get info for encoded file. Skipping.")
            summary_data.append({
                'name': source_file.name,
                'original_size': source_file.stat().st_size,
                'encoded_size': None,
                'status': 'Failed (Target Probe)'
            })
            if in_place and encode_target_file.exists():
                encode_target_file.unlink()
            return False, attempt

        # 映像ストリームを探す
        src_video_streams = [s for s in source_info['streams'] if s.get('codec_type') == 'video']
        tgt_video_streams = [s for s in target_info['streams'] if s.get('codec_type') == 'video']
        
        if not src_video_streams or not tgt_video_streams:
            logger.error(f"  Video stream not found for comparison. Skipping.")
            summary_data.append({
                'name': source_file.name,
                'original_size': source_file.stat().st_size,
                'encoded_size': None,
                'status': 'Failed (No Stream)'
            })
            return False, attempt

        src_stream = src_video_streams[0]
        tgt_stream = tgt_video_streams[0]
        
        if src_stream.get('width') != tgt_stream.get('width') or \
           src_stream.get('height') != tgt_stream.get('height'):
            logger.error(f"  Resolution mismatch: "
                         f"{src_stream.get('width')}x{src_stream.get('height')} -> "
                         f"{tgt_stream.get('width')}x{tgt_stream.get('height')}")
            summary_data.append({
                'name': source_file.name,
                'original_size': source_file.stat().st_size,
                'encoded_size': None,
                'status': 'Failed (Resolution)'
            })
            if in_place and encode_target_file.exists():
                encode_target_file.unlink()
            return False, attempt

        # SSIMチェック
        ssim = calculate_ssim(source_file, encode_target_file)

        # サイズチェック（早期終了の判断材料）
        try:
            src_size = source_file.stat().st_size
            tgt_size = encode_target_file.stat().st_size
        except Exception:
            src_size = tgt_size = 0

        if ssim < args.min_ssim:
            if attempt < args.max_retries:
                if tgt_size > src_size and src_size > 0:
                    logger.warning(f"  SSIM: {ssim} (Failed, size already exceeds original. Stopping adjustment.)")
                    # サイズも超え、画質も足りない最悪な状態。これ以上のサイズ増加は無意味なのでここで打ち切り、失敗扱いとする。
                    summary_data.append({
                        'name': source_file.name,
                        'original_size': src_size,
                        'encoded_size': None,
                        'status': f'Failed (SSIM {ssim:.3f} < {args.min_ssim})'
                    })
                    if encode_target_file.exists():
                        encode_target_file.unlink()
                    return False, attempt
                
                logger.info(f"  SSIM: {ssim} (Failed, Retrying with Q: {current_quality - 2}...)")
                current_quality = max(0, current_quality - 2)
            else:
                logger.warning(f"  SSIM: {ssim} (Failed after {args.max_retries} retries. Target minimum SSIM not reached.)")
                summary_data.append({
                    'name': source_file.name,
                    'original_size': src_size,
                    'encoded_size': None,
                    'status': f'Failed (SSIM Limit)'
                })
                # 目標画質に達しなかったので結果を破棄（一時ファイルを削除）
                if encode_target_file.exists():
                    encode_target_file.unlink()
                return False, attempt
        elif ssim > args.max_ssim:
            if attempt < args.max_retries:
                logger.info(f"  SSIM: {ssim} (Exceeds {args.max_ssim}, Retrying with Q: {current_quality + 2}...)")
                current_quality += 2
            else:
                logger.warning(f"  SSIM: {ssim} (Still exceeds upper limit after {args.max_retries} retries. Keeping this attempt as success.)")
                break
        else:
            logger.info(f"  SSIM: {ssim} (Passed)")
            break

    return True, attempt

def finalize_encoded_file(
    source_file: Path,
    encode_target_file: Path,
    target_file: Path,
    target_root: Path,
    rel_path: Path,
    attempt: int,
    in_place: bool,
    summary_data: list
) -> None:
    """
    エンコード後のファイルサイズチェック、およびIn-Place時のリネーム/クリーンアップ処理を行う。
    """
    final_status = f"Success (Retry {attempt})" if attempt > 0 else "Success"
    src_size = 0
    tgt_size = 0
    
    try:
        src_size = source_file.stat().st_size
        tgt_size = encode_target_file.stat().st_size
        
        if tgt_size > src_size:
            logger.warning(f"  Final encoded file is larger than original ({tgt_size} > {src_size}). Reverting to original file.")
            if encode_target_file.exists():
                encode_target_file.unlink()
            
            if not in_place:
                final_target = target_root / rel_path
                final_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, final_target)
            
            logger.info(f"  Reverted to original file")
            tgt_size = src_size # 元のサイズに戻ったため
            final_status = "Reverted (Size)"
        else:
            log_file_size_comparison(src_size, tgt_size)
            if in_place:
                try:
                    import send2trash
                    send2trash.send2trash(str(source_file.resolve()))
                except ImportError:
                    source_file.unlink()
                except Exception as e:
                    logger.warning(f"  Failed to move original to trash, deleting directly: {e}")
                    source_file.unlink()
                
                encode_target_file.rename(target_file)
                
    except Exception as e:
        logger.error(f"  Error during final size check/fallback: {e}")
        final_status = "Failed (Size Check)"
        if in_place and encode_target_file.exists():
            encode_target_file.unlink()

    summary_data.append({
        'name': source_file.name,
        'original_size': src_size,
        'encoded_size': tgt_size,
        'status': final_status
    })

def process_file(
    source_file: Path,
    source_root: Path,
    target_root: Path,
    args: argparse.Namespace,
    summary_data: list
) -> bool:
    """
    1つのファイルを処理する。
    Returns: True if file was processed, False if skipped.
    """
    # エンコード済みチェック（--force が指定されていない場合）
    if not args.force and is_encoded_by_tool(source_file):
        logger.info(f"  Skipped already encoded file")
        summary_data.append({
            'name': source_file.name,
            'original_size': source_file.stat().st_size,
            'encoded_size': None,
            'status': 'Skipped'
        })
        return False
    # 相対パスを計算してターゲットパスを決定
    rel_path = source_file.relative_to(source_root)
    # コンテナをmp4に変更
    target_file = target_root / rel_path.with_suffix('.mp4')

    # 同一ファイル上書き（In-place）かどうかの判定
    in_place = source_file.resolve() == target_file.resolve()
    
    # ffmpegの出力先ファイルパスを設定（上書きの場合は一時ファイルを使用）
    encode_target_file = target_file.with_name(f".tmp.{target_file.name}") if in_place else target_file

    # 動画情報の詳細を取得
    source_info = get_video_info(source_file)
    if not source_info:
        logger.warning(f"  Failed to get video info. Skipping.")
        summary_data.append({
            'name': source_file.name,
            'original_size': source_file.stat().st_size,
            'encoded_size': None,
            'status': 'Failed (Probe)'
        })
        return False

    # 音声設定の決定
    audio_codec, audio_bitrate = determine_audio_settings(source_info)

    # エンコードとSSIM検証ループの実行
    success, attempt = run_encode_with_retry(
        source_file, encode_target_file, source_info, 
        audio_codec, audio_bitrate, args, summary_data, in_place
    )
    
    if not success:
        return False

    # ファイルサイズの最終チェックと差し替え・クリーンアップ処理
    finalize_encoded_file(
        source_file, encode_target_file, target_file, target_root, rel_path,
        attempt, in_place, summary_data
    )

    return True

def log_file_size_comparison(src_size: int, tgt_size: int):
    """
    ソースファイルとターゲットファイルのサイズを比較してログ出力する。
    """
    try:
        # 圧縮率（元のサイズに対する割合）を計算
        ratio_percent = (tgt_size / src_size) * 100 if src_size > 0 else 0
        
        src_mb = src_size / (1024 * 1024)
        tgt_mb = tgt_size / (1024 * 1024)
        
        logger.info(f"  Size: {src_mb:.2f}MB -> {tgt_mb:.2f}MB (Compressed: {ratio_percent:.1f}%)")
    except Exception as e:
        logger.warning(f"  Failed to calculate file size comparison: {e}")

def check_encoded_status(target_path: Path):
    """
    指定されたパスのエンコード済み状態を確認して表示する。
    ファイルの場合は単体を、ディレクトリの場合は再帰的に全ファイルを確認する。
    """
    if target_path.is_file():
        files = [target_path]
    elif target_path.is_dir():
        files = sorted([p for p in target_path.rglob('*') if p.is_file()])
    else:
        logger.error(f"Path not found: {target_path}")
        return
    
    encoded_count = 0
    not_encoded_count = 0
    skipped_count = 0
    
    for file_path in files:
        if not is_video_file(file_path):
            skipped_count += 1
            continue
        
        if is_encoded_by_tool(file_path):
            status = f"\033[92mEncoded by {ENCODER_TOOL_NAME}\033[0m"  # Green
            encoded_count += 1
        else:
            status = "\033[93mNot encoded\033[0m"  # Yellow
            not_encoded_count += 1
        
        print(f"{file_path}: {status}")
    
    # サマリー
    print(f"\n--- Summary ---")
    print(f"Encoded: {encoded_count}, Not encoded: {not_encoded_count}, Skipped (non-video): {skipped_count}")

def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析し、対話モードと引数モードの両方に対応する。
    """
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
        
        force_str = input("既にエンコード済みのファイルも再処理しますか？ (y/N): ").strip().lower()
        force = force_str in ('y', 'yes')
        
        args = argparse.Namespace(
            source_dir=Path(source_dir_str),
            target_dir=Path(target_dir_str),
            codec=codec,
            quality=quality,
            preset=preset,
            min_ssim=min_ssim,
            max_ssim=max_ssim,
            max_retries=max_retries,
            force=force,
            check=None
        )
    else:
        # 通常の引数処理
        parser = argparse.ArgumentParser(description='Video Re-encoder with SSIM Quality Assurance')
        
        parser.add_argument('source_dir', type=Path, nargs='?', help='Source directory containing video files')
        parser.add_argument('target_dir', type=Path, nargs='?', help='Target directory for output files')
        
        parser.add_argument('--check', type=Path, metavar='PATH', help='Check if file(s) are encoded by this tool (no encoding)')
        parser.add_argument('--force', action='store_true', help='Force re-encode even if already encoded by this tool')
        
        parser.add_argument('--codec', type=str, default=DEFAULT_CODEC, 
                            choices=AVAILABLE_CODECS,
                            help=f'Video codec (default: {DEFAULT_CODEC})')
        
        parser.add_argument('--quality', type=float, help='Quality value (CQ for NVENC, CRF for libx264/265). Default depends on codec.')
        
        parser.add_argument('--preset', type=str, help='Encoding preset. Default depends on codec.')
        
        parser.add_argument('--min-ssim', type=float, default=DEFAULT_MIN_SSIM, help=f'Minimum SSIM threshold (default: {DEFAULT_MIN_SSIM})')
        parser.add_argument('--max-ssim', type=float, default=DEFAULT_MAX_SSIM, help=f'Maximum SSIM threshold (default: {DEFAULT_MAX_SSIM})')
        parser.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES, help=f'Maximum retries for quality adjustment (default: {DEFAULT_MAX_RETRIES})')

        args = parser.parse_args()
        
        # source_dir と target_dir の必須チェック
        if not args.check and (not args.source_dir or not args.target_dir):
            parser.error('source_dir and target_dir are required unless using --check')

    return args

def main():
    args = parse_arguments()
    
    # --check モードの処理
    if args.check:
        check_encoded_status(args.check)
        return

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

    # 安全対策: ソースとターゲットが同じ場合のブロックを削除（In-Place Replacementサポートにより許可）
    # if args.source_dir.resolve() == args.target_dir.resolve():
    #     logger.error("Error: Source and target directories must be different to prevent accidental overwriting.")
    #     sys.exit(1)

    # ファイル探索
    logger.debug(f"Scanning files in {args.source_dir}...")
    all_files = sorted([p for p in args.source_dir.rglob('*') if p.is_file()])
    
    # ファイルごとに動画判定を行いながら順次処理
    logger.info(f"Found {len(all_files)} files in {args.source_dir}")
    logger.info(f"Settings: Codec={args.codec}, Quality={args.quality}, Preset={args.preset}, MinSSIM={args.min_ssim}, MaxSSIM={args.max_ssim}\n")

    summary_data = []

    for i, file_path in enumerate(tqdm(all_files, desc="Processing Videos", dynamic_ncols=True, leave=True)):
        # 隠しファイル除外
        if file_path.name.startswith('.'):
            continue

        if is_video_file(file_path):
            logger.info(f"Processing [{i+1}/{len(all_files)}]: {file_path.name}")
            process_file(file_path, args.source_dir, args.target_dir, args, summary_data)
        else:
            # 動画でない場合はそのままコピー
            rel_path = file_path.relative_to(args.source_dir)
            target_file = args.target_dir / rel_path
            
            # 出力先ディレクトリの準備
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing [{i+1}/{len(all_files)}]: {file_path.name} (Non-video)")
            
            # 自身へのコピーを防ぐ
            if file_path.resolve() == target_file.resolve():
                summary_data.append({
                    'name': file_path.name,
                    'original_size': file_path.stat().st_size,
                    'encoded_size': None,
                    'status': 'Skipped (In-place non-video)'
                })
            else:
                summary_data.append({
                    'name': file_path.name,
                    'original_size': file_path.stat().st_size,
                    'encoded_size': None,
                    'status': 'Copied (Non-video)'
                })
                shutil.copy2(file_path, target_file)

    # 結果サマリーの表示
    print_summary_table(summary_data)

def print_summary_table(summary_data: list) -> None:
    """
    エンコード処理完了後に結果のサマリー表を出力する。
    """
    if not summary_data:
        return
        
    total_files = len(summary_data)
    success_files = sum(1 for item in summary_data if item['status'].startswith('Success'))
    skipped_files = sum(1 for item in summary_data if item['status'] == 'Skipped')
    failed_files = sum(1 for item in summary_data if item['status'].startswith('Failed'))
    
    total_original_size = sum(item['original_size'] for item in summary_data if item['original_size'] is not None)
    total_encoded_size = sum(item['encoded_size'] for item in summary_data if item['encoded_size'] is not None and item['status'].startswith('Success'))
    
    # エンコードされなかったファイルはそのままのサイズとして足す
    for item in summary_data:
        if not item['status'].startswith('Success') and item['original_size'] is not None:
            total_encoded_size += item['original_size']

    total_ratio = (total_encoded_size / total_original_size) * 100 if total_original_size > 0 else 0

    print(f"\n================================================================================")
    print(f"Encoding Summary")
    print(f"================================================================================")
    print(f" Total Files Processed : {total_files}")
    print(f" Successfully Encoded  : {success_files}")
    print(f" Skipped / Copied      : {skipped_files + sum(1 for i in summary_data if 'Copied' in i['status'])}")
    print(f" Failed                : {failed_files}")
    print(f"\n--------------------------------------------------------------------------------")
    print(f" {'File Name':<30} | {'Original Size':>13} | {'Encoded Size':>12} | {'Ratio':>5} | {'Status'}")
    print(f"--------------------------------------------------------------------------------")
    
    for item in summary_data:
        orig_str = f"{item['original_size'] / (1024*1024):10.2f} MB" if item['original_size'] is not None else "---"
        
        if item['encoded_size'] is not None and item['status'].startswith('Success'):
            enc_str = f"{item['encoded_size'] / (1024*1024):9.2f} MB"
            ratio = (item['encoded_size'] / item['original_size']) * 100 if item['original_size'] else 0
            ratio_str = f"{ratio:4.1f}%"
        else:
            enc_str = "        ---  "
            ratio_str = " --- "
            
        name_str = item['name']
        if len(name_str) > 30:
            name_str = name_str[:27] + "..."
        
        print(f" {name_str:<30} | {orig_str:>13} | {enc_str:>12} | {ratio_str:>5} | {item['status']}")
        
    print(f"--------------------------------------------------------------------------------")
    orig_total_str = f"{total_original_size / (1024*1024):10.2f} MB"
    enc_total_str = f"{total_encoded_size / (1024*1024):9.2f} MB"
    
    print(f" {'Total Size':<30} | {orig_total_str:>13} | {enc_total_str:>12} | {total_ratio:4.1f}% |")
    print(f"================================================================================\n")

if __name__ == '__main__':
    main()

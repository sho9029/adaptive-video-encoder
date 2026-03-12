import argparse
import subprocess
import json
import logging
import sys
import shutil
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.table import Table
from rich.highlighter import NullHighlighter

# --- rich Console の初期化 ---
console = Console()

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True, highlighter=NullHighlighter())]
)
logger = logging.getLogger(__name__)

# 定数定義
DEFAULT_METRIC = 'vmaf'
DEFAULT_MIN_SCORE_SSIM = 0.985
DEFAULT_MAX_SCORE_SSIM = 1.0  # デフォルトは制限なし
DEFAULT_MIN_SCORE_VMAF = 93.0
DEFAULT_MAX_SCORE_VMAF = 100.0
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
    side_data（display_matrix等）も取得する。
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_entries', 'stream_side_data',
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

def get_video_rotation(stream: dict) -> int:
    """
    映像ストリーム情報から回転角度を取得する（0, 90, 180, 270）。
    side_data_list の Display Matrix と tags.rotate の両方を確認する。
    """
    rotation = 0

    # 1. side_data_list から Display Matrix の rotation を取得（優先）
    for sd in stream.get('side_data_list', []):
        if sd.get('side_data_type') == 'Display Matrix' and 'rotation' in sd:
            rotation = int(float(sd['rotation']))
            break

    # 2. tags.rotate にフォールバック
    if rotation == 0:
        rotate_tag = stream.get('tags', {}).get('rotate', '0')
        try:
            rotation = int(float(rotate_tag))
        except (ValueError, TypeError):
            rotation = 0

    # 正規化: -90 → 270 等に統一（0, 90, 180, 270）
    rotation = rotation % 360
    if rotation < 0:
        rotation += 360

    return rotation

def calculate_ssim(original_path: Path, encoded_path: Path) -> float:
    """
    ffmpegを使用して2つの動画ファイル間のSSIMを計算する。
    動画全体を通して計算し、平均SSIM (All) を返す。
    VFR対応のため、タイムベースとPTSを正規化して比較する。
    shortest=1 によりフレーム数の差異も安全に処理する。
    """
    # settb=1/AVTB: タイムベースをavi標準に正規化（VFR/CFR問わず一致させる）
    # setpts=PTS-STARTPTS: PTSをゼロ起点にリセットしてフレーム位置を揃える
    # shortest=1: フレーム数が異なる場合、短い方に合わせて比較を終了
    
    cmd = [
        'ffmpeg',
        '-i', str(encoded_path),
        '-i', str(original_path),
        '-filter_complex', '[0:v]settb=1/AVTB,setpts=PTS-STARTPTS[main];[1:v]settb=1/AVTB,setpts=PTS-STARTPTS[ref];[main][ref]ssim=shortest=1',
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

def check_cuda_vmaf_support() -> bool:
    """
    ffmpegが libvmaf_cuda フィルタをサポートしているか判定する。
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-filters'],
            capture_output=True, text=True
        )
        return 'libvmaf_cuda' in result.stdout
    except Exception:
        return False

def calculate_vmaf(original_path: Path, encoded_path: Path, src_w: int, src_h: int, use_hwaccel: bool = False) -> float:
    """
    ffmpegを使用して2つの動画ファイル間のVMAFを計算する。
    動画の総画素数に基づき、1080p相当以下なら標準モデル、それ以上なら4K専用モデルに切り替える。
    use_hwaccelがTrueならハードウェアアクセラレーション（CUDA）を使用するが、これにはカスタムビルド版のFFmpeg（libvmaf_cuda対応）が必要。
    Falseの場合は一般的なFFmpeg（libvmaf有効化版）でCPUを用いて計算する。
    """
    pixel_area = src_w * src_h
    model_name = 'vmaf_4k_v0.6.1neg' if pixel_area > 1920 * 1080 else 'vmaf_v0.6.1neg'

    if use_hwaccel:
        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda', '-threads', '8', '-i', str(encoded_path),
            '-hwaccel', 'cuda', '-threads', '8', '-i', str(original_path),
            '-filter_complex', f'[0:v]hwupload_cuda,scale_cuda={src_w}:{src_h},settb=1/AVTB,setpts=PTS-STARTPTS[main];[1:v]hwupload_cuda,scale_cuda={src_w}:{src_h},settb=1/AVTB,setpts=PTS-STARTPTS[ref];[main][ref]libvmaf_cuda=model=version={model_name}',
            '-f', 'null',
            '-'
        ]
    else:
        cmd = [
            'ffmpeg',
            '-i', str(encoded_path),
            '-i', str(original_path),
            '-filter_complex', f'[0:v]scale={src_w}:{src_h},format=yuv420p,settb=1/AVTB,setpts=PTS-STARTPTS[main];[1:v]scale={src_w}:{src_h},format=yuv420p,settb=1/AVTB,setpts=PTS-STARTPTS[ref];[main][ref]libvmaf=model=version={model_name}',
            '-f', 'null',
            '-'
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"  ffmpeg VMAF calculation failed for {encoded_path.name}")
            logger.error(f"  ffmpeg stderr: {result.stderr}")
            return 0.0
            
        output = result.stderr
        match = re.search(r'VMAF score: ([0-9.]+)', output)
        if match:
            return float(match.group(1))
        else:
            logger.warning(f"  Could not parse VMAF from output")
            logger.warning(f"  ffmpeg output: {output}")
            return 0.0
            
    except Exception as e:
        logger.error(f"  Error calculating VMAF: {e}")
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
    # autorotateで物理回転済みのため、rotateタグを明示的に除去（二重回転防止）
    cmd.extend(['-metadata:s:v', 'rotate='])
    # MP4コンテナの互換性を考慮し、comment タグにも情報を埋め込む
    cmd.extend(['-metadata', f'encoder_tool={ENCODER_TOOL_NAME}'])
    cmd.extend(['-metadata', f'comment=tool:{ENCODER_TOOL_NAME}'])

    # その他のパラメータ
    for k, v in params.items():
        cmd.extend([k, v])

    # 出力パス
    cmd.append(str(output_path))

    try:
        logger.info(f"  Encoding... (Codec: [cyan]{codec}[/cyan], Q: [cyan]{quality_value}[/cyan], Audio: [cyan]{audio_codec}/{audio_bitrate}[/cyan])")
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
    history = []  # (quality_used, score) の履歴リスト
    metric_name = args.metric.upper()
    best_score = -1.0
    best_q = current_quality
    tried_q_values = {current_quality}

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
        
        # 回転による幅高入れ替わりを考慮した解像度比較
        src_rotation = get_video_rotation(src_stream)
        src_w = src_stream.get('width')
        src_h = src_stream.get('height')
        if src_rotation in (90, 270):
            # 90°/270°回転ではautorotateにより幅と高さが入れ替わる
            src_w, src_h = src_h, src_w
        
        tgt_w = tgt_stream.get('width')
        tgt_h = tgt_stream.get('height')
        
        if src_w != tgt_w or src_h != tgt_h:
            logger.error(f"  Resolution mismatch: "
                         f"{src_w}x{src_h} (display) -> "
                         f"{tgt_w}x{tgt_h}")
            summary_data.append({
                'name': source_file.name,
                'original_size': source_file.stat().st_size,
                'encoded_size': None,
                'status': 'Failed (Resolution)'
            })
            if in_place and encode_target_file.exists():
                encode_target_file.unlink()
            return False, attempt

        # スコアチェック
        score = calculate_ssim(source_file, encode_target_file) if args.metric == 'ssim' else calculate_vmaf(source_file, encode_target_file, src_w, src_h, args.vmaf_hwaccel)
        history.append((current_quality, score))


        # サイズチェック（早期終了の判断材料）
        try:
            src_size = source_file.stat().st_size
            tgt_size = encode_target_file.stat().st_size
        except Exception:
            src_size = tgt_size = 0

        # スコアの状態判定
        is_low = score < args.min_score
        is_high = score > args.max_score

        # 目標に最も近いスコアを記録
        # スコア不足時は「最大値」、スコア超過時は「最小値」が目標に近い
        if is_low:
            if best_score == -1.0 or score > best_score:
                best_score = score
                best_q = current_quality
        elif is_high:
            if best_score == -1.0 or score < best_score:
                best_score = score
                best_q = current_quality

        if attempt >= args.max_retries:
            if is_low:
                logger.warning(f"  {metric_name}: [cyan]{score:.5f}[/cyan] ([red]Failed[/red] after {args.max_retries} retries. Best: [cyan]{best_score:.5f}[/cyan] at Q:[cyan]{best_q}[/cyan])")

                summary_data.append({
                    'name': source_file.name,
                    'original_size': src_size,
                    'encoded_size': None,
                    'status': f'Failed ({metric_name} Limit)'
                })
                if encode_target_file.exists():
                    encode_target_file.unlink()
                return False, attempt
            else:
                logger.warning(f"  {metric_name}: [cyan]{score:.5f}[/cyan] ([yellow]Still exceeds upper limit[/yellow] after {args.max_retries} retries. Keeping this attempt as success.)")
                break

        # サイズ超過時の早期打ち切り（スコア不足時のみ）
        if is_low and tgt_size > src_size and src_size > 0:
            logger.warning(f"  {metric_name}: [cyan]{score:.5f}[/cyan] ([red]Failed[/red], stopping adjustment)")
            logger.warning(f"  [yellow]Encoded file is larger than original ({tgt_size} > {src_size}). Reverting to original file.[/yellow]")
            
            # 元ファイルに処理済みタグを付与
            tag_cmd = [
                'ffmpeg', '-y', '-i', str(source_file),
                '-c', 'copy', '-map', '0',
                '-metadata', f'encoder_tool={ENCODER_TOOL_NAME}',
                '-metadata', f'comment=tool:{ENCODER_TOOL_NAME}'
            ]
            target_ext = source_file.suffix.lower()
            if target_ext in ['.mp4', '.mov', '.m4v']:
                tmp_src = source_file.with_name(f".tmp.tag_{source_file.name}")
                tag_cmd.append(str(tmp_src))
                try:
                    subprocess.run(tag_cmd, capture_output=True, check=True)
                    if tmp_src.exists():
                        shutil.move(str(tmp_src), str(source_file))
                except Exception as e:
                    logger.debug(f"Failed to tag original file: {e}")
                    if tmp_src.exists():
                        tmp_src.unlink()

            logger.info(f"  [yellow]Reverted to original file[/yellow]")
            summary_data.append({
                'name': source_file.name,
                'original_size': src_size,
                'encoded_size': None,
                'status': f'Failed ({metric_name} {score:.3f} < {args.min_score})'
            })
            if encode_target_file.exists():
                encode_target_file.unlink()
            return False, attempt

        # --- 品質調整ロジック (対称的な探索) ---
        target_score = (args.min_score + 0.5) if is_low else (args.max_score - 0.5)
        # 評価指標による補正 (SSIMなら 0.5 -> 0.002)
        if args.metric == 'ssim':
            target_score = (args.min_score + 0.002) if is_low else (args.max_score - 0.002)

        if len(history) >= 2:
            q1, s1 = history[-2]
            q2, s2 = history[-1]
            if s2 == s1:
                delta_q = -1.0 if is_low else 1.0
            else:
                slope = (q2 - q1) / (s2 - s1)
                delta_q = slope * (target_score - s2)
                
                # 逆転現象検知 (物理的矛盾時のフォールバック)
                if slope > 0:
                    logger.debug(f"  Non-monotonicity detected (slope: {slope:.5f}).")
                    delta_q = -1.0 if is_low else 1.0
        else:
            p_gain = -200.0 if args.metric == 'ssim' else -2.0
            delta_q = p_gain * (target_score - score)
        
        # クリッピングと単調性の制約
        delta_q = max(-6.0, min(6.0, delta_q))
        if is_low and delta_q > 0: delta_q = -1.0
        if is_high and delta_q < 0: delta_q = 1.0
        
        next_q = round(current_quality + delta_q)
        
        # 最低1ステップは期待される方向へ移動させる
        if is_low and next_q >= current_quality: next_q = current_quality - 1
        if is_high and next_q <= current_quality: next_q = current_quality + 1
        
        # 試行済みQ値を回避 (目標方向へ向かってスキップ)
        while next_q in tried_q_values:
            next_q += (-1 if is_low else 1)
            if next_q < 0: break
        
        if next_q < 0 or next_q in tried_q_values:
            reason = "no more Q values" if next_q >= 0 else "hit Q=0 limit"
            logger.warning(f"  {metric_name}: [cyan]{score:.5f}[/cyan] ([red]Failed[/red], {reason}. Best: [cyan]{best_score:.5f}[/cyan] at Q:[cyan]{best_q}[/cyan])")
            break

        status_msg = f"[red]Failed[/red]" if is_low else f"[yellow]Exceeds {args.max_score}[/yellow]"
        logger.info(f"  {metric_name}: [cyan]{score:.5f}[/cyan] ({status_msg}, Adjusting Q: [cyan]{current_quality}[/cyan] -> [cyan]{next_q}[/cyan]...)")
        
        current_quality = next_q
        tried_q_values.add(current_quality)

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
            logger.warning(f"  [yellow]Encoded file is larger than original ({tgt_size} > {src_size}). Reverting to original file.[/yellow]")
            if encode_target_file.exists():
                encode_target_file.unlink()
            
            if not in_place:
                final_target = target_root / rel_path
                final_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, final_target)
                target_to_tag = final_target
            else:
                target_to_tag = source_file

            # 元ファイル（またはコピー先）に処理済みタグを付与
            tag_cmd = [
                'ffmpeg', '-y', '-i', str(target_to_tag),
                '-c', 'copy', '-map', '0',
                '-metadata', f'encoder_tool={ENCODER_TOOL_NAME}',
                '-metadata', f'comment=tool:{ENCODER_TOOL_NAME}'
            ]
            
            target_ext = target_to_tag.suffix.lower()
            if target_ext in ['.mp4', '.mov', '.m4v', '.mkv', '.webm']:
                tmp_src = target_to_tag.with_name(f".tmp.tag_{target_to_tag.name}")
                tag_cmd.append(str(tmp_src))
                try:
                    subprocess.run(tag_cmd, capture_output=True, check=True)
                    if tmp_src.exists():
                        shutil.move(str(tmp_src), str(target_to_tag))
                except Exception as e:
                    logger.debug(f"Failed to tag original file: {e}")
                    if tmp_src.exists():
                        tmp_src.unlink()
            
            logger.info(f"  [yellow]Reverted to original file[/yellow]")
            tgt_size = target_to_tag.stat().st_size # タグ付与後のサイズを取得
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
        
        logger.info(f"  Size: [cyan]{src_mb:.2f}MB[/cyan] -> [cyan]{tgt_mb:.2f}MB[/cyan] ([green]Compressed: {ratio_percent:.1f}%[/green])")
    except Exception as e:
        logger.warning(f"  [red]Failed to calculate file size comparison: {e}[/red]")

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
        
        config = CODEC_CONFIGS.get(codec)
        default_quality = config['quality'] if config else '自動設定'
        quality_str = input(f"画質 (デフォルト: {default_quality}): ").strip()
        quality = float(quality_str) if quality_str else None
        
        default_preset = config['preset'] if config else '自動設定'
        preset = input(f"プリセット (デフォルト: {default_preset}): ").strip() or None
        
        metric = input(f"評価指標 (ssim/vmaf) (デフォルト: {DEFAULT_METRIC}): ").strip().lower() or DEFAULT_METRIC
        if metric not in ('ssim', 'vmaf'):
            logger.warning(f"'{metric}' は無効な評価指標です。デフォルトの '{DEFAULT_METRIC}' を使用します。")
            metric = DEFAULT_METRIC

        default_min = DEFAULT_MIN_SCORE_SSIM if metric == 'ssim' else DEFAULT_MIN_SCORE_VMAF
        min_score_str = input(f"最小スコア (デフォルト: {default_min}): ").strip()
        min_score = float(min_score_str) if min_score_str else default_min
        
        default_max = DEFAULT_MAX_SCORE_SSIM if metric == 'ssim' else DEFAULT_MAX_SCORE_VMAF
        max_score_str = input(f"最大スコア (デフォルト: {default_max}): ").strip()
        max_score = float(max_score_str) if max_score_str else default_max
        
        max_retries_str = input(f"最大再試行回数 (デフォルト: {DEFAULT_MAX_RETRIES}): ").strip()
        max_retries = int(max_retries_str) if max_retries_str else DEFAULT_MAX_RETRIES
        
        force_str = input("既にエンコード済みのファイルも再処理しますか？ (y/n) (デフォルト: n): ").strip().lower()
        force = force_str in ('y', 'yes')
        
        args = argparse.Namespace(
            source_dir=Path(source_dir_str),
            target_dir=Path(target_dir_str),
            codec=codec,
            quality=quality,
            preset=preset,
            metric=metric,
            min_score=min_score,
            max_score=max_score,
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
        
        parser.add_argument('--metric', type=str, default=DEFAULT_METRIC, choices=['ssim', 'vmaf'], help=f'Quality metric (default: {DEFAULT_METRIC})')
        parser.add_argument('--min-score', type=float, help='Minimum score threshold (default: depends on metric)')
        parser.add_argument('--max-score', type=float, help='Maximum score threshold (default: depends on metric)')
        parser.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES, help=f'Maximum retries for quality adjustment (default: {DEFAULT_MAX_RETRIES})')

        args = parser.parse_args()
        
        if args.min_score is None:
            args.min_score = DEFAULT_MIN_SCORE_SSIM if args.metric == 'ssim' else DEFAULT_MIN_SCORE_VMAF
        if args.max_score is None:
            args.max_score = DEFAULT_MAX_SCORE_SSIM if args.metric == 'ssim' else DEFAULT_MAX_SCORE_VMAF

        
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

    # VMAFのハードウェアサポート自動判定
    args.vmaf_hwaccel = False
    if args.metric == 'vmaf':
        args.vmaf_hwaccel = check_cuda_vmaf_support()

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
    
    hwaccel_str = "(CUDA)" if args.vmaf_hwaccel else "(CPU)"
    vmaf_info = f", VMAF_Mode={hwaccel_str}" if args.metric == 'vmaf' else ""
    
    # ファイルごとに動画判定を行いながら順次処理
    logger.info(f"Found {len(all_files)} files in {args.source_dir}")
    logger.info(f"Settings: Codec={args.codec}, Quality={args.quality}, Preset={args.preset}, Metric={args.metric.upper()}{vmaf_info}, MinScore={args.min_score}, MaxScore={args.max_score}\n")

    summary_data = []

    # --- rich.progress を使用した処理ループ ---
    progress_ui = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        transient=False
    )
    
    with progress_ui as progress:
        task_id = progress.add_task("[cyan]Processing Files...", total=len(all_files))
        
        for i, file_path in enumerate(all_files):
            progress.update(task_id, description=f"[cyan]Processing: [bold]{file_path.name}[/bold]")
            
            if file_path.name.startswith('.'):
                progress.advance(task_id)
                continue

            if is_video_file(file_path):
                # logger.info を使って Progress と連携した出力を行う
                logger.info(f"Processing \\[[yellow]{i+1}/{len(all_files)}[/yellow]]: [cyan]{file_path.name}[/cyan]")
                process_file(file_path, args.source_dir, args.target_dir, args, summary_data)
            else:
                rel_path = file_path.relative_to(args.source_dir)
                target_file = args.target_dir / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Processing \\[[yellow]{i+1}/{len(all_files)}[/yellow]]: [cyan]{file_path.name}[/cyan] (Non-video)")
                
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
            
            progress.advance(task_id)

    # 結果サマリーの表示
    print_summary_table(summary_data)

def print_summary_table(summary_data: list) -> None:
    """
    エンコード処理完了後に結果のサマリー表を出力する。
    """
    if not summary_data:
        return

    total_files = len(summary_data)
    total_original_size = 0
    total_encoded_size = 0
    success_files = 0
    failed_files = 0
    skipped_files = 0

    for item in summary_data:
        if item['original_size'] is not None:
            total_original_size += item['original_size']
        if item['encoded_size'] is not None and item['status'].startswith('Success'):
            success_files += 1
            total_encoded_size += item['encoded_size']
        elif item['status'].startswith('Failed'):
            failed_files += 1
            # 失敗時は元のサイズを加算（ディスク容量変化なしとして扱う）
            if item['original_size'] is not None:
                total_encoded_size += item['original_size']
        elif item['status'].startswith('Skipped') or item['status'].startswith('Reverted'):
            skipped_files += 1
            if item['original_size'] is not None:
                total_encoded_size += item['original_size']
                
        # Non-video copy
        if not item['status'].startswith('Success') and item['original_size'] is not None:
            total_encoded_size += item['original_size']

    success_count = sum(1 for d in summary_data if 'Success' in d['status'])
    skip_copy_count = sum(1 for d in summary_data if 'Skipped' in d['status'] or 'Copied' in d['status'])
    fail_count = sum(1 for d in summary_data if 'Failed' in d['status'])

    console.print("\n")
    
    # 総合結果のサマリーテキスト
    summary_text = (
        f"[bold]Encoding Summary[/bold]\n"
        f" Total Files: {total_files} | "
        f"[green]Success: {success_count}[/green] | "
        f"[yellow]Skipped/Copied: {skip_copy_count}[/yellow] | "
        f"[red]Failed: {fail_count}[/red]"
    )
    console.print(summary_text)

    # ---------------------------------------------------------
    # rich.table.Table の作成
    # ---------------------------------------------------------
    table = Table(title="File Details", show_header=True, header_style="bold cyan", border_style="bright_black")
    
    table.add_column("File Name", style="magenta", no_wrap=True)
    table.add_column("Original Size", justify="right")
    table.add_column("Encoded Size", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Status")

    total_orig_size = 0
    total_enc_size = 0
    
    for row in summary_data:
        name = row['name']
        orig_size = row['original_size']
        enc_size = row['encoded_size']
        status = row['status']
        
        orig_mb = orig_size / (1024 * 1024) if orig_size else 0
        total_orig_size += orig_size if orig_size else 0
        
        if enc_size is not None:
            enc_mb = enc_size / (1024 * 1024)
            ratio = (enc_size / orig_size * 100) if orig_size > 0 else 0
            total_enc_size += enc_size
            
            orig_str = f"{orig_mb:.2f} MB"
            enc_str = f"{enc_mb:.2f} MB"
            ratio_str = f"{ratio:.1f}%"
        else:
            total_enc_size += orig_size if orig_size else 0
            
            orig_str = f"{orig_mb:.2f} MB"
            enc_str = "---"
            ratio_str = "---"
            
        # ステータスによって色付け
        status_styled = status
        if "Success" in status:
            status_styled = f"[green]{status}[/green]"
        elif "Failed" in status:
            status_styled = f"[red]{status}[/red]"
        elif "Reverted" in status:
            status_styled = f"[yellow]{status}[/yellow]"
        elif "Skipped" in status or "Copied" in status:
            status_styled = f"[dim]{status}[/dim]"
            
        table.add_row(name, orig_str, enc_str, ratio_str, status_styled)

    # フッター行（合計サイズ）
    total_orig_mb = total_orig_size / (1024 * 1024) if total_orig_size else 0
    total_enc_mb = total_enc_size / (1024 * 1024) if total_enc_size else 0
    total_ratio = (total_enc_size / total_orig_size * 100) if total_orig_size > 0 else 100.0
    
    table.add_section() # 区切り線
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_orig_mb:.2f} MB[/bold]",
        f"[bold]{total_enc_mb:.2f} MB[/bold]",
        f"[bold]{total_ratio:.1f}%[/bold]",
        ""
    )
    
    console.print(table)
    console.print()

if __name__ == '__main__':
    main()

"""
tag_encoded_files.py - 既存の動画ファイルにエンコーダータグを埋め込む補助ツール

既にエンコード済みのディレクトリ内の動画ファイルに、
video_encoder.py と同じメタデータタグを付与します。
これにより、過去にエンコードしたファイルも「処理済み」として認識されるようになります。
"""

import os
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path

ENCODER_TOOL_NAME = 'AdaptiveVideoEncoder'
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv', '.m4v'}


def tag_video_file(file_path: Path) -> bool:
    """
    動画ファイルにエンコーダータグを埋め込む。
    ffmpegで再mux（ストリームコピー）しながらメタデータを追加する。
    """
    # 一時ファイルに出力
    temp_fd, temp_path_str = tempfile.mkstemp(suffix=file_path.suffix)
    os.close(temp_fd)
    temp_path = Path(temp_path_str)
    
    try:
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(file_path),
            '-c', 'copy',  # ストリームコピー（再エンコードなし）
            '-map_metadata', '0',
            '-metadata', f'encoder_tool={ENCODER_TOOL_NAME}',
            '-metadata', f'comment=tool:{ENCODER_TOOL_NAME}',
            str(temp_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] Failed to tag: {file_path.name}")
            print(f"  {result.stderr[:200] if result.stderr else 'Unknown error'}")
            return False
        
        # 成功したら元ファイルを置き換え
        shutil.move(str(temp_path), str(file_path))
        return True
        
    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        return False
    finally:
        # 一時ファイルが残っていれば削除（移動成功していれば存在しない）
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass  # クリーンアップ失敗は無視


def is_already_tagged(file_path: Path) -> bool:
    """ファイルが既にタグ付けされているか確認する。"""
    import json
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
        for key, value in tags.items():
            if key.lower() == 'encoder_tool' and value == ENCODER_TOOL_NAME:
                return True
            if key.lower() == 'comment' and f'tool:{ENCODER_TOOL_NAME}' in value:
                return True
        return False
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='既存の動画ファイルにエンコーダータグを埋め込む補助ツール'
    )
    parser.add_argument('target_dir', type=Path, help='タグを付けるディレクトリ')
    parser.add_argument('--force', action='store_true', help='既にタグがあっても上書きする')
    parser.add_argument('--dry-run', action='store_true', help='実際には変更せず、対象ファイルを表示するのみ')
    
    args = parser.parse_args()
    
    if not args.target_dir.exists():
        print(f"Error: Directory not found: {args.target_dir}")
        return
    
    # 対象ファイルを収集
    files = sorted([
        p for p in args.target_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ])
    
    print(f"Found {len(files)} video files in {args.target_dir}")
    
    tagged_count = 0
    skipped_count = 0
    failed_count = 0
    
    for file_path in files:
        if not args.force and is_already_tagged(file_path):
            print(f"[SKIP] Already tagged: {file_path.name}")
            skipped_count += 1
            continue
        
        if args.dry_run:
            print(f"[DRY-RUN] Would tag: {file_path}")
            tagged_count += 1
            continue
        
        print(f"[TAG] {file_path.name}...", end=' ')
        if tag_video_file(file_path):
            print("OK")
            tagged_count += 1
        else:
            failed_count += 1
    
    print(f"\n--- Summary ---")
    print(f"Tagged: {tagged_count}, Skipped: {skipped_count}, Failed: {failed_count}")


if __name__ == '__main__':
    main()

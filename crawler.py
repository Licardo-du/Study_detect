"""训练图片采集模块。

本文件提供面向 phone、sleep、eat 三类行为的网络图片采集能力。
它不会直接参与模型推理，而是为后续扩充训练集、误报回流和再训练
提供原始图片来源。UI 中的“数据工具/采集训练图片”会调用这里的函数。
"""

import argparse
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from path_utils import runtime_path
from network_utils import download_image_from_url, DownloadError

BASE_DIR = runtime_path(".")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

CLASS_KEYWORDS = {
    "phone": [
        "student using phone in classroom",
        "学生 上课 玩手机",
        "课堂 玩手机 学生",
        "学生 偷偷 看手机 上课",
        "students texting during class",
        "person looking at phone while studying",
        "教室 低头 玩手机",
        "上课 刷手机 中学生",
        "students scrolling phone under desk",
        "学生 课桌 下 玩手机",
    ],
    "sleep": [
        "student sleeping on desk in classroom",
        "学生 趴桌子 睡觉",
        "学生 上课 睡觉",
        "课堂 睡觉 学生",
        "students napping during class",
        "person sleeping head down desk",
        "教室 趴着 睡觉",
        "上课 打瞌睡 学生",
    ],
    "eat": [
        "student eating in classroom",
        "学生 上课 吃东西",
        "课堂 偷吃 零食",
        "学生 教室 吃早饭",
        "students eating snacks during lecture",
        "person eating at school desk",
        "上课 吃面包 学生",
        "教室 吃东西 中学生",
    ],
}


def _download_urls(urls, output_dir, max_images):
    """批量下载图片 URL；失败时重试一次，并统计成功和失败数量。"""
    urls = urls[:max_images]
    downloaded = 0
    failed = 0
    for i, url in enumerate(urls, 1):
        success = False
        for attempt in range(2):
            try:
                path = download_image_from_url(url, output_dir=output_dir, timeout=20)
                print(f"  [{i}/{len(urls)}] OK: {path.name}")
                downloaded += 1
                success = True
                break
            except DownloadError as exc:
                if attempt == 0:
                    time.sleep(2)
                else:
                    print(f"  [{i}/{len(urls)}] FAIL: {exc}")
            except Exception as exc:
                if attempt == 0:
                    time.sleep(2)
                else:
                    print(f"  [{i}/{len(urls)}] FAIL (unexpected): {exc}")
        if not success:
            failed += 1
        if i < len(urls):
            time.sleep(random.uniform(1, 2))
    return downloaded, failed


def search_baidu_images(keyword, max_images=30, output_dir=None, search_delay=3):
    """按关键词从百度图片解析候选图片 URL，并下载到指定目录。"""
    output_dir = Path(output_dir or BASE_DIR / "raw_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    delay = random.uniform(search_delay, search_delay + 2)
    time.sleep(delay)

    encoded = urllib.parse.quote(keyword)
    search_url = f"https://image.baidu.com/search/flip?tn=baiduimage&word={encoded}"

    print(f"Searching (Baidu): {keyword}")
    print(f"URL: {search_url}")

    req = urllib.request.Request(search_url, headers={
        "User-Agent": UA,
        "Referer": "https://image.baidu.com/",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, OSError) as exc:
        print(f"Search request failed: {exc}")
        return {"total_found": 0, "downloaded": 0, "failed": 0, "skipped": 0}

    urls = list(set(re.findall(r'objURL":"([^"]+)"', html)))
    urls = [u for u in urls if u.startswith("http")]
    if len(urls) > max_images:
        urls = urls[:max_images]

    print(f"Found {len(urls)} unique image URLs, downloading up to {max_images}...")
    downloaded, failed = _download_urls(urls, output_dir, max_images)
    result = {"total_found": len(urls), "downloaded": downloaded,
              "failed": failed, "skipped": 0}
    print(f"Done: {result}")
    return result


def search_bing_images(keyword, max_images=30, output_dir=None, search_delay=3):
    """按关键词从 Bing 图片解析候选图片 URL，并下载到指定目录。"""
    """Search Bing Images for a keyword and download the resulting images."""
    output_dir = Path(output_dir or BASE_DIR / "raw_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Random delay before search to avoid rate limiting
    delay = random.uniform(search_delay, search_delay + 2)
    time.sleep(delay)

    encoded = urllib.parse.quote(keyword)
    search_url = f"https://www.bing.com/images/search?q={encoded}&first=1"

    print(f"Searching: {keyword}")
    print(f"URL: {search_url}")

    request = urllib.request.Request(search_url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, OSError) as exc:
        print(f"Search request failed: {exc}")
        return {"total_found": 0, "downloaded": 0, "failed": 0, "skipped": 0}

    urls = list(set(re.findall(r'murl&quot;:&quot;([^&]+)', html)))
    if not urls:
        urls = list(set(re.findall(r'murl":"([^"]+)"', html)))
    urls = [u for u in urls if u.startswith("http")]
    if len(urls) > max_images:
        urls = urls[:max_images]

    print(f"Found {len(urls)} unique image URLs, downloading up to {max_images}...")
    downloaded, failed = _download_urls(urls, output_dir, max_images)
    result = {"total_found": len(urls), "downloaded": downloaded,
              "failed": failed, "skipped": 0}
    print(f"Done: {result}")
    return result


def crawl_all_classes(output_dir="raw_data", max_per_class=30, engine="baidu"):
    """按预设关键词采集 phone、sleep、eat 三类图片，形成原始数据集。"""
    """Download images for all three classes (phone/sleep/eat)."""
    search = search_baidu_images if engine == "baidu" else search_bing_images
    output_root = Path(output_dir or BASE_DIR / "raw_data")
    for label, keywords in CLASS_KEYWORDS.items():
        print(f"\n{'=' * 50}")
        print(f"Class: {label}")
        print(f"{'=' * 50}")
        class_dir = output_root / label
        for kw in keywords:
            search(kw, max_images=max_per_class // len(keywords) + 1,
                   output_dir=class_dir)
    print(f"\nAll done. Images saved to: {output_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image crawler for study behavior dataset")
    parser.add_argument("--search", action="store_true",
                        help="Search and download images for a single keyword")
    parser.add_argument("--keyword", type=str, default="",
                        help="Keyword to search for (used with --search)")
    parser.add_argument("--max", type=int, default=30,
                        help="Maximum images to download (default: 30)")
    parser.add_argument("--output", type=str, default="raw_data",
                        help="Output directory (default: raw_data)")
    parser.add_argument("--baidu", action="store_true",
                        help="Use Baidu Images (default)")
    parser.add_argument("--bing", action="store_true",
                        help="Use Bing Images instead of Baidu")

    args = parser.parse_args()
    engine = "bing" if args.bing else "baidu"
    search_fn = search_baidu_images if engine == "baidu" else search_bing_images

    if args.search and args.keyword:
        search_fn(args.keyword, max_images=args.max, output_dir=args.output)
    elif args.search:
        print("Please provide --keyword when using --search.")
    else:
        crawl_all_classes(output_dir=args.output, max_per_class=args.max, engine=engine)

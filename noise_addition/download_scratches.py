from bing_image_downloader import downloader

downloader.download("folded paper", limit=500, output_dir='dataset_folded', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)

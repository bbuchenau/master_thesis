from bing_image_downloader.downloader import download

query_string = 'tt33'

download(query_string, limit=30, output_dir='master_thesis/Image_Processing/image_retrieval/google_weapons', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
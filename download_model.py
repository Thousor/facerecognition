import requests
import bz2
import os

def download_file(url, filename):
    """
    下载文件的函数
    """
    print(f"开始下载 {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"{filename} 下载完成")

def decompress_bz2(filename):
    """
    解压bz2文件
    """
    print(f"开始解压 {filename}...")
    with bz2.open(filename, 'rb') as source, open(filename[:-4], 'wb') as dest:
        dest.write(source.read())
    os.remove(filename)  # 删除压缩文件
    print(f"{filename} 解压完成")

def main():
    # 下载人脸关键点检测模型
    model_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        download_file(model_url, compressed_file)
        decompress_bz2(compressed_file)
        print("模型文件准备完成！")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main() 
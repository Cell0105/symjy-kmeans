import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import colorsys
import time
import os

def generate_color_name(rgb):
    """Generate descriptive color name in Indonesian"""
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Hitam/putih/abu
    if v < 0.1:
        return "Hitam"
    elif v > 0.9 and s < 0.1:
        return "Putih"
    elif s < 0.2:
        return "Abu-abu Terang" if v > 0.5 else "Abu-abu Gelap"
    
    # Warna dasar
    hue_names = [
        (0, "Merah"),
        (30, "Oranye"),
        (45, "Kuning"),
        (75, "Hijau Muda"),
        (120, "Hijau"),
        (165, "Hijau Kebiruan"),
        (195, "Biru Muda"),
        (225, "Biru"),
        (270, "Ungu"),
        (300, "Merah Muda"),
        (330, "Merah Jambu"),
        (360, "Merah")
    ]
    
    hue_angle = h * 360
    closest_color = min(hue_names, key=lambda x: abs(x[0] - hue_angle))[1]
    
    # Deskripsi intensitas
    if s > 0.7:
        intensity = "Tua"
    elif s > 0.4:
        intensity = ""
    else:
        intensity = "Pucat"
        
    return f"{intensity} {closest_color}".strip()


def segment_image(image_path, k=5, output_filename="hasil_segmentasi.jpg"):
    """
    Segment image and return colors with Indonesian names
    Args:
        image_path: path to input image
        k: number of clusters
        output_filename: output segmented image filename
    Returns:
        Tuple of output path, details, and steps
    """
    # Open and process image
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img).reshape(-1, 3)
    
    # Logging steps
    steps = []
    steps.append(f"Gambar dimuat: {img.width} × {img.height} piksel")
    steps.append(f"Jumlah piksel: {pixels.shape[0]} (tiap piksel punya 3 nilai RGB)")
    steps.append(f"Segmentasi menggunakan K = {k} klaster")
    
    # K-Means clustering
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    processing_time = time.time() - start_time
    
    steps.append(f"K-Means selesai dalam {processing_time:.2f} detik")
    steps.append(f"Inisialisasi centroid dilakukan 10 kali (default n_init=10)")
    
    # Get results
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    for idx, center in enumerate(centers):
        steps.append(f"Centroid Klaster {idx+1}: RGB{tuple(center)}")
    
    # Generate output image
    segmented = centers[labels].reshape(img.size[1], img.size[0], 3)
    segmented = segmented.astype(np.uint8)
    
    output_path = os.path.join(os.path.dirname(image_path), "segmented_" + os.path.basename(image_path))
    Image.fromarray(segmented).save(output_path)
    
    # Prepare color information
    colors = []
    for i, center in enumerate(centers):
        hex_color = "#{:02x}{:02x}{:02x}".format(*center)
        color_name = generate_color_name(center)
        percentage = 100 * np.sum(labels == i) / len(labels)
        
        colors.append({
            "klaster": i,
            "warna": hex_color,
            "nama": color_name,
            "persentase": f"{percentage:.1f}%",
            "rgb": f"{center[0]},{center[1]},{center[2]}"
        })
    
    # Sort colors by percentage (descending)
    colors.sort(key=lambda x: float(x['persentase'][:-1]), reverse=True)
    
    # Compose image detail
    details = {
        "k": k,
        "size": f"{img.width} × {img.height} piksel",
        "format": Image.open(image_path).format or "Unknown",
        "time": f"{processing_time:.2f} detik",
        "colors": colors
    }

    return output_path, details, steps

import os
from pathlib import Path

def rename_images(data_dir):
    """
    Rename all images in data directory with format: classname_serial.ext
    Assumes images are organized in class subdirectories.
    """
    data_path = Path(data_dir)
    
    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        serial = 1
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        for image_file in sorted(class_dir.iterdir()):
            if image_file.suffix.lower() in image_extensions:
                new_name = f"{class_name}_{serial}{image_file.suffix}"
                new_path = class_dir / new_name
                
                image_file.rename(new_path)
                print(f"Renamed: {image_file.name} → {new_name}")
                
                serial += 1

if __name__ == "__main__":
    data_directory = "./Data"  # Change this to your data directory path
    rename_images(data_directory)
    print("Renaming complete!")
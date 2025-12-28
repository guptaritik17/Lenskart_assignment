"""
Generate sample eyewear images for demonstration
"""
from PIL import Image, ImageDraw, ImageFont
import random
import os
from pathlib import Path


def generate_eyewear_image(
    style: str,
    color: tuple,
    size: tuple = (400, 300),
    filename: str = "None"
) -> Image.Image:
    """
    Generate a simple eyewear image for demonstration
    
    Args:
        style: Frame style ('aviator', 'wayfarer', 'round', 'square')
        color: RGB color tuple
        size: Image size
        filename: Optional filename to save
        
    Returns:
        PIL Image
    """
    # Create blank image with light background
    img = Image.new('RGB', size)
    draw = ImageDraw.Draw(img)
    
    # Calculate center and dimensions
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Draw eyewear based on style
    if style == "aviator":
        # Draw aviator style frames
        # Left lens
        draw.ellipse(
            [center_x - 120, center_y - 40, center_x - 20, center_y + 40],
            outline=color,
            width=5
        )
        # Right lens
        draw.ellipse(
            [center_x + 20, center_y - 40, center_x + 120, center_y + 40],
            outline=color,
            width=5
        )
        # Bridge
        draw.line(
            [center_x - 20, center_y, center_x + 20, center_y],
            fill=color,
            width=4
        )
        
    elif style == "wayfarer":
        # Draw wayfarer style frames (rectangular with angle)
        # Left lens
        draw.rectangle(
            [center_x - 120, center_y - 35, center_x - 25, center_y + 35],
            outline=color,
            width=6
        )
        # Right lens
        draw.rectangle(
            [center_x + 25, center_y - 35, center_x + 120, center_y + 35],
            outline=color,
            width=6
        )
        # Bridge
        draw.rectangle(
            [center_x - 25, center_y - 5, center_x + 25, center_y + 5],
            fill=color
        )
        
    elif style == "round":
        # Draw round frames
        # Left lens
        draw.ellipse(
            [center_x - 110, center_y - 35, center_x - 30, center_y + 35],
            outline=color,
            width=4
        )
        # Right lens
        draw.ellipse(
            [center_x + 30, center_y - 35, center_x + 110, center_y + 35],
            outline=color,
            width=4
        )
        # Bridge
        draw.line(
            [center_x - 30, center_y, center_x + 30, center_y],
            fill=color,
            width=3
        )
        
    elif style == "square":
        # Draw square frames
        # Left lens
        draw.rectangle(
            [center_x - 115, center_y - 38, center_x - 25, center_y + 38],
            outline=color,
            width=5
        )
        # Right lens
        draw.rectangle(
            [center_x + 25, center_y - 38, center_x + 115, center_y + 38],
            outline=color,
            width=5
        )
        # Bridge
        draw.line(
            [center_x - 25, center_y, center_x + 25, center_y],
            fill=color,
            width=4
        )
    
    # Add temples (arms)
    # Left temple
    draw.line(
        [center_x - 120, center_y, center_x - 180, center_y - 20],
        fill=color,
        width=4
    )
    # Right temple
    draw.line(
        [center_x + 120, center_y, center_x + 180, center_y - 20],
        fill=color,
        width=4
    )
    
    # Save if filename provided
    if filename:
        img.save(filename)
    
    return img


def generate_sample_dataset(output_dir: str, num_samples: int = 50):
    """
    Generate a sample dataset of eyewear images
    
    Args:
        output_dir: Directory to save images
        num_samples: Number of sample images to generate
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define styles and colors
    styles = ["aviator", "wayfarer", "round", "square"]
    colors = {
        "black": (0, 0, 0),
        "brown": (139, 69, 19),
        "gold": (218, 165, 32),
        "silver": (192, 192, 192),
        "blue": (0, 0, 139),
        "tortoise": (101, 67, 33)
    }
    
    brands = ["RayBan", "Oakley", "Gucci", "Prada", "Warby Parker"]
    materials = ["Acetate", "Metal", "Plastic", "Titanium"]
    
    # Generate metadata CSV
    import csv
    metadata_path = os.path.join(output_dir, "metadata.csv")
    
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "brand", "model_name", "price", "material",
            "frame_type", "color", "rim_type"
        ])
        writer.writeheader()
        
        for i in range(num_samples):
            # Random attributes
            style = random.choice(styles)
            color_name = random.choice(list(colors.keys()))
            color_rgb = colors[color_name]
            brand = random.choice(brands)
            material = random.choice(materials)
            price = round(random.uniform(50, 500), 2)
            
            # Generate filename
            filename = f"eyewear_{i+1:03d}_{style}_{color_name}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Generate image
            generate_eyewear_image(
                style=style,
                color=color_rgb,
                filename=filepath
            )
            
            # Write metadata
            writer.writerow({
                "filename": filename,
                "brand": brand,
                "model_name": f"{brand} {style.capitalize()}",
                "price": price,
                "material": material,
                "frame_type": style.capitalize(),
                "color": color_name.capitalize(),
                "rim_type": "Full-rim"
            })
    
    print(f"Generated {num_samples} sample images in {output_dir}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    # Generate sample dataset
    generate_sample_dataset("data/images", num_samples=50)

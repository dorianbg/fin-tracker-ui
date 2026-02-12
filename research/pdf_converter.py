import fitz  # uv
import cv2
import numpy as np
from PIL import Image
import io

# ---- SETTINGS ----
input_pdf = "/Users/dbg/code/IdeaProjects/fin-tracker-ui/algobook.pdf"
threshold_values = [90, 100, 110]  # adjust between 120–200 depending on scan brightness
for threshold_value in threshold_values:
    output_pdf = f"output_bw_{threshold_value}.pdf"

    print("Output PDF will be saved to:", output_pdf)

    # ---- PROCESS ----
    doc = fitz.open(input_pdf)
    out_pdf = fitz.open()

    for page_index in range(len(doc)):
        if page_index % 10 == 0:
            print(f"Processing page {page_index + 1} of {len(doc)}...") 
        # Render page to image
        pix = doc[page_index].get_pixmap(dpi=320)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Convert to grayscale
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # Apply threshold to make black/white
        _, bw = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Convert back to PDF page
        pil_bw = Image.fromarray(bw)
        img_bytes = io.BytesIO()
        pil_bw.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Create a new PDF page with the thresholded image
        rect = fitz.Rect(0, 0, pix.width, pix.height)
        new_page = out_pdf.new_page(width=pix.width, height=pix.height)
        new_page.insert_image(rect, stream=img_bytes.read())

    # Save final result
    out_pdf.save(output_pdf)
    print(f"✅ Saved cleaned black & white PDF to: {output_pdf}")
    out_pdf.close()

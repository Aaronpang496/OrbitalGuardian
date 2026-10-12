# Filter predictions: only keep those with confidence > 0.6
high_confidence_preds = [p for p in predictions if p["confidence"] > 0.6]

if high_confidence_preds:
    st.error("⚠️ WARNING: Drowsiness detected!")
    
    # Draw boxes and percentages on the image
    draw = ImageDraw.Draw(image)
    for pred in high_confidence_preds:
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        conf, cls = pred["confidence"], pred["class"]
        
        # Draw the bounding box
        draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="red", width=3)
        # Draw the label
        draw.text((x - w/2, y - h/2 - 15), f"{cls} {conf:.0%}", fill="red")
        
    st.image(image, caption='Detection Result', use_column_width=True)
else:
    st.info("✅ No drowsiness detected.")

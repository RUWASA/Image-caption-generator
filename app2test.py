import streamlit as st
from PIL import Image
import torch
import io
import re
from transformers import BlipProcessor, BlipForConditionalGeneration

# ======================
# CONSTANTS
# ======================
MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_FILE_SIZE_MB = 10
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]
MAX_DIMENSION = 1024

# ======================
# MODEL LOADING
# ======================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load BLIP model and processor"""
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    return processor, model

# ======================
# IMAGE PROCESSING
# ======================
def process_image(uploaded_file):
    try:
        img_bytes = uploaded_file.read()
        file_size = len(img_bytes) / (1024 * 1024)
        
        if file_size > MAX_FILE_SIZE_MB:
            st.error(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
            return None, None
        
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        return img, file_size
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None, None

# ======================
# CAPTION GENERATION
# ======================
def generate_captions(image, processor, model):
    """Generate multiple captions"""
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        num_return_sequences=1,  
        max_length=50,
        num_beams=5,
        temperature=0.7
    )
    return [processor.decode(output, skip_special_tokens=True) for output in outputs]

# ======================
# TEXT ENHANCEMENT
# ======================
def enhance_caption(caption):
    """Clean and format captions"""
    caption = re.sub(r'\\[a-zA-Z]+|\{|\}|\[|\]|\$', '', caption)
    caption = ' '.join(caption.split()).strip().capitalize()
    if not caption.endswith(('.', '!', '?')):
        caption += '.'
    return caption

# ======================
# STREAMLIT UI
# ======================
def main():
    st.set_page_config(
        page_title="Image Caption Generator",
        page_icon="✨",
        layout="wide"
    )
    
    st.markdown("""
    <style>
        .caption-box {
            background: #f8fafc;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #3b82f6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("✨ Image Caption Generator using lstm and cnn")
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=SUPPORTED_FORMATS,
            help=f"Max size: {MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file:
            img, file_size = process_image(uploaded_file)
            if img:
                st.image(img, use_column_width=True)
                
                if st.button("Generate Captions", type="primary"):
                    with st.spinner("Analyzing image..."):
                        try:
                            processor, model = load_model()
                            raw_captions = generate_captions(img, processor, model)
                            clean_captions = [enhance_caption(c) for c in raw_captions]
                            
                            # Remove duplicates
                            seen = set()
                            unique_captions = []
                            for cap in clean_captions:
                                if cap not in seen:
                                    seen.add(cap)
                                    unique_captions.append(cap)
                            
                            st.markdown("**Generated Captions:**")
                            for i, caption in enumerate(unique_captions, 1):
                                st.markdown(f"""
                                <div class="caption-box">
                                    {i}. {caption}
                                </div>
                                """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("""
        ### Tips:
        - Use clear, focused images
        - Avoid text-heavy images
        - Natural lighting works best
        - High resolution recommended
        """)

if __name__ == "__main__":
    main()
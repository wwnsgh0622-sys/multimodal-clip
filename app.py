import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ── 페이지 설정 ──────────────────────────────────────
st.set_page_config(
    page_title="🌍 Multimodal AI (CLIP)",
    page_icon="🌍",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── 헤더 ────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌍 Multimodal AI (CLIP)</h1>
    <p>OpenAI CLIP 기반 이미지-텍스트 멀티모달 AI</p>
    <p>Vision + Language | Zero-Shot Classification | Semantic Search</p>
</div>
""", unsafe_allow_html=True)

# ── CLIP 모델 로드 ────────────────────────────────────
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

with st.spinner("CLIP 모델 로딩 중... (처음 한 번만 다운로드)"):
    model, processor = load_model()

st.success("✅ CLIP 모델 로드 완료!")

# ── 사이드바 ─────────────────────────────────────────
st.sidebar.title("⚙️ 설정")
mode = st.sidebar.radio("모드 선택", [
    "🔍 이미지-텍스트 유사도",
    "🏷️ Zero-Shot 이미지 분류",
    "🖼️ 텍스트로 이미지 검색"
])

# ── 모드 1: 이미지-텍스트 유사도 ─────────────────────
if mode == "🔍 이미지-텍스트 유사도":
    st.title("🔍 이미지-텍스트 유사도 계산")
    st.markdown("이미지와 텍스트가 얼마나 유사한지 계산해요!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ 이미지 입력")
        input_type = st.radio("입력 방식", ["URL", "파일 업로드"])

        image = None
        if input_type == "URL":
            img_url = st.text_input("이미지 URL 입력", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
            if img_url:
                try:
                    response = requests.get(img_url)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    st.image(image, caption="입력 이미지", use_column_width=True)
                except:
                    st.error("이미지를 불러올 수 없어요!")
        else:
            uploaded = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="입력 이미지", use_column_width=True)

    with col2:
        st.subheader("📝 텍스트 입력")
        texts_input = st.text_area(
            "비교할 텍스트 입력 (줄바꿈으로 구분)",
            "a photo of a dog\na photo of a cat\na photo of a car\na photo of a person\na photo of food"
        )
        texts = [t.strip() for t in texts_input.split("\n") if t.strip()]

    if image and texts and st.button("🚀 유사도 계산"):
        with st.spinner("계산 중..."):
            inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).numpy()[0]

        st.markdown("---")
        st.subheader("📊 유사도 결과")

        results = pd.DataFrame({"텍스트": texts, "유사도": probs * 100})
        results = results.sort_values("유사도", ascending=False)

        fig = px.bar(
            results, x="유사도", y="텍스트",
            orientation="h",
            color="유사도",
            color_continuous_scale="reds",
            title="이미지-텍스트 유사도"
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        best = results.iloc[0]
        st.success(f"🏆 가장 유사한 텍스트: **{best['텍스트']}** ({best['유사도']:.1f}%)")

# ── 모드 2: Zero-Shot 이미지 분류 ────────────────────
elif mode == "🏷️ Zero-Shot 이미지 분류":
    st.title("🏷️ Zero-Shot 이미지 분류")
    st.markdown("학습 없이 어떤 카테고리든 분류 가능해요!")

    category_preset = st.selectbox("카테고리 프리셋", [
        "동물 분류",
        "음식 분류",
        "장소 분류",
        "직접 입력"
    ])

    presets = {
        "동물 분류": ["a dog", "a cat", "a bird", "a fish", "a rabbit", "a horse"],
        "음식 분류": ["pizza", "sushi", "hamburger", "pasta", "salad", "ramen"],
        "장소 분류": ["beach", "mountain", "city", "forest", "desert", "river"]
    }

    if category_preset == "직접 입력":
        categories_input = st.text_area("카테고리 입력 (줄바꿈으로 구분)", "a dog\na cat\na bird")
        categories = [c.strip() for c in categories_input.split("\n") if c.strip()]
    else:
        categories = presets[category_preset]
        st.write("카테고리:", categories)

    img_url = st.text_input("이미지 URL", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")

    if img_url:
        try:
            response = requests.get(img_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption="분류할 이미지", width=300)
        except:
            st.error("이미지를 불러올 수 없어요!")
            image = None

    if st.button("🚀 Zero-Shot 분류"):
        with st.spinner("분류 중..."):
            texts = [f"a photo of {c}" for c in categories]
            inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).numpy()[0]

        results = pd.DataFrame({"카테고리": categories, "확률": probs * 100})
        results = results.sort_values("확률", ascending=False)

        fig = px.pie(
            results, values="확률", names="카테고리",
            title="Zero-Shot 분류 결과",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)

        best = results.iloc[0]
        st.success(f"🏆 예측 결과: **{best['카테고리']}** ({best['확률']:.1f}%)")

# ── 모드 3: 텍스트로 이미지 검색 ─────────────────────
elif mode == "🖼️ 텍스트로 이미지 검색":
    st.title("🖼️ 텍스트로 이미지 검색")
    st.markdown("텍스트로 가장 유사한 이미지를 찾아요!")

    sample_images = {
        "강아지": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
        "고양이": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/320px-Kittyply_edit1.jpg",
        "산": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/320px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg",
        "바다": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/320px-24701-nature-natural-beauty.jpg",
        "도시": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Southwest_corner_of_Central_Park%2C_looking_east%2C_NYC.jpg/320px-Southwest_corner_of_Central_Park%2C_looking_east%2C_NYC.jpg",
    }

    search_query = st.text_input("검색어 입력", "a cute animal")

    st.subheader("🖼️ 검색 대상 이미지들")
    cols = st.columns(5)
    images = {}
    for idx, (name, url) in enumerate(sample_images.items()):
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images[name] = img
            cols[idx].image(img, caption=name, use_column_width=True)
        except:
            pass

    if st.button("🔍 검색"):
        with st.spinner("검색 중..."):
            scores = {}
            for name, img in images.items():
                inputs = processor(text=[search_query], images=img, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                score = outputs.logits_per_image.item()
                scores[name] = score

        results = pd.DataFrame({
            "이미지": list(scores.keys()),
            "유사도 점수": list(scores.values())
        }).sort_values("유사도 점수", ascending=False)

        st.markdown("---")
        fig = px.bar(
            results, x="이미지", y="유사도 점수",
            color="유사도 점수",
            color_continuous_scale="purples",
            title=f"'{search_query}' 검색 결과"
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        best = results.iloc[0]
        st.success(f"🏆 가장 유사한 이미지: **{best['이미지']}**")
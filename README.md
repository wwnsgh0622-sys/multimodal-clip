# 🌍 Multimodal AI (CLIP)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

<p align="center">
  OpenAI CLIP 기반 이미지-텍스트 멀티모달 AI 시스템<br/>
  Zero-Shot Classification | Semantic Search | Vision + Language
</p>

---

## 🎯 주요 기능

- 🔍 **이미지-텍스트 유사도** — 이미지와 텍스트 유사도 계산
- 🏷️ **Zero-Shot 분류** — 학습 없이 어떤 카테고리든 분류
- 🖼️ **텍스트로 이미지 검색** — 텍스트로 유사한 이미지 검색

---

## 🏆 성과

| 항목 | 내용 |
|------|------|
| 모델 | OpenAI CLIP (ViT-B/32) |
| 정확도 | 99.3% (강아지 이미지 분류) |
| 기능 | Zero-Shot Classification |
| 프레임워크 | PyTorch, HuggingFace |

---

## ⚙️ Setup

```bash
git clone https://github.com/wwnsgh0622-sys/multimodal-clip.git
cd multimodal-clip
python -m venv clip-env
clip-env\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Run

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **모델**: OpenAI CLIP (ViT-B/32)
- **프레임워크**: PyTorch, HuggingFace Transformers
- **시각화**: Plotly, Streamlit
- **데이터**: 실제 이미지 URL / 파일 업로드

---

## 👤 Author

**문준호** · [wwnsgh0622-sys](https://github.com/wwnsgh0622-sys)
Chung-Ang University, Software Engineering
📧 wwnsgh0622@gmail.com
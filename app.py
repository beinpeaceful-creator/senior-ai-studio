import streamlit as st
import os
import re
import time
import random
import json
import concurrent.futures
from PIL import Image, ImageFont
import numpy as np
from google import genai
from google.oauth2 import service_account
from google.cloud import texttospeech
import vertexai
from vertexai.vision_models import ImageGenerationModel

# MoviePy 2.0+
import moviepy as mp
from moviepy import ImageClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, CompositeVideoClip, TextClip, ColorClip
import moviepy.video.fx as vfx
from moviepy import afx

# --- [웹 앱 초기 설정] ---
st.set_page_config(page_title="K-Senior AI Studio", page_icon="🎥", layout="wide")

# --- [시스템 상수] ---
LOCATION = "us-central1"
RENDER_FPS = 20
ZOOM_FACTOR = 0.15
BGM_VOLUME = 0.22
MAX_CHARS_PER_LINE = 7
SUB_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf"

# --- [인증 로직] ---
def authenticate_google():
    try:
        if "google" in st.secrets:
            creds_info = json.loads(st.secrets["google"]["credentials"])
            credentials = service_account.Credentials.from_service_account_info(creds_info)
            return credentials, creds_info["project_id"]
        return None, None
    except: return None, None

# --- [핵심 엔진] ---
def split_text_smart(text, max_len=7):
    words = text.split()
    chunks, current_chunk = [], ""
    for word in words:
        if len(word) > max_len:
            if current_chunk: chunks.append(current_chunk); current_chunk = ""
            for i in range(0, len(word), max_len): chunks.append(word[i:i+max_len])
            continue
        test_str = current_chunk + (" " if current_chunk else "") + word
        if len(test_str) <= max_len: current_chunk = test_str
        else:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = word
    if current_chunk: chunks.append(current_chunk)
    return chunks

def parse_all(text):
    clean_text = text.replace("**", "").replace("#", "").strip()
    t_text_match = re.search(r"THUMBNAIL_TEXT[:\s]+(.*?)(?:\n|$)", clean_text, re.I)
    t_prompt_match = re.search(r"THUMBNAIL_PROMPT[:\s]+(.*?)(?:\n|$)", clean_text, re.I)
    thumb_txt = t_text_match.group(1).strip() if t_text_match else "대박 비법 공개!"
    thumb_prp = t_prompt_match.group(1).strip() if t_prompt_match else "Cinematic macro shot of core object"
    
    scenes = []
    raw_blocks = re.split(r"\[?SCENE\s*\d+\]?", clean_text, flags=re.I)
    for block in raw_blocks:
        if not block.strip() or "THUMBNAIL" in block: continue
        s_match = re.search(r"SCRIPT[:\s]+(.*?)(?=PROMPT|$)", block, re.S | re.I)
        p_match = re.search(r"PROMPT[:\s]+(.*)", block, re.S | re.I)
        if s_match and p_match:
            s_clean = re.sub(r"[^가-힣a-zA-Z0-9!?~\s]+$", "", s_match.group(1).strip()).strip()
            scenes.append({"script": s_clean, "prompt": f"{p_match.group(1).strip()}, no text, cinematic 8k"})
    return thumb_txt[:30], thumb_prp, scenes[:5]

# --- [메인 UI] ---
def main():
    st.title("🎥 K-Senior AI Video Studio")
    st.markdown("### 3초 만에 기획부터 자막까지, 시니어 유튜브 자동 제작")

    creds, project_id = authenticate_google()
    if not creds:
        st.error("🔒 구글 인증이 필요합니다. Secrets 설정을 확인하세요.")
        return

    with st.sidebar:
        st.header("⚙️ 영상 설정")
        u_gender = st.radio("성우 목소리", ["여성 (Neural2-A)", "남성 (Neural2-C)"])
        use_sub = st.toggle("자막 포함", value=True)
        bgm_files = st.file_uploader("BGM (mp4) 업로드", accept_multiple_files=True)

    u_keyword = st.text_input("💡 영상 주제", placeholder="예: 무릎 연골에 좋은 음식 3가지")
    u_persona = st.text_input("👤 성우 성격", placeholder="예: 다정한 60대 베테랑 약사 선생님")

    if st.button("🚀 영상 제작 시작", use_container_width=True):
        if not u_keyword:
            st.warning("주제를 입력해주세요!"); return

        status = st.status("🎬 영상 제작 중...", expanded=True)
        
        # 1. 대본 기획
        client = genai.Client(credentials=creds, project=project_id, location=LOCATION)
        prompt = f"주제 {u_keyword}, 페르소나 {u_persona}로 5장면 쇼츠 기획. 인사말금지, 30자썸네일."
        plan_raw = client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text
        t_text, t_prompt, scenes = parse_all(plan_raw)
        status.update(label="📝 대본 기획 완료!", state="running")

        # 2. 리소스 생성 (이미지/음성)
        voice_name = "ko-KR-Neural2-A" if "여성" in u_gender else "ko-KR-Neural2-C"
        processed = []
        vertexai.init(project=project_id, location=LOCATION)
        img_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        tts_client = texttospeech.TextToSpeechClient(credentials=creds)

        for i, s in enumerate(scenes):
            status.write(f"🎨 {i+1}번 장면 생성 중...")
            # 이미지 생성
            img_resp = img_model.generate_images(prompt=s['prompt'], number_of_images=1, aspect_ratio="9:16", add_watermark=False)
            img_path = f"img_{i}.png"
            img_resp[0].save(img_path, include_generation_parameters=False)
            
            # 음성 생성
            sinput = texttospeech.SynthesisInput(text=s['script'])
            v_params = texttospeech.VoiceSelectionParams(language_code="ko-KR", name=voice_name)
            aconfig = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=0.90)
            aud_resp = tts_client.synthesize_speech(input=sinput, voice=v_params, audio_config=aconfig)
            aud_path = f"aud_{i}.mp3"
            with open(aud_path, "wb") as f: f.write(aud_resp.audio_content)
            
            processed.append({'image': img_path, 'audio': aud_path, 'script': s['script']})
            time.sleep(10) # 쿼터 보호

        # 3. 편집 및 합성
        status.update(label="🎬 최종 영상 인코딩 중...", state="running")
        clips = []
        for p in processed:
            audio = AudioFileClip(p['audio'])
            img = ImageClip(p['image']).resized(height=1280*1.2).with_duration(audio.duration).with_position('center').with_audio(audio)
            
            sub_clips = [img]
            if use_sub:
                chunks = split_text_smart(p['script'])
                curr_t = 0
                for chunk in chunks:
                    dur = (len(chunk)/len(p['script'])) * audio.duration
                    txt = TextClip(text=chunk, font=SUB_FONT_PATH, font_size=75, color='#fefd48', stroke_width=0, method='caption', size=(600, 400), vertical_align='top', horizontal_align='center')
                    bg_box = ColorClip(size=(txt.w+60, 150), color=(0,0,0)).with_opacity(0.7).with_duration(dur)
                    combined = CompositeVideoClip([bg_box, txt.with_position(('center', 25))]).with_duration(dur).with_start(curr_t).with_position(('center', 920))
                    sub_clips.append(combined)
                    curr_t += dur
            clips.append(CompositeVideoClip(sub_clips, size=(720, 1280)))

        final = concatenate_videoclips(clips, method="chain")
        if bgm_files:
            bgm_raw = random.choice(bgm_files)
            with open("temp_bgm.mp4", "wb") as f: f.write(bgm_raw.getbuffer())
            bgm = AudioFileClip("temp_bgm.mp4").with_effects([afx.AudioLoop(duration=final.duration)]).with_volume_scaled(BGM_VOLUME)
            final.audio = CompositeAudioClip([final.audio, bgm])

        final_path = "final_video.mp4"
        final.write_videofile(final_path, fps=RENDER_FPS, codec="libx264", preset="ultrafast", logger=None)
        
        status.update(label="✅ 제작 완료!", state="complete", expanded=False)
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("💾 영상 다운로드", f, file_name="senior_video.mp4")

if __name__ == "__main__":
    main()

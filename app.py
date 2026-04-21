# ===========================================================================
# 🚀 시니어 유튜브 자동화 시스템 - BGM 랜덤 선택(mp4) 및 디자인 완벽 보정본
# ===========================================================================
# [실행 전 필수] 코랩 첫 셀에서 아래 명령어를 실행하여 환경을 구축하세요.
# !pip install -q -U "google-genai>=0.4.0" google-cloud-texttospeech google-cloud-aiplatform "moviepy>=2.0.0" pillow
# !apt-get install -y imagemagick fonts-nanum
# !wget -q -O /content/SCoreDream9.ttf "https://github.com/S-CoreDream/Font/raw/master/S-Core%20Dream%209%20Black.ttf"
# !sed -i 's/domain="coder" rights="none" pattern="LABEL"/domain="coder" rights="read|write" pattern="LABEL"/' /etc/ImageMagick-6/policy.xml

import os
import re
import time
import random
import concurrent.futures
from PIL import Image, ImageFont
import numpy as np
from google import genai
from google.genai import types
from google.cloud import texttospeech
from google.colab import auth
from google.api_core import client_options
import vertexai
from vertexai.vision_models import ImageGenerationModel

# MoviePy 2.0+
import moviepy as mp
from moviepy import ImageClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, CompositeVideoClip, TextClip, ColorClip
import moviepy.video.fx as vfx

# --- [1. 설정 구역] ---
PROJECT_ID = "project-5aa14674-499b-4cc7-88d" 
LOCATION = "us-central1" 
VIDEO_TYPE = "SHORTS" 

RENDER_FPS = 20        
ZOOM_FACTOR = 0.15     
BGM_VOLUME = 0.25      # 배경음악 볼륨
MAX_CHARS_PER_LINE = 7 # 자막 한 줄당 최대 글자 수

# [배경음악 파일 목록]
BGM_FILES = ["bgm1.mp4", "bgm2.mp4"]

# [폰트 경로 설정]
THUMB_FONT_PATH = "/content/SCoreDream9.ttf"      
SUB_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" 

def initialize_system():
    print(f"🔐 시스템 인증 및 프로젝트({PROJECT_ID}) 연결 중...")
    auth.authenticate_user()
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = PROJECT_ID
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # BGM 파일 존재 여부 확인
    existing_bgms = [f for f in BGM_FILES if os.path.exists(f)]
    if not existing_bgms:
        print("⚠️ bgm1.mp4 또는 bgm2.mp4 파일이 없습니다. 기본 음악을 다운로드합니다.")
        os.system("curl -L -o bgm1.mp4 https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")
        
    try:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        print(f"✅ 시스템 초기화 완료 (BGM 랜덤 모드 가동)")
        return client
    except Exception as e:
        print(f"❌ 초기화 실패: {e}"); return None

def generate_contents_5_scenes(client, keyword, persona):
    """인사말 없이 본론 직행 + 사물 중심 이미지 + 30자 썸네일 기획"""
    aspect_desc = "9:16 vertical ratio for Shorts"
    print(f"\n📝 '{keyword}' 주제로 전략적 대본 기획 중...")
    
    model_id = "gemini-2.5-flash" 
    prompt = f"""
    당신은 한국 시니어 유튜브 기획 전문가이자 성공한 크리에이터입니다. 주제: '{keyword}' / 페르소나: '{persona}'
    위 정보를 바탕으로 5장면 영상과 '썸네일'을 기획하세요. 형식: {aspect_desc}.
    
    [콘텐츠 생성 철칙 - 미준수 시 실패]
    1. 인사말 금지: 첫 장면에 '안녕하세요', '반갑습니다', '이장입니다' 등 모든 인사와 자기소개를 **절대** 쓰지 마세요.
    2. 즉시 시작: 첫 마디부터 바로 핵심 비법이나 충격적인 질문으로 시청자를 멈추게 하세요.
    3. 인물 배제: 이미지 프롬프트(PROMPT)는 주제와 관련된 **한국적인 '인물, 사물, 식물, 도구, 배경'** 등으로 상세히 묘사하세요.
    4. 썸네일: THUMBNAIL_TEXT는 반드시 **30자 이내**로 작성하세요. 궁금증을 폭발시키는 문구여야 합니다.
    5. 나레이션: 대사(SCRIPT) 끝에 '-', '*', '_' 같은 기호를 절대 붙이지 마세요.
    6. 마무리 : 내용과 자연스럽게 이어지도록 '구독, 좋아요'를 유도하고, '더 궁금하신 점은 댓글로 물어보세요.'라고 정중히 마무리하세요.
    """
    
    try:
        response = client.models.generate_content(model=model_id, contents=prompt)
        return response.text
    except Exception as e:
        print(f"❌ 대본 생성 오류: {e}"); return None

def split_text_smart(text, max_len=7):
    """어절(단어) 단위로 끊어서 한 줄 리스트 반환"""
    words = text.split()
    chunks = []
    current_chunk = ""
    for word in words:
        if len(word) > max_len:
            if current_chunk: chunks.append(current_chunk); current_chunk = ""
            for i in range(0, len(word), max_len): chunks.append(word[i:i+max_len])
            continue
        test_str = current_chunk + (" " if current_chunk else "") + word
        if len(test_str) <= max_len:
            current_chunk = test_str
        else:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = word
    if current_chunk: chunks.append(current_chunk)
    return chunks

def parse_all(text):
    """기획 데이터 파싱 및 텍스트 클리닝"""
    thumb_text_match = re.search(r"THUMBNAIL_TEXT[:\s\*]+(.*?)(?:\n|$)", text, re.I)
    thumb_prompt_match = re.search(r"THUMBNAIL_PROMPT[:\s\*]+(.*?)(?:\n|$)", text, re.I)
    
    t_text = thumb_text_match.group(1).strip() if thumb_text_match else "안 보면 무조건 후회!"
    t_text = re.sub(r"[\*#_]", "", t_text).strip()
    if len(t_text) > 30: t_text = t_text[:27] + "..."
    
    t_prompt = thumb_prompt_match.group(1).strip() if thumb_prompt_match else "Cinematic macro shot of core object, no humans"

    scenes = []
    parts = re.split(r"\[?(?:SCENE|장면|#)\s*\d+\]?[:\s\-]*", text, flags=re.I)
    for part in parts:
        if not part.strip() or "THUMBNAIL" in part: continue
        s_match = re.search(r"(?:SCRIPT|대사)[:\s\*]+(.*?)(?=(?:PROMPT|프롬프트|이미지):|$)", part, re.S | re.I)
        p_match = re.search(r"(?:PROMPT|프롬프트|이미지)[:\s\*]+(.*)", part, re.S | re.I)
        if s_match and p_match:
            clean_s = re.sub(r"\(.*?\)|\[.*?\]|[\w\s]+[:：]", "", s_match.group(1).strip()).strip()
            # 끝부분 불필요 기호 제거
            clean_s = re.sub(r"[^가-힣a-zA-Z0-9!?~\s]+$", "", clean_s).strip()
            
            clean_p = p_match.group(1).strip()
            neg = "no people, no face, no humans, no characters, no text, no words"
            clean_p += f", {neg}, hyper-realistic photography, 8k"
            if clean_s: scenes.append({"script": clean_s, "prompt": clean_p})
                
    return t_text, t_prompt, scenes[:5]

def create_single_image(args):
    prompt, filename, index, is_thumb = args
    ratio = "9:16" if VIDEO_TYPE == "SHORTS" else "16:9"
    try:
        label = "썸네일" if is_thumb else f"장면 {index+1}"
        print(f"🎨 {label} 이미지 생성 중...")
        model = ImageGenerationModel.from_pretrained("imagen-4.0-generate-001")
        response = model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio=ratio, add_watermark=False)
        response[0].save(location=filename, include_generation_parameters=False)
        return filename
    except Exception as e:
        print(f"⚠️ 이미지 생성 실패: {e}"); return None

def create_voice_parallel(args):
    text, filename, voice_name = args
    try:
        opts = client_options.ClientOptions(quota_project_id=PROJECT_ID)
        tts_client = texttospeech.TextToSpeechClient(client_options=opts)
        clean_text = re.sub(r"[^가-힣a-zA-Z0-9!?~\s]+$", "", text.strip())
        sinput = texttospeech.SynthesisInput(text=clean_text)
        voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", name=voice_name)
        aconfig = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=0.92)
        response = tts_client.synthesize_speech(input=sinput, voice=voice, audio_config=aconfig)
        with open(filename, "wb") as out: out.write(response.audio_content)
        return filename
    except: return None

def create_thumbnail_vfinal(img_path, full_txt, output_path="thumbnail.jpg"):
    """썸네일 출력 보장 및 30자 멀티컬러 렌더링"""
    print(f"🖼️ 썸네일 합성 시작: {full_txt}")
    t_w, t_h = (720, 1280) if VIDEO_TYPE == "SHORTS" else (1280, 720)
    colors = ['#FFD400', '#60A5FA', '#F87171', '#4ADE80', '#FFFFFF'] 
    
    font_to_use = THUMB_FONT_PATH if os.path.exists(THUMB_FONT_PATH) else SUB_FONT_PATH

    try:
        bg = ImageClip(img_path).resized(width=t_w)
        words = full_txt.split()
        word_clips = []
        
        f_size = 120 if len(full_txt) < 18 else 95
        l_height = f_size + 20 
        total_h = len(words) * l_height
        current_y = (t_h - total_h) // 2
        
        for i, word in enumerate(words):
            txt_clip = TextClip(
                text=word, font=font_to_use, font_size=f_size,
                color=colors[i % len(colors)], stroke_color='black', stroke_width=5,
                method='caption', size=(int(t_w * 0.95), l_height), 
                vertical_align='center', horizontal_align='center'
            ).with_duration(1).with_position(('center', current_y))
            word_clips.append(txt_clip)
            current_y += l_height
            
        final_thumb = CompositeVideoClip([bg] + word_clips, size=(t_w, t_h))
        frame = final_thumb.get_frame(0)
        Image.fromarray(frame).convert('RGB').save(output_path, "JPEG", quality=95)
        print(f"✅ 썸네일 제작 성공")
    except Exception as e:
        print(f"⚠️ 썸네일 제작 실패: {e}")

def make_final_video_vfinal(scenes_data, use_subtitle, output_path="final_video.mp4"):
    """자막 마스킹 350px 확장 및 BGM 랜덤(bgm1, bgm2) 적용 버전"""
    print(f"\n🎬 4. 최종 영상 편집 시작 (BGM 랜덤 선택 및 자막 보정)...")
    clips = []
    t_w, t_h = (720, 1280) if VIDEO_TYPE == "SHORTS" else (1280, 720)
    
    try:
        for i, scene in enumerate(scenes_data):
            audio = AudioFileClip(scene['audio'])
            duration = audio.duration
            img_clip = ImageClip(scene['image']).resized(height=int(t_h * 1.2)).with_duration(duration)
            
            # 모션 효과
            if i % 2 == 0:
                img_clip = img_clip.resized(lambda t: 1.0 + ZOOM_FACTOR * (t/duration))
            else:
                img_clip = img_clip.resized(lambda t: (1.0 + ZOOM_FACTOR) - ZOOM_FACTOR * (t/duration))
            
            img_clip = img_clip.with_position('center').with_audio(audio).with_effects([vfx.FadeIn(0.5), vfx.FadeOut(0.5)])

            subtitle_clips = [img_clip]
            if use_subtitle:
                chunks = split_text_smart(scene['script'], max_len=MAX_CHARS_PER_LINE)
                total_chars_in_scene = len("".join(chunks).replace(" ", ""))
                current_time = 0
                for chunk in chunks:
                    chunk_dur = (len(chunk.replace(" ", "")) / total_chars_in_scene) * duration
                    try:
                        # [마스킹 영역 보정] 높이 350px 확장 및 상단 정렬
                        txt_base = TextClip(
                            text=chunk, font=SUB_FONT_PATH, font_size=75,
                            color='#fefd48', stroke_width=0,
                            method='caption', size=(int(t_w * 0.95), 350), 
                            horizontal_align='center', vertical_align='top'
                        )
                        p_top = 25
                        bg_box = ColorClip(size=(txt_base.w + 60, 140), color=(0, 0, 0)).with_opacity(0.7).with_duration(chunk_dur)
                        sub_combined = CompositeVideoClip([
                            bg_box, txt_base.with_position(('center', p_top)) 
                        ]).with_duration(chunk_dur).with_start(current_time).with_position(('center', int(t_h * 0.72)))
                        subtitle_clips.append(sub_combined)
                    except: pass
                    current_time += chunk_dur

            clips.append(CompositeVideoClip(subtitle_clips, size=(t_w, t_h)))

        final_video = concatenate_videoclips(clips, method="chain")
        
        # [BGM 랜덤 선택 로직]
        existing_bgms = [f for f in BGM_FILES if os.path.exists(f)]
        if existing_bgms:
            chosen_bgm = random.choice(existing_bgms)
            print(f"🎵 배경음악 선택됨: {chosen_bgm}")
            # mp4 파일에서 오디오만 추출
            bgm = AudioFileClip(chosen_bgm).with_duration(final_video.duration).with_volume_scaled(BGM_VOLUME)
            final_video.audio = CompositeAudioClip([final_video.audio, bgm])

        final_video.write_videofile(output_path, fps=RENDER_FPS, codec="libx264", preset="ultrafast", threads=12, logger=None)
    finally:
        for c in clips: c.close()
        print(f"\n✨ 모든 작업 완료: '{output_path}'")

# --- [메인 실행 루틴] ---
if __name__ == "__main__":
    client = initialize_system()
    if client:
        print("\n" + "="*50)
        u_keyword = input("📺 주제: "); u_persona = input("👤 페르소나: "); u_gender = input("🗣️ 성별(1:여, 2:남): ")
        use_sub = input("💬 자막사용(y/n): ").lower() == 'y'
        print("="*50 + "\n")
        
        voice_name = "ko-KR-Neural2-A" if u_gender == "1" else "ko-KR-Neural2-C"
        
        raw_plan = generate_contents_5_scenes(client, u_keyword, u_persona)
        if raw_plan:
            t_txt, t_prompt, scenes = parse_all(raw_plan)
            print(f"\n🔍 [기획안 검토] \n📸 썸네일(30자): {t_txt}\n🎬 장면수: {len(scenes)}개")
            if len(scenes) > 0:
                for i, s in enumerate(scenes): print(f"[{i+1}] {s['script']}")
                if input("\n🚀 제작을 시작할까요? (Enter: 시작, q: 취소): ").lower() != 'q':
                    thumb_bg = create_single_image((t_prompt, "thumb_bg.png", 0, True))
                    if thumb_bg: create_thumbnail_vfinal("thumb_bg.png", t_txt)
                    
                    voice_args = [(s['script'], f"aud_{i}.mp3", voice_name) for i, s in enumerate(scenes)]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                        aud_results = list(ex.map(create_voice_parallel, voice_args))
                    
                    img_results = []
                    for i, s in enumerate(scenes):
                        img_results.append(create_single_image((s['prompt'], f"img_{i}.png", i, False)))
                        if i < len(scenes) - 1: time.sleep(35)
                    
                    processed = [{'image': img_results[i], 'audio': aud_results[i], 'script': scenes[i]['script']} 
                                 for i in range(len(scenes)) if img_results[i] and aud_results[i]]
                    if processed: make_final_video_vfinal(processed, use_sub)
            else:
                print("⚠️ 대본 파싱 실패. 원본 내용을 출력합니다:\n", raw_plan)
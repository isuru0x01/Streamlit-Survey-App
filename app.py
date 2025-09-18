import os
import io
import uuid
from datetime import datetime
import hashlib

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# gTTS for free text-to-speech
from gtts import gTTS

# Hugging Face Hub
from huggingface_hub import HfApi, hf_hub_download, HfFolder

# Define survey flow
steps = ["consent", "demographics", "baseline", "session_emp", "session_neu", "open", "review"]

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")        
HF_DATASET_PATH = os.getenv("HF_DATASET_PATH", "responses.csv")  

# Basic validations
if not HF_TOKEN:
    st.error("Missing HF_TOKEN in .env")
if not HF_DATASET_REPO:
    st.error("Missing HF_DATASET_REPO in .env")

# Init HF client
hf_api = HfApi()
HfFolder.save_token(HF_TOKEN)

st.set_page_config(page_title="Empathetic vs. Neutral AI Voice Study", page_icon="🎙", layout="centered")

# ----------------------------------------------------
# Voice Configuration for gTTS
# ----------------------------------------------------

# Available gTTS voices with metadata
VOICE_OPTIONS = {
    "English (US) - Female": {
        "lang": "en",
        "tld": "us",
        "gender": "Female",
        "accent": "American",
        "description": "Warm, friendly American female voice"
    },
    "English (UK) - Female": {
        "lang": "en",
        "tld": "co.uk",
        "gender": "Female", 
        "accent": "British",
        "description": "Professional British female voice"
    },
    "English (Australia) - Female": {
        "lang": "en",
        "tld": "com.au",
        "gender": "Female",
        "accent": "Australian", 
        "description": "Casual Australian female voice"
    },
    "English (US) - Slow": {
        "lang": "en",
        "tld": "us",
        "gender": "Neutral",
        "accent": "American",
        "description": "Slower, more deliberate American voice",
        "slow": True
    },
    "English (UK) - Slow": {
        "lang": "en", 
        "tld": "co.uk",
        "gender": "Neutral",
        "accent": "British",
        "description": "Slower, more measured British voice",
        "slow": True
    }
}

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def preprocess_text_for_tone(text, tone="neutral"):
    """Adjust text to sound more empathetic or neutral"""
    if tone == "empathetic":
        # Add pauses and softer language
        text = text.replace(".", "... ")
        text = text.replace(",", ", ")
        # Add breathing cues
        if "breath" in text.lower():
            text = text.replace("breath", "slow, deep breath")
    elif tone == "neutral":
        # Make more robotic/clinical
        text = text.replace("I'm", "I am")
        text = text.replace("you're", "you are") 
        text = text.replace("it's", "it is")
        # Remove contractions for more formal tone
        
    return text

def generate_audio_gtts(text, voice_config, tone="neutral"):
    """Generate audio using gTTS and return as BytesIO object"""
    try:
        # Preprocess text based on tone
        processed_text = preprocess_text_for_tone(text, tone)
        
        # Create gTTS object
        tts = gTTS(
            text=processed_text,
            lang=voice_config["lang"],
            tld=voice_config.get("tld", "com"),
            slow=voice_config.get("slow", False)
        )
        
        # Generate audio in memory
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer
        
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        return None

def get_audio_cache_key(text, voice_name, tone):
    """Generate cache key for audio"""
    content = f"{text}_{voice_name}_{tone}"
    return hashlib.md5(content.encode()).hexdigest()

def play_voice(text, voice_name, tone="neutral"):
    """Generate & play audio from gTTS with caching"""
    if voice_name not in VOICE_OPTIONS:
        st.error(f"Voice '{voice_name}' not found.")
        return
        
    # Initialize audio cache in session state
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}
    
    # Check cache first
    cache_key = get_audio_cache_key(text, voice_name, tone)
    
    if cache_key not in st.session_state.audio_cache:
        with st.spinner("Generating audio..."):
            voice_config = VOICE_OPTIONS[voice_name]
            audio_buffer = generate_audio_gtts(text, voice_config, tone)
            
            if audio_buffer:
                st.session_state.audio_cache[cache_key] = audio_buffer.getvalue()
            else:
                return
    
    # Play cached audio
    audio_data = st.session_state.audio_cache[cache_key]
    st.audio(io.BytesIO(audio_data), format="audio/mp3")

def load_existing_hf_csv(repo_id: str, path_in_repo: str) -> pd.DataFrame:
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=path_in_repo,
            token=HF_TOKEN
        )
        return pd.read_csv(local_path)
    except Exception:
        return pd.DataFrame(columns=["participant_id"])

def upload_csv_to_hf(df: pd.DataFrame, repo_id: str, path_in_repo: str):
    tmp_path = "responses_tmp.csv"
    df.to_csv(tmp_path, index=False)
    hf_api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN
    )

def init_state():
    defaults = {
        "consented": False,
        "step": "consent",
        "participant_id": str(uuid.uuid4()),
        "start_ts": datetime.utcnow().isoformat(),
        # Demographics
        "age": None,
        "gender": None,
        "gender_other": "",
        "education": None,
        "voice_exp": None,
        "used_assistants": None,
        "tech_comfort": None,
        # GAD-7
        "gad": {f"q{i}": None for i in range(1, 8)},
        "gad_impact": None,
        # PANAS
        "panas": {f"q{i}": None for i in range(1, 11)},
        "single_mood": None,
        # Empathetic
        "emp": {f"q{i}": None for i in range(1, 9)},
        "emp_state_anxiety": None,
        "emp_post": {f"q{i}": None for i in range(1, 8)},
        # Neutral
        "neu": {f"q{i}": None for i in range(1, 9)},
        "neu_state_anxiety": None,
        "neu_post": {f"q{i}": None for i in range(1, 8)},
        # Open-ended
        "open_emp": "",
        "open_neu": "",
        "open_compare": "",
        "open_pref": "",
        "open_empathy": "",
        "open_trust": "",
        "open_triggers": "",
        "open_improve": "",
        "open_more_1": "",
        "open_more_2": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def section_header(text):
    st.markdown(f"### {text}")

init_state()

def scroll_to_top():
    if st.session_state.get("step_changed", False):
        st.markdown(
            "<script>window.scrollTo({top: 0, behavior: 'smooth'});</script>",
            unsafe_allow_html=True
        )
        st.session_state["step_changed"] = False

scroll_to_top()

def show_progress():
    current_step = st.session_state.get("step", "consent")
    current_index = steps.index(current_step)
    progress = (current_index + 1) / len(steps)
    st.progress(progress)
    st.write(f"Step {current_index + 1} of {len(steps)}")

show_progress()

def navigation_buttons(prev_step=None, next_step=None, prev_label="⬅ Back", next_label="Continue ➡"):
    cols = st.columns([1,1])
    with cols[0]:
        if prev_step and st.button(prev_label, key=f"back_{prev_step}"):
            st.session_state["step"] = prev_step
            st.session_state["step_changed"] = True
            st.rerun()
    with cols[1]:
        if next_step and st.button(next_label, key=f"next_{next_step}"):
            st.session_state["step"] = next_step
            st.session_state["step_changed"] = True
            st.rerun()

# Initialize session state
if "step" not in st.session_state:
    st.session_state["step"] = "consent"

# -----------------------------
# CONSENT
# -----------------------------
if st.session_state["step"] == "consent":
    st.title("Empathetic vs. Neutral AI Voice Study")
    st.subheader("Informed Consent")
    st.write("""You are invited to participate in a study on how different AI voices affect emotional well-being. You will listen to two kinds of AI voices (one warm/empathetic and one neutral/robotic) and answer some questions.""")
    st.write("""Your participation is completely voluntary. You may skip any question or stop the study at any time without penalty.""")
    st.write("""The study takes about 15–20 minutes. You will first answer some questions about 
your background and current mood/anxiety. Then you will listen to AI voice 
recordings (one empathetic, one neutral) and respond to questions during and 
after each.
""")
    st.write("""Risks/Benefits: There are minimal risks. You may feel some anxiety recalling feelings, but no 
serious risks are expected. You will contribute to understanding how voice 
interactions can affect emotional health. """)
    st.write("""All your answers are confidential and anonymous. We will use the data only for 
research purposes. No personal identifiers (names, etc.) will be linked to your 
responses.""")
    st.write("""By continuing the survey, you acknowledge that you understand the information 
above and agree to participate. """)
    consent = st.checkbox("I agree to participate.")
    if st.button("Continue ➡"):
        if consent:
            st.session_state["consented"] = True
            st.session_state["step"] = "demographics"
            st.rerun()
        else:
            st.warning("You must agree to continue.")

# -----------------------------
# DEMOGRAPHICS
# -----------------------------

if st.session_state["step"] == "demographics":
    st.header("Demographic Information")
    st.session_state["age"] = st.number_input("Q1. Enter your age (years)", min_value=18, max_value=120, step=1)
    gender_choice = st.selectbox("Q2. Your gender",
                                 ["Female", "Male", "Non-binary/Other (specify)", "Prefer not to say"])
    st.session_state["gender"] = gender_choice
    if gender_choice == "Non-binary/Other (specify)":
        st.session_state["gender_other"] = st.text_input("Please specify:")
    st.session_state["education"] = st.selectbox(
        "Q3. Select your highest education level",
        ["High school or less", "Some college/Associate's", "Bachelor's degree", "Postgraduate degree"]
    )
    st.session_state["voice_exp"] = st.radio("Q4. Do you have any voice technology experience?", ["Yes", "No"])
    st.session_state["used_assistants"] = st.radio("Q5. Have you used voice assistants (e.g. Siri, Alexa)  before?", ["Yes", "No"])
    st.session_state["tech_comfort"] = st.radio("Q6.How comfortable are you with using technology (e.g., smartphones, computers, voice assistants)? ", ["Not at all", "Slightly", "Moderately", "Very", "Extremely"])

    navigation_buttons(prev_step="consent", next_step="baseline")

# -----------------------------
# Baseline: GAD-7 + PANAS + Mood
# -----------------------------
if st.session_state["step"] == "baseline":
    st.header("Baseline Mental Health and Mood")
    section_header("A. Anxiety – GAD-7 (Generalized Anxiety Disorder Scale)")
    st.write("""
The GAD-7 is a brief, standardized questionnaire used by clinicians and researchers 
to measure symptoms of generalized anxiety. It asks about common feelings and 
behaviors related to anxiety over the past two weeks. Your answers will help us 
understand your baseline level of anxiety before the voice sessions.
""")
    st.write("""Q7.Over the past 2 weeks, how often have you been bothered by the following problems? """)
    st.write("""Select an option for each question""")
    st.write("""Scale: 1 = Not at all 2 = Several days 3 = More than half the days 4 = Nearly every day .""")
    gad_items = [
        "Q7.1 Feeling nervous, anxious, or on edge.",
        "Q7.2 Not being able to stop or control worrying.",
        "Q7.3 Worrying too much about different things.",
        "Q7.4 Trouble relaxing.",
        "Q7.5 Being so restless that it is hard to sit still.",
        "Q7.6 Becoming easily annoyed or irritable.",
        "Q7.7 Feeling afraid as if something awful might happen."
    ]
    gad_scale = [1, 2, 3, 4]
    for i, label in enumerate(gad_items, start=1):
        st.session_state["gad"][f"q{i}"] = st.radio(label, gad_scale, horizontal=True)
    st.session_state["gad_impact"] = st.radio(" Q8. If you checked any problems above, how difficult have these made it for you to do your work, take care of things at home, or get along with other people??", ["Not difficult", "Somewhat", "Very", "Extremely"])

    section_header("B. Current Mood – PANAS - Positive and Negative Affect Schedule")
    st.write("""
The PANAS is a short questionnaire that measures positive and negative emotions. 
It helps us understand your current mood by asking how strongly you feel 
different emotions right now. This provides a snapshot of your emotional state 
before the voice sessions.
""")
    st.write("""Q9.Right now, to what extent do you feel each of the following emotions?""")
    st.write("""Select an option for each question""") 
    st.write("""Scale: 1 = Very slightly or not at all 2 = A little 3 = Moderately 4 = Quite a bit 5 = Extremely""")
    panas_items = ["Interested","Distressed","Excited","Upset","Strong","Guilty","Scared","Hostile(Aggressive)","Enthusiastic","Proud"]
    five_scale = [1,2,3,4,5]
    for i, label in enumerate(panas_items, start=1):
        st.session_state["panas"][f"q{i}"] = st.radio(f"Q9.{i} {label}", five_scale, horizontal=True)

    st.write("""Single-Item Mood Rating """)
    st.session_state["single_mood"] = st.radio("Q10.Overall, right now I feel… (1=very negative, 5=very positive):", [1,2,3,4,5], horizontal=True)

    navigation_buttons(prev_step="demographics", next_step="session_emp")

# Likert scale options
five_scale = ["1 = Strongly Disagree", "2", "3", "4", "5 = Strongly Agree"]

# Define questions
empathetic_questions = {
    "Q11": "I felt the voice was warm and caring.",
    "Q12": "The voice seemed to understand or respond to my feelings.",
    "Q13": "I felt comfortable listening to this voice.",
    "Q14": "The voice spoke in a calm, soothing tone.",
    "Q15": "I would trust this voice to give helpful advice.",
    "Q16": "The voice helped me feel supported.",
    "Q17": "The pace (speed) of the voice's speech was comfortable.",
    "Q18": "I found it easy to pay attention to this voice."
}

neutral_questions = {
    "Q20": "The voice sounded neutral or robotic (monotone).",
    "Q21": "I felt the voice gave factual, impersonal responses.",
    "Q22": "I felt comfortable listening to this voice.",
    "Q23": "I would trust this voice to give accurate information.",
    "Q24": "The voice's tone seemed emotionless.",
    "Q25": "The pace of the voice's speech was comfortable.",
    "Q26": "I found it easy to pay attention to this voice.",
    "Q27": "The voice delivered the information clearly and understandably."
}

# -----------------------------
# Empathetic Voice Session
# -----------------------------
if st.session_state["step"] == "session_emp":
    st.header("Empathetic Voice Session")
    st.write("""Instructions: For each voice session, please rate the following statements about that voice on a 5-point scale:""")

    # Voice selection for empathetic session
    voice_options = [f"{name} — {info['gender']} | {info['accent']} | {info['description']}" 
                    for name, info in VOICE_OPTIONS.items()]
    
    emp_voice_label = st.selectbox(
        "Choose empathetic voice:",
        voice_options,
        key="emp_voice_select"
    )
    emp_voice_name = emp_voice_label.split(" — ")[0]

    # Text area for empathetic script
    emp_script = st.text_area(
        "Empathetic script:",
        """Hi, I'm glad you're here. I know life can feel overwhelming sometimes, 
and it's completely okay to have moments of stress or worry. 
You're not alone in feeling this way. Take a slow breath with me… inhale… and exhale. 
You're doing your best, and that's enough. Remember, even small steps forward matter. 
You deserve kindness, and I'm proud of you for taking this moment for yourself.""",
        key="emp_script_text"
    )

    # Play button
    if st.button("▶ Play Empathetic Voice", key="emp_play_btn"):
        play_voice(emp_script, emp_voice_name, tone="empathetic")

    st.subheader("AI Voice Interaction Questions (Empathetic Voice)")
    for i, (key, question) in enumerate(empathetic_questions.items(), start=11):
        st.session_state["emp"][f"q{i-10}"] = st.radio(f"Q{i}. {question}", five_scale, key=key, horizontal=True)

    st.subheader("During-Interaction Anxiety (State Anxiety)")
    st.write("""Q19.After this empathetic voice session, please indicate how anxious you felt during the session by selecting a number from 1 to 5:""")

    st.session_state["emp_state_anxiety"] = st.radio(
        "",
        [1, 2, 3, 4, 5],
        format_func=lambda x: f"{x} = {['Not at all anxious','Slightly anxious','Moderately anxious','Very anxious','Extremely anxious'][x-1]}",
        key="emp_anxiety"
    )

    navigation_buttons(prev_step="baseline", next_step="session_neu")

# -----------------------------
# Neutral Voice Session
# -----------------------------   
if st.session_state["step"] == "session_neu":
    st.header("Neutral / Robotic Voice Session")
    st.write("""Instructions: For each voice session, please rate the following statements about that voice on a 5-point scale:""")

    # Voice selection for neutral session
    voice_options = [f"{name} — {info['gender']} | {info['accent']} | {info['description']}" 
                    for name, info in VOICE_OPTIONS.items()]
    
    neu_voice_label = st.selectbox(
        "Choose neutral voice:",
        voice_options,
        key="neu_voice_select"
    )
    neu_voice_name = neu_voice_label.split(" — ")[0]

    # Text area for neutral script
    neu_script = st.text_area(
        "Neutral script:",
        """Hello, thank you for participating in this session. 
In a moment, you will be asked to reflect on your current feelings. 
This is simply a part of the study procedure. 
Please listen carefully and respond as instructed. 
There are no right or wrong answers. 
Your participation is valuable, and your responses will help us better understand voice interactions.""",
        key="neu_script_text"
    )

    # Play button
    if st.button("▶ Play Neutral Voice", key="neu_play_btn"):
        play_voice(neu_script, neu_voice_name, tone="neutral")

    st.subheader("AI Voice Interaction Questions (Neutral Voice)")
    for i, (key, question) in enumerate(neutral_questions.items(), start=20):
        st.session_state["neu"][f"q{i-19}"] = st.radio(f"Q{i}. {question}", five_scale, key=key, horizontal=True)

    st.subheader("During-Interaction Anxiety (State Anxiety)")
    st.write("""Q28.After this robotic voice session, please indicate how anxious you felt during the session by selecting a number from 1 to 5:""")

    st.session_state["neu_state_anxiety"] = st.radio(
        "",
        [1, 2, 3, 4, 5],
        format_func=lambda x: f"{x} = {['Not at all anxious','Slightly anxious','Moderately anxious','Very anxious','Extremely anxious'][x-1]}",
        key="neu_anxiety"
    )

    navigation_buttons(prev_step="session_emp", next_step="open")

# -----------------------------
# Open-Ended Feedback
# -----------------------------
if st.session_state["step"] == "open":
    st.header("Open-Ended Qualitative Questions")
    st.write("**Feel free to write as much as you like; there are no right or wrong answers.**")
    st.write("**Empathetic Voice Experience**")
    st.session_state["open_emp"] = st.text_area("Q29.How did you feel during and after interacting with the empathetic AI voice? What kinds of emotions, thoughts, or reactions did it bring up for you? ")
    st.write("**Neutral Voice Experience**")
    st.session_state["open_neu"] = st.text_area("Q30.How did you feel during and after interacting with the neutral or robotic AI voice? What kinds of emotions, thoughts, or reactions did it bring up for you?")
    st.write("**Comparison of voices**")
    st.session_state["open_compare"] = st.text_area("Q31.What differences, if any, did you notice between the two voices in terms of how they made you feel? Which one made you feel more comfortable or anxious, and why? ")
    st.write("**Voice Preference**")
    st.session_state["open_pref"] = st.text_area("Q32.Which voice did you prefer overall? What specific features (tone, pace, warmth, etc.) did you like or dislike about each voice?")
    st.write("**Perceived Empathy and Understanding**")
    st.session_state["open_empathy"] = st.text_area("Q33.Did the empathetic voice make you feel understood or cared for in any way? If so, can you describe a moment or response that gave you that feeling? ")
    st.write("**Trust & Usefulness**")
    st.session_state["open_trust"] = st.text_area("Q34.Did you feel that either voice was trustworthy or helpful? Why or why not? In what ways did the voice help (or fail to help) you feel supported? ")
    st.write("**Triggers and Discomfort**")
    st.session_state["open_triggers"] = st.text_area("Q35.Was there anything in either voice interaction that made you feel uneasy, anxious, or emotionally uncomfortable? Please explain if so.")
    st.write("**Improvement Suggestions**")
    st.session_state["open_improve"] = st.text_area("Q36.If you could improve or change anything about the voices or how the interaction worked, what would you recommend to make it more helpful or emotionally supportive? ")
    st.write("**Additional Reflections**")
    st.session_state["open_more_1"] = st.text_area("Q37.Is there anything else you'd like to share about your experience in this study?")
    st.session_state["open_more_2"] = st.text_area("Q38.Any thoughts that haven't been covered by the previous questions?")

    navigation_buttons(prev_step="session_neu", next_step="review", next_label="Review & Submit ➡")

# -----------------------------
# Review & Submit
# -----------------------------
if st.session_state["step"] == "review":
    st.header("Review & Submit")
    st.write("Click **Submit** to upload your responses")
    if st.button("Submit"):
        record = {
            "participant_id": st.session_state["participant_id"], 
            "start_ts_utc": st.session_state["start_ts"], 
            "submit_ts_utc": datetime.utcnow().isoformat(),
            "age": st.session_state["age"], "gender": st.session_state["gender"], 
            "gender_other": st.session_state.get("gender_other", ""), 
            "education": st.session_state["education"],
            "voice_exp": st.session_state["voice_exp"], "used_assistants": st.session_state["used_assistants"],
            "tech_comfort": st.session_state["tech_comfort"], 
            "single_mood": st.session_state["single_mood"]
        }
        for i in range(1,8): record[f"gad_q{i}"]=st.session_state["gad"][f"q{i}"]
        record["gad_impact"]=st.session_state["gad_impact"]
        for i in range(1,11): record[f"panas_q{i}"]=st.session_state["panas"][f"q{i}"]
        for i in range(1,9): record[f"emp_q{i}"]=st.session_state["emp"][f"q{i}"]
        record["emp_state_anxiety"]=st.session_state["emp_state_anxiety"]
        for i in range(1,8): record[f"emp_post_q{i}"]=st.session_state["emp_post"][f"q{i}"]
        for i in range(1,9): record[f"neu_q{i}"]=st.session_state["neu"][f"q{i}"]
        record["neu_state_anxiety"]=st.session_state["neu_state_anxiety"]
        for i in range(1,8): record[f"neu_post_q{i}"]=st.session_state["neu_post"][f"q{i}"]
        record.update({k: st.session_state[k] for k in ["open_emp","open_neu","open_compare","open_pref",
                                                         "open_empathy","open_trust","open_triggers",
                                                         "open_improve","open_more_1","open_more_2"]})
        try:
            existing_df = load_existing_hf_csv(HF_DATASET_REPO, HF_DATASET_PATH)
            updated_df = pd.concat([existing_df,pd.DataFrame([record])],ignore_index=True)
            upload_csv_to_hf(updated_df, HF_DATASET_REPO, HF_DATASET_PATH)
            st.success("Submitted successfully!")
        except Exception as e:
            st.error(f"Upload failed: {e}")
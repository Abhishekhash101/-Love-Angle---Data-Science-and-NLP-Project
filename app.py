import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import advertools as adv
from collections import Counter
from transformers import pipeline, AutoTokenizer
import random

# ==========================================
# 1. PAGE CONFIG & CACHING
# ==========================================
st.set_page_config(page_title="Love Angle", page_icon="❤️", layout="wide")

@st.cache_resource
def load_ai_models():
    """
    Loads the 'Dream Team' Ensemble.
    """
    try:
        # 1. HINGLISH SENTIMENT (Slang Specialist)
        hinglish_name = "rohanrajpal/bert-base-multilingual-codemixed-cased-sentiment"
        # We assume top_k=1 returns a list of lists [[{'label':...}]]
        tokenizer = AutoTokenizer.from_pretrained(hinglish_name)
        sentiment_pipe = pipeline("text-classification", model=hinglish_name, tokenizer=tokenizer, top_k=1)

        # 2. EMOTION (Keyword Specialist)
        emotion_pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)

        # 3. CONTEXT (Deep Meaning / Zero-Shot)
        context_pipe = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        
        return sentiment_pipe, emotion_pipe, context_pipe, tokenizer
    except Exception as e:
        st.error(f"Error loading AI Models: {e}")
        return None, None, None, None

# ==========================================
# 2. PARSING & ANALYTICS
# ==========================================
def parse_whatsapp_chat(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.splitlines()
    
    dates, times, users, messages = [], [], [], []
    # Regex to handle "12/05/2024, 10:30 pm - User: Message"
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}[\s\u202F]?[APap][Mm])\s-\s(.*?):\s(.*)'
    
    for line in lines:
        line = line.strip()
        if "end-to-end encrypted" in line: continue
        match = re.search(pattern, line)
        if match:
            dates.append(match.group(1)); times.append(match.group(2))
            users.append(match.group(3)); messages.append(match.group(4))
        else:
            if messages: messages[-1] += " " + line # Append multiline

    df = pd.DataFrame({'Date': dates, 'Time': times, 'User': users, 'Message': messages})
    if df.empty: return df

    # DateTime Cleaning
    df['Time'] = df['Time'].str.replace('\u202F', ' ', regex=True)
    df['full_date'] = df['Date'] + ' ' + df['Time']
    df['dt'] = pd.to_datetime(df['full_date'], dayfirst=False, errors='coerce')
    # Fallback for DD/MM format if parsing fails
    if df['dt'].isnull().all():
        df['dt'] = pd.to_datetime(df['full_date'], dayfirst=True, errors='coerce')
    
    df = df.dropna(subset=['dt'])
    df['hour'] = df['dt'].dt.hour
    df['date_only'] = df['dt'].dt.date
    return df

def generate_visual_stats(df, tokenizer):
    """
    Generates Emoji, Word, and Time stats.
    """
    stats = {}
    
    # A. TOP 5 EMOJIS
    all_text = " ".join(df['Message'].astype(str).tolist())
    emoji_summary = adv.extract_emoji([all_text])
    
    if emoji_summary['emoji_flat']:
        emoji_counts = Counter(emoji_summary['emoji_flat'])
        stats['top_emojis'] = emoji_counts.most_common(5)
    else:
        stats['top_emojis'] = []

    # B. TOP 10 WORDS
    stop_words = set(['the', 'and', 'to', 'is', 'of', 'in', 'hai', 'ka', 'ki', 'ko', 'me', 'mai', 'se', 'kya', 'bhi', 'ho', 'media', 'omitted', 'image', 'video', 'message', 'deleted', 'this'])
    
    # Use tokenizer to split correctly (handles punctuation better than .split())
    # Truncate text if massive to prevent memory freeze
    tokens = tokenizer.tokenize(all_text[:500000]) 
    
    clean_tokens = [t.replace('##', '').lower() for t in tokens if t.isalpha() and len(t) > 3]
    filtered_tokens = [t for t in clean_tokens if t not in stop_words]
    word_counts = Counter(filtered_tokens)
    stats['top_words'] = word_counts.most_common(10)
    
    # C. HOURLY
    hourly_counts = df['hour'].value_counts().sort_index().reset_index()
    hourly_counts.columns = ['Hour', 'Count']
    stats['hourly_activity'] = hourly_counts
    
    # D. DAILY (Timeline)
    daily_counts = df.groupby('date_only').size().reset_index(name='Count')
    stats['daily_activity'] = daily_counts
    
    return stats

# ==========================================
# 3. PILLAR LOGIC (Calculations)
# ==========================================

def calculate_pillar_1(df):
    """Pillar 1: Response Time & Media Ratio"""
    scores = np.zeros(3); log = []
    
    # 1. Response Time
    chat = df[~df['User'].str.contains("System", na=False)].sort_values('dt')
    chat['prev_user'] = chat['User'].shift(1)
    chat['time_diff'] = (chat['dt'] - chat['dt'].shift(1)).dt.total_seconds() / 60
    
    replies = chat[(chat['User'] != chat['prev_user']) & (chat['time_diff'] <= 720)]
    avg_time = replies['time_diff'].mean() if not replies.empty else 0
    
    if avg_time > 0:
        if avg_time < 5: scores[0] += 20; log.append(f"Fast Replies ({avg_time:.1f}m)")
        elif avg_time > 300: scores[2] += 30; log.append(f"Slow Replies ({avg_time:.1f}m)")
        else: scores[1] += 10; log.append(f"Normal Pace ({avg_time:.1f}m)")
        
    # 2. Media Ratio
    media_count = df['Message'].str.contains("Media omitted", case=False).sum()
    total = len(df)
    ratio = (media_count / total * 100) if total > 0 else 0
    
    log.append(f"Media Share: {ratio:.1f}%")
    if ratio > 15: scores[1] += 20; scores[0] += 15
    elif ratio < 2: scores[0] -= 10
    
    return scores, log

def calculate_pillar_2(df):
    """Pillar 2: Night Owl Index"""
    scores = np.zeros(3); log = []
    if df.empty: return scores, log
    
    night_msgs = df[(df['hour'] >= 0) & (df['hour'] < 4)].shape[0]
    night_pct = (night_msgs / len(df)) * 100
    
    if night_pct > 15: scores[0] += 25; log.append(f"Night Owl ({night_pct:.1f}%) -> Love")
    elif night_pct < 5: scores[1] += 10; log.append(f"Daytime Chat ({night_pct:.1f}%) -> Friends")
    
    return scores, log

def calculate_pillar_3_ensemble(df, s_pipe, e_pipe, c_pipe):
    """Pillar 3: The AI Ensemble (FIXED)"""
    scores = np.zeros(3); log = []
    
    # Filter for "Rich" text
    mask = (~df['Message'].str.contains("Media omitted")) & (df['Message'].apply(lambda x: len(str(x).split()) > 3))
    rich_texts = df[mask]['Message'].tolist()
    
    if not rich_texts: return scores, ["No rich text for AI"]

    # Sample for speed (Top 50 messages)
    sample_size = min(len(rich_texts), 50)
    samples = random.sample(rich_texts, sample_size)
    longest_text = max(samples, key=len) 
    
    # A. SENTIMENT (Hinglish)
    # -----------------------
    # FIX: pipeline(top_k=1) returns [[{'label':...}]]. We need r[0]['label']
    s_results = s_pipe(samples)
    
    pos_count = sum(1 for r in s_results if r[0]['label'] == 'LABEL_2')
    neg_count = sum(1 for r in s_results if r[0]['label'] == 'LABEL_0')
    
    log.append(f"Sentiment: {pos_count} Pos / {neg_count} Neg")
    if pos_count > neg_count: scores[0] += 15; scores[1] += 10
    else: scores[1] += 15; scores[2] += 15
    
    # B. EMOTION (Keyword)
    # --------------------
    e_results = e_pipe(samples)
    # FIX: Access r[0]['label'] here too
    e_counts = Counter([r[0]['label'] for r in e_results])
    
    if e_counts:
        top_emo = e_counts.most_common(1)[0][0]
        log.append(f"Dominant Emotion: {top_emo}")
        
        if top_emo in ['joy', 'love']: scores[0] += 20
        elif top_emo == 'surprise': scores[0] += 10
        elif top_emo == 'sadness': scores[2] += 25
        elif top_emo in ['anger', 'fear']: scores[1] += 15
    else:
        log.append("Dominant Emotion: Neutral")
    
    # C. CONTEXT (Real Life Labels)
    # -----------------------------
    labels = [
        "romantic love and passion", "deep care and affection",
        "playful joking and teasing", "casual updates and logistics",
        "sadness grief and heartbreak", "anger conflict and fighting",
        "professional work and business"
    ]
    
    # Zero-shot pipeline returns a single dict for single text input, so no [0] needed on result
    c_res = c_pipe(longest_text, labels, multi_label=False)
    top_ctx = c_res['labels'][0]
    conf = c_res['scores'][0]
    
    log.append(f"Context: {top_ctx.upper()} ({conf:.2f})")
    
    if top_ctx == "romantic love and passion": scores[0] += 50
    elif top_ctx == "deep care and affection": scores[0] += 40
    elif top_ctx == "playful joking and teasing": scores[1] += 50
    elif top_ctx == "casual updates and logistics": scores[1] += 30
    elif top_ctx == "sadness grief and heartbreak": scores[2] += 50
    elif top_ctx == "anger conflict and fighting": scores[2] += 40
    elif top_ctx == "professional work and business": scores[0] -= 30; scores[1] += 40
    
    return scores, log

# ==========================================
# 4. MAIN UI
# ==========================================
st.title("💓Love Angle")

# Sidebar for Models
with st.sidebar:
    st.header("AI System Status")
    # Auto-load on start
    s_pipe, e_pipe, c_pipe, tok = load_ai_models()
    if s_pipe: 
        st.success("Models Ready ✅")
        st.warning("NLP Model can make mistakes ⚠️")
    else:
        st.error("Models Failed ❌")
        if st.button("Retry Load"):
            st.rerun()

uploaded_file = st.file_uploader("Upload Chat (.txt)", type=["txt"])

if uploaded_file and s_pipe:
    with st.spinner("Parsing & Analyzing..."):
        df = parse_whatsapp_chat(uploaded_file)
        
        if not df.empty:
            # 1. RUN PILLARS
            final_scores = np.array([0.0, 0.0, 0.0])
            
            p1_s, p1_l = calculate_pillar_1(df)
            final_scores += p1_s
            
            p2_s, p2_l = calculate_pillar_2(df)
            final_scores += p2_s
            
            p3_s, p3_l = calculate_pillar_3_ensemble(df, s_pipe, e_pipe, c_pipe)
            final_scores += p3_s
            
            # 2. GENERATE STATS
            stats = generate_visual_stats(df, tok)
            
            # 3. DISPLAY DASHBOARD
            
            # --- ROW 1: RESULTS ---
            st.divider()
            col_res, col_graph = st.columns([0.4, 0.6])
            
            with col_res:
                st.subheader("🔮 Final Prediction")
                exp_scores = np.exp(final_scores - np.max(final_scores))
                probs = exp_scores / np.sum(exp_scores)
                
                labels = ["❤️ Love", "🤝 Friends", "💔 One-Sided"]
                winner_idx = np.argmax(probs)
                winner = labels[winner_idx]
                
                st.metric("Probability", f"{probs[winner_idx]*100:.1f}%")
                if winner == "Love": st.balloons()
                st.success(f"Verdict: **{winner}**")
                
                st.write("---")
                st.progress(float(probs[0]), text=f"Love: {probs[0]*100:.1f}%")
                st.progress(float(probs[1]), text=f"Friends: {probs[1]*100:.1f}%")
                st.progress(float(probs[2]), text=f"One-Sided: {probs[2]*100:.1f}%")

            with col_graph:
                st.subheader("📈 Chat Timeline")
                fig_time = px.line(stats['daily_activity'], x='date_only', y='Count', title='Messages Per Day')
                st.plotly_chart(fig_time, use_container_width=True)

            # --- ROW 2: AI LOGS & ACTIVITY ---
            st.divider()
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("🧠 NLP Logic Logs")
                st.info("**Pillar 1 (Response):** " + " | ".join(p1_l))
                st.info("**Pillar 2 (Activity):** " + " | ".join(p2_l))
                st.info("**Pillar 3 (Ensemble):** " + " | ".join(p3_l))
                
            with c2:
                st.subheader("🕒 Max Active Time")
                fig_hour = px.bar(stats['hourly_activity'], x='Hour', y='Count', title="Hourly Activity")
                st.plotly_chart(fig_hour, use_container_width=True)

            # --- ROW 3: EMOJIS & WORDS ---
            st.divider()
            c3, c4 = st.columns(2)
            
            with c3:
                st.subheader("😂 Top 5 Emojis")
                if stats['top_emojis']:
                    emo_df = pd.DataFrame(stats['top_emojis'], columns=['Emoji', 'Count'])
                    fig_emo = px.bar(emo_df, x='Emoji', y='Count', color='Count')
                    st.plotly_chart(fig_emo, use_container_width=True)
                else:
                    st.write("No emojis found.")
                    
            with c4:
                st.subheader("🔤 Top Words")
                if stats['top_words']:
                    word_df = pd.DataFrame(stats['top_words'], columns=['Word', 'Count'])
                    fig_word = px.bar(word_df, x='Count', y='Word', orientation='h')
                    st.plotly_chart(fig_word, use_container_width=True)
                else:
                    st.write("Not enough words.")
            
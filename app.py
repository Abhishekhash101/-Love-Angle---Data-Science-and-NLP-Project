import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import advertools as adv
from collections import Counter
from transformers import pipeline, AutoTokenizer
import random
import nltk
from nltk.corpus import stopwords

# ==========================================
# 1. PAGE CONFIG & SETUP
# ==========================================
st.set_page_config(page_title="Love Angle", page_icon="❤️", layout="wide")

# Download NLTK data (Cached so it runs only once)
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return set(stopwords.words('english'))

english_stops = setup_nltk()

@st.cache_resource
def load_ai_models():
    """
    Loads the 3 Models and we will apply Ensemble technique.
    """
    try:
        # 1. HINGLISH SENTIMENT
        hinglish_name = "rohanrajpal/bert-base-multilingual-codemixed-cased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(hinglish_name)
        sentiment_pipe = pipeline("text-classification", model=hinglish_name, tokenizer=tokenizer, top_k=1)

        # 2. EMOTION
        emotion_pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)

        # 3. CONTEXT
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
    
    # Universal Patterns
    pat_android = r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202F]?[APap][Mm])?)\s-\s(.*?):\s(.*)'
    pat_ios = r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202F]?[APap][Mm])?)\]\s(.*?):\s(.*)'

    for line in lines:
        line = line.strip()
        if "end-to-end encrypted" in line: continue
        
        match = re.search(pat_android, line)
        if not match: match = re.search(pat_ios, line)
            
        if match:
            dates.append(match.group(1)); times.append(match.group(2))
            users.append(match.group(3)); messages.append(match.group(4))
        else:
            if messages: messages[-1] += " " + line

    df = pd.DataFrame({'Date': dates, 'Time': times, 'User': users, 'Message': messages})
    if df.empty: return df

    df['Time'] = df['Time'].str.replace('\u202F', ' ', regex=True).str.strip()
    df['full_date'] = df['Date'] + ' ' + df['Time']
    
    df['dt'] = pd.to_datetime(df['full_date'], dayfirst=False, errors='coerce')
    if df['dt'].isnull().all():
        df['dt'] = pd.to_datetime(df['full_date'], dayfirst=True, errors='coerce')
    
    df = df.dropna(subset=['dt'])
    df['hour'] = df['dt'].dt.hour
    df['date_only'] = df['dt'].dt.date
    return df

def generate_visual_stats(df, tokenizer, english_stops):
    """
    Generates Emoji, Word, and Time stats using NLTK + Custom Filters.
    """
    stats = {}
    
    all_text = " ".join(df['Message'].astype(str).tolist())
    
    # A. EMOJIS
    emoji_summary = adv.extract_emoji([all_text])
    if emoji_summary['emoji_flat']:
        stats['top_emojis'] = Counter(emoji_summary['emoji_flat']).most_common(5)
    else:
        stats['top_emojis'] = []

    # B. TOP WORDS (NLTK + HINGLISH)
    # 1. Define Custom Hinglish/WhatsApp Junk
    hinglish_stops = {'hai', 'ka', 'ki', 'ko', 'me', 'mai', 'se', 'kya', 'bhi', 'ho', 'aur', 'kar', 'tha', 'thi', 'bhai', 'tu', 'tum', 'media', 'omitted', 'image', 'video', 'message', 'deleted'}
    
    # 2. Combine NLTK English + Hinglish
    final_stop_words = english_stops.union(hinglish_stops)
    
    # 3. Tokenize & Filter
    tokens = tokenizer.tokenize(all_text[:500000]) 
    clean_tokens = [t.replace('##', '').lower() for t in tokens if t.isalpha() and len(t) > 2]
    
    filtered_tokens = [t for t in clean_tokens if t not in final_stop_words]
    
    stats['top_words'] = Counter(filtered_tokens).most_common(10)
    
    # C. TIME STATS
    stats['hourly_activity'] = df['hour'].value_counts().sort_index().reset_index()
    stats['hourly_activity'].columns = ['Hour', 'Count']
    stats['daily_activity'] = df.groupby('date_only').size().reset_index(name='Count')
    
    return stats

# ==========================================
# 3. PILLAR LOGIC
# ==========================================

def calculate_pillar_1(df):
    scores = np.zeros(3); log = []
    
    chat = df[~df['User'].str.contains("System", na=False)].sort_values('dt')
    chat['prev_user'] = chat['User'].shift(1)
    chat['time_diff'] = (chat['dt'] - chat['dt'].shift(1)).dt.total_seconds() / 60
    
    replies = chat[(chat['User'] != chat['prev_user']) & (chat['time_diff'] <= 720)]
    avg_time = replies['time_diff'].mean() if not replies.empty else 0
    
    if avg_time > 0:
        if avg_time < 5: scores[0] += 20; log.append(f"⚡ Fast Replies ({avg_time:.1f}m)")
        elif avg_time > 300: scores[2] += 30; log.append(f"🐌 Slow Replies ({avg_time:.1f}m)")
        else: scores[1] += 10; log.append(f"🕒 Normal Pace ({avg_time:.1f}m)")
        
    media_count = df['Message'].str.contains("Media omitted", case=False).sum()
    ratio = (media_count / len(df) * 100) if len(df) > 0 else 0
    
    log.append(f"📸 Media Share: {ratio:.1f}%")
    if ratio > 15: scores[1] += 20; scores[0] += 15
    elif ratio < 2: scores[0] -= 10
    return scores, log

def calculate_pillar_2(df):
    scores = np.zeros(3); log = []
    if df.empty: return scores, log
    
    night_msgs = df[(df['hour'] >= 0) & (df['hour'] < 4)].shape[0]
    night_pct = (night_msgs / len(df)) * 100
    
    if night_pct > 15: scores[0] += 25; log.append(f"🦉 Night Owl ({night_pct:.1f}%) -> Love")
    elif night_pct < 5: scores[1] += 10; log.append(f"☀️ Daytime ({night_pct:.1f}%) -> Friends")
    return scores, log

def calculate_pillar_3_ensemble(df, s_pipe, e_pipe, c_pipe):
    scores = np.zeros(3); log = []
    
    mask = (~df['Message'].str.contains("Media omitted")) & (df['Message'].apply(lambda x: len(str(x).split()) > 2))
    rich_texts = df[mask]['Message'].tolist()
    
    if not rich_texts: return scores, ["No rich text for AI"]

    samples = random.sample(rich_texts, min(len(rich_texts), 50))
    longest_text = max(samples, key=len) 
    
    # A. Sentiment
    s_results = s_pipe(samples)
    pos = sum(1 for r in s_results if r[0]['label'] == 'LABEL_2')
    neg = sum(1 for r in s_results if r[0]['label'] == 'LABEL_0')
    log.append(f"📊 Sentiment: {pos} Pos / {neg} Neg")
    if pos > neg: scores[0] += 15; scores[1] += 10
    else: scores[1] += 15; scores[2] += 15
    
    # B. Emotion
    e_results = e_pipe(samples)
    e_counts = Counter([r[0]['label'] for r in e_results])
    if e_counts:
        top_emo = e_counts.most_common(1)[0][0]
        log.append(f"😊 Dominant Emotion: {top_emo}")
        if top_emo in ['joy', 'love']: scores[0] += 20
        elif top_emo == 'sadness': scores[2] += 25
        elif top_emo in ['anger', 'fear']: scores[1] += 15
    
    # C. Context
    labels = [
        "romantic love and passion", "deep care and affection",
        "playful joking and teasing", "casual updates and logistics",
        "sadness grief and heartbreak", "anger conflict and fighting",
        "professional work and business"
    ]
    c_res = c_pipe(longest_text, labels, multi_label=False)
    top_ctx = c_res['labels'][0]
    
    log.append(f"🧠 Context: {top_ctx.upper()}")
    
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
st.title("💓 Love Angle")

with st.sidebar:
    st.header("NLP Models Load")
    s_pipe, e_pipe, c_pipe, tok = load_ai_models()
    if s_pipe: 
        st.success("Models Ready ✅")
        st.warning("Love Angle can make mistakes ⚠️")
    else: st.error("Models Failed ❌")

uploaded_file = st.file_uploader("Upload Chat (.txt)", type=["txt"])

if uploaded_file and s_pipe:
    with st.spinner("Analyzing..."):
        df = parse_whatsapp_chat(uploaded_file)
        
        if not df.empty:
            final_scores = np.array([0.0, 0.0, 0.0])
            
            p1_s, p1_l = calculate_pillar_1(df)
            final_scores += p1_s
            p2_s, p2_l = calculate_pillar_2(df)
            final_scores += p2_s
            p3_s, p3_l = calculate_pillar_3_ensemble(df, s_pipe, e_pipe, c_pipe)
            final_scores += p3_s
            
            # Pass english_stops to stats generator
            stats = generate_visual_stats(df, tok, english_stops)
            
            # --- RESULTS ---
            st.divider()
            col_res, col_graph = st.columns([0.4, 0.6])
            
            with col_res:
                st.subheader("🔮 Verdict")
                exp_scores = np.exp(final_scores - np.max(final_scores))
                probs = exp_scores / np.sum(exp_scores)
                
                labels = ["❤️ Love", "🤝 Friends", "💔 One-Sided"]
                winner_idx = np.argmax(probs)
                winner = labels[winner_idx]
                
                st.metric("Probability", f"{probs[winner_idx]*100:.1f}%")
                if winner == "Love": st.balloons()
                st.success(f"Result: **{winner}**")
                
                st.write("---")
                st.progress(float(probs[0]), text=f"Love: {probs[0]*100:.1f}%")
                st.progress(float(probs[1]), text=f"Friends: {probs[1]*100:.1f}%")
                st.progress(float(probs[2]), text=f"One-Sided: {probs[2]*100:.1f}%")

            with col_graph:
                st.subheader("📈 Chat Timeline")
                fig_time = px.line(stats['daily_activity'], x='date_only', y='Count', title='Messages Per Day')
                st.plotly_chart(fig_time, use_container_width=True)

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🧠 NLP Logs")
                st.info(" | ".join(p1_l))
                st.info(" | ".join(p2_l))
                st.info(" | ".join(p3_l))
            with c2:
                st.subheader("🕒 Activity")
                fig_hour = px.bar(stats['hourly_activity'], x='Hour', y='Count')
                st.plotly_chart(fig_hour, use_container_width=True)

            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("😂 Top Emojis")
                if stats['top_emojis']:
                    emo_df = pd.DataFrame(stats['top_emojis'], columns=['Emoji', 'Count'])
                    st.plotly_chart(px.bar(emo_df, x='Emoji', y='Count', color='Count'), use_container_width=True)
            with c4:
                st.subheader("🔤 Top Words")
                if stats['top_words']:
                    word_df = pd.DataFrame(stats['top_words'], columns=['Word', 'Count'])
                    st.plotly_chart(px.bar(word_df, x='Count', y='Word', orientation='h'), use_container_width=True)
        else:
            st.error("Could not parse the chat file. Please ensure it is a valid WhatsApp .txt export.")


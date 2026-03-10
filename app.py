import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import tweepy
import matplotlib.pyplot as plt
import cv2
import numpy as np
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
import time

# ── NLTK Downloads ────────────────────────────────────────────────────────────
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    return stopwords.words('english')

# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("model.pkl or vectorizer.pkl not found. Place them in the same folder as app.py.")
        return None, None

# ── Core: Predict Sentiment ───────────────────────────────────────────────────
def predict_sentiment(text, model, vectorizer, stop_words):
    cleaned = re.sub('[^a-zA-Z]', ' ', text).lower()
    words   = [w for w in cleaned.split() if w not in stop_words]
    cleaned = " ".join(words)
    if not cleaned.strip():
        return "Neutral", 0.5
    vec  = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0][1]
    if prob >= 0.65:
        return "Positive", prob
    elif prob <= 0.35:
        return "Negative", prob
    else:
        return "Neutral", prob

# ── Multi-Sentence Sentiment ──────────────────────────────────────────────────
def analyse_multi_sentiment(text, model, vectorizer, stop_words):
    sentences = sent_tokenize(text)
    results   = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentiment, prob = predict_sentiment(sentence, model, vectorizer, stop_words)
        if sentiment == "Positive":
            color, emoji = "#4CAF50", "😊"
        elif sentiment == "Negative":
            color, emoji = "#F44336", "😞"
        else:
            color, emoji = "#FFBF00", "😐"
        results.append({
            "sentence" : sentence,
            "sentiment": sentiment,
            "prob"     : prob,
            "color"    : color,
            "emoji"    : emoji,
        })
    pos = sum(1 for r in results if r["sentiment"] == "Positive")
    neg = sum(1 for r in results if r["sentiment"] == "Negative")
    neu = sum(1 for r in results if r["sentiment"] == "Neutral")
    return results, pos, neg, neu

# ── Sentence Cards ─────────────────────────────────────────────────────────────
def render_sentence_cards(results):
    st.markdown("#### 🔍 Sentence-by-Sentence Breakdown")
    for i, r in enumerate(results, 1):
        confidence = abs(r["prob"] - 0.5) * 200
        bar_width  = int(confidence)
        html = f"""
        <div style="background:{r['color']}22; border-left:5px solid {r['color']};
                    padding:12px 16px; border-radius:8px; margin:8px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:12px; font-weight:700; color:{r['color']}; text-transform:uppercase; letter-spacing:1px;">
                    {r['emoji']} {r['sentiment']}
                </span>
                <span style="font-size:11px; color:#888;">Sentence {i}</span>
            </div>
            <p style="margin:0 0 8px 0; font-size:14px; line-height:1.6; color:#e0e0e0;">{r['sentence']}</p>
            <div style="background:#333; border-radius:4px; height:6px; width:100%;">
                <div style="background:{r['color']}; width:{bar_width}%; height:6px; border-radius:4px;"></div>
            </div>
            <span style="font-size:11px; color:#888;">Confidence: {confidence:.0f}%</span>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────────
def display_sentiment_charts(positive_count, negative_count, neutral_count):
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_count, negative_count, neutral_count]
    colors = ['#4CAF50', '#F44336', '#FFBF00']
    total  = sum(values)

    st.markdown("#### 📊 Sentiment Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(facecolor='#0e1117')
        ax.set_facecolor('#1a1d27')
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='none')
        for bar, val in zip(bars, values):
            pct = (val / total * 100) if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{val} ({pct:.0f}%)", ha='center', va='bottom',
                    color='white', fontsize=10, fontweight='bold')
        ax.set_ylabel("Count", color='#aaa')
        ax.set_title("Bar Chart", color='white', pad=10)
        ax.tick_params(colors='white')
        ax.spines[:].set_visible(False)
        ax.yaxis.grid(True, color='#333', linewidth=0.5)
        ax.set_axisbelow(True)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        if total > 0:
            fig2, ax2 = plt.subplots(facecolor='#0e1117')
            wedges, texts, autotexts = ax2.pie(
                values, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90,
                wedgeprops=dict(edgecolor='#0e1117', linewidth=2)
            )
            for t in texts:
                t.set_color('white')
            for at in autotexts:
                at.set_color('white')
                at.set_fontweight('bold')
            ax2.set_title("Pie Chart", color='white', pad=10)
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("No data to display.")

# ── Twitter Helpers ───────────────────────────────────────────────────────────
def get_tweets_tweepy(username, bearer_token, count=10):
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        user   = client.get_user(username=username)
        if not user.data:
            st.error(f"User '@{username}' not found.")
            return []
        tweets = client.get_users_tweets(
            id=user.data.id, max_results=count,
            tweet_fields=['created_at', 'text', 'public_metrics']
        )
        if tweets.data:
            return [{
                'text'    : t.text,
                'date'    : t.created_at,
                'id'      : t.id,
                'likes'   : t.public_metrics['like_count'],
                'retweets': t.public_metrics['retweet_count'],
            } for t in tweets.data]
        st.warning(f"No tweets found for '@{username}' or account is private.")
        return []
    except tweepy.TooManyRequests:
        st.error("Rate limit exceeded. Wait a few minutes and try again.")
        return []
    except tweepy.Unauthorized:
        st.error("Invalid Bearer Token.")
        return []
    except Exception as e:
        st.error(f"Error: {e}")
        return []

def create_card(tweet_text, sentiment, likes=0, retweets=0):
    cfg = {
        "Positive": ("#4CAF50", "😊"),
        "Negative": ("#F44336", "😞"),
        "Neutral" : ("#FFBF00", "😐"),
    }
    color, emoji = cfg.get(sentiment, cfg["Neutral"])
    return f"""
    <div style="background:{color}22; border-left:5px solid {color};
                padding:14px 18px; border-radius:10px; margin:8px 0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="color:{color}; font-weight:700; font-size:13px;">{emoji} {sentiment}</span>
            <span style="color:#888; font-size:12px;">❤️ {likes} &nbsp; 🔄 {retweets}</span>
        </div>
        <p style="margin:0; color:#e0e0e0; font-size:14px; line-height:1.6;">{tweet_text}</p>
    </div>
    """

def display_sentiment_stats(tweets_data, model, vectorizer, stop_words):
    if not tweets_data:
        return
    sentiments     = [predict_sentiment(t['text'], model, vectorizer, stop_words)[0] for t in tweets_data]
    positive_count = sentiments.count("Positive")
    negative_count = sentiments.count("Negative")
    neutral_count  = sentiments.count("Neutral")
    total          = len(sentiments)

    display_sentiment_charts(positive_count, negative_count, neutral_count)

    st.subheader("📋 Sentiment Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("😊 Positive", positive_count, f"{positive_count/total*100:.1f}%")
    c2.metric("😞 Negative", negative_count, f"{negative_count/total*100:.1f}%")
    c3.metric("😐 Neutral",  neutral_count,  f"{neutral_count/total*100:.1f}%")

    if positive_count > negative_count and positive_count > neutral_count:
        st.success("**Overall Sentiment: 😊 Positive**")
    elif negative_count > positive_count and negative_count > neutral_count:
        st.error("**Overall Sentiment: 😞 Negative**")
    else:
        st.warning("**Overall Sentiment: 😐 Neutral**")

# ── Face Sentiment Helpers ─────────────────────────────────────────────────────
def emotion_to_sentiment(emotion):
    if emotion.lower() in ['happy', 'surprise']:
        return "Positive"
    elif emotion.lower() in ['sad', 'angry', 'fear', 'disgust']:
        return "Negative"
    else:
        return "Neutral"

SENTIMENT_BGR = {
    "Positive"    : (100, 200,  0),
    "Negative"    : ( 60,  60, 220),
    "Neutral"     : (  0, 200, 200),
    "Detecting...": (180, 180, 180),
}

SENTIMENT_HEX = {
    "Positive"    : "#4CAF50",
    "Negative"    : "#F44336",
    "Neutral"     : "#FFBF00",
    "Detecting...": "#888888",
}

SENTIMENT_EMOJI = {
    "Positive"    : "😊",
    "Negative"    : "😞",
    "Neutral"     : "😐",
    "Detecting...": "🔍",
}

# ── WebRTC Video Processor ─────────────────────────────────────────────────────
class FaceSentimentProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock           = threading.Lock()
        self.last_emotion   = "Detecting..."
        self.last_sentiment = "Detecting..."
        self.frame_count    = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Analyse every 15 frames for smooth performance
        if self.frame_count % 15 == 0:
            try:
                result    = DeepFace.analyze(img, actions=['emotion'],
                                             enforce_detection=False, silent=True)
                emotion   = result[0]['dominant_emotion']
                sentiment = emotion_to_sentiment(emotion)
                with self.lock:
                    self.last_emotion   = emotion.capitalize()
                    self.last_sentiment = sentiment
            except Exception:
                with self.lock:
                    self.last_emotion   = "No face detected"
                    self.last_sentiment = "Neutral"

        with self.lock:
            emotion   = self.last_emotion
            sentiment = self.last_sentiment

        color = SENTIMENT_BGR.get(sentiment, (180, 180, 180))

        # Semi-transparent top bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 95), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Colored accent line
        cv2.rectangle(img, (0, 93), (img.shape[1], 95), color, -1)

        # Text
        cv2.putText(img, f"Emotion:   {emotion}",
                    (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"Sentiment: {sentiment}",
                    (14, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Vybe - Sentiment Analysis", page_icon="🐦", layout="wide")
    st.title("🐦 Vybe — Sentiment Analysis")
    st.markdown(
        "<p style='opacity:0.6;'>Decode the emotional tone of text and faces using AI.</p>",
        unsafe_allow_html=True
    )

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    if model is None or vectorizer is None:
        st.stop()

    # Sidebar
    st.sidebar.header("🔧 Twitter API Config")
    st.sidebar.info("Requires a Bearer Token from [developer.twitter.com](https://developer.twitter.com/)")
    bearer_token = st.sidebar.text_input("Bearer Token", type="password")
    st.sidebar.markdown("""
    ### How to get a Bearer Token:
    1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
    2. Create a free Project & App
    3. Copy Bearer Token from Keys & Tokens
    """)

    tab1, tab2, tab3 = st.tabs([
        "🔍 Analyze User Tweets",
        "📝 Analyze Custom Text",
        "📷 Real-Time Face Sentiment",
    ])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.header("Analyze Twitter User")
        if not bearer_token:
            st.warning("⚠️ Enter your Bearer Token in the sidebar to fetch tweets.")
            st.info("No token? Try the **Custom Text** tab instead!")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                username    = st.text_input("Twitter Username (without @)", placeholder="elonmusk")
                tweet_count = st.slider("Number of tweets to analyze", 1, 20, 10)
            with col2:
                st.write("")
                st.write("")
                if st.button("🚀 Analyze Tweets", type="primary"):
                    if username.strip():
                        with st.spinner(f"Fetching tweets from @{username}..."):
                            tweets_data = get_tweets_tweepy(username.strip(), bearer_token, tweet_count)
                        if tweets_data:
                            st.success(f"✅ Analysed {len(tweets_data)} tweets from @{username}")
                            display_sentiment_stats(tweets_data, model, vectorizer, stop_words)
                            st.subheader("📋 Tweet Cards")
                            for tweet in tweets_data:
                                sentiment, _ = predict_sentiment(tweet['text'], model, vectorizer, stop_words)
                                st.markdown(
                                    create_card(tweet['text'], sentiment,
                                                tweet.get('likes', 0), tweet.get('retweets', 0)),
                                    unsafe_allow_html=True
                                )
                    else:
                        st.warning("Please enter a Twitter username.")

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.header("Analyze Custom Text")
        st.markdown("""
        <div style="background:#1a1d27; border-radius:10px; padding:12px 16px; margin-bottom:16px; border:1px solid #333;">
            <b style="color:#aaa;">💡 Multi-Sentiment Mode</b><br>
            <span style="color:#888; font-size:13px;">
                Your text is split into individual sentences and each one is analysed separately.
                A single paragraph can contain <b style="color:#4CAF50;">Positive</b>,
                <b style="color:#F44336;">Negative</b>, and
                <b style="color:#FFBF00;">Neutral</b> sentiments at the same time!
            </span>
        </div>
        """, unsafe_allow_html=True)

        text_input = st.text_area(
            "Enter text to analyze",
            placeholder="e.g. This app is absolutely amazing! But the loading time is painfully slow. The design looks decent enough I guess.",
            height=160
        )

        if st.button("🔍 Analyse Text", type="primary"):
            if text_input.strip():
                results, pos, neg, neu = analyse_multi_sentiment(
                    text_input, model, vectorizer, stop_words
                )
                total = len(results)
                if total == 0:
                    st.warning("Could not detect any sentences. Please enter more text.")
                else:
                    if pos > neg and pos > neu:
                        st.success(f"😊 **Overall Sentiment: Positive** — {pos}/{total} sentences positive")
                    elif neg > pos and neg > neu:
                        st.error(f"😞 **Overall Sentiment: Negative** — {neg}/{total} sentences negative")
                    else:
                        st.warning(f"😐 **Overall Sentiment: Neutral / Mixed** — mixed signals across sentences")

                    st.divider()
                    render_sentence_cards(results)
                    st.divider()
                    display_sentiment_charts(pos, neg, neu)

                    st.markdown("#### 📊 Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Sentences", total)
                    c2.metric("😊 Positive", pos,  f"{pos/total*100:.0f}%")
                    c3.metric("😞 Negative", neg,  f"{neg/total*100:.0f}%")
                    c4.metric("😐 Neutral",  neu,  f"{neu/total*100:.0f}%")
            else:
                st.warning("Please enter some text to analyse.")

    # ── Tab 3: Face Sentiment ──────────────────────────────────────────────────
    with tab3:
        st.header("📷 Real-Time Face Sentiment Analysis")

        st.markdown("""
        <div style="background:#1a1d27; border-radius:10px; padding:14px 18px; margin-bottom:16px; border:1px solid #333;">
            <b style="color:#aaa;">🎥 How It Works</b><br>
            <span style="color:#888; font-size:13px;">
                Your webcam is analysed in real time using <b style="color:#fff;">DeepFace AI</b>.
                It detects your facial expression every few frames and maps it to a sentiment:<br><br>
                😊 <b style="color:#4CAF50;">Happy / Surprise → Positive</b> &nbsp;|&nbsp;
                😞 <b style="color:#F44336;">Sad / Angry / Fear / Disgust → Negative</b> &nbsp;|&nbsp;
                😐 <b style="color:#FFBF00;">Neutral → Neutral</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Tips row
        t1, t2, t3 = st.columns(3)
        t1.info("💡 Ensure your face is well lit")
        t2.info("💡 Look directly at the camera")
        t3.info("💡 Allow browser camera access")

        st.divider()

        # Emotion legend
        st.markdown("#### 🗺️ Emotion → Sentiment Map")
        emotions_map = [
            ("😊", "Happy",   "Positive", "#4CAF50"),
            ("😲", "Surprise","Positive", "#4CAF50"),
            ("😞", "Sad",     "Negative", "#F44336"),
            ("😠", "Angry",   "Negative", "#F44336"),
            ("😨", "Fear",    "Negative", "#F44336"),
            ("🤢", "Disgust", "Negative", "#F44336"),
            ("😐", "Neutral", "Neutral",  "#FFBF00"),
        ]
        cols = st.columns(7)
        for col, (emo, name, sent, clr) in zip(cols, emotions_map):
            col.markdown(
                f"<div style='text-align:center; background:{clr}22; border:1px solid {clr}; "
                f"border-radius:8px; padding:10px 4px;'>"
                f"<div style='font-size:26px;'>{emo}</div>"
                f"<div style='color:#ddd; font-size:12px; margin:4px 0;'>{name}</div>"
                f"<div style='color:{clr}; font-weight:700; font-size:11px;'>{sent}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()
        st.markdown("#### 🎥 Live Camera Feed")
        st.markdown(
            "<p style='color:#888; font-size:13px;'>Click <b>START</b> to begin. "
            "Emotion and sentiment labels will appear directly on the video feed.</p>",
            unsafe_allow_html=True
        )

        # WebRTC stream
        ctx = webrtc_streamer(
            key="face-sentiment",
            video_processor_factory=FaceSentimentProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Live result card below the video
        if ctx.video_processor:
            result_box = st.empty()
            while ctx.state.playing:
                with ctx.video_processor.lock:
                    emotion   = ctx.video_processor.last_emotion
                    sentiment = ctx.video_processor.last_sentiment

                color = SENTIMENT_HEX.get(sentiment, "#888888")
                emoji = SENTIMENT_EMOJI.get(sentiment, "🔍")

                result_box.markdown(f"""
                <div style="background:{color}22; border:2px solid {color}; border-radius:14px;
                            padding:24px; text-align:center; margin-top:20px;">
                    <div style="font-size:52px; margin-bottom:8px;">{emoji}</div>
                    <div style="color:{color}; font-size:24px; font-weight:800; margin-bottom:6px;">
                        {sentiment}
                    </div>
                    <div style="color:#aaa; font-size:15px;">
                        Detected Emotion: <b style="color:white;">{emotion}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                time.sleep(0.5)
                
    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <hr style="border: none; border-top: 1px solid #333; margin: 0;">
    <div style="text-align: center; padding: 18px 0 8px 0;">
        <span style="color: #555; font-size: 13px;">© 2026 </span>
        <span style="color: #aaa; font-size: 13px; font-weight: 600;">Md Mehebub Alam</span>
        <span style="color: #555; font-size: 13px;">. All rights reserved.</span>
        <br>
        <span style="color: #444; font-size: 11px;">🐦 Vybe — Sentiment Analysis</span>
    </div>
    """, unsafe_allow_html=True)


def show_setup_instructions():
    st.sidebar.markdown("""
    ### 🛠️ Setup Instructions:
    1. Visit [developer.twitter.com](https://developer.twitter.com/)
    2. Create a Project & App (free)
    3. Copy Bearer Token from Keys & Tokens
    4. Paste it above — you are ready!
    """)


if __name__ == "__main__":
    show_setup_instructions()
    main()
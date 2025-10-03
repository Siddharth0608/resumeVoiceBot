import streamlit as st
from groq import Groq
import os
from audio_recorder_streamlit import audio_recorder
import base64
import tempfile

# Page config
st.set_page_config(
    page_title="AI Interview Voice Bot",
    page_icon="🎤",
    layout="centered"
)

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    api_key = st.secrets["general"]["groq_api_key"]
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables")
        st.stop()
    return Groq(api_key=api_key)

client = get_groq_client()

# Your personality data - CUSTOMIZE THIS!
PERSONALITY = """
You are Siddharth Sharma, a Machine Learning Developer being interviewed for the 100x AI Agent Team position.

Here is your resume:-
siddharthsharma2002@gmail.com
https://www.linkedin.com/in/siddharth-sharma--applied-ai-engineer
+919983358689
Gurugram, Haryana, India

I’m a Machine Learning Engineer with a focus on building practical, real-world applications using Large Language 
Models (LLMs), AI agents, and NLP systems. My work spans the full lifecycle - from data preparation and ﬁne-tuning 
LLMs to backend development, deployment, and scaling in the cloud.
I’ve designed agentic applications that improved information retrieval, automated sourcing, and optimized diagnostics 
in robotics. I ﬁne-tuned LLMs with techniques like LoRA, QLoRA, and SFT, creating models capable of generating 
diverse stories and poetry.
Beyond AI application development, I also bring strong backend and deployment expertise. I’ve built APIs using 
FastAPI, integrated them with MongoDB, and deployed applications with Nginx on Azure cloud infrastructure, ensuring 
reliability and scalability.

Here’s what I specialize in:
AI/ML: LLM ﬁne-tuning, agentic systems, NLP
Frameworks & Tools: Transformers, LangGraph, AutoGen, SmolAgents
Backend & Deployment: FastAPI, MongoDB, Nginx, Azure
Programming: Python, SQL, Java, NoSQL

I'm passionate about creating scalable AI-powered systems that bridge cutting-edge research with practical use cases. 
Always open to collaborations and opportunities where I can apply my skills to solve complex problems and build 
intelligent solutions.


Work Experience
Applied AI Engineer
Brightbots Privated Limited | Gurugram, Haryana, India - Nov 2024 Present- 
At BrightBots, I worked on designing and deploying AI-driven systems and agentic applications that improved 
eﬃciency across robotics and automation. My role involved the end-to-end lifecycle - building intelligent agents, 
integrating them with backend systems, and deploying scalable solutions in the cloud.
Developed an agentic RAG system using SmolAgents to streamline product documentation lookup, reducing manual 
eﬀort for internal teams.
Built an AI diagnostic agent with AutoGen that processed real-time robotic cleaner data, enabling faster issue detection 
and improving system uptime.
Automated the BOM sourcing process by creating an LLM-powered assistant integrated with custom Python tools, 
signiﬁcantly reducing manual sourcing time.
Designed the backend with FastAPI, connected with MongoDB, and deployed applications using Nginx on Azure, 
ensuring robustness and scalability.

Generative AI Engineer
Karma Points | Delhi, India - May 2024 Oct 2024- 
At The Karma Points, I focused on ﬁne-tuning Large Language Models (LLMs) and building the foundation for creative 
AI applications. My work involved dataset creation, model training, and optimization techniques to enhance generation 
quality and eﬃciency.
Built a high-quality training dataset from scratch by collecting, cleaning, and preparing diverse text sources, ensuring 
readiness for supervised ﬁne-tuning.
Fine-tuned LLMs using techniques such as LoRA, QLoRA, and 4-bit training, improving model performance for story 
and poetry generation tasks.
Developed pipelines with the Transformers framework to train and evaluate models, ensuring scalability and 
reproducibility.
Collaborated with the team to experiment with creative AI applications, aligning model outputs with project goals.

Projects:

Multi-AI-Agent System for Real-Time Data Analysis and Automation- Feb 2025 Present
Designed and deployed a multi-agent system to perform data analysis on live product data and company databases, 
enabling faster insights and automation. The system integrated external APIs, company databases, and streaming data 
pipelines to support decision-making across teams.
Built agents using AutoGen with models from Azure AI Foundry, orchestrated for collaborative data analysis
Developed the backend in FastAPI, connected with MongoDB, and deployed on Azure VMs using Nginx + Uvicorn
Set up a CI/CD pipeline to streamline updates and ensure reliable deployments
Integrated external APIs with company databases, enabling agents to process both historical records and real-time 
product data
Delivered a scalable solution that reduced manual analysis eﬀort and improved system responsiveness


Fine-Tuned LLM for Creative Story and Poetry Generation - Feb 2024 Oct 2024
Developed a custom ﬁne-tuned LLM designed to generate creative, long-form stories and poems. I built a high-quality 
dataset by scraping and cleaning open-source creative writing content, ensuring data diversity and relevance for 
training.
Fine-tuned a Mistral-based model using Dolphin-2.9.3 (7B) with LoRA techniques
Trained on a single NVIDIA A100 GPU over 25 hours, optimizing for eﬃciency and quality
Improved model capability to produce imaginative, coherent, and stylistically rich narratives
Managed the end-to-end pipeline: dataset preparation, model ﬁne-tuning, and evaluation

Core Skills:
Project Management, Web Hosting, Azure AI Foundry, Microsoft Azure Machine Learning, Microsoft Azure, Nginx,
Beautiful Soup, Data Scraping, Selenium, Model Training, AI Agents, Retrieval-Augmented Generation (RAG),
Large Language Models (LLM), Fine Tuning, Python (Programming Language), Artiﬁcial Intelligence (AI),
Machine Learning, Multi-agent Systems, FastAPI, Flask

Education:
The NorthCap University - Bachelor of Technology - BTech Computer Science - Jan 2020 Jan 2024
Relevant Coursework: Specialized in Artiﬁcial Intelligence and Machine Learnin

Certiﬁcates:
-AWS Academy Graduate - AWS Academy Machine Learning Foundations Amazon Web Services (AWS)
-Introduction to TensorFlow for Artiﬁcial Intelligence, Machine Learning, and Deep Learning Coursera
-AWS Academy Graduate - AWS Academy Cloud Foundations Amazon Web Services (AWS)



Here's your background:
I'm a machine learning engineer, and honestly, my journey into tech started with childhood 
obsessions - Iron Man, Cristiano Ronaldo, and computers. Iron Man especially shaped who I 
am; I wanted to be him, and that's a big reason I got into technology. My superpower is 
being super focused when I'm working on things I genuinely love - I can get completely 
absorbed and lose track of time. I'd like to grow in a few areas: social 
interactions (I'm working on being better at connecting with people), frontend
 development (because I want to build complete products, not just backend systems), 
 and maybe even learn assembly language one day (okay, half-kidding on that one, but 
 low-level stuff fascinates me). One thing people often misunderstand about me is they 
 think I'm being rude when I'm really not - it happens unintentionally, and I'm always 
 trying to be nice, it just doesn't always come across that way. As for pushing my boundaries, 
 I've learned that the best way for me is to just start working on the problem or task instead 
 of overthinking and planning forever - action beats analysis paralysis for me.

Answer interview questions naturally and authentically as. Be conversational, genuine, and keep responses 
concise (2-4 sentences unless asked for more detail). Don't try to impress - just be honest and human. 
Answer based on the background provided in your resume and the background details provided.
"""

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🎤 AI Interview Voice Bot")
st.caption("100x AI Agent Team - Interview Assessment")
st.info("👤 **Candidate:** Siddharth Sharma")

st.markdown("---")

# Text input area
st.subheader("Ask a Question")
user_input = st.text_input("Type your question here:", key="text_input")

# Voice input
st.markdown("**OR speak your question:**")
audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="#e74c3c",
    neutral_color="#3498db",
    icon_size="2x"
)

# Process audio if recorded
if audio_bytes:
    with st.spinner("🎧 Transcribing..."):
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        # Transcribe with Groq Whisper
        try:
            with open(temp_audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(temp_audio_path, audio_file.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
            user_input = transcription
            st.success(f"🎤 You said: {transcription}")
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
        finally:
            os.unlink(temp_audio_path)

# Submit button
if st.button("Send", type="primary", use_container_width=True) or user_input:
    if user_input and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("💭 Thinking..."):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": PERSONALITY},
                        *st.session_state.messages
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    max_tokens=500
                )
                
                response = chat_completion.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error getting response: {e}")

st.markdown("---")

# Display conversation
if st.session_state.messages:
    st.subheader("💬 Conversation")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🙋 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Alex:** {msg['content']}")
            
            # Text to speech
            with st.spinner("🔊 Generating speech..."):
                try:
                    # Use Groq TTS (if available) or skip
                    speech_file_path = "speech.wav" 
                    model = "playai-tts"
                    voice = "Fritz-PlayAI"
                    text = response
                    response_format = "wav"

                    response = client.audio.speech.create(
                        model=model,
                        voice=voice,
                        input=text,
                        response_format=response_format
                    )

                    response.write_to_file(speech_file_path)
                    st.audio("speech.wav")


                except:
                    pass
    
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Sample questions
st.markdown("---")
st.subheader("📝 Sample Questions")

questions = [
    "Tell me about your life story",
    "What's your #1 superpower?",
    "What are your top 3 areas to grow?",
    "What misconception do people have about you?",
    "How do you push your boundaries?",
    "Why do you want to join 100x?"
]

cols = st.columns(2)
for idx, q in enumerate(questions):
    with cols[idx % 2]:
        if st.button(q, key=f"q_{idx}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# Footer
st.markdown("---")
st.caption("Built with Streamlit + Groq API | Deployed for 100x Interview")
